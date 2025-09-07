import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from src.utils.transformer_helper import combine_encoding


class SwiGLU(nn.Module):
    """
    SwiGLU activation function as used in LLaMA models.
    Applies: SiLU(gate(x)) * up(x) -> down()
    """
    def __init__(self, embed_size, dropout_rate=0.0):
        super().__init__()
        # Following LLaMA's convention: intermediate_size = (2/3) * 4 * embed_size
        # This maintains similar parameter count to standard FFN
        intermediate_size = int(2 * (4 * embed_size) / 3)
        # Round to nearest multiple of 8 for efficiency
        intermediate_size = (intermediate_size + 7) // 8 * 8
        
        self.gate_proj = nn.Linear(embed_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(embed_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, embed_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
    def forward(self, x):
        # SwiGLU: gate with SiLU activation
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        intermediate = gate * up
        
        if self.dropout:
            intermediate = self.dropout(intermediate)
            
        return self.down_proj(intermediate)


class FlashMultiheadAttention(nn.Module):
    """
    Multi-head attention using PyTorch's scaled_dot_product_attention
    which automatically uses Flash Attention when available.
    """
    _logged_backend = False  # Class variable to track if we've already logged
    
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Scaling factor for attention scores
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        B, L, D = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # [B, L, 3*D]
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)  # [B, L, 3, H, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is [B, H, L, head_dim]
        
        # One-time logging of which attention backend is being used
        if not FlashMultiheadAttention._logged_backend:
            FlashMultiheadAttention._logged_backend = True
            if q.dtype in [torch.float16, torch.bfloat16]:
                try:
                    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
                        _ = F.scaled_dot_product_attention(q[:1,:1], k[:1,:1], v[:1,:1], is_causal=True)
                    print("🚀 Flash Attention is ENABLED (using Flash SDPA backend)")
                except:
                    try:
                        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]):
                            _ = F.scaled_dot_product_attention(q[:1,:1], k[:1,:1], v[:1,:1], is_causal=True)
                        print("⚡ Memory Efficient Attention is ENABLED")
                    except:
                        print("🔢 Using Math SDPA backend (baseline)")
            else:
                print(f"⚠️ Flash Attention DISABLED - using {q.dtype} (need float16/bfloat16)")
        
        # Apply scaled dot-product attention with causal mask
        # SDPA automatically selects the best implementation (flash, mem-efficient, or math)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,  # Use built-in causal masking
            scale=self.scale
        )  # [B, H, L, head_dim]
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2)  # [B, L, H, head_dim]
        attn_output = attn_output.contiguous().view(B, L, D)  # [B, L, D]
        output = self.out_proj(attn_output)  # [B, L, D]
        
        return output


class MultiQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) / Multi-Query Attention (MQA)
    - When num_kv_heads = 1: MQA (maximum memory savings)
    - When num_kv_heads = num_heads: Standard multi-head attention
    - When 1 < num_kv_heads < num_heads: GQA (balanced tradeoff)
    """
    _logged_backend = False  # Class variable to track if we've already logged
    
    def __init__(self, embed_dim, num_heads, num_kv_heads=1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        
        # Separate projections for Q and KV
        # Q: multi-head projection
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # K, V: grouped projection (num_kv_heads groups)
        self.kv_proj = nn.Linear(embed_dim, 2 * num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Scaling factor for attention scores
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        B, L, D = x.shape
        
        # Project Q with multiple heads
        q = self.q_proj(x)  # [B, L, D]
        q = q.reshape(B, L, self.num_heads, self.head_dim)  # [B, L, H, head_dim]
        q = q.transpose(1, 2)  # [B, H, L, head_dim]
        
        # Project K, V with grouped heads
        kv = self.kv_proj(x)  # [B, L, 2*num_kv_heads*head_dim]
        kv = kv.reshape(B, L, 2, self.num_kv_heads, self.head_dim)  # [B, L, 2, num_kv_heads, head_dim]
        kv = kv.permute(2, 0, 3, 1, 4)  # [2, B, num_kv_heads, L, head_dim]
        k, v = kv[0], kv[1]  # Each is [B, num_kv_heads, L, head_dim]
        
        # Repeat K, V to match number of Q heads if using GQA/MQA
        if self.num_kv_heads < self.num_heads:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)  # [B, H, L, head_dim]
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)  # [B, H, L, head_dim]
        
        # One-time logging of which attention backend is being used
        if not MultiQueryAttention._logged_backend:
            MultiQueryAttention._logged_backend = True
            if q.dtype in [torch.float16, torch.bfloat16]:
                try:
                    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
                        _ = F.scaled_dot_product_attention(q[:1,:1], k[:1,:1], v[:1,:1], is_causal=True)
                    print("🚀 Flash Attention is ENABLED (using Flash SDPA backend)")
                except:
                    try:
                        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]):
                            _ = F.scaled_dot_product_attention(q[:1,:1], k[:1,:1], v[:1,:1], is_causal=True)
                        print("⚡ Memory Efficient Attention is ENABLED")
                    except:
                        print("🔢 Using Math SDPA backend (baseline)")
            else:
                print(f"⚠️ Flash Attention DISABLED - using {q.dtype} (need float16/bfloat16)")
        
        # Apply scaled dot-product attention with causal mask
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,  # Use built-in causal masking
            scale=self.scale
        )  # [B, H, L, head_dim]
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2)  # [B, L, H, head_dim]
        attn_output = attn_output.contiguous().view(B, L, D)  # [B, L, D]
        output = self.out_proj(attn_output)  # [B, L, D]
        
        return output


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout_rate=0.1, num_kv_heads=1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiQueryAttention(embed_dim=embed_size, num_heads=heads, num_kv_heads=num_kv_heads)
        self.norm1 = nn.RMSNorm(embed_size, eps=1e-6)
        self.norm2 = nn.RMSNorm(embed_size, eps=1e-6)
        self.feed_forward = SwiGLU(embed_size, dropout_rate=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        # Pre-LN for attention
        norm_x = self.norm1(x)
        # Flash attention with built-in causal mask
        attention_out = self.attention(norm_x)  # (N, seq_length, embed_size)
        x = self.dropout(attention_out) + x if self.dropout else attention_out + x  # (N, seq_length, embed_size)

        # Pre-LN for feed-forward
        out = self.feed_forward(self.norm2(x)) + x  # (N, seq_length, embed_size)
        return out  # (N, seq_length, embed_size)
    
class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, *,
                 max_length = 2048, dropout_rate=0.05, jupyter_debug=False, num_kv_heads=1):
        super().__init__()
        assert embed_size % 8 == 0
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.max_grid_size = 64 # 30 is the max edge size from the data. add 1 for ROW_SEPARATOR and 1 for buffer, charts id requires more
        self.grid_embedding_size = embed_size // 4
        self.register_buffer("grid_encoding", self.generate_positional_encoding(self.max_grid_size, self.grid_embedding_size, True))

        # Create transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_size, 
                heads, 
                dropout_rate=dropout_rate,
                num_kv_heads=num_kv_heads
            ) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, vocab_size)

        # Add learnable grid_scale parameter
        self.grid_scale = nn.Parameter(torch.tensor([2.4002, 2.6549, 1.2614])) # previously tuned

        # self.embedding.weight = self.fc_out.weight # NOT GOOD! see special_checkpoints/weight_tying

        self.dropout = nn.Dropout(dropout_rate)
        self.heads = heads

    def set_dropout_rate(self, dropout_rate):
        """
        Set the dropout rate for all dropout layers in the model.
        
        Args:
            dropout_rate (float): New dropout probability between 0 and 1
        """
        # Update main dropout layer
        self.dropout.p = dropout_rate
        
        # Update dropout in transformer blocks
        for layer in self.layers:
            # Update dropout in feed forward network
            for module in layer.feed_forward:
                if isinstance(module, nn.Dropout):
                    module.p = dropout_rate
            
            # Update dropout after attention
            if layer.dropout is not None:
                layer.dropout.p = dropout_rate        

    def generate_positional_encoding(self, max_length, embed_size, reverse = False):
        pe = torch.zeros(max_length, embed_size) # (max_length, embed_size)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1) # (max_length, 1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size)) #  (embed_size // 2,)
        pe[:, 0::2] = torch.sin(position * div_term) # (max_length, embed_size // 2)
        pe[:, 1::2] = torch.cos(position * div_term) # (max_length, embed_size // 2)
        if reverse:
            pe = torch.flip(pe, [1])  # Flip the positional encoding along the sequence dimension
        return pe.unsqueeze(0) # (1, max_length, embed_size)
    
    def forward(self, x):
        batch_size, seq_length, cell_size = x.shape  # x: (N, seq_length, 5)
        assert cell_size == 5 

        combined_encodings = combine_encoding(x, batch_size, seq_length, self.max_grid_size, self.grid_scale, self.grid_encoding)
            
        initial_tensor = self.embedding(x[:, :, 0]) + combined_encodings # (N, seq_length, embed_size)

        x = self.dropout(initial_tensor)  # (N, seq_length, embed_size)

        for layer in self.layers:
            x = layer(x)  # (N, seq_length, embed_size)

        out = self.fc_out(x)  # (N, seq_length, vocab_size)
        return out  # (N, seq_length, vocab_size)
        
