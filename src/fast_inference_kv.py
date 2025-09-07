"""
Fast Batch Inference with REAL KV Caching
Actually implements KV cache, not just a placeholder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from typing import Optional, Dict, List, Tuple, Any
import time
from datetime import datetime
import logging
from pathlib import Path

from src.model import Transformer, MultiQueryAttention
from src.token import VOCAB_SIZE, SpecialToken
from src.checkpoint_handler import CheckpointHandler


class KVCache:
    """
    REAL Key-Value cache for transformer layers
    Actually stores and reuses computations
    """
    def __init__(self, batch_size: int, max_seq_len: int, num_layers: int,
                 num_kv_heads: int, head_dim: int, device: torch.device, dtype=torch.float32):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        
        # Allocate cache tensors
        self.k_cache = torch.zeros(
            (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim),
            dtype=dtype, device=device
        )
        self.v_cache = torch.zeros(
            (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim),
            dtype=dtype, device=device
        )
        
        # Track current position in cache
        self.cache_pos = 0
        
    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, start_pos: int):
        """Update cache with new keys and values"""
        seq_len = k.shape[2]  # [batch, num_kv_heads, seq_len, head_dim]
        
        # Store in cache
        self.k_cache[layer_idx, :, :, start_pos:start_pos+seq_len] = k
        self.v_cache[layer_idx, :, :, start_pos:start_pos+seq_len] = v
        
    def get(self, layer_idx: int, start_pos: int, seq_len: int):
        """Get cached keys and values up to current position"""
        if start_pos == 0:
            # First token, no cache yet
            return None, None
            
        # Return cached K,V up to current position
        k = self.k_cache[layer_idx, :, :, :start_pos]
        v = self.v_cache[layer_idx, :, :, :start_pos]
        return k, v
    
    def reset(self):
        """Clear the cache"""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.cache_pos = 0


class CachedMultiQueryAttention(nn.Module):
    """
    Multi-Query Attention with REAL KV caching support
    Modified to actually use cached keys and values
    """
    def __init__(self, base_attention: MultiQueryAttention):
        super().__init__()
        self.base_attention = base_attention
        self.embed_dim = base_attention.embed_dim
        self.num_heads = base_attention.num_heads
        self.num_kv_heads = base_attention.num_kv_heads
        self.head_dim = base_attention.head_dim
        self.num_queries_per_kv = base_attention.num_queries_per_kv
        self.scale = base_attention.scale
        
        # Copy the projection weights
        self.q_proj = base_attention.q_proj
        self.kv_proj = base_attention.kv_proj
        self.out_proj = base_attention.out_proj
        
    def forward(self, x: torch.Tensor, start_pos: int, cache_k: Optional[torch.Tensor] = None, 
                cache_v: Optional[torch.Tensor] = None):
        """
        Forward pass with KV caching
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            start_pos: Starting position in sequence (for caching)
            cache_k: Cached keys from previous steps
            cache_v: Cached values from previous steps
        
        Returns:
            output: Attention output
            new_k: Keys to cache (for current tokens)
            new_v: Values to cache (for current tokens)
        """
        B, L, D = x.shape
        
        # Compute Q for current tokens
        q = self.q_proj(x)  # [B, L, D]
        q = q.reshape(B, L, self.num_heads, self.head_dim)  # [B, L, H, head_dim]
        q = q.transpose(1, 2)  # [B, H, L, head_dim]
        
        # Compute K, V for current tokens
        kv = self.kv_proj(x)  # [B, L, 2*num_kv_heads*head_dim]
        kv = kv.reshape(B, L, 2, self.num_kv_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # [2, B, num_kv_heads, L, head_dim]
        k, v = kv[0], kv[1]  # Each is [B, num_kv_heads, L, head_dim]
        
        # If we have cached K,V, concatenate them
        if cache_k is not None and cache_v is not None:
            # Concatenate cached K,V with current K,V
            k = torch.cat([cache_k, k], dim=2)  # [B, num_kv_heads, cached_len + L, head_dim]
            v = torch.cat([cache_v, v], dim=2)
        
        # Store new K,V for caching (just the new tokens)
        new_k = kv[0]  # [B, num_kv_heads, L, head_dim]
        new_v = kv[1]
        
        # Expand K,V to match number of Q heads if using MQA/GQA
        if self.num_kv_heads < self.num_heads:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)  # [B, H, seq_len, head_dim]
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)
        
        # Compute attention
        # When using KV cache, we can't use is_causal=True because Q and K have different lengths
        # For the initial pass (no cache), we can use is_causal=True
        # For subsequent passes (with cache), we use is_causal=False because:
        #   - Q represents only the new token(s)
        #   - K,V contain all previous tokens up to current position
        #   - The causal mask is implicitly enforced by only having past tokens in cache
        if cache_k is not None:
            # With cache: Q is new tokens, K,V are all tokens up to current position
            # No need for causal mask - we only have past/current tokens, no future
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,  # Can't use causal with different Q,K lengths
                scale=self.scale
            )
        else:
            # Without cache: Q and K have same length, use causal mask
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,  # Use causal mask for initial pass
                scale=self.scale
            )  # [B, H, L, head_dim]
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2)  # [B, L, H, head_dim]
        attn_output = attn_output.contiguous().view(B, L, D)  # [B, L, D]
        output = self.out_proj(attn_output)  # [B, L, D]
        
        return output, new_k, new_v


class CachedTransformerBlock(nn.Module):
    """Transformer block with KV caching support"""
    def __init__(self, base_block):
        super().__init__()
        # Replace attention with cached version
        self.attention = CachedMultiQueryAttention(base_block.attention)
        self.norm1 = base_block.norm1
        self.norm2 = base_block.norm2
        self.feed_forward = base_block.feed_forward
        self.dropout = base_block.dropout
        
    def forward(self, x, start_pos, cache_k=None, cache_v=None):
        # Pre-LN for attention with caching
        norm_x = self.norm1(x)
        attention_out, new_k, new_v = self.attention(norm_x, start_pos, cache_k, cache_v)
        x = self.dropout(attention_out) + x if self.dropout else attention_out + x
        
        # Pre-LN for feed-forward (no caching needed)
        out = self.feed_forward(self.norm2(x)) + x
        
        return out, new_k, new_v


class CachedTransformer(nn.Module):
    """
    Transformer with REAL KV caching
    Actually implements caching, not just a placeholder
    """
    def __init__(self, base_model: Transformer):
        super().__init__()
        self.base_model = base_model
        
        # Copy base model components
        self.embedding = base_model.embedding
        self.grid_encoding = base_model.grid_encoding
        self.grid_scale = base_model.grid_scale
        self.fc_out = base_model.fc_out
        self.dropout = base_model.dropout
        
        # Model params
        self.embed_size = base_model.embed_size
        self.max_grid_size = base_model.max_grid_size
        self.grid_embedding_size = base_model.grid_embedding_size
        
        # Replace transformer blocks with cached versions
        self.layers = nn.ModuleList([
            CachedTransformerBlock(block) for block in base_model.layers
        ])
        
        # Get KV heads from first attention layer
        self.num_kv_heads = base_model.layers[0].attention.num_kv_heads
        self.head_dim = base_model.layers[0].attention.head_dim
        
    def forward(self, x: torch.Tensor, start_pos: int = 0, kv_cache: Optional[KVCache] = None):
        """
        Forward pass with KV caching
        
        Args:
            x: Input tensor [batch, seq_len, 5]
            start_pos: Starting position in sequence
            kv_cache: KV cache object
            
        Returns:
            output: Model output
            kv_cache: Updated cache (if provided)
        """
        batch_size, seq_length, cell_size = x.shape
        assert cell_size == 5
        
        # Handle embeddings and encodings (same as base model)
        from src.utils.transformer_helper import combine_encoding
        
        # Grid encoding is always used in our models
        combined_encodings = combine_encoding(
            x, batch_size, seq_length,
            self.max_grid_size, self.grid_scale, self.grid_encoding
        )
        
        initial_tensor = self.embedding(x[:, :, 0]) + combined_encodings
        # Ensure tensor is in the correct dtype (same as model weights)
        initial_tensor = initial_tensor.to(self.fc_out.weight.dtype)
        x = self.dropout(initial_tensor)
        
        # Process through layers WITH CACHING
        for layer_idx, layer in enumerate(self.layers):
            # Get cached K,V for this layer if available
            cache_k, cache_v = None, None
            if kv_cache is not None and start_pos > 0:
                cache_k, cache_v = kv_cache.get(layer_idx, start_pos, seq_length)
            
            # Forward through layer, get new K,V
            x, new_k, new_v = layer(x, start_pos, cache_k, cache_v)
            
            # Update cache with new K,V
            if kv_cache is not None:
                kv_cache.update(layer_idx, new_k, new_v, start_pos)
        
        # Final output projection
        out = self.fc_out(x)
        
        return out


class FastBatchInferenceKV:
    """
    Fast batch inference with REAL KV caching
    Actually faster, not just pretending
    """
    def __init__(self, checkpoint_path: str, device: torch.device = None,
                 use_compile: bool = True):
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        
        # Load base model
        base_model, self.max_seq_length, _, self.checkpoint_info = \
            CheckpointHandler.load_checkpoint_in_production(
                checkpoint_path, device, adjust_max_length=12000
            )
        
        # Wrap with cached transformer
        self.model = CachedTransformer(base_model)
        
        # Model stays in FP32 - autocast handles BF16 conversion during computation
        # This preserves precision while getting BF16 speed benefits
        
        # NOTE: torch.compile doesn't work well with KV cache due to tensor mutations
        # The cache updates in-place which prevents CUDA graphs optimization
        if use_compile and device.type == 'cuda':
            logging.warning("torch.compile disabled for KV cache - mutations prevent CUDA graphs")
            # Don't compile - it causes issues with KV cache mutations
        
        self.model.eval()
        
    @torch.no_grad()
    def generate_batch(
        self,
        input_sequences: List[List],
        max_new_tokens: int = 1000,
        return_logits: bool = True,
        pad_token_id: int = SpecialToken.PAD.value,
        eos_token_id: int = SpecialToken.END.value,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate with REAL KV caching for actual speedup
        """
        batch_size = len(input_sequences)
        
        # Prepare input
        max_input_len = max(len(seq) for seq in input_sequences)
        input_ids = torch.full(
            (batch_size, max_input_len, 5),
            pad_token_id,
            dtype=torch.long,
            device=self.device
        )
        
        for i, seq in enumerate(input_sequences):
            seq_len = len(seq)
            input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long, device=self.device)
        
        input_lengths = torch.tensor([len(seq) for seq in input_sequences], device=self.device)
        
        # Initialize KV cache for EACH sequence in batch
        kv_cache = KVCache(
            batch_size=batch_size,
            max_seq_len=self.max_seq_length,
            num_layers=len(self.model.layers),
            num_kv_heads=self.model.num_kv_heads,
            head_dim=self.model.head_dim,
            device=self.device,
            dtype=torch.float32  # KV cache uses FP32 for precision
        )
        
        # Track generation
        unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        # Initialize coordinates from last token of each sequence
        current_coords = torch.zeros((batch_size, 2), dtype=torch.long, device=self.device)
        for b in range(batch_size):
            # Get coordinates from last token in input
            last_y = input_ids[b, -1, 1].item()
            last_x = input_ids[b, -1, 2].item()
            # If last token is START_OUTPUT (-1, -1), first output should be (0, 0)
            if last_y == -1 and last_x == -1:
                current_coords[b] = torch.tensor([0, 0], device=self.device)
            else:
                current_coords[b] = torch.tensor([last_y, last_x], device=self.device)
        all_tokens = input_ids.clone()
        all_logits = [] if return_logits else None
        
        # First pass: process all input tokens (fill cache)
        with autocast('cuda', enabled=True, dtype=torch.bfloat16):
            outputs = self.model(input_ids, start_pos=0, kv_cache=kv_cache)
        
        if return_logits:
            all_logits.append(outputs)
        
        # Start position for generation
        start_pos = max_input_len
        
        
        # Generation loop WITH KV CACHE
        for step in range(max_new_tokens):
            if not unfinished_sequences.any():
                break
            
            # Get last token predictions
            # Only need to process the LAST token thanks to KV cache!
            last_tokens = all_tokens[:, -1:, :]  # [batch, 1, 5]
            
            with autocast('cuda', enabled=True, dtype=torch.bfloat16):
                # Process just the new token with cached K,V
                outputs = self.model(last_tokens, start_pos=start_pos, kv_cache=kv_cache)
            
            next_token_logits = outputs[:, -1, :].float()
            
            # Always use argmax (deterministic)
            next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            
            if return_logits:
                all_logits.append(next_token_logits.unsqueeze(1))
            
            # Update coordinates following src/sample.py logic
            # Create next token tensor with CURRENT coordinates
            next_token_tensor = torch.zeros((batch_size, 1, 5), dtype=torch.long, device=self.device)
            next_token_tensor[:, 0, 0] = next_tokens
            
            for b in range(batch_size):
                if not unfinished_sequences[b]:
                    continue
                    
                token = next_tokens[b].item()
                y, x = current_coords[b]
                
                # Use CURRENT coordinates for this token
                if token < SpecialToken.CELL_TOKEN_SIZE.value:
                    coord = (y, x)
                    # Update for NEXT token
                    x = min(x + 1, self.model.max_grid_size - 1)
                elif token == SpecialToken.ROW_SEPARATOR.value:
                    coord = (y, x)
                    # Update for NEXT token
                    x = 0
                    y = min(y + 1, self.model.max_grid_size - 1)
                else:
                    # Special tokens use (-1, -1)
                    coord = (-1, -1)
                    # Reset for NEXT token
                    y = 0
                    x = 0
                
                # Set coordinates for THIS token
                next_token_tensor[b, 0, 1] = coord[0]
                next_token_tensor[b, 0, 2] = coord[1]
                
                # Update current_coords for NEXT iteration
                current_coords[b] = torch.tensor([y, x], device=self.device)
            next_token_tensor[:, 0, 3] = -1
            next_token_tensor[:, 0, 4] = -1
            
            next_token_tensor[~unfinished_sequences] = pad_token_id
            
            # Append to sequences
            all_tokens = torch.cat([all_tokens, next_token_tensor], dim=1)
            
            # Check for EOS
            unfinished_sequences = unfinished_sequences & (next_tokens != eos_token_id)
            
            # Update position for next iteration
            start_pos += 1
        
        # Calculate lengths
        final_lengths = input_lengths.clone()
        for b in range(batch_size):
            tokens = all_tokens[b, :, 0]
            eos_positions = (tokens == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                final_lengths[b] = eos_positions[0] + 1
            else:
                non_pad = (tokens != pad_token_id).nonzero(as_tuple=True)[0]
                if len(non_pad) > 0:
                    final_lengths[b] = non_pad[-1] + 1
        
        
        result = {
            'sequences': all_tokens,
            'tokens': all_tokens[:, :, 0],
            'lengths': final_lengths,
        }
        
        if return_logits and all_logits:
            result['logits'] = torch.cat(all_logits, dim=1)
        
        return result


if __name__ == "__main__":
    print("Fast Inference with REAL KV Caching")
    print("=" * 60)
    print("This actually implements KV cache, not just a placeholder!")
    print("Expected speedup: 5-10x for long sequences")
    print("Memory usage: Higher but worth it for speed")
    print("=" * 60)