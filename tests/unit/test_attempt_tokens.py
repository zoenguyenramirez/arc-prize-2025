#!/usr/bin/env python3
"""
Unit tests for ATTEMPT_START and ATTEMPT_END tokens
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.token import SpecialToken, VOCAB_SIZE


class TestAttemptTokens(unittest.TestCase):
    """Test suite for ATTEMPT tokens"""
    
    def test_attempt_tokens_exist(self):
        """Test that ATTEMPT_START and ATTEMPT_END tokens are defined"""
        self.assertTrue(hasattr(SpecialToken, 'ATTEMPT_START'))
        self.assertTrue(hasattr(SpecialToken, 'ATTEMPT_END'))
    
    def test_attempt_token_values(self):
        """Test that tokens have correct values"""
        self.assertEqual(SpecialToken.ATTEMPT_START.value, 19)
        self.assertEqual(SpecialToken.ATTEMPT_END.value, 20)
        self.assertEqual(SpecialToken.COUNT_OF_TOKENS.value, 21)
    
    def test_vocab_size_updated(self):
        """Test that VOCAB_SIZE is correctly updated"""
        self.assertEqual(VOCAB_SIZE, 21)
    
    def test_no_token_conflicts(self):
        """Test that there are no conflicting token values"""
        token_values = {}
        for token in SpecialToken:
            if token.name == 'CELL_TOKEN_SIZE':
                continue
            
            # Check for duplicates
            if token.value in token_values:
                self.fail(f"Token conflict: {token.name} and {token_values[token.value]} "
                         f"both have value {token.value}")
            token_values[token.value] = token.name
        
        # Verify we have the expected number of unique tokens
        expected_tokens = SpecialToken.COUNT_OF_TOKENS.value - SpecialToken.CELL_TOKEN_SIZE.value
        self.assertEqual(len(token_values), expected_tokens)
    
    def test_token_ordering(self):
        """Test that ATTEMPT tokens come before COUNT_OF_TOKENS"""
        self.assertLess(SpecialToken.ATTEMPT_START.value, SpecialToken.COUNT_OF_TOKENS.value)
        self.assertLess(SpecialToken.ATTEMPT_END.value, SpecialToken.COUNT_OF_TOKENS.value)
        self.assertLess(SpecialToken.ATTEMPT_START.value, SpecialToken.ATTEMPT_END.value)


if __name__ == '__main__':
    unittest.main()