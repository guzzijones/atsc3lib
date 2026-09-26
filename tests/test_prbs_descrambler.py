"""Unit tests for PRBS descrambler."""

import pytest
import numpy as np
from atsc3lib.prbs_descrambler import (
    PRBSDescrambler, BitScrambler,
    descramble, scramble
)


class TestPRBSDescrambler:
    """Test PRBS descrambler."""
    
    def test_init_default(self):
        """Test default initialization."""
        descrambler = PRBSDescrambler()
        assert descrambler.seed == 0x7FFF  # Default all-ones
    
    def test_init_custom_seed(self):
        """Test custom seed initialization."""
        descrambler = PRBSDescrambler(seed=0x1234)
        assert descrambler.seed == 0x1234
    
    def test_init_seed_masked(self):
        """Test that seed is masked to 15 bits."""
        descrambler = PRBSDescrambler(seed=0xFFFFF)  # More than 15 bits
        assert descrambler.seed == 0x7FFF  # Masked to 15 bits
    
    def test_generate_prbs_length(self):
        """Test PRBS generation length."""
        descrambler = PRBSDescrambler()
        
        prbs = descrambler._generate_prbs(100)
        
        assert len(prbs) == 100
        assert np.all(np.isin(prbs, [0, 1]))
    
    def test_generate_prbs_binary(self):
        """Test PRBS values are binary."""
        descrambler = PRBSDescrambler()
        
        prbs = descrambler._generate_prbs(1000)
        
        assert np.all((prbs == 0) | (prbs == 1))
    
    def test_generate_prbs_reproducible(self):
        """Test PRBS is reproducible with same seed."""
        descrambler = PRBSDescrambler(seed=0x4321)
        
        prbs1 = descrambler._generate_prbs(100)
        prbs2 = descrambler._generate_prbs(100)
        
        assert np.array_equal(prbs1, prbs2)
    
    def test_generate_prbs_different_seeds(self):
        """Test different seeds produce different PRBS."""
        descrambler = PRBSDescrambler()
        
        prbs1 = descrambler._generate_prbs(100, seed=0x1111)
        prbs2 = descrambler._generate_prbs(100, seed=0x2222)
        
        # Should be different (with very high probability)
        assert not np.array_equal(prbs1, prbs2)
    
    def test_descramble(self):
        """Test descrambling operation."""
        descrambler = PRBSDescrambler()
        
        # Create test data
        original = np.random.randint(0, 2, 100, dtype=np.uint8)
        
        # Scramble first
        scrambled = descrambler.scramble(original)
        
        # Descramble
        descrambled = descrambler.descramble(scrambled)
        
        # Should recover original
        assert np.array_equal(original, descrambled)
    
    def test_descramble_length(self):
        """Test descrambling preserves length."""
        descrambler = PRBSDescrambler()
        
        bits = np.random.randint(0, 2, 500, dtype=np.uint8)
        result = descrambler.descramble(bits)
        
        assert len(result) == len(bits)
    
    def test_scramble_descramble_symmetry(self):
        """Test that scramble and descramble are symmetric (both XOR)."""
        descrambler = PRBSDescrambler()
        
        bits = np.random.randint(0, 2, 100, dtype=np.uint8)
        
        # Scramble then descramble
        scrambled = descrambler.scramble(bits)
        descrambled = descrambler.descramble(scrambled)
        
        assert np.array_equal(bits, descrambled)
        
        # Descramble then scramble (should also work)
        descrambled2 = descrambler.descramble(bits)
        scrambled2 = descrambler.scramble(descrambled2)
        
        assert np.array_equal(bits, scrambled2)
    
    def test_descramble_bytes(self):
        """Test descrambling byte data."""
        descrambler = PRBSDescrambler()
        
        # Test data
        original = bytes([0x00, 0x55, 0xAA, 0xFF])
        
        # Convert to bits and scramble
        bits = np.zeros(32, dtype=np.uint8)
        for i, byte in enumerate(original):
            for j in range(8):
                bits[i * 8 + j] = (byte >> (7 - j)) & 1
        
        scrambled_bits = descrambler.scramble(bits)
        
        # Convert scrambled bits back to bytes
        scrambled_bytes = bytearray()
        for i in range(0, 32, 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | scrambled_bits[i + j]
            scrambled_bytes.append(byte)
        
        # Descramble bytes
        descrambled = descrambler.descramble_bytes(bytes(scrambled_bytes))
        
        # Should recover original
        assert descrambled == original
    
    def test_descramble_bytes_length(self):
        """Test descramble_bytes preserves length."""
        descrambler = PRBSDescrambler()
        
        data = bytes(range(256))
        result = descrambler.descramble_bytes(data)
        
        assert len(result) == len(data)


class TestBitScrambler:
    """Test bit scrambler class."""
    
    def test_init(self):
        """Test scrambler initialization."""
        scrambler = BitScrambler()
        assert scrambler.descrambler is not None
    
    def test_scramble(self):
        """Test scrambling."""
        scrambler = BitScrambler()
        
        bits = np.random.randint(0, 2, 100, dtype=np.uint8)
        scrambled = scrambler.scramble(bits)
        
        assert len(scrambled) == len(bits)
        assert np.all(np.isin(scrambled, [0, 1]))
    
    def test_descramble(self):
        """Test descrambling."""
        scrambler = BitScrambler()
        
        original = np.random.randint(0, 2, 100, dtype=np.uint8)
        scrambled = scrambler.scramble(original)
        descrambled = scrambler.descramble(scrambled)
        
        assert np.array_equal(original, descrambled)


class TestConvenienceFunctions:
    """Test module convenience functions."""
    
    def test_descramble_function(self):
        """Test descramble convenience function."""
        bits = np.random.randint(0, 2, 100, dtype=np.uint8)
        result = descramble(bits)
        
        assert len(result) == len(bits)
        assert np.all(np.isin(result, [0, 1]))
    
    def test_scramble_function(self):
        """Test scramble convenience function."""
        bits = np.random.randint(0, 2, 100, dtype=np.uint8)
        result = scramble(bits)
        
        assert len(result) == len(bits)
    
    def test_roundtrip_functions(self):
        """Test roundtrip using convenience functions."""
        original = np.random.randint(0, 2, 100, dtype=np.uint8)
        
        scrambled = scramble(original)
        descrambled = descramble(scrambled)
        
        assert np.array_equal(original, descrambled)


class TestIntegration:
    """Test integration with LDPC workflow."""
    
    def test_descramble_after_ldpc(self):
        """Test descrambling after LDPC decoding."""
        np.random.seed(42)
        
        # Simulate LDPC output (scrambled bits)
        n_bits = 1296
        scrambled_bits = np.random.randint(0, 2, n_bits, dtype=np.uint8)
        
        # Descramble
        descrambler = PRBSDescrambler()
        descrambled = descrambler.descramble(scrambled_bits)
        
        # Should be same length
        assert len(descrambled) == len(scrambled_bits)
        
        # Should be binary
        assert np.all((descrambled == 0) | (descrambled == 1))
        
        # Check bit balance (scrambling should randomize)
        ones_ratio = np.sum(descrambled) / len(descrambled)
        # Should be roughly 50/50 for random data
        assert 0.4 < ones_ratio < 0.6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
