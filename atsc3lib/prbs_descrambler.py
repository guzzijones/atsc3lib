"""PRBS descrambler for ATSC 3.0.

ATSC 3.0 uses a pseudo-random binary sequence (PRBS) scrambler for spectral shaping.
The scrambler XORs the data with a PRBS to ensure sufficient bit transitions.

This module implements the descrambler to recover original data.

Reference: ATSC A/322 Physical Layer Specification Section 6.4
"""

import numpy as np
from typing import Optional


class PRBSDescrambler:
    """ATSC 3.0 PRBS descrambler.
    
    The ATSC 3.0 scrambler uses a linear feedback shift register (LFSR)
    with polynomial: x^15 + x^14 + 1
    
    The descrambler XORs received bits with the same PRBS sequence.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize PRBS descrambler.
        
        Args:
            seed: Initial LFSR state (default: all ones as per A/322)
        """
        # Default seed is 15 ones (all registers set to 1)
        self.default_seed = 0x7FFF  # 15 bits all 1
        
        if seed is None:
            self.seed = self.default_seed
        else:
            self.seed = seed & 0x7FFF  # Ensure 15 bits
        
        # PRBS polynomial: x^15 + x^14 + 1
        self.polynomial = 0xC001  # Bits 15 and 14 set
    
    def _generate_prbs(self, length: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate PRBS sequence.
        
        Args:
            length: Number of bits to generate
            seed: Initial LFSR state (optional, uses default if not provided)
            
        Returns:
            PRBS sequence as binary array
        """
        if seed is None:
            seed = self.seed
        
        prbs = np.zeros(length, dtype=np.uint8)
        shift_reg = seed
        
        for i in range(length):
            # Output is LSB of shift register
            prbs[i] = shift_reg & 1
            
            # Calculate feedback: XOR of bits 15 and 14 (polynomial x^15 + x^14 + 1)
            feedback = ((shift_reg >> 14) ^ (shift_reg >> 13)) & 1
            
            # Shift left and insert feedback at LSB
            shift_reg = ((shift_reg << 1) | feedback) & 0x7FFF
        
        return prbs
    
    def descramble(self, bits: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """
        Descramble bits using PRBS.
        
        Args:
            bits: Scrambled input bits
            seed: Initial LFSR state (optional)
            
        Returns:
            Descrambled bits
        """
        prbs = self._generate_prbs(len(bits), seed)
        return bits ^ prbs
    
    def scramble(self, bits: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """
        Scramble bits using PRBS (for testing/encoding).
        
        Scrambling and descrambling use the same operation (XOR).
        
        Args:
            bits: Input bits
            seed: Initial LFSR state (optional)
            
        Returns:
            Scrambled bits
        """
        return self.descramble(bits, seed)
    
    def descramble_bytes(self, data: bytes, seed: Optional[int] = None) -> bytes:
        """
        Descramble byte data.
        
        Args:
            data: Scrambled byte data
            seed: Initial LFSR state (optional)
            
        Returns:
            Descrambled byte data
        """
        # Convert bytes to bits
        bits = np.zeros(len(data) * 8, dtype=np.uint8)
        for i, byte in enumerate(data):
            for j in range(7, -1, -1):
                bits[i * 8 + (7 - j)] = (byte >> j) & 1
        
        # Descramble
        descrambled_bits = self.descramble(bits, seed)
        
        # Convert back to bytes
        result = bytearray()
        for i in range(0, len(descrambled_bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(descrambled_bits):
                    byte = (byte << 1) | descrambled_bits[i + j]
            result.append(byte)
        
        return bytes(result)


class BitScrambler:
    """ATSC 3.0 PRBS bit scrambler (for testing/encoding)."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize bit scrambler."""
        self.descrambler = PRBSDescrambler(seed)
    
    def scramble(self, bits: np.ndarray) -> np.ndarray:
        """
        Scramble bits (transmitter side).
        
        Args:
            bits: Input bits
            
        Returns:
            Scrambled bits
        """
        return self.descrambler.scramble(bits)
    
    def descramble(self, bits: np.ndarray) -> np.ndarray:
        """
        Descramble bits (receiver side).
        
        Args:
            bits: Scrambled bits
            
        Returns:
            Descrambled bits
        """
        return self.descrambler.descramble(bits)


def descramble(bits: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """
    Convenience function to descramble bits.
    
    Args:
        bits: Scrambled input bits
        seed: Initial LFSR state (optional)
        
    Returns:
        Descrambled bits
    """
    descrambler = PRBSDescrambler(seed)
    return descrambler.descramble(bits)


def scramble(bits: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """
    Convenience function to scramble bits.
    
    Args:
        bits: Input bits
        seed: Initial LFSR state (optional)
        
    Returns:
        Scrambled bits
    """
    descrambler = PRBSDescrambler(seed)
    return descrambler.scramble(bits)
