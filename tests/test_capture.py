"""Unit tests for capture module."""

import pytest
import numpy as np
from atsc3lib.capture import capture


class TestCapture:
    """Test capture functionality."""
    
    def test_detect_sdr_tools(self):
        """Test SDR tool detection."""
        from atsc3lib.capture import _detect_sdr_tools
        
        tools = _detect_sdr_tools()
        
        # Should return dict
        assert isinstance(tools, dict)
        
        # May have hackrf, airspy, or soapysdr
        # At least check it doesn't crash
        for tool_name in tools:
            assert tool_name in ['hackrf', 'airspy', 'soapysdr']
    
    def test_capture_hackrf_function(self):
        """Test HackRF capture function exists."""
        from atsc3lib.capture import _capture_hackrf
        
        # Just test it's callable (can't actually run without hardware)
        assert callable(_capture_hackrf)
    
    def test_capture_airspy_function(self):
        """Test Airspy capture function exists."""
        from atsc3lib.capture import _capture_airspy
        
        assert callable(_capture_airspy)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
