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

        # May have sdrplay or airspy; hackrf is gone.
        for tool_name in tools:
            assert tool_name in ['sdrplay', 'airspy']

    def test_detect_sdr_tools_prefers_sdrplay(self, monkeypatch):
        """SDRplay must be ordered before Airspy when both are present."""
        import shutil
        from atsc3lib.capture import _detect_sdr_tools

        monkeypatch.setattr(shutil, 'which',
                            lambda name: '/usr/bin/' + name)
        tools = _detect_sdr_tools()
        assert list(tools.keys()) == ['sdrplay', 'airspy']

    def test_capture_sdrplay_function(self):
        """Test SDRplay capture function exists."""
        from atsc3lib.capture import _capture_sdrplay

        # Just test it's callable (can't actually run without hardware)
        assert callable(_capture_sdrplay)

    def test_capture_airspy_function(self):
        """Test Airspy capture function exists."""
        from atsc3lib.capture import _capture_airspy

        assert callable(_capture_airspy)

    def test_hackrf_capture_removed(self):
        """The HackRF capture path is gone."""
        import atsc3lib.capture as cap
        assert not hasattr(cap, '_capture_hackrf')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
