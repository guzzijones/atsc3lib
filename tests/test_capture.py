"""Unit tests for the capture module."""

import importlib

import pytest

from atsc3lib.capture import capture

#: ``atsc3lib.capture`` is shadowed by the same-named ``capture`` function
#: re-exported from ``atsc3lib/__init__.py``; import the submodule by hand.
capture_module = importlib.import_module("atsc3lib.capture")


class TestCapture:
    """Test capture functionality."""

    def test_detect_sdr_tools(self):
        """Test SDR detection returns a DetectedSdr record."""
        from atsc3lib.capture import _detect_sdr_tools, DetectedSdr

        detected = _detect_sdr_tools()
        assert isinstance(detected, DetectedSdr)
        assert isinstance(detected.sdrplay, bool)

    def test_detect_sdrplay_via_bindings(self, monkeypatch):
        """The SDRplay is detected through the sdrbindings extension."""
        class _FakeBindings:
            pass

        monkeypatch.setattr(capture_module, '_bindings',
                            lambda: _FakeBindings())
        monkeypatch.setattr('shutil.which', lambda name: None)
        assert capture_module._detect_sdr_tools().sdrplay

    def test_capture_sdrplay_function(self):
        """Test SDRplay capture function exists."""
        from atsc3lib.capture import _capture_sdrplay

        assert callable(_capture_sdrplay)

    def test_capture_sdrplay_uses_bindings(self, monkeypatch):
        """_capture_sdrplay must route through sdrbindings.capture_iq."""
        calls = {}

        class _FakeBindings:
            @staticmethod
            def capture_iq(freq, out, **kw):
                calls['freq'] = freq
                calls['out'] = out
                calls.update(kw)

        monkeypatch.setattr(capture_module, '_bindings',
                            lambda: _FakeBindings())
        capture_module._capture_sdrplay(587_000_000, 10_000_000, 45, 2,
                                        '/tmp/x.iq', 0, 3, True)
        assert calls['freq'] == 587_000_000
        assert calls['ifgr'] == 45
        assert calls['rfgr'] == 3
        assert calls['cs8'] is True
        assert calls['bandwidth_hz'] == capture_module.SDRPLAY_BANDWIDTH_HZ
        assert calls['driver'] == capture_module.SDRPLAY_DRIVER

    def test_detected_sdr_bool(self):
        """DetectedSdr truthiness tracks SDRplay availability."""
        from atsc3lib.capture import DetectedSdr
        assert DetectedSdr(sdrplay=True)
        assert not DetectedSdr(sdrplay=False)

    def test_capture_sdrplay_without_bindings(self, monkeypatch):
        """A clear error is raised when sdrbindings is unavailable."""
        monkeypatch.setattr(capture_module, '_bindings', lambda: None)
        with pytest.raises(RuntimeError):
            capture_module._capture_sdrplay(587_000_000, 10_000_000, 45, 2,
                                            '/tmp/x.iq', 0, 3, False)

    def test_capture_rejects_unknown_device(self, monkeypatch):
        """Only 'sdrplay' is accepted as a device type."""
        monkeypatch.setattr(capture_module, '_detect_sdr_tools',
                            lambda: capture_module.DetectedSdr(sdrplay=True))
        with pytest.raises(RuntimeError):
            capture_module.capture(587_000_000, 10_000_000, '/tmp/x.iq',
                                   duration_sec=1, device_type='airspy')

    def test_hackrf_capture_removed(self):
        """The HackRF capture path is gone."""
        assert not hasattr(capture_module, '_capture_hackrf')

    def test_airspy_capture_removed(self):
        """The Airspy capture path is gone."""
        assert not hasattr(capture_module, '_capture_airspy')

    def test_soapy_capture_helper_removed(self):
        """The standalone soapy_capture C helper is gone."""
        assert not hasattr(capture_module, '_ensure_soapy_capture_built')
        assert not hasattr(capture_module, '_soapy_capture_path')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
