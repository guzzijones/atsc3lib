"""Tests for A/322 spec lookup helpers (guard interval, carriers)."""

import pytest

from atsc3lib import spec


class TestGuardInterval:
    def test_signalling_value_6_is_1536(self):
        # A/322 Table 8.6: value 6 = GI6_1536.  The list index is NOT the
        # signalling value (value 1 = 192 is index 0).
        assert spec.guard_interval(8192, 6) == 1536
        assert spec.guard_interval(16384, 6) == 1536
        assert spec.GUARD_INTERVALS[8192][6] == 2048      # the old bug

    def test_192_and_2048(self):
        assert spec.guard_interval(8192, 1) == 192
        assert spec.guard_interval(8192, 7) == 2048

    @pytest.mark.parametrize("value", [0, 13, 14, 15])
    def test_reserved_values_rejected(self, value):
        with pytest.raises(ValueError):
            spec.guard_interval(8192, value)

    def test_illegal_for_fft_size_rejected(self):
        with pytest.raises(ValueError):
            spec.guard_interval(8192, 8)         # 2432 not allowed at 8K


class TestNoc:
    def test_8k_cred0(self):
        assert spec.noc(8192, 0) == 6913
