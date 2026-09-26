"""ATSC 3.0 LDPC decoder using exact A/322 Annex A tables.

Implements the exact LDPC codes defined in ATSC A/322:
- Type A (rates 2/15-5/15): IRA structure with parity accumulator addresses
- Type B (rates 6/15-13/15): QC-LDPC with base graph + Qldpc lifting

Tables extracted from official ATSC A/322:2024-04 Annex A (Table A.2.1-A.2.12)
for Ninner=16200 (short frames).

Reference: ATSC A/322:2024-04 Physical Layer Protocol, Section 6.1.3, Annex A
"""

import numpy as np
import json
import os
from typing import Tuple, Dict, Optional, List

# Type A parameters (Table 6.6): rate -> (m1, m2, Q1, Q2)
TYPE_A_PARAMS = {
    2: (3240, 10800, 9, 30),
    3: (1080, 11880, 3, 33),
    4: (1080, 10800, 3, 30),
    5: (720, 10080, 2, 28),
}

# Type B parameters (Table 6.7): rate -> Qldpc
TYPE_B_QLDPC = {6: 27, 7: 24, 8: 21, 9: 18, 10: 15, 11: 12, 12: 9, 13: 6}

_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def load_tables() -> Dict:
    """Load official A/322 Annex A tables for Ninner=16200."""
    with open(os.path.join(_DATA_DIR, 'ldpc_tables_N16200.json')) as f:
        tables = json.load(f)
    return {int(k): v for k, v in tables.items()}


def get_code_params(rate: int, n: int = 16200) -> Tuple[int, int, str]:
    """
    Get K, M, and type for a given rate.

    Returns:
        (K, M, type) where type is 'A' or 'B'
    """
    K = n * rate // 15
    M = n - K
    code_type = 'A' if rate in TYPE_A_PARAMS else 'B'
    return K, M, code_type


class ATSC3LDPCExact:
    """Exact ATSC 3.0 LDPC decoder/encoder for Ninner=16200.

    Uses official A/322 Annex A parity tables. Implements the encoder
    (Section 6.1.3) to generate valid codewords for testing, and a
    belief-propagation decoder.
    """

    def __init__(self, rate: int, n: int = 16200, max_iterations: int = 50):
        """
        Initialize exact LDPC codec.

        Args:
            rate: Code rate numerator (2-13, denominator is 15)
            n: Codeword length (16200 for short frames)
            max_iterations: Max BP decoding iterations
        """
        if rate not in range(2, 14):
            raise ValueError(f"Rate {rate}/15 not supported (use 2-13)")
        if n != 16200:
            raise ValueError(f"Only Ninner=16200 supported (got {n})")

        self.rate = rate
        self.n = n
        self.K = n * rate // 15
        self.M = n - self.K
        self.code_type = 'A' if rate in TYPE_A_PARAMS else 'B'
        self.max_iterations = max_iterations

        # Load official table (keys are table indices 1-12; index = rate - 1)
        tables = load_tables()
        table = tables[rate - 1]
        self.rows = table['rows']
        assert table['rate'] == rate, f"Table rate mismatch: {table['rate']} != {rate}"

        # Split rows into info rows and parity rows.
        # Type A (A/322 6.1.3.1): the Annex A table has Kldpc/360 information
        # rows PLUS Q1 additional rows (used in step vii). Type B (6.1.3.2)
        # has exactly Kldpc/360 rows.
        n_info_rows = self.K // 360
        self.n_info_rows = n_info_rows
        self.info_rows = self.rows[:n_info_rows]
        if self.code_type == 'A':
            self.parity_rows = self.rows[n_info_rows:]
            self.m1, self.m2, self.q1, self.q2 = TYPE_A_PARAMS[rate]
        else:
            self.parity_rows = []
            self.m1, self.m2 = 0, self.M
            self.q1, self.q2 = 0, TYPE_B_QLDPC[rate]

        # Build sparse H for decoding (from encoder structure)
        self._build_sparse_h()

    def _addr(self, x: int, l: int) -> int:
        """Parity-accumulator address (A/322 6.1.3.1 step ii)."""
        if x < self.m1:
            return (x + l * self.q1) % self.m1
        return self.m1 + ((x - self.m1) + l * self.q2) % self.m2

    def _pi(self, a: int, variant: str = 'standard') -> int:
        """Map a parity-accumulator index to its codeword index.

        A/322 6.1.3.1 steps vi/viii: for Type A the M1 (dual-diagonal) parity
        is emitted as ``lambda_{K+360t+s} = p_{Q1*s+t}`` and the M2 parity as
        ``lambda_{K+M1+360t+s} = p_{M1+Q2*s+t}``.  Type B has no interleave, so
        this is the identity.
        """
        if self.code_type == 'B':
            return self.K + a
        if a < self.m1:
            if variant == 'standard':
                s, t = divmod(a, self.q1)
                return self.K + 360 * t + s
            t, s = divmod(a, 360)
            return self.K + self.q1 * s + t
        b = a - self.m1
        if variant == 'standard':
            s, t = divmod(b, self.q2)
            return self.K + self.m1 + 360 * t + s
        t, s = divmod(b, 360)
        return self.K + self.m1 + self.q2 * s + t

    def _build_sparse_h(self, variant: str = 'standard'):
        """Build check-node connections, one list of variable indices per check.

        Derived from the encoder steps so that ``H . encode(s) == 0``; verified
        for every rate in ``tests/test_ldpc_exact.py``.
        """
        K, n_info, M1, M2 = self.K, self.n_info_rows, self.m1, self.m2
        checks: List[List[int]] = [[] for _ in range(self.M)]

        # Information part (steps i-ii).
        for k in range(K):
            row = self.rows[k // 360]
            l = k % 360
            for x in row:
                checks[self._addr(x, l)].append(k)

        if self.code_type == 'A':
            # Step vii: the M1 dual-diagonal codeword bits feed back through
            # the extra Q1 rows.  These are CODEWORD bits (K+i), not p-indices.
            for i in range(M1):
                row = self.rows[n_info + i // 360]
                l = i % 360
                for x in row:
                    checks[self._addr(x, l)].append(K + i)
            # Steps v/vi: dual diagonal over the M1 parity bits.
            for a in range(M1):
                checks[a].append(self._pi(a, variant))
                if a >= 1:
                    checks[a].append(self._pi(a - 1, variant))
            # Step viii: identity over the M2 parity bits.
            for a in range(M1, M1 + M2):
                checks[a].append(self._pi(a, variant))
        else:
            for a in range(self.M):
                checks[a].append(self._pi(a, variant))
                if a >= 1:
                    checks[a].append(self._pi(a - 1, variant))

        # A column appearing twice in the same check cancels over GF(2).
        check_vars: List[List[int]] = []
        for c in checks:
            c.sort()
            d = []
            i = 0
            while i < len(c):
                if i + 1 < len(c) and c[i] == c[i + 1]:
                    i += 2
                else:
                    d.append(c[i])
                    i += 1
            check_vars.append(d)

        self.check_vars = check_vars
        self.var_checks: List[List[int]] = [[] for _ in range(self.n)]
        for c, vs in enumerate(check_vars):
            for v in vs:
                self.var_checks[v].append(c)

        self._build_packed(check_vars)

    def _build_packed(self, check_vars: List[List[int]]):
        """Precompute rectangular edge arrays for vectorized decoding.

        ATSC 3.0 codes have a small maximum check degree (<= 16 for
        Ninner=16200), so the whole Tanner graph fits in a few rectangular
        arrays and one BP iteration becomes a handful of NumPy operations
        instead of a Python loop over ~13000 checks.

        Sets:
            check_idx   (n_checks, dmax) int32 - variable column per edge slot
            check_mask  (n_checks, dmax) bool  - False on padding slots
            edge_var    (n_edges,)  int32      - variable per edge, check-major
            edge_ptr    (n_checks+1,) int32    - CSR row pointers into edge_var
            var_edges   (n_edges,)  int64      - edge ids grouped by variable,
                                                 in check-major order within a
                                                 variable (for scatter-add)
        """
        n_checks = len(check_vars)
        dmax = max((len(v) for v in check_vars), default=0)
        self.dmax = dmax
        check_idx = np.zeros((n_checks, dmax), dtype=np.int32)
        check_mask = np.zeros((n_checks, dmax), dtype=bool)
        flat = []
        row_len = np.zeros(n_checks, dtype=np.int32)
        for c, vs in enumerate(check_vars):
            row_len[c] = len(vs)
            check_idx[c, :len(vs)] = vs
            check_mask[c, :len(vs)] = True
            flat.extend(vs)
        self.check_idx = check_idx
        self.check_mask = check_mask
        self.edge_var = np.asarray(flat, dtype=np.int32)
        self.edge_ptr = np.concatenate([[0], np.cumsum(row_len)]).astype(np.int32)
        # Edge ids grouped by variable, preserving check-major order, so a
        # segment sum reproduces the per-variable addend order.
        self.var_edges = np.argsort(self.edge_var, kind='stable').astype(np.int64)
        counts = np.bincount(self.edge_var, minlength=self.n)
        self.var_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)


    def encode(self, info_bits: np.ndarray,
               variant: str = 'standard') -> np.ndarray:
        """Encode information bits into a full codeword (A/322 6.1.3).

        Implements Type A steps (i)-(viii) and Type B steps (i)-(iii).

        Args:
            info_bits: K information bits
            variant: Type A parity-interleave index order ('standard' or
                     'swapped'); the air arbitrates when it matters.

        Returns:
            N-bit codeword.
        """
        if len(info_bits) != self.K:
            raise ValueError(f"Expected {self.K} info bits, got {len(info_bits)}")

        s = np.asarray(info_bits, dtype=np.uint8)
        p = np.zeros(self.M, dtype=np.uint8)

        # Steps i-ii: accumulate info bits into parity accumulators.
        for k in range(self.K):
            if not s[k]:
                continue
            row = self.rows[k // 360]
            l = k % 360
            for x in row:
                p[self._addr(x, l)] ^= 1

        codeword = np.zeros(self.n, dtype=np.uint8)
        codeword[:self.K] = s

        if self.code_type == 'A':
            # Step v: running XOR over the M1 dual-diagonal accumulators.
            np.bitwise_xor.accumulate(p[:self.m1], out=p[:self.m1])
            # Step vi: emit the M1 parity bits.
            for a in range(self.m1):
                codeword[self._pi(a, variant)] = p[a]
            # Step vii: feed the M1 codeword bits back through the Q1 rows.
            for i in range(self.m1):
                if not codeword[self.K + i]:
                    continue
                row = self.rows[self.n_info_rows + i // 360]
                l = i % 360
                for x in row:
                    p[self._addr(x, l)] ^= 1
            # Step viii: emit the M2 parity bits.
            for a in range(self.m1, self.m1 + self.m2):
                codeword[self._pi(a, variant)] = p[a]
        else:
            # Type B step iii: running XOR over the whole parity accumulator.
            np.bitwise_xor.accumulate(p, out=p)
            for a in range(self.M):
                codeword[self._pi(a, variant)] = p[a]
        return codeword

    def check_syndrome(self, codeword: np.ndarray) -> bool:
        """Return True if the codeword satisfies every parity check."""
        return self.check_syndrome_bits(np.asarray(codeword, dtype=np.uint8))

    def check_syndrome_bits(self, bits: np.ndarray) -> bool:
        """Vectorized syndrome check.

        Each check equation XORs the bits at its variable slots; padding slots
        contribute 0 (multiplied out by the mask).  Returns True iff every
        check is satisfied.
        """
        bits = np.asarray(bits, dtype=np.uint8)
        gathered = bits[self.check_idx] * self.check_mask  # padding -> 0
        syndrome = np.bitwise_xor.reduce(gathered, axis=1)
        return not np.any(syndrome)

    def _syndrome_unsatisfied(self, hard: np.ndarray) -> np.ndarray:
        """Per-check syndrome (1 = unsatisfied) for a hard-decision array."""
        gathered = hard[self.check_idx] * self.check_mask
        return np.bitwise_xor.reduce(gathered, axis=1)

    def decode(self, llrs: np.ndarray, max_iterations: Optional[int] = None,
               alpha: float = 0.75) -> Tuple[np.ndarray, bool]:
        """Decode with vectorized normalized min-sum belief propagation.

        The whole Tanner graph is held in rectangular edge arrays (see
        :meth:`_build_packed`), so each iteration is a few array operations
        rather than a Python loop over every check node.

        Args:
            llrs: N LLRs (soft information). Convention follows the library:
                  LLR > 0 => bit likely 1, LLR < 0 => bit likely 0.
            max_iterations: Override the instance iteration limit.
            alpha: Normalized min-sum scaling factor (0 < alpha <= 1).

        Returns:
            (decoded_info_bits, converged)
        """
        if len(llrs) != self.n:
            raise ValueError(f"Expected {self.n} LLRs, got {len(llrs)}")

        iters = max_iterations if max_iterations is not None else self.max_iterations

        # Work in the conventional L'>0 => bit 0 domain by negating the
        # channel LLRs, then map back to the library's LLR>0 => bit 1 output.
        channel = -np.asarray(llrs, dtype=np.float64)
        total = channel.copy()

        hard = (total < 0).astype(np.uint8)
        if self.check_syndrome_bits(hard):
            return hard[:self.K], True

        n_checks, dmax = self.check_idx.shape
        check_degree = self.check_mask.sum(axis=1)

        # Check -> variable messages, one value per edge (check-major order,
        # so M reshapes directly onto the rectangular check_idx slots).
        M = np.zeros((n_checks, dmax), dtype=np.float64)
        # Dummy column to keep padding slots finite in the gather.
        total_pad = np.empty(self.n + 1, dtype=np.float64)
        total_pad[:self.n] = total
        total_pad[self.n] = 1e9
        idx_pad = self.check_idx.copy()
        idx_pad[~self.check_mask] = self.n
        row_index = np.arange(n_checks)

        for _ in range(iters):
            # Intrinsic messages W = posterior - incoming check message.
            W = total_pad[idx_pad] - M
            mags = np.abs(W)
            mags[~self.check_mask] = np.inf      # padding never wins the min
            order = np.argpartition(mags, 1, axis=1)
            first = order[:, 0]
            second = order[:, 1]
            min1 = mags[row_index, first]
            min2 = mags[row_index, second]
            # A degree-1 check carries no extrinsic information: min2 = min1.
            min2 = np.where(check_degree == 1, min1, min2)
            # Outgoing magnitude: min2 on the edge attaining min1, else min1.
            out_mag = np.where(np.arange(dmax)[None, :] == first[:, None],
                               min2[:, None], min1[:, None])
            # Sign: product of all W signs, times this edge's own sign.
            neg = W < 0
            parity = np.bitwise_xor.reduce(neg & self.check_mask, axis=1)
            sign = np.where(parity[:, None] ^ neg, -1.0, 1.0)
            M = out_mag * sign * alpha
            M[~self.check_mask] = 0.0            # padding contributes nothing

            # Variable node update: posterior = channel + sum of edge messages.
            total = channel.copy()
            np.add.at(total, self.edge_var, M[self.check_mask])
            total_pad[:self.n] = total

            hard = (total < 0).astype(np.uint8)
            if self.check_syndrome_bits(hard):
                return hard[:self.K], True

        return hard[:self.K], False