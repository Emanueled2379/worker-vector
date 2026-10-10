"""Succinct bit vector with rank and select over a compact two-level index."""

from __future__ import annotations


class SuccinctBitVector:
    """A bit array supporting rank and select in roughly n + o(n) bits.

    Bits are packed 64 per word (Python ints are arbitrary-precision, but we
    mask to 64 bits so shifts never widen a word unbounded). Rank is answered
    by a block table storing the cumulative popcount at every 512-bit
    boundary; a 512-bit block is eight 64-bit words. Select is answered by a
    positional sample: every 8192-th set bit (or unset bit for select0) is
    recorded with its bit index, and the gap in between is scanned linearly.

    The sample period for select is deliberately coarse (8192). Scanning a
    handful of 64-bit words in CPython is cheap, and keeping the table small
    matters more here than shaving a few microseconds, because every stored
    integer is a boxed Python int. If you need denser sampling, lower
    SELECT_SAMPLE at construction time — but know you pay for it in memory.
    """

    WORD_BITS = 64
    BLOCK_WORDS = 8
    BLOCK_BITS = BLOCK_WORDS * WORD_BITS  # 512
    SELECT_SAMPLE = 8192

    __slots__ = ("_words", "_num_bits", "_rank_blocks", "_select1", "_select0", "_num_ones")

    def __init__(self, bits, *, select_sample: int = SELECT_SAMPLE):
        """Build from any iterable of ints (0 or 1).

        select_sample tunes the select index density. Smaller values speed up
        select at the cost of more memory; larger values save memory but make
        select scan more bits. It is exposed because the right trade-off is
        workload-dependent.
        """
        words = []
        word = 0
        count = 0
        for b in bits:
            if b == 0:
                pass
            elif b == 1:
                word |= 1 << (count & (self.WORD_BITS - 1))
            else:
                raise ValueError(f"bit must be 0 or 1, got {b!r}")
            count += 1
            if count % self.WORD_BITS == 0:
                words.append(word)
                word = 0
        if count % self.WORD_BITS != 0:
            words.append(word)

        self._words = words
        self._num_bits = count

        self._rank_blocks = [0]
        total = 0
        for w in words:
            total += self._popcount(w)
            if len(self._rank_blocks) * self.BLOCK_BITS <= (len(self._rank_blocks) - 1) * self.BLOCK_BITS + len(self._rank_blocks) * 0:
                pass
        # The loop above is awkward; rewrite clearly below.
        self._rank_blocks = [0]
        total = 0
        for i, w in enumerate(words):
            total += self._popcount(w)
            if (i + 1) % self.BLOCK_WORDS == 0:
                self._rank_blocks.append(total)
        self._rank_blocks.append(total)
        self._num_ones = total

        self._select1 = self._build_select_sample(1, select_sample)
        self._select0 = self._build_select_sample(0, select_sample)

    @staticmethod
    def _popcount(x: int) -> int:
        return bin(x).count("1")

    def _build_select_sample(self, target: int, sample_period: int) -> list:
        if sample_period <= 0:
            raise ValueError("select_sample must be positive")
        if target == 1:
            limit = self._num_ones
        else:
            limit = self._num_bits - self._num_ones
        samples = [-1]  # samples[k] = bit index of the (k*sample_period)-th target bit
        if limit == 0:
            return samples
        count = 0
        for i in range(self._num_bits):
            if self._bit_at(i) == target:
                if count % sample_period == 0:
                    samples.append(i)
                count += 1
                if count == limit:
                    break
        return samples

    def _bit_at(self, i: int) -> int:
        w = i // self.WORD_BITS
        off = i % self.WORD_BITS
        return (self._words[w] >> off) & 1

    def __len__(self) -> int:
        return self._num_bits

    def rank1(self, i: int) -> int:
        """Number of set bits in [0, i)."""
        if i < 0:
            raise ValueError("i must be non-negative")
        if i > self._num_bits:
            i = self._num_bits
        if i == 0:
            return 0
        block = i // self.BLOCK_BITS
        r = self._rank_blocks[block]
        start = block * self.BLOCK_WORDS
        end = i // self.WORD_BITS
        for j in range(start, end):
            r += self._popcount(self._words[j])
        rem = i % self.WORD_BITS
        if rem:
            r += self._popcount(self._words[end] & ((1 << rem) - 1))
        return r

    def rank0(self, i: int) -> int:
        """Number of unset bits in [0, i)."""
        return i - self.rank1(i)

    def select1(self, k: int) -> int:
        """Index of the (k+1)-th set bit. k is 0-based."""
        if k < 0:
            raise ValueError("k must be non-negative")
        if k >= self._num_ones:
            raise ValueError("k out of range")
        s = self._select1
        period = len(s) - 1  # number of samples taken minus the sentinel
        # samples are at 0, period, 2*period, ...; find the largest sample index <= k
        idx = k // self.SELECT_SAMPLE
        start = s[idx] if idx < len(s) else s[-1]
        # scan from start
        remaining = k - idx * self.SELECT_SAMPLE
        i = start
        while remaining > 0:
            i += 1
            if self._bit_at(i) == 1:
                remaining -= 1
        return i

    def select0(self, k: int) -> int:
        """Index of the (k+1)-th unset bit. k is 0-based."""
        total0 = self._num_bits - self._num_ones
        if k < 0:
            raise ValueError("k must be non-negative")
        if k >= total0:
            raise ValueError("k out of range")
        s = self._select0
        idx = k // self.SELECT_SAMPLE
        start = s[idx] if idx < len(s) else s[-1]
        remaining = k - idx * self.SELECT_SAMPLE
        i = start
        while remaining > 0:
            i += 1
            if self._bit_at(i) == 0:
                remaining -= 1
        return i

    def access(self, i: int) -> int:
        """Return the bit at position i (0-based)."""
        if i < 0 or i >= self._num_bits:
            raise ValueError("index out of range")
        return self._bit_at(i)
