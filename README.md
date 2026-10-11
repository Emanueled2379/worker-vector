# worker-vector

A succinct bit vector supporting `rank` and `select` operations in compact space.

```python
from worker_vector import SuccinctBitVector

sbv = SuccinctBitVector([1, 0, 1, 1, 0, 0, 1, 0, 1])
print(sbv.rank1(5))   # 3 — ones in [0,5)
print(sbv.select1(2))  # 3 — index of the 3rd one (0-based k=2)
print(sbv.access(4))  # 0
```

## Why

Rank and select are the two primitives underneath suffix-array lookups, FM-indexes, and
wavelet trees. A plain Python list of booleans costs roughly 8 bytes per bit; this library
packs 64 bits per word and keeps a two-level index (one rank block per 512 bits, one select
sample per 8192-th bit) so the overhead stays sublinear.

The trade-off is simplicity over density. We store sampled integers as boxed Python ints,
so `o(n)` has a large constant. For a few million bits it is fine; for a few hundred million
you want a C extension.

## Edge cases

`rank1(i)` counts ones in the half-open range `[0, i)`, so `rank1(0)` is always 0 and
`rank1(len)` equals the total popcount. Positions `i` past the end are clamped to `len`.

`select1(k)` / `select0(k)` are 0-based: `select1(0)` returns the index of the first set bit.
They raise `ValueError` when `k` exceeds the available count. An empty vector has no
selectable bits, so every `select` call raises.
