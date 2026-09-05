#!/usr/bin/env python3
"""z3 proof of the comparison identity used by lc3_get_symbol().

Claim: for the unsigned values the LC3 arithmetic decoder maintains,

    low < range * L      <=>      low / range < L

so the quotient low/range, computed once, can replace the per-probe
multiply range*symbols[s].low in the symbol binary search.

The C arithmetic is unsigned 32-bit, but no expression can wrap:
  low   < 2^24                (ac->low is masked to 24 bits everywhere)
  range in [0x40, 0x3fff]     (ac->range is initialized to 0xffffff and
                               every renorm restores >= 0x10000, so
                               (ac->range >> 10) & 0xffff >= 0x40)
  L     < 2^16                (uint16_t cumulative frequency)
  range * L < 2^30            (0x3fff * 0xffff)
so 32-bit unsigned ops coincide with integer arithmetic on this domain,
and the identity is proved over the integers. The quotient q = low/range
is characterized by the division axioms low = q*range + rem, 0 <= rem <
range, which C unsigned division satisfies by definition.
"""
import z3

low, rng, L, q, rem = z3.Ints("low rng L q rem")

pre = z3.And(
    low >= 0, low < 2**24,
    rng >= 0x40, rng <= 0x3FFF,
    L >= 0, L < 2**16,
    low == q * rng + rem,      # division axioms: q, rem are THE
    rem >= 0, rem < rng,       # quotient and remainder of low / rng
    q >= 0,
)

mul_form = low < rng * L
div_form = q < L

s = z3.Solver()
s.add(pre, mul_form != div_form)
res = s.check()
print("negation of the identity:", res)
assert res == z3.unsat, "identity REFUTED"
print("PROVED: low < range*L  <=>  low/range < L on the decoder's domain")
