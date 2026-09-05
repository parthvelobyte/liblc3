#!/usr/bin/env python3
"""z3 proof for rewrite R1 (lc3_get_symbol, src/bits.h) — division-axiom form.

Claim: with q = a / r (unsigned integer division, r >= 1):
    a < r * L   <=>   q < L

Rather than using z3's UDiv (which is expensive), characterize q by the
division axioms: a = q*r + rem with 0 <= rem < r. Any (q, rem) satisfying
them IS the unsigned quotient/remainder pair, so proving the equivalence
for all such (q, rem) proves it for the actual division.

Ranges (bits.h invariants): a < 2^24, 0x40 <= r <= 0xffff, L < 2^16.
40-bit vectors: every product q*r + rem <= a < 2^24 and r*L < 2^32 is
exact, no wraparound in the constrained space.
"""
import z3

a = z3.BitVec("a", 40)
r = z3.BitVec("r", 40)
L = z3.BitVec("L", 40)
q = z3.BitVec("q", 40)
rem = z3.BitVec("rem", 40)

pre = z3.And(
    z3.ULT(a, 1 << 24),
    z3.UGE(r, 0x40), z3.ULE(r, 0xFFFF),
    z3.ULE(L, 0xFFFF),
    a == q * r + rem,          # division axioms: q, rem are THE
    z3.ULT(rem, r),            # quotient and remainder of a / r
    z3.ULE(q, a),              # (implied; keeps products in range)
)

mul_form = z3.ULT(a, r * L)
div_form = z3.ULT(q, L)

s = z3.Solver()
s.add(pre, mul_form != div_form)
res = s.check()
print("negation:", res)
assert res == z3.unsat, "R1 identity REFUTED"
print("R1 PROVED: a < r*L  <=>  a/r < L on the decoder's ranges")
