# Python encodings of the value-level identities behind liblc3 rewrites,
# mirrors tools/identities.c for the Python ESBMC lane.

def lc3_symbol_probe(low: int, range_: int, L: int) -> int:
    range_ = range_ & 0x3FFF
    if range_ < 0x40:
        range_ = 0x40
    low = low & 0xFFFFFF
    L = L & 0xFFFF
    if low < range_ * L:
        return 1
    return 0
