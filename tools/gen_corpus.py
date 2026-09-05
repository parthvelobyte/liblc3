#!/usr/bin/env python3
"""Deterministic benchmark corpus: 48 kHz mono 16-bit, 180 s.

Three fixed tones plus LCG noise (seed 0x1234567, the classic
Numerical Recipes constants), so every machine regenerates the
byte-identical file. Regenerate with:  python3 tools/gen_corpus.py
"""
import math
import os
import struct
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "test", "corpus48k.wav")

SR = 48000
DUR = 180

s = 0x1234567


def rnd():
    global s
    s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
    return (s >> 8) / 16777216.0 - 0.5


frames = bytearray()
for i in range(SR * DUR):
    t = i / SR
    v = (0.35 * math.sin(2 * math.pi * 440 * t)
         + 0.2 * math.sin(2 * math.pi * 1873 * t)
         + 0.15 * math.sin(2 * math.pi * 7900 * t)
         + 0.12 * rnd())
    frames += struct.pack("<h", max(-32768, min(32767, int(v * 32767))))

w = wave.open(OUT, "wb")
w.setnchannels(1)
w.setsampwidth(2)
w.setframerate(SR)
w.writeframes(bytes(frames))
w.close()
print("wrote", OUT)
