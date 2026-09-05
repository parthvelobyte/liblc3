#!/usr/bin/env python3
"""Deterministic LC3 benchmark for Evo region tuning.

Runs a fixed encode/decode matrix over test/corpus48k.wav (itself generated
deterministically by tools/gen_corpus.py), byte-checks every output against
tools/lc3_golden.txt, and prints one line:

    elapsed_ns=<median over repeats of the summed matrix wall time>

Any checksum mismatch exits non-zero with a diagnostic on stderr; a trial
that changed the encoded bits is a failed trial, never a fast one.
"""
import hashlib
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ELC3 = os.path.join(ROOT, "bin", "elc3")
DLC3 = os.path.join(ROOT, "bin", "dlc3")
CORPUS = os.path.join(ROOT, "test", "corpus48k.wav")
GOLDEN = os.path.join(ROOT, "tools", "lc3_golden.txt")

# (name, kind, bitrate, frame_ms) — fixed matrix, standard configs.
CONFIGS = [
    ("enc_32k_10ms", "enc", "32000", "10"),
    ("enc_96k_10ms", "enc", "96000", "10"),
    ("enc_96k_7m5", "enc", "96000", "7.5"),
    ("dec_96k_10ms", "dec", "96000", "10"),
]

REPEATS = int(os.environ.get("LC3_BENCH_REPEATS", "5"))
WARMUP = 1


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_matrix(tmp):
    """Run the full matrix once; return (total_ns, {name: sha})."""
    total = 0
    sums = {}
    enc_out = {}
    for name, kind, bitrate, frame in CONFIGS:
        if kind == "enc":
            out = os.path.join(tmp, name + ".lc3")
            argv = [ELC3, "-b", bitrate, "-m", frame, CORPUS, out]
        else:
            src = enc_out[(bitrate, frame)]
            out = os.path.join(tmp, name + ".wav")
            argv = [DLC3, src, out]
        t0 = time.perf_counter_ns()
        subprocess.run(argv, check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        total += time.perf_counter_ns() - t0
        if kind == "enc":
            enc_out[(bitrate, frame)] = out
        sums[name] = sha(out)
    return total, sums


def main():
    write_golden = "--write-golden" in sys.argv
    with tempfile.TemporaryDirectory() as tmp:
        for _ in range(WARMUP):
            _, sums = run_matrix(tmp)
        times = []
        for _ in range(REPEATS):
            t, sums = run_matrix(tmp)
            times.append(t)

    if write_golden:
        with open(GOLDEN, "w") as f:
            for name, digest in sorted(sums.items()):
                f.write(f"{name} {digest}\n")
        print(f"wrote {GOLDEN}", file=sys.stderr)
    else:
        golden = {}
        with open(GOLDEN) as f:
            for line in f:
                name, digest = line.split()
                golden[name] = digest
        bad = [n for n in golden if sums.get(n) != golden[n]]
        if bad or set(sums) != set(golden):
            print(f"OUTPUT MISMATCH: {bad or 'config set changed'}",
                  file=sys.stderr)
            sys.exit(1)

    times.sort()
    print(f"elapsed_ns={times[len(times) // 2]}")


if __name__ == "__main__":
    main()
