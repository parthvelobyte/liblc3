#!/usr/bin/env python3
# hostsim -- a Target Adapter (protocol 2) for target profile `host`.
#
# Measures liblc3 on this machine. One `execute` is one round:
#   1. build the BASELINE arm: the project's ordinary `make -B` (all objects,
#      bin/liblc3.so, bin/elc3, bin/dlc3, upstream flags incl. -flto -ffast-math)
#   2. build the TUNED arm: compile the scope's translation unit with the
#      workload's declared compile line plus the emitted control flags, delete
#      bin/liblc3.so, then run the declared rebuild (`make`) to relink it
#   3. run the declared entry (tools/bench_lc3.py) for both arms, interleaved,
#      swapping the arm's liblc3.so into bin/ before each run and reading
#      `elapsed_ns=` from stdout
#   4. correctness: every run's checksum lines must equal tools/lc3_golden.txt
#      and the tuned arm's must equal the baseline arm's, byte for byte
#      (the entry also self-checks and exits non-zero on any mismatch)
#   5. answer the matching `poll` with cost (median tuned ns), baseline_cost
#      (median baseline ns), correct, and the digest of the tuned library
#
# Load-bearing rules, from the protocol document:
#   - one JSON line in, one JSON line out, then exit; never crash
#   - a poll for a handle this adapter did not issue is REFUSED
#   - correctness is never inferred from a cost
# This adapter is declared `physical_risk: none`. It runs nothing but this
# project's own benchmark on this host.

import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import uuid

PROTOCOL = 2
ADAPTER_ID = "hostsim"
TARGET_PROFILE = "host"
SOURCE_PATH = "adapters/hostsim/adapter.py"
ADAPTER_VERSION = "0.1.0"
BACKEND = "host-subprocess"
RUNNER = ADAPTER_ID

DEVICE_ID = "host"
STATE_DIR = os.path.join(".evolve", "hostsim-adapter")
HANDLE_RE = re.compile(r"^hostsim-(execute|cost)-[0-9a-f]{12}$")
WORKLOADS = os.path.join(".evolve", "workloads.json")
GOLDEN = os.path.join("tools", "lc3_golden.txt")
METRIC_KEY = "elapsed_ns"
DEFAULT_REPEATS = 5
DEFAULT_WARMUP = 1
# The FP/link contract lives entirely in the workload's declared compile line
# (upstream ships -ffast-math -flto); nothing is appended invariantly here.
INVARIANT_TAIL = []
# The artifact both tools load at runtime; the unit being retuned relinks it.
LIB = os.path.join("bin", "liblc3.so")

PARAMETERS = [
    {"control_id": "compiler.optimization_level", "mutability": "mutable",
     "kind": "enum", "members": ["O0", "O1", "O2", "O3"]},
    {"control_id": "compiler.loop_unroll", "mutability": "mutable", "kind": "boolean"},
    {"control_id": "compiler.loop_vectorize", "mutability": "mutable", "kind": "boolean"},
    {"control_id": "compiler.slp_vectorize", "mutability": "mutable", "kind": "boolean"},
]
DIAGNOSTICS = [
    {"id": "runs", "unit": "count", "cost": "free"},
    {"id": "baseline_runs", "unit": "count", "cost": "free"},
]


class Refusal(Exception):
    """fault=None -> protocol-level refusal; fault=<name> -> outcome refusal."""
    FAULTS = ("artifact_rejected", "execution_faulted", "parameters_refused",
              "device_not_ready", "stale_artifact")

    def __init__(self, detail, fault=None):
        if fault is not None and fault not in Refusal.FAULTS:
            raise ValueError("fault %r is outside the closed set" % fault)
        super().__init__(detail)
        self.detail = detail
        self.fault = fault


def refused(reason):
    return {"protocol": PROTOCOL, "reply": "refused", "reason": reason}


def provenance(run_id):
    return {"target_profile": TARGET_PROFILE, "backend": BACKEND, "runner": RUNNER,
            "adapter_version": ADAPTER_VERSION, "run_id": run_id}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def log(msg):
    sys.stderr.write("hostsim: %s\n" % msg)


def config_int(config, key, default):
    value = config.get(key, default) if isinstance(config, dict) else default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value or value < 0:
        raise Refusal("config `%s` must be a non-negative whole number, got %r" % (key, value))
    return int(value)


def probed_chip():
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5)
            text = out.stdout.decode("utf-8", "replace").strip()
            if text:
                return "%s (%s)" % (text, platform.machine())
    except Exception:
        pass
    return platform.machine() or "unknown"


# ---------------------------------------------------------------------------
# The round
# ---------------------------------------------------------------------------

def run_cmd(argv, what):
    log("run: %s" % " ".join(shlex.quote(a) for a in argv))
    try:
        run = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        raise Refusal("%s could not be started (%s): %s" % (what, exc, argv), fault="artifact_rejected")
    if run.returncode != 0:
        raise Refusal("%s exited %d:\n%s" % (
            what, run.returncode, run.stderr.decode("utf-8", "replace")[-2000:]),
            fault="artifact_rejected")
    return run


def load_workload(workload_id):
    try:
        with open(WORKLOADS) as f:
            doc = json.load(f)
    except Exception as exc:
        raise Refusal("%s could not be read (%s)" % (WORKLOADS, exc), fault="artifact_rejected")
    for w in doc.get("workloads", []):
        if w.get("workload_id") == workload_id:
            return w
    raise Refusal("workload `%s` is not declared in %s" % (workload_id, WORKLOADS),
                  fault="artifact_rejected")


def control_argv(controls):
    """The emitted flags, as argv tokens. Only plain `-flag` tokens pass."""
    argv = []
    for row in controls:
        payload = row.get("payload", "")
        if not isinstance(payload, str):
            raise Refusal("control %r carries a non-string payload" % row.get("control_id"),
                          fault="parameters_refused")
        for tok in shlex.split(payload):
            if not tok.startswith("-") or any(c in tok for c in ";&|<>`$\n"):
                raise Refusal("control %r payload token %r is not a plain compiler flag"
                              % (row.get("control_id"), tok), fault="parameters_refused")
            argv.append(tok)
    return argv


def read_bundle(artifact_path):
    if not isinstance(artifact_path, str) or not artifact_path:
        raise Refusal("this execute carried no `artifact_path`, and a host build needs the bundle",
                      fault="artifact_rejected")
    if os.path.isabs(artifact_path) or ".." in artifact_path.split(os.sep):
        raise Refusal("artifact_path %r must be project-relative" % artifact_path,
                      fault="artifact_rejected")
    try:
        with open(os.path.join(artifact_path, "controls.json")) as f:
            controls = json.load(f).get("controls", [])
        with open(os.path.join(artifact_path, "manifest.json")) as f:
            manifest = json.load(f)
    except Exception as exc:
        raise Refusal("bundle %r is unreadable (%s)" % (artifact_path, exc), fault="artifact_rejected")
    for rel in manifest.get("sources", []):
        if not rel.startswith("source/"):
            continue
        project_file = rel[len("source/"):]
        bundled = os.path.join(artifact_path, rel)
        if not os.path.isfile(project_file) or not os.path.isfile(bundled):
            raise Refusal("bundle names %r but this checkout has no such file" % project_file,
                          fault="stale_artifact")
        if sha256_file(bundled) != sha256_file(project_file):
            raise Refusal("bundled %r differs from this checkout's copy; the bundle is stale"
                          % project_file, fault="stale_artifact")
    workload_id = manifest.get("facts", {}).get("workload_id")
    if not workload_id:
        raise Refusal("bundle manifest names no workload_id", fault="artifact_rejected")
    return controls, manifest, workload_id


def run_entry(workload, lib_path):
    """Swap the arm's library into place, run the entry, parse its stdout."""
    shutil.copy2(lib_path, LIB)
    entry = workload["entry"]
    argv = [os.path.join(".", entry["artifact"])] + list(entry.get("argv", []))
    env = dict(os.environ, LC3_BENCH_REPEATS="1")
    try:
        run = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    except OSError as exc:
        raise Refusal("entry %r could not be started (%s)" % (argv, exc), fault="execution_faulted")
    if run.returncode != 0:
        raise Refusal("entry exited %d for %s:\n%s" % (
            run.returncode, lib_path, run.stderr.decode("utf-8", "replace")[-2000:]),
            fault="execution_faulted")
    lines = run.stdout.decode("utf-8", "replace").splitlines()
    metric = None
    checksum_lines = []
    for line in lines:
        m = re.match(r"^\s*%s\s*=\s*(\d+)\s*$" % re.escape(METRIC_KEY), line)
        if m:
            if metric is not None:
                raise Refusal("entry printed `%s` twice" % METRIC_KEY, fault="execution_faulted")
            metric = int(m.group(1))
        else:
            checksum_lines.append(line)
    if metric is None:
        raise Refusal("entry printed no `%s=` line" % METRIC_KEY, fault="execution_faulted")
    return metric, "\n".join(checksum_lines).strip()


def execute_round(request, handle):
    config = request.get("config") or {}
    repeats = config_int(config, "repeats", DEFAULT_REPEATS)
    warmup = config_int(config, "warmup", DEFAULT_WARMUP)
    if repeats < 1:
        raise Refusal("config `repeats` must be at least 1")

    controls, manifest, workload_id = read_bundle(request.get("artifact_path"))
    workload = load_workload(workload_id)
    compile_ = workload["compile"]
    rebuild = workload["rebuild"]
    entry_artifact = workload["entry"]["artifact"]
    if workload["entry"].get("kind") != "built_artifact":
        raise Refusal("entry kind %r is not built_artifact" % workload["entry"].get("kind"),
                      fault="artifact_rejected")
    if not os.access(entry_artifact, os.X_OK):
        raise Refusal("entry %r is not executable" % entry_artifact, fault="artifact_rejected")

    extra = control_argv(controls)
    tuned_object = None
    argv = list(compile_["argv"])
    if "-o" in argv:
        tuned_object = argv[argv.index("-o") + 1]

    os.makedirs(STATE_DIR, exist_ok=True)
    baseline_lib = os.path.join(STATE_DIR, handle + "-baseline.so")
    tuned_lib = os.path.join(STATE_DIR, handle + "-tuned.so")

    try:
        # Arm 1: the ordinary build.
        run_cmd([rebuild["program"], "-B"] + list(rebuild["argv"]), "baseline rebuild")
        if not os.path.isfile(LIB):
            raise Refusal("the rebuild produced no %s" % LIB, fault="artifact_rejected")
        shutil.copy2(LIB, baseline_lib)

        # Arm 2: tuned translation unit, then the declared rebuild relinks it.
        # GNU make 3.81 (macOS) compares mtimes at whole-second granularity,
        # so remove the library first so the rebuild must relink it.
        run_cmd([compile_["program"]] + argv + INVARIANT_TAIL + extra, "tuned compile")
        if os.path.isfile(LIB):
            os.remove(LIB)
        run_cmd([rebuild["program"]] + list(rebuild["argv"]), "tuned rebuild")
        if not os.path.isfile(LIB):
            raise Refusal("the tuned rebuild produced no %s" % LIB, fault="artifact_rejected")
        shutil.copy2(LIB, tuned_lib)
        materialised = sha256_file(tuned_lib)
        identical = materialised == sha256_file(baseline_lib)

        golden = None
        if os.path.isfile(GOLDEN):
            with open(GOLDEN) as f:
                golden = f.read().strip()

        base_times, tuned_times = [], []
        base_out, tuned_out = [], []
        for _ in range(warmup):
            run_entry(workload, baseline_lib)
            run_entry(workload, tuned_lib)
        for i in range(repeats):
            order = [("b", baseline_lib), ("t", tuned_lib)]
            if i % 2:
                order.reverse()
            for arm, lib in order:
                ns, text = run_entry(workload, lib)
                (base_times if arm == "b" else tuned_times).append(ns)
                (base_out if arm == "b" else tuned_out).append(text)
    finally:
        # Leave the tree byte-identical to the registered baseline state:
        # baseline library back in place and the tuned object recompiled with
        # the workload's declared (baseline) compile line.
        try:
            if os.path.isfile(baseline_lib):
                shutil.copy2(baseline_lib, LIB)
            if tuned_object:
                subprocess.run([compile_["program"]] + list(compile_["argv"]),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as exc:
            log("cleanup: %s" % exc)

    problems = []
    if golden is not None and any(o != golden for o in base_out):
        problems.append("baseline output differs from tools/lc3_golden.txt")
    if any(o != base_out[0] for o in base_out):
        problems.append("baseline output is not stable across runs")
    if any(o != base_out[0] for o in tuned_out):
        problems.append("tuned output differs from baseline output")
    correct = not problems
    detail = ("tuned checksums identical to baseline%s across %d runs; tuned library %s baseline library" % (
        " and tools/lc3_golden.txt" if golden is not None else "", repeats,
        "byte-identical to" if identical else "differs from")
        if correct else "; ".join(problems))
    return {
        "cost": int(statistics.median(tuned_times)),
        "baseline_cost": int(statistics.median(base_times)),
        "unit": "ns",
        "correct": correct,
        "correctness_detail": detail,
        "materialised": materialised,
        "runs": len(tuned_times),
        "baseline_runs": len(base_times),
        "tuned_times": tuned_times,
        "baseline_times": base_times,
        "flags": extra,
        "binary_identical": identical,
        "workload_id": workload_id,
    }


# ---------------------------------------------------------------------------
# Handles and state
# ---------------------------------------------------------------------------

def new_handle(kind):
    return "hostsim-%s-%s" % (kind, uuid.uuid4().hex[:12])


def state_path(handle):
    return os.path.join(STATE_DIR, handle + ".json")


def save_state(handle, record):
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = state_path(handle) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f)
    os.replace(tmp, state_path(handle))


def completed_outcome(request, handle, result):
    requested = request.get("requested_fidelity", "measured")
    accepted = "measured" if requested == "measured" else requested
    out = {
        "protocol": PROTOCOL, "reply": "outcome", "state": "completed",
        "requested_fidelity": requested, "accepted_fidelity": accepted,
        "cost": result["cost"], "baseline_cost": result["baseline_cost"], "unit": result["unit"],
        "correct": result["correct"], "correctness_detail": result["correctness_detail"],
        "artifact_digests": {"submitted": request.get("artifact"), "materialised": result["materialised"]},
    }
    if accepted == "measured":
        out["wall_ns"] = result["cost"]
    wanted = request.get("diagnostics") or []
    readings = []
    for d in DIAGNOSTICS:
        if d["id"] in wanted and d["id"] in result:
            readings.append({"id": d["id"], "value": int(result[d["id"]]), "unit": d["unit"]})
    if readings:
        out["diagnostics"] = readings
    out.update(provenance(handle))
    return out


def refused_outcome(handle, fault, detail):
    out = {"protocol": PROTOCOL, "reply": "outcome", "state": "refused",
           "fault": fault, "detail": detail}
    out.update(provenance(handle))
    return out


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def answer(request):
    protocol = request.get("protocol")
    if protocol != PROTOCOL:
        return refused("adapter `%s` speaks adapter protocol %s and this request declared protocol %s"
                       % (ADAPTER_ID, PROTOCOL, protocol))
    operation = request.get("operation")

    if operation == "describe":
        return {"protocol": PROTOCOL, "reply": "payload", "adapter_id": ADAPTER_ID,
                "adapter_version": ADAPTER_VERSION, "target_profile": TARGET_PROFILE}

    if operation == "devices":
        return {"protocol": PROTOCOL, "reply": "payload", "devices": [{
            "device_id": DEVICE_ID, "transport": "local", "status": "ready",
            "probed_chip": probed_chip()}]}

    if operation == "parameters":
        device = request.get("device")
        if device not in (None, DEVICE_ID):
            return refused("device %r is not one this adapter lists; it lists %r" % (device, DEVICE_ID))
        return {"protocol": PROTOCOL, "reply": "payload", "parameters": PARAMETERS}

    if operation == "diagnostics.catalogue":
        return {"protocol": PROTOCOL, "reply": "payload", "diagnostics": DIAGNOSTICS}

    if operation == "cost":
        handle = new_handle("cost")
        save_state(handle, {"kind": "cost", "request": request})
        return {"protocol": PROTOCOL, "reply": "handle", "handle": handle}

    if operation == "execute":
        handle = new_handle("execute")
        device = request.get("device")
        if device != DEVICE_ID:
            save_state(handle, {"kind": "execute", "outcome": refused_outcome(
                handle, "device_not_ready", "device %r is not attached; this adapter lists %r"
                % (device, DEVICE_ID))})
            return {"protocol": PROTOCOL, "reply": "handle", "handle": handle}
        try:
            result = execute_round(request, handle)
            outcome = completed_outcome(request, handle, result)
            save_state(handle, {"kind": "execute", "outcome": outcome, "result": result})
        except Refusal as r:
            fault = r.fault or "execution_faulted"
            save_state(handle, {"kind": "execute", "outcome": refused_outcome(handle, fault, r.detail)})
        except Exception as exc:
            save_state(handle, {"kind": "execute", "outcome": refused_outcome(
                handle, "execution_faulted", "adapter defect in %s: %s: %s" % (SOURCE_PATH, type(exc).__name__, exc))})
        return {"protocol": PROTOCOL, "reply": "handle", "handle": handle}

    if operation == "poll":
        handle = request.get("handle")
        if not isinstance(handle, str) or not HANDLE_RE.match(handle):
            return refused("handle %r was not issued by adapter `%s`; an outcome for a submission "
                           "that never happened is a fabricated result" % (handle, ADAPTER_ID))
        if not os.path.isfile(state_path(handle)):
            return refused("handle %r is not one adapter `%s` has a record of issuing" % (handle, ADAPTER_ID))
        with open(state_path(handle)) as f:
            record = json.load(f)
        if record.get("kind") == "cost":
            out = {"protocol": PROTOCOL, "reply": "outcome", "state": "completed",
                   "requested_fidelity": record["request"].get("requested_fidelity", "estimate"),
                   "accepted_fidelity": "estimate",
                   "artifact_digests": {"submitted": record["request"].get("artifact")},
                   "detail": "this host target has no cost model; nothing was run and no cost is reported"}
            out.update(provenance(handle))
            return out
        return record["outcome"]

    return refused("%r is not an operation adapter protocol %s defines" % (operation, PROTOCOL))


def main():
    line = sys.stdin.readline()
    try:
        request = json.loads(line)
        reply = answer(request) if isinstance(request, dict) else refused(
            "the request line is not a JSON object: %.120s" % line.strip())
    except Exception as exc:
        reply = refused("adapter `%s` could not handle that request (%s: %s); a defect in %s"
                        % (ADAPTER_ID, type(exc).__name__, exc, SOURCE_PATH))
    sys.stdout.write(json.dumps(reply) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
