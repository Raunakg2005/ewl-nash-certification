"""Run the certification circuits and an idle-delay sweep on an IBM quantum processor.

Usage (the IBM account must have been saved beforehand with QiskitRuntimeService.save_account; this
script never asks for or stores an API key):

  python hardware_run.py --dry-run                      # offline rehearsal on a device snapshot
  python hardware_run.py                                # plan only: backend, pair, circuits, QPU-time estimate
  python hardware_run.py --submit                       # submit ONE job and wait for the result

Design: one qubit pair (the 'typical' pair of the selection rule used in the paper, evaluated on the
backend's calibration data at run time), idle delays tau = f * tau_c^pred with f in FRACTIONS, the 11
certification circuits per delay plus two readout-calibration circuits (|00>, |11>) per delay.
Dynamical decoupling and twirling are switched OFF, because they would suppress the very dephasing
during the idle period that the experiment measures.  Everything needed for the analysis (job id,
calibration snapshot, circuit list, raw counts) is written to results/hardware_*.json.
"""
import argparse
import datetime as dt
import json
import os
import time
import warnings

import numpy as np
from qiskit import QuantumCircuit, transpile

import common as C
from ewlcert import device as Dv
from ewlcert import game as G
from ewlcert import stats as S

warnings.filterwarnings("ignore")
FRACTIONS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5]


def read_env(path):
    """Minimal KEY=VALUE reader for redo/.env (values are never printed)."""
    out = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def calibration_snapshot(backend, pair):
    t = backend.target
    snap = {"backend": backend.name, "taken_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "qubits": {}}
    for q in pair:
        p = t.qubit_properties[q]
        m = t["measure"][(q,)]
        snap["qubits"][str(q)] = {"T1": p.t1, "T2": p.t2, "readout_error": m.error if m else None}
    snap["err2q"] = Dv.edge_error(backend, *pair)
    try:
        props = backend.properties()
        snap["properties_last_update"] = str(props.last_update_date) if props else None
    except Exception:
        snap["properties_last_update"] = None
    return snap


def measured(qc_logical):
    qc = qc_logical.copy()
    qc.measure_all()
    return qc


def build(backend, pair, tc):
    """Logical->physical circuits with metadata."""
    circs, meta = [], []
    for f in FRACTIONS:
        tau = f * tc
        for k, (a, b) in enumerate(S.circuit_strategies()):
            circs.append(measured(Dv.logical_circuit(a, b, tau, backend=backend)))
            meta.append({"f": f, "tau": tau, "kind": "game", "index": k})
        cal0 = QuantumCircuit(2)
        cal1 = QuantumCircuit(2)
        cal1.x([0, 1])
        for c, kind in ((cal0, "cal00"), (cal1, "cal11")):
            circs.append(measured(c))
            meta.append({"f": f, "tau": tau, "kind": kind, "index": None})
    phys = transpile(circs, backend, initial_layout=list(pair), optimization_level=1, seed_transpiler=7)
    return phys, meta


def counts_to_vec(counts):
    """Qiskit bitstring 'ba' (clbit 0 = Alice, rightmost) -> our order CC, CD, DC, DD = index 2a+b."""
    v = np.zeros(4, int)
    for key, n in counts.items():
        bits = key.replace(" ", "")[-2:]
        a, b = int(bits[1]), int(bits[0])
        v[2 * a + b] += n
    return v


def qpu_seconds_estimate(meta, shots, rep_delay=250e-6):
    per_shot = [m["tau"] + 5e-6 + rep_delay for m in meta]
    return shots * float(np.sum(per_shot)) + 5.0 * len(meta) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None, help="IBM backend name (default: least busy available)")
    ap.add_argument("--shots", type=int, default=5000)
    ap.add_argument("--submit", action="store_true", help="actually submit the job")
    ap.add_argument("--dry-run", default=None, nargs="?", const="fake_fez",
                    help="offline rehearsal on a device snapshot (default fake_fez)")
    ap.add_argument("--fetch", default=None, help="collect the result of an already submitted job: results/<file>.json")
    ap.add_argument("--pair", default=None, help="force a qubit pair, e.g. 16,23 (default: selection rule)")
    ap.add_argument("--delays-us", default=None, help="absolute idle delays start:stop:step in us, e.g. 0:100:5")
    ap.add_argument("--max-seconds", type=int, default=480, help="hard cap on QPU execution time")
    args = ap.parse_args()
    C.memory_guard(0.8)          # single light process (~0.4 GB); heavy simulations use the 1.5 GB default

    if args.fetch:
        from qiskit_ibm_runtime import QiskitRuntimeService
        env = read_env(os.path.join(C.ROOT, ".env"))
        service = (QiskitRuntimeService(channel="ibm_quantum_platform", token=env["IBM_QUANTUM_TOKEN"],
                                        instance=env.get("IBM_QUANTUM_INSTANCE") or None)
                   if env.get("IBM_QUANTUM_TOKEN") else QiskitRuntimeService())
        record = C.load(os.path.basename(args.fetch))
        job = service.job(record["job_id"])
        print("job", record["job_id"], "status:", job.status(), flush=True)
        result = job.result()
        record["counts"] = [counts_to_vec(r.data.meas.get_counts()).tolist() for r in result]
        record["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        try:
            record["usage_seconds"] = job.usage()
        except Exception:
            record["usage_seconds"] = None
        C.save(os.path.basename(args.fetch), record)
        print("counts saved to", args.fetch)
        return

    if args.dry_run:
        backend = Dv.backend_by_name(args.dry_run)
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        env = read_env(os.path.join(C.ROOT, ".env"))
        if env.get("IBM_QUANTUM_TOKEN"):                      # credentials from redo/.env (never printed)
            print("using credentials from .env")
            service = QiskitRuntimeService(channel="ibm_quantum_platform", token=env["IBM_QUANTUM_TOKEN"],
                                           instance=env.get("IBM_QUANTUM_INSTANCE") or None)
        else:
            service = QiskitRuntimeService()                  # locally saved account
        backend = service.backend(args.backend) if args.backend else service.least_busy(operational=True, simulator=False)

    global FRACTIONS
    picks, info = C.select_pairs(backend)
    pick = [p for p in picks if p["label"] == "typical"][0]
    if args.pair:
        want = tuple(sorted(int(q) for q in args.pair.split(",")))
        pick = dict([r for r in C.pair_table(backend) if tuple(sorted(r["pair"])) == want][0])
        pick["label"] = "forced"
    pair, tc = tuple(pick["pair"]), pick["tc_theory"]
    if args.delays_us:
        a, b, st = (float(x) for x in args.delays_us.split(":"))
        FRACTIONS = [round(d * 1e-6 / tc, 6) for d in np.arange(a, b + st / 2, st)]
    snap = calibration_snapshot(backend, pair)
    phys, meta = build(backend, pair, tc)
    est = qpu_seconds_estimate(meta, args.shots)
    print(f"backend {backend.name}; pair {pair}; T1 {[round(x*1e6) for x in pick['T1']]} us; "
          f"T2 {[round(x*1e6) for x in pick['T2']]} us; predicted tau_c {tc*1e6:.1f} us")
    print(f"{len(phys)} circuits x {args.shots} shots; estimated QPU time ~{est/60:.1f} min")

    tag = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    record = dict(tag=tag, backend=backend.name, pair=pair, tc_pred=tc, fractions=FRACTIONS, shots=args.shots,
                  calibration=snap, selection=info, pick={k: pick[k] for k in ("T1", "T2", "T2eff", "readout", "err2q")},
                  meta=meta, dry_run=bool(args.dry_run))
    # device-model prediction from the same calibration data (reduced two-qubit simulation)
    dp = Dv.DevicePair(backend, pair)
    grid = np.round(np.linspace(0, 1.6, 17), 3).tolist()
    model = []
    for f in grid:
        P = dp.run([dp.physical(a, b, f * tc) for a, b in S.circuit_strategies()])
        model.append({"f": f, "gap": S.signed_gap(P), "dgain": S.mean_D_gain(P), "eps": S.eps_from_probs(P),
                      "gap_readout": S.signed_gap(dp.with_readout(P))})
    model_tc = np.nan
    for i in range(len(model) - 1):
        if model[i]["dgain"] < 0 <= model[i + 1]["dgain"]:
            g0, g1 = model[i]["dgain"], model[i + 1]["dgain"]
            model_tc = tc * (grid[i] + (grid[i + 1] - grid[i]) * (-g0) / (g1 - g0))
            break
    record["model"] = model
    record["model_tc"] = model_tc
    print(f"device-model prediction: tau_c ~ {model_tc*1e6:.1f} us")

    if args.dry_run:
        # offline rehearsal: (1) bit-order check on an ideal simulator, (2) synthetic counts from the
        # validated reduced device model, sampled like real data
        from qiskit_aer import AerSimulator
        ideal = AerSimulator()
        test = [measured(Dv.logical_circuit(a, b, 0.0)) for a, b in S.circuit_strategies()]
        res = ideal.run(transpile(test, ideal), shots=4000, seed_simulator=1).result()
        for i, (a, b) in enumerate(S.circuit_strategies()):
            v = counts_to_vec(res.get_counts(i)) / 4000
            assert np.allclose(v, G.probabilities(G.U_x(a), G.U_x(b)), atol=0.04), "bit-order mismatch"
        print("bit-order check passed")
        rng = np.random.default_rng(99)
        counts = []
        for m in meta:
            if m["kind"] == "game":
                a, b = S.circuit_strategies()[m["index"]]
                p = dp.with_readout(dp.run([dp.physical(a, b, m["tau"])]))[0]
            else:
                p = dp.confusion[:, 0 if m["kind"] == "cal00" else 3]
            counts.append(rng.multinomial(args.shots, np.clip(p, 0, None) / np.clip(p, 0, None).sum()).tolist())
        record["counts"] = counts
        record["job_id"] = "dry-run"
        C.save(f"hardware_dryrun_{backend.name}.json", record)
        print("dry run complete:", os.path.join(C.RESULTS, f"hardware_dryrun_{backend.name}.json"))
        return

    if not args.submit:
        print("plan only; re-run with --submit to send the job")
        return

    from qiskit_ibm_runtime import SamplerV2
    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = args.shots
    sampler.options.dynamical_decoupling.enable = False
    sampler.options.twirling.enable_gates = False
    sampler.options.twirling.enable_measure = False
    sampler.options.max_execution_time = args.max_seconds      # hard cap on QPU time
    job = sampler.run(phys)
    record["job_id"] = job.job_id()
    record["submitted_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    C.save(f"hardware_job_{tag}.json", record)
    print("submitted job", job.job_id(), "- saved metadata; waiting for the result ...", flush=True)
    result = job.result()
    record["counts"] = [counts_to_vec(r.data.meas.get_counts()).tolist() for r in result]
    record["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        record["usage_seconds"] = job.usage()
    except Exception:
        record["usage_seconds"] = None
    C.save(f"hardware_job_{tag}.json", record)
    print("done; counts saved to", os.path.join(C.RESULTS, f"hardware_job_{tag}.json"))


if __name__ == "__main__":
    main()
