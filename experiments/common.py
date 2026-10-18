"""Shared configuration: device snapshots and the deterministic qubit-pair selection rule."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

from ewlcert import device as Dv          # noqa: E402
from ewlcert import theory as Th          # noqa: E402

BACKENDS = ["fake_torino", "fake_fez", "fake_marrakesh",          # IBM Heron (CZ)
            "fake_brisbane", "fake_sherbrooke", "fake_kyiv"]      # IBM Eagle (ECR)
QUANTILES = (0.1, 0.5, 0.9)                                       # short / typical / long coherence


def eff_T2(T1, T2):
    return min(T2, 2 * T1)                # the device noise model caps T2 at 2 T1


def pair_table(backend):
    """All couplers with their calibration data and the predicted critical idle time."""
    from qiskit_aer.noise import NoiseModel
    nm = NoiseModel.from_backend(backend)
    rows = []
    for a, b in Dv.candidate_pairs(backend):
        pa, pb = Dv.qubit_params(backend, a), Dv.qubit_params(backend, b)
        vals = [pa["T1"], pa["T2"], pb["T1"], pb["T2"]]
        if any(v is None for v in vals) or not all(np.isfinite(vals)):   # live devices: missing calibrations
            continue
        ro = [float(nm._local_readout_errors[(q,)].probabilities[0][1]) if (q,) in nm._local_readout_errors else np.nan
              for q in (a, b)]
        tc = Th.critical_idle_time(pa["T1"], eff_T2(pa["T1"], pa["T2"]), pb["T1"], eff_T2(pb["T1"], pb["T2"]))
        rows.append(dict(pair=(a, b), T1=(pa["T1"], pb["T1"]), T2=(pa["T2"], pb["T2"]),
                         T2eff=(eff_T2(pa["T1"], pa["T2"]), eff_T2(pb["T1"], pb["T2"])),
                         readout=ro, err2q=Dv.edge_error(backend, a, b), tc_theory=tc))
    return rows


def select_pairs(backend):
    """Couplers whose 2q-gate error and both readout errors are at or below the device median; among
    them the pairs at the 10th/50th/90th percentile of the predicted critical idle time."""
    rows = pair_table(backend)
    e2 = np.nanmedian([r["err2q"] for r in rows])
    ro = np.nanmedian([max(r["readout"]) for r in rows])
    good = [r for r in rows if r["err2q"] <= e2 and max(r["readout"]) <= ro and np.isfinite(r["tc_theory"])]
    good.sort(key=lambda r: r["tc_theory"])
    picks = []
    for qnt, label in zip(QUANTILES, ("short", "typical", "long")):
        r = dict(good[int(round(qnt * (len(good) - 1)))])
        r["label"] = label
        picks.append(r)
    return picks, dict(n_couplers=len(rows), n_eligible=len(good), median_err2q=e2, median_readout=ro)


def memory_guard(min_free_gb=1.5):
    """Abort instead of letting the machine run out of memory."""
    import psutil
    free = psutil.virtual_memory().available / 2**30
    if free < min_free_gb:
        raise MemoryError(f"only {free:.1f} GB RAM free; stopping to protect the machine")


def save(name, obj):
    def conv(o):
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, tuple):
            return list(o)
        raise TypeError(type(o))
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=1, default=conv)


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)
