"""Device-calibrated simulation of the EWL certification circuits with Qiskit Aer.

Each circuit is compiled for a fixed physical qubit pair of an IBM device snapshot (qiskit-ibm-runtime
fake backends), scheduled so that idle periods carry thermal relaxation, optionally folded for
zero-noise extrapolation (G -> G (G^dagger G)^k, noise scale 2k+1), and simulated with the device
noise model (gate depolarizing + thermal relaxation) in density-matrix mode.  Readout errors are
applied afterwards through the device's confusion matrices, which is equivalent to Aer's classical
readout model and yields exact outcome distributions (finite-shot data are sampled from them).
"""
from __future__ import annotations

import warnings

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RYYGate, UnitaryGate
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from . import game as G

warnings.filterwarnings("ignore", category=DeprecationWarning)
QISKIT_ORDER = [0, 2, 1, 3]       # our index 2a+b  <-  qiskit index a + 2b (Alice = first qubit)


def backend_by_name(name: str):
    from qiskit_ibm_runtime import fake_provider as fp
    cls = "Fake" + "".join(w.capitalize() for w in name.replace("fake_", "").split("_"))
    return getattr(fp, cls)()


def qubit_params(backend, q: int) -> dict:
    p = backend.target.qubit_properties[q]
    return {"T1": p.t1, "T2": p.t2}


def two_qubit_gate(backend) -> str:
    return "ecr" if "ecr" in backend.operation_names else "cz"


def edge_error(backend, a: int, b: int) -> float:
    g = two_qubit_gate(backend)
    props = backend.target[g]
    e = props.get((a, b)) or props.get((b, a))
    return float(e.error) if e is not None and e.error is not None else np.nan


def readout_matrices(backend, noise_model: NoiseModel, pair) -> np.ndarray:
    """4x4 confusion matrix P(measured | true) in our (a, b) ordering."""
    mats = []
    for q in pair:
        ro = noise_model._local_readout_errors.get((q,))
        if ro is None:
            mats.append(np.eye(2))
        else:
            mats.append(np.asarray(ro.probabilities).T)      # columns = true state
    return np.kron(mats[0], mats[1])


def candidate_pairs(backend):
    seen = set()
    for a, b in backend.coupling_map.get_edges():
        e = tuple(sorted((a, b)))
        if e not in seen:
            seen.add(e)
            yield e


def delay_ticks(backend, tau: float) -> int:
    """Idle time in device clock ticks, rounded to the device's timing granularity."""
    g = backend.target.timing_constraints().granularity
    return int(round(tau / backend.dt / g)) * g


def logical_circuit(xA, xB, tau: float, gamma: float = np.pi / 2, backend=None) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.append(RYYGate(gamma), [0, 1])                        # J(gamma) = exp(-i gamma/2 Y(x)Y)
    if tau > 0:
        if backend is None:
            qc.delay(tau, [0, 1], unit="s")
        else:
            n = delay_ticks(backend, tau)
            if n > 0:
                qc.delay(n, [0, 1], unit="dt")
    qc.append(UnitaryGate(G.U_x(xA), label="A"), [0])
    qc.append(UnitaryGate(G.U_x(xB), label="B"), [1])
    qc.append(RYYGate(-gamma), [0, 1])
    return qc


class DevicePair:
    """A physical qubit pair of a device snapshot; Alice = pair[0], Bob = pair[1].

    Circuits are compiled and scheduled for the full device, then REDUCED to the two active qubits:
    every gate is followed by the device noise model's own error channel for that gate and those
    qubits, and every scheduled idle period (delay) is replaced by the thermal-relaxation channel of
    its duration (T2 capped at 2 T1, as in Aer's device model).  Simulating this two-qubit circuit is
    equivalent to simulating the full-width circuit with AerSimulator.from_backend (checked in
    tests/test_device_stats.py) but needs megabytes instead of gigabytes of memory."""

    def __init__(self, backend, pair):
        from qiskit_aer.noise import thermal_relaxation_error
        self._relax = thermal_relaxation_error
        self.backend = backend
        self.pair = tuple(int(q) for q in pair)
        self.noise = NoiseModel.from_backend(backend)
        self.gate_errors = {name: {k: v for k, v in errs.items() if set(k) <= set(self.pair)}
                            for name, errs in self.noise._local_quantum_errors.items()}
        self.sim = AerSimulator(method="density_matrix", max_parallel_threads=1)
        self.n = backend.num_qubits
        self.confusion = readout_matrices(backend, self.noise, self.pair)
        self.qA, self.qB = qubit_params(backend, self.pair[0]), qubit_params(backend, self.pair[1])
        self.t1t2 = {q: (p["T1"], min(p["T2"], 2 * p["T1"])) for q, p in zip(self.pair, (self.qA, self.qB))}

    def scheduled(self, xA, xB, tau, fold=0, gamma=np.pi / 2) -> QuantumCircuit:
        """Full-width compiled and scheduled circuit (no measurement)."""
        G_t = transpile(logical_circuit(xA, xB, tau, gamma, self.backend), self.backend, initial_layout=list(self.pair),
                        optimization_level=1, seed_transpiler=7)
        if fold:
            inv = transpile(G_t.inverse(), self.backend, initial_layout=list(range(self.n)), optimization_level=0)
            body = G_t.copy()
            for _ in range(fold):
                body.barrier()
                body = body.compose(inv)
                body.barrier()
                body = body.compose(G_t)
            G_t = body
        return transpile(G_t, self.backend, initial_layout=list(range(self.n)), optimization_level=0,
                         scheduling_method="alap")

    def reduce(self, sched: QuantumCircuit, noisy: bool = True) -> QuantumCircuit:
        out = QuantumCircuit(2)
        for inst in sched.data:
            qs = tuple(sched.find_bit(q).index for q in inst.qubits)
            if inst.operation.name == "barrier" or not set(qs) & set(self.pair):
                continue
            if not set(qs) <= set(self.pair):
                raise RuntimeError(f"gate {inst.operation.name} couples the pair to qubits {qs}")
            loc = [self.pair.index(q) for q in qs]
            op = inst.operation
            if op.name == "barrier":
                continue
            if op.name == "delay":
                dur = float(op.params[0]) * (self.backend.dt if op.unit == "dt" else
                                             {"s": 1, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}[op.unit])
                if noisy and dur > 0:
                    for q, l in zip(qs, loc):
                        out.append(self._relax(*self.t1t2[q], dur).to_instruction(), [l])
                continue
            out.append(op, loc)
            err = self.gate_errors.get(op.name, {}).get(qs)
            if noisy and err is not None:
                out.append(err.to_instruction(), loc)
        out.save_probabilities([0, 1])
        return out

    def physical(self, xA, xB, tau, fold=0, gamma=np.pi / 2, noisy=True) -> QuantumCircuit:
        return self.reduce(self.scheduled(xA, xB, tau, fold, gamma), noisy=noisy)

    def run(self, circuits) -> np.ndarray:
        res = self.sim.run(circuits).result()
        P = np.array([np.asarray(res.data(i)["probabilities"])[QISKIT_ORDER] for i in range(len(circuits))])
        return P                                              # before readout error

    def run_full_width(self, xA, xB, tau, fold=0) -> np.ndarray:
        """Reference: full-width simulation with AerSimulator.from_backend (memory heavy; validation only)."""
        sim = AerSimulator.from_backend(self.backend, method="density_matrix", max_parallel_threads=1)
        sched = self.scheduled(xA, xB, tau, fold)
        sched.save_probabilities(list(self.pair))
        return np.asarray(sim.run(sched).result().data(0)["probabilities"])[QISKIT_ORDER]

    def with_readout(self, P: np.ndarray) -> np.ndarray:
        return P @ self.confusion.T

    def calibration_probs(self) -> tuple[np.ndarray, np.ndarray]:
        """Exact measured distributions of |00> and |11> (readout calibration circuits)."""
        e = np.zeros(4)
        e[0] = 1
        f = np.zeros(4)
        f[3] = 1
        return self.confusion @ e, self.confusion @ f
