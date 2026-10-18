# ewl-nash-certification

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23447741.svg)](https://doi.org/10.5281/zenodo.23447741)

Code and data for **"Certifying quantum Nash equilibria under hardware noise: a finite-circuit test, T1/T2
stability conditions and an experiment on a superconducting processor"** (R. K. Gupta, V. P. Vasani,
D. Zaware, S. Pawar; manuscript under preparation).

The repository regenerates every number, table and figure of the paper from scratch, and contains the raw
counts of the experiment on IBM Quantum hardware.

## What is in here

| Path | Content |
|---|---|
| `ewlcert/game.py` | exact density-matrix model of the Eisert–Wilkens–Lewenstein Prisoner's Dilemma with noise; quadratic-form (six-circuit) certification of the incentive to deviate |
| `ewlcert/theory.py` | closed-form stability condition `7 c_A c_B >= (1-g_A)(1-g_B) + g_A g_B` and the critical idle time from T1, T2 |
| `ewlcert/stats.py` | finite-shot estimation, readout mitigation, bootstrap bounds, violation test |
| `ewlcert/device.py` | IBM device-calibrated simulation: compiled and scheduled circuits, zero-noise-extrapolation folding, reduced two-qubit model |
| `experiments/e1_threshold.py` … `e4_coherent.py` | simulation experiments E1–E4 |
| `experiments/hardware_*.py` | hardware run (IBM Quantum), analysis, noise-model fit, estimator validation |
| `experiments/make_figures.py`, `make_hardware_figure.py` | all figures (with an automatic check for overlapping or clipped text) |
| `results/` | raw results (JSON) of all experiments, including the hardware counts |
| `figures/` | figures as PDF and PNG |
| `tests/` | 17 tests (`python -m pytest`) |

## Hardware data

Two runs on `ibm_kingston` (IBM Heron r2), qubits 16 and 23, 18 October 2026, dynamical decoupling and
twirling switched off:

| Run | File | IBM job id | Circuits | Shots/circuit |
|---|---|---|---|---|
| 1 | `results/hardware_job_20261018_111455.json` | dba5pps24mrc7384ul5g | 117 (9 delays) | 5000 |
| 2 | `results/hardware_job_20261018_113856.json` | dba6521gv2cc73e6dtd0 | 273 (0–100 µs, 5 µs steps) | 1500 |

Each file holds the circuit list (`meta`), the counts in the order CC, CD, DC, DD, and the calibration data
recorded at submission. The `_analysis.json` and `_fit.json` files contain the derived results.

## Reproduce

```
pip install -r requirements.txt
python -m pytest                     # 17 tests
cd experiments
python e1_threshold.py               # E1: T1/T2 threshold vs device model (18 qubit pairs, 6 IBM snapshots)
python e2_certification.py           # E2: finite-shot certification
python e3_zne.py                     # E3: zero-noise extrapolation
python e4_coherent.py                # E4: coherent errors
python durations.py
python hardware_analyze.py hardware_job_20261018_111455.json
python hardware_analyze.py hardware_job_20261018_113856.json
python hardware_fit.py hardware_job_20261018_113856_fit.json hardware_job_20261018_113856.json
python hardware_vertex.py hardware_job_20261018_111455.json hardware_job_20261018_113856.json
python make_figures.py && python make_hardware_figure.py
```

All simulations together take well under an hour on a laptop and need at most about 2 GB of RAM (each
worker simulates a reduced two-qubit circuit that is equivalent to the full-device Aer simulation; see
`tests/test_device_stats.py`).

To repeat the hardware experiment with your own IBM Quantum account, put your API key in a file `.env`
(`IBM_QUANTUM_TOKEN=...`, never commit it) and run `python experiments/hardware_run.py` (plan only) and then
`python experiments/hardware_run.py --submit`.

Software used: Python 3.12.10, NumPy 2.4.1, SciPy 1.16.3, SymPy 1.13.1, Qiskit 2.1.2, Qiskit Aer 0.17.2,
qiskit-ibm-runtime 0.41.1, Matplotlib 3.10.8.

## Acknowledgement

We acknowledge the use of IBM Quantum services for this work. The views expressed are those of the authors and
do not reflect the official policy or position of IBM or the IBM Quantum team.

## License

MIT, see `LICENSE`. If you use the code or data, please cite it (see `CITATION.cff`).
