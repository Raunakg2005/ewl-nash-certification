import numpy as np
import pytest

from ewlcert import game as G
from ewlcert import stats as S

rng = np.random.default_rng(7)


def test_circuit_set_and_forms_exact():
    P = np.array([G.probabilities(G.U_x(a), G.U_x(b)) for a, b in S.circuit_strategies()])
    assert len(P) == 11
    assert S.eps_from_probs(P) == pytest.approx(0, abs=1e-12)
    f = S.forms_from_probs(P)
    assert np.allclose(f["B"], G.exact_forms()["B"]) and np.allclose(f["A"], G.exact_forms()["A"])


def test_readout_mitigation_is_exact_with_exact_calibration():
    conf = np.kron(np.array([[0.97, 0.06], [0.03, 0.94]]), np.array([[0.99, 0.02], [0.01, 0.98]]))
    P = np.array([G.probabilities(G.U_x(a), G.U_x(b)) for a, b in S.circuit_strategies()])
    Pm = P @ conf.T
    n0, n1 = conf[:, 0] * 1e6, conf[:, 3] * 1e6
    assert np.allclose(S.mitigate(Pm, S.confusion_from_cal(n0, n1)), P, atol=1e-9)


def test_bootstrap_bounds_cover_truth_noiseless():
    P = np.array([G.probabilities(G.U_x(a), G.U_x(b)) for a, b in S.circuit_strategies()])
    counts = S.sample_counts(P, 4000, rng)
    pt, lo, hi = S.bootstrap(counts, B=200, rng=rng)
    assert lo <= pt <= hi and hi < 0.2


def test_violation_test_detects_clear_violation():
    k = G.thermal_relaxation(0, 0.2)                      # strong dephasing: c^2 = 0.04 < 1/7
    P = np.array([G.probabilities(G.U_x(a), G.U_x(b), pre=G.local(k, k)) for a, b in S.circuit_strategies()])
    assert S.violation_test(S.sample_counts(P, 4000, rng))[0]
    P0 = np.array([G.probabilities(G.U_x(a), G.U_x(b)) for a, b in S.circuit_strategies()])
    assert not S.violation_test(S.sample_counts(P0, 4000, rng))[0]


@pytest.mark.slow
def test_device_circuits_match_exact_model_when_noiseless():
    from ewlcert import device as Dv
    be = Dv.backend_by_name("fake_brisbane")
    dp = Dv.DevicePair(be, next(Dv.candidate_pairs(be)))
    strategies = S.circuit_strategies()[:4]
    for fold in (0, 1):
        circs = [dp.physical(a, b, tau=2e-6, fold=fold, noisy=False) for a, b in strategies]
        P = dp.run(circs)
        exact = np.array([G.probabilities(G.U_x(a), G.U_x(b)) for a, b in strategies])
        assert np.allclose(P, exact, atol=1e-9)


@pytest.mark.slow
def test_reduced_two_qubit_model_equals_full_width_device_simulation():
    """One full-width simulation per case (memory heavy: run this test on its own)."""
    from ewlcert import device as Dv
    be = Dv.backend_by_name("fake_fez")
    dp = Dv.DevicePair(be, (73, 74))
    for (a, b), tau, fold in [(S.circuit_strategies()[0], 40e-6, 0), (S.circuit_strategies()[3], 5e-6, 1)]:
        reduced = dp.run([dp.physical(a, b, tau, fold)])[0]
        full = dp.run_full_width(a, b, tau, fold)
        assert np.allclose(reduced, full, atol=1e-6)
