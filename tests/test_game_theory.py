import itertools
import numpy as np
import pytest

from ewlcert import game as G
from ewlcert import theory as Th

PI = np.pi
rng = np.random.default_rng(123)


def test_classical_game_embedded_for_all_gamma():
    for gamma in np.linspace(0, PI / 2, 7):
        for ua, ub, pa, pb in [(G.C, G.C, 3, 3), (G.C, G.D, 0, 5),
                               (G.D, G.C, 5, 0), (G.D, G.D, 1, 1)]:
            assert G.payoff("A", ua, ub, gamma=gamma) == pytest.approx(pa)
            assert G.payoff("B", ua, ub, gamma=gamma) == pytest.approx(pb)


def test_ewl_quantum_results_at_max_entanglement():
    assert G.payoff("A", G.Q, G.Q) == pytest.approx(3)
    assert G.payoff("A", G.Q, G.D) == pytest.approx(5)       # Q beats D (EWL 1999)
    assert G.payoff("B", G.Q, G.D) == pytest.approx(0)


def test_ewl_parametrisation():
    for t, f in [(0.3, 0.2), (2.0, 1.4), (PI, 0), (0, PI / 2)]:
        assert np.allclose(G.U_ewl(t, f), G.U_x(G.x_ewl(t, f)))


def test_QQ_is_NE_noiseless_restricted_but_not_full_su2():
    assert G.epsilon(G.RESTRICTED) == pytest.approx(0, abs=1e-12)
    assert G.epsilon(G.FULL) == pytest.approx(2.0)          # i sigma_x earns 5 vs 3 (Benjamin & Hayden 2001)


def random_two_qubit_channel(k=3):
    A = rng.normal(size=(4 * k, 4)) + 1j * rng.normal(size=(4 * k, 4))
    V, _ = np.linalg.qr(A)
    return [V[4 * i:4 * i + 4] for i in range(k)]


def test_quadratic_form_lemma_under_arbitrary_noise():
    pre, post = random_two_qubit_channel(), random_two_qubit_channel()
    conf = rng.dirichlet(np.ones(4), size=4).T
    forms = G.exact_forms(G.FULL, pre=pre, post=post, confusion=conf)
    for _ in range(300):
        x = rng.normal(size=4)
        x /= np.linalg.norm(x)
        ub = G.payoff("B", G.Q, G.U_x(x), pre=pre, post=post, confusion=conf)
        assert ub == pytest.approx(x @ forms["B"] @ x, abs=1e-12)


def test_max_on_orthant_matches_dense_search():
    for _ in range(20):
        A = rng.normal(size=(3, 3))
        M = (A + A.T) / 2
        best, _ = G.max_on_orthant(M)
        th, ph = np.meshgrid(np.linspace(0, PI, 301), np.linspace(0, PI / 2, 151))
        x = np.stack([np.cos(th / 2) * np.cos(ph), np.sin(th / 2), np.cos(th / 2) * np.sin(ph)], -1)
        dense = np.einsum("...i,ij,...j->...", x, M, x).max()
        assert best >= dense - 1e-9 and best <= dense + 1e-3


@pytest.mark.parametrize("stage", ["pre", "post"])
def test_closed_form_gains_match_model(stage):
    for _ in range(25):
        gA, gB = rng.uniform(0, 1, 2)
        cA, cB = rng.uniform(0, 1, 2) * np.sqrt([1 - gA, 1 - gB])
        rA, rB = rng.uniform(0, 1, 2)
        kA = G.compose(G.depolarizing(rA), G.thermal_relaxation(gA, cA))
        kB = G.compose(G.depolarizing(rB), G.thermal_relaxation(gB, cB))
        noise = {stage: G.local(kA, kB)}
        f = G.exact_forms(G.RESTRICTED, **noise)["B"]
        assert np.allclose(f - np.diag(np.diag(f)), 0, atol=1e-12)          # diagonal
        mc, md, mq = np.diag(f)
        assert md - mq == pytest.approx(Th.gain_D(gA, cA, gB, cB, rA, rB, stage), abs=1e-12)
        assert mc - mq == pytest.approx(Th.gain_C(gA, cA, gB, cB, rA, rB, stage), abs=1e-12)
        fa = G.exact_forms(G.RESTRICTED, **noise)["A"]
        assert np.diag(fa)[1] - np.diag(fa)[2] == pytest.approx(Th.gain_D(gA, cA, gB, cB, rA, rB, stage), abs=1e-12)


def test_threshold_and_limits():
    T1A, T2A, T1B, T2B = 200e-6, 150e-6, 120e-6, 60e-6
    tc = Th.critical_idle_time(T1A, T2A, T1B, T2B)
    for t, sign in [(0.97 * tc, -1), (1.03 * tc, +1)]:
        gA, cA = Th.idle_params(t, T1A, T2A)
        gB, cB = Th.idle_params(t, T1B, T2B)
        kA, kB = G.thermal_relaxation(gA, cA), G.thermal_relaxation(gB, cB)
        eps = G.epsilon(G.RESTRICTED, pre=G.local(kA, kB))
        assert (eps > 1e-9) == (sign > 0)
    big = 1e3
    assert Th.critical_idle_time(big, 50e-6, big, 80e-6) == pytest.approx(
        Th.critical_idle_time_dephasing_limit(50e-6, 80e-6), rel=1e-4)
    a = Th.AD_THRESHOLD_SYMMETRIC
    for p, sign in [(a - 1e-3, -1), (a + 1e-3, 1)]:
        k = G.thermal_relaxation(p, np.sqrt(1 - p))
        assert (G.epsilon(G.RESTRICTED, pre=G.local(k, k)) > 1e-9) == (sign > 0)


def test_diagonal_forms_with_noise_at_both_stages():
    P1 = [G.I2, G.X, G.Y, G.Z]
    P2 = [np.kron(a, b) for a in P1 for b in P1]
    for _ in range(50):
        # correlated two-qubit Pauli channels at both stages
        w1, w2 = rng.dirichlet(np.ones(16) * 0.3), rng.dirichlet(np.ones(16) * 0.3)
        pre = [np.sqrt(w) * p for w, p in zip(w1, P2)]
        post = [np.sqrt(w) * p for w, p in zip(w2, P2)]
        for M in G.exact_forms(G.FULL, pre=pre, post=post).values():
            assert np.allclose(M - np.diag(np.diag(M)), 0, atol=1e-12)
        # local relaxation-dephasing + depolarizing at both stages
        ks = []
        for _stage in range(2):
            g = rng.uniform(0, 0.6, 2)
            c = rng.uniform(0.3, 1, 2) * np.sqrt(1 - g)
            r = rng.uniform(0, 0.3, 2)
            ks.append(G.local(G.compose(G.depolarizing(r[0]), G.thermal_relaxation(g[0], c[0])),
                              G.compose(G.depolarizing(r[1]), G.thermal_relaxation(g[1], c[1]))))
        for M in G.exact_forms(G.FULL, pre=ks[0], post=ks[1]).values():
            assert np.allclose(M - np.diag(np.diag(M)), 0, atol=1e-12)


def test_depolarizing_never_breaks_equilibrium():
    for r in [0.1, 0.5, 0.9, 0.999]:
        for stage in ["pre", "post"]:
            k = G.local(G.depolarizing(r), G.depolarizing(r / 2))
            assert G.epsilon(G.RESTRICTED, **{stage: k}) == pytest.approx(0, abs=1e-12)
