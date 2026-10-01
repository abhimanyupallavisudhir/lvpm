import itertools

import numpy as np
import pytest

from lvpm import model as M


def all_x(n):
    return np.array(list(itertools.product([0, 1], repeat=n)), float)


@pytest.mark.parametrize("k", [1, 2])
def test_factor_loglik_normalises_and_matches_copula(k):
    rng = np.random.default_rng(k)
    n = 5
    p = rng.uniform(0.1, 0.9, n)
    L = rng.uniform(-0.6, 0.6, (n, k))
    xs = all_x(n)
    lf = np.array([M.factor_loglik(x, p, L) for x in xs])
    assert np.exp(lf).sum() == pytest.approx(1, abs=1e-6)
    lc = np.array([M.gaussian_copula_loglik(x, p, M.factor_corr(L)) for x in xs])
    assert np.allclose(lf, lc, atol=2e-3)
    # exact marginals
    for i in range(n):
        assert np.exp(lf)[xs[:, i] == 1].sum() == pytest.approx(p[i], abs=1e-6)


def test_missing_indicators_are_marginalised():
    p, L = np.array([0.3, 0.6, 0.8]), np.array([0.7, 0.5, -0.4])
    x = np.array([1, np.nan, 0])
    full = [M.factor_loglik([1, v, 0], p, L) for v in (0, 1)]
    assert M.factor_loglik(x, p, L) == pytest.approx(np.logaddexp(*full), abs=1e-8)


def test_positive_loading_means_more_likely_when_latent_high():
    # with a common positive factor, X1 = 1 should raise P(X2 = 1)
    p, L = np.array([0.5, 0.5]), np.array([0.8, 0.8])
    p11 = np.exp(M.factor_loglik([1, 1], p, L))
    assert p11 > 0.25 + 0.1


def test_rho_conversions_roundtrip():
    for pi, pj, r in [(0.3, 0.7, 0.5), (0.1, 0.2, -0.3), (0.5, 0.5, 0.9)]:
        p11 = M.p11_from_rho(pi, pj, r)
        assert M.rho_from_p11(pi, pj, p11) == pytest.approx(r, abs=1e-3)
        q1, q0 = p11 / pi, (pj - p11) / (1 - pi)
        assert M.rho_from_conditionals(pi, pj, q1, q0) == pytest.approx(r, abs=2e-3)


def test_loading_from_conditionals_roundtrip():
    from scipy.stats import norm
    p, lam = 0.3, 0.6
    tau = norm.ppf(p)
    u = np.sqrt(1 - lam ** 2)
    q_hi, q_lo = norm.cdf((tau + lam) / u), norm.cdf((tau - lam) / u)
    assert M.loading_from_conditionals(p, q_hi, q_lo) == pytest.approx(lam, abs=1e-4)


def test_fit_factor_and_gauge_alignment():
    L = np.array([0.8, 0.7, 0.6, 0.5, -0.4])[:, None]
    Lhat = M.fit_factor(M.factor_corr(L), k=1)
    assert np.allclose(M.align(Lhat, L), L, atol=1e-3)
    assert np.allclose(M.align(-Lhat, L), L, atol=1e-3)  # sign flip is pure gauge


def test_nearest_correlation_is_psd_unit_diag():
    R = np.array([[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]])
    C = M.nearest_correlation(R)
    assert np.allclose(np.diag(C), 1) and np.linalg.eigvalsh(C).min() > -1e-8
