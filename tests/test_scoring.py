import numpy as np
import pytest

from lvpm import covariation as C
from lvpm import model as M
from lvpm import scoring as S


def test_structure_score_zero_for_independence_and_positive_on_average_for_truth():
    rng = np.random.default_rng(0)
    p, lam = np.array([0.3, 0.5, 0.6, 0.7]), np.array([0.7, 0.6, 0.8, 0.5])
    R = M.factor_corr(lam)
    xs = M.sample(p, R, 3000, rng)
    assert S.structure_score(xs[0], p, R=np.eye(4)) == 0
    s = [S.structure_score(x, p, L=lam) for x in xs]
    assert np.mean(s) > 0.05  # expected score = total correlation > 0


def test_genz_and_quadrature_agree_through_scoring_api():
    p, lam = np.array([0.2, 0.5, 0.7]), np.array([0.6, -0.5, 0.4])
    x = np.array([1.0, 0.0, 1.0])
    assert S.joint_loglik(x, p, L=lam) == pytest.approx(S.joint_loglik(x, p, R=M.factor_corr(lam)), abs=2e-3)


def test_calibration_slope_recovers_one_for_true_model():
    rng = np.random.default_rng(1)
    p, lam = np.array([0.4, 0.5, 0.6]), np.array([0.8, 0.7, 0.6])
    R = M.factor_corr(lam)
    pairs = [S.covariance_pairs(x, p, R) for x in M.sample(p, R, 4000, rng)]
    beta, se = S.calibration_slope(pairs)
    assert abs(beta - 1) < 3 * se + 0.05


def test_martingale_quadratic_covariation_matches_outcome_covariance():
    # two markets driven by one shared coin plus own coins; p_t = E_t[X]
    rng = np.random.default_rng(2)
    qv, prod = [], []
    for _ in range(20000):
        shared, a, b = rng.random(3) < 0.5
        x1, x2 = float(shared or a), float(shared or b)
        P = np.array([[0.75, 0.75], [1.0 if shared else 0.5, 1.0 if shared else 0.5], [x1, x2]])
        qv.append(C.realised_covariation(P)[0, 1])
        prod.append((x1 - 0.75) * (x2 - 0.75))
    assert np.mean(qv) == pytest.approx(np.mean(prod), abs=0.01)
    assert np.mean(qv) == pytest.approx(0.5 + 0.5 * 0.25 - 0.75 ** 2, abs=0.01)  # exact Cov


def test_daily_grid_forward_fills_and_respects_start():
    h = [[(100, 0.2), (100 + 2 * C.DAY, 0.4)]]
    G = C.daily_grid(h, 100 - C.DAY, 100 + 3 * C.DAY)[:, 0]
    assert np.isnan(G[0]) and list(G[1:]) == [0.2, 0.2, 0.4, 0.4]
