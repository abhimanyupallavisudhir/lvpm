"""Numerical checks and figures for notes/latent-meaning.md.

Run:  PYTHONPATH=. .venv/bin/python experiments/latent_meaning.py
Writes notes/figures/*.png and prints every number quoted in the note.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lvpm.latent_bottleneck import (bottleneck_curve, em, entropy, log_loss,
                                    model_joint, posterior, stats)

FIG = Path(__file__).resolve().parent.parent / "notes" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def h(x):
    return entropy([x, 1 - x])


def bern(v):
    v = np.asarray(v, dtype=float)
    return np.stack([1 - v, v], -1)


# ---------------------------------------------------------------- 1. DSBS
def dsbs():
    p = 0.1
    py = np.array([[1 - p, p], [p, 1 - p]]) / 2
    i12 = 1 - h(p)
    a = (1 - np.sqrt(1 - 2 * p)) / 2
    c_wyner = 1 + h(p) - 2 * h(a)

    taus = np.concatenate([np.linspace(0.02, 0.9, 45), 1 - np.geomspace(0.1, 1e-5, 30)])
    curve = bottleneck_curve(py, 4, taus, restarts=10, seed=0)
    res = np.array([c["residual"] for c in curve])
    rate = np.array([c["rate"] for c in curve])

    # Information bottleneck (latent forced to be read from Y1): T = Y1 through
    # a BSC(c), optimal for this source by Mrs Gerber's lemma.
    cs = np.linspace(0, 0.5, 400)
    ib_rate = np.array([1 - h(c) for c in cs])
    ib_res = np.array([h(c * (1 - p) + (1 - c) * p) - h(p) for c in cs])

    print("[1] DSBS(p=0.1)")
    print(f"    I(Y1;Y2) = {i12:.4f}  Wyner C (closed form) = {c_wyner:.4f}  "
          f"latent bottleneck at tau=1-1e-5: rate {rate[-1]:.4f}, residual {res[-1]:.1e}")
    print(f"    IB endpoint (minimal sufficient statistic = Y1): {ib_rate[0]:.4f}")
    ib_at = np.interp(res[::-1], ib_res[::-1], ib_rate[::-1])[::-1]
    print(f"    latent bottleneck <= IB everywhere on the grid: {bool(np.all(rate <= ib_at + 1e-6))}")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.grid(color=GRID, lw=0.6)
    xs = np.linspace(0, i12, 2)
    ax.plot(xs, i12 - xs, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.plot(ib_res, ib_rate, color=ORANGE, lw=2, label="Information bottleneck  (X read from Y1)")
    ax.plot(res, rate, color=BLUE, lw=2, label="Latent bottleneck  (X free)")
    ax.plot([0], [c_wyner], "o", ms=8, color=BLUE, mec=SURFACE, mew=2, zorder=5)
    ax.plot([0], [1], "o", ms=8, color=ORANGE, mec=SURFACE, mew=2, zorder=5)
    ax.annotate(f"Wyner common information {c_wyner:.3f}", (0, c_wyner), (0.11, 0.97),
                color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.annotate("X = Y1", (0, 1), (0.015, 1.04), color=INK, fontsize=9, va="center")
    ax.text(0.06, 0.482, "I(Y1;Y2) − residual  (lower bound)", color=INK2, fontsize=8.5,
            rotation=-16, rotation_mode="anchor")
    ax.set_xlabel("unexplained dependence  I(Y1;Y2 | X)  [bits]")
    ax.set_ylabel("information cost  I(X; Y1,Y2)  [bits]")
    ax.set_title("A latent is cheaper than any representation of one view")
    ax.set_xlim(-0.01, i12 + 0.02)
    ax.set_ylim(-0.02, 1.08)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 0.84), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "bottleneck_dsbs.png", dpi=180)
    plt.close(fig)


# -------------------------------------------- 2+3. two views vs three views
PI = np.array([0.4, 0.6])
A, B, C = bern([0.2, 0.85]), bern([0.3, 0.9]), bern([0.1, 0.7])


def fiber(n=301):
    """All two-state latent models reproducing P(Y1,Y2) exactly, indexed by
    a = (P(Y1=1|X=0), P(Y1=1|X=1)) with a0 < a1."""
    p2 = model_joint(PI, [A, B]).sum(0)
    g = np.linspace(0, 1, n)
    a0, a1 = np.meshgrid(g, g, indexing="ij")
    ok = a0 < a1
    Am = np.stack([np.stack([1 - a0, 1 - a1], -1), np.stack([a0, a1], -1)], -2)  # (n,n,y1,x)
    det = Am[..., 0, 0] * Am[..., 1, 1] - Am[..., 0, 1] * Am[..., 1, 0]
    inv = np.stack([np.stack([Am[..., 1, 1], -Am[..., 0, 1]], -1),
                    np.stack([-Am[..., 1, 0], Am[..., 0, 0]], -1)], -2) / np.where(ok, det, 1)[..., None, None]
    M = inv @ p2  # (n,n,x,y2) = pi_x * P(y2|x)
    ok &= np.all(M >= -1e-12, axis=(-1, -2))
    M = np.clip(M, 0, None)
    pis = M.sum(-1)
    ok &= np.all(pis > 1e-9, -1)
    Bs = M / np.where(pis > 0, pis, 1)[..., None]
    Al = np.moveaxis(Am, -1, -2)  # (n,n,x,y1)
    return g, ok, pis, Al, Bs


def fit_third_view(p3, pis, Al, Bs, iters=4000):
    """For each fixed (pi, P(Y1|X), P(Y2|X)), the best P(Y3|X): EM over C only
    (the log-likelihood is concave in C, so this is the global optimum)."""
    G = pis.shape[0]
    base = pis[:, :, None, None] * Al[:, :, :, None] * Bs[:, :, None, :]  # (G,x,y1,y2)
    Cs = np.full((G, 2, 2), 0.5)
    Cs[:, 0, 1], Cs[:, 1, 1] = 0.3, 0.7
    Cs[:, :, 0] = 1 - Cs[:, :, 1]
    for _ in range(iters):
        joint = base[..., None] * Cs[:, :, None, None, :]  # (G,x,y1,y2,y3)
        m = joint.sum(1, keepdims=True)
        w = joint / np.maximum(m, 1e-300) * p3[None, None]
        Cs = w.sum((2, 3))
        Cs /= np.maximum(Cs.sum(-1, keepdims=True), 1e-300)
    m = (base[..., None] * Cs[:, :, None, None, :]).sum(1)
    ll = -(p3[None] * np.log2(np.maximum(m, 1e-300))).sum((1, 2, 3))
    return ll - entropy(p3), Cs


def views():
    p2 = model_joint(PI, [A, B]).sum(0)
    p3 = model_joint(PI, [A, B, C]).sum(0)
    g, ok, pis, Al, Bs = fiber()
    idx = np.argwhere(ok)
    P, AA, BB = pis[ok], Al[ok], Bs[ok]

    rates, post, ll2 = [], [], []
    for pi_, a_, b_ in zip(P, AA, BB):
        rates.append(stats(model_joint(pi_, [a_, b_]))["rate"])
        post.append(posterior(pi_, [a_, b_])[1, 1, 1])
        ll2.append(log_loss(p2, pi_, [a_, b_]) - entropy(p2))
    rates, post, ll2 = map(np.array, (rates, post, ll2))
    excess3, _ = fit_third_view(p3, P, AA, BB)

    t = np.argmin(np.abs(idx[:, 0] - 0.2 * (len(g) - 1)) + np.abs(idx[:, 1] - 0.85 * (len(g) - 1)))
    w = np.argmin(rates)
    print("[2] Two views: every model on the fiber reproduces P(Y1,Y2)")
    print(f"    feasible two-state models on grid: {len(P)};  max log-score regret on P(Y1,Y2): {ll2.max():.1e} bits")
    print(f"    P(X=1 | y1=1, y2=1) ranges over [{post.min():.3f}, {post.max():.3f}]  (true model: {post[t]:.3f})")
    print(f"    I(X;Y) ranges over [{rates.min():.4f}, {rates.max():.4f}];  true {rates[t]:.4f};  "
          f"min at a=({g[idx[w,0]]:.3f}, {g[idx[w,1]]:.3f})")
    print("[3] Three views: best achievable log-score regret on P(Y1,Y2,Y3), per fiber point")
    print(f"    at the true model: {excess3[t]:.1e} bits;  min elsewhere (a more than 0.05 from truth): "
          f"{excess3[np.hypot(g[idx[:,0]]-0.2, g[idx[:,1]]-0.85) > 0.05].min():.1e} bits")
    print(f"    at the 2-view Wyner point: {excess3[w]:.4f} bits")

    rng = np.random.default_rng(0)
    fits = []
    for _ in range(200):
        pi_, cs = em(p3, 2, rng)
        if pi_[0] > pi_[1]:  # undo the relabelling symmetry
            pi_, cs = pi_[::-1], [c[::-1] for c in cs]
        fits.append((log_loss(p3, pi_, cs), np.concatenate([pi_, *[c[:, 1] for c in cs]])))
    best = min(f[0] for f in fits)
    good = [f[1] for f in fits if f[0] < best + 1e-8]
    truth = np.concatenate([PI, A[:, 1], B[:, 1], C[:, 1]])
    print(f"    EM from 200 random starts: {len(good)} reach the optimum; max |param - truth| among them: "
          f"{max(np.abs(x - truth).max() for x in good):.1e}")


# ------------------------------------------- 4. scoring an instance's latent
def probe():
    q_true = posterior(PI, [A, B])[1, 1, 1]  # P(X=1 | y1=1, y2=1)
    qs = np.linspace(0.001, 0.999, 999)

    def expected_score(Cm):
        pred_true = (1 - q_true) * Cm[0] + q_true * Cm[1]
        pred = (1 - qs)[:, None] * Cm[0] + qs[:, None] * Cm[1]
        return (pred_true[None] * np.log2(pred)).sum(1)

    s_inf, s_flat = expected_score(C), expected_score(bern([0.4, 0.4]))
    print("[4] Scoring a belief about this unit's X via a held-out view Y3")
    print(f"    true posterior P(X=1|y1=1,y2=1) = {q_true:.3f};  argmax of expected log score "
          f"(informative probe) = {qs[np.argmax(s_inf)]:.3f};  spread with uninformative probe = "
          f"{np.ptp(s_flat):.1e} bits")

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.grid(color=GRID, lw=0.6)
    ax.plot(qs, s_inf, color=BLUE, lw=2, label="probe with distinct rows  P(Y3|X)")
    ax.plot(qs, s_flat, color=ORANGE, lw=2, label="probe with equal rows")
    ax.axvline(q_true, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.text(q_true - 0.015, s_inf.min() + 0.02, f"true posterior {q_true:.2f}", ha="right",
            color=INK2, fontsize=8.5)
    ax.set_xlabel("reported P(X=1 | this unit's y1, y2)")
    ax.set_ylabel("expected log score of Y3  [bits]")
    ax.set_title("A belief about a latent is scored through a view it predicts")
    ax.legend(frameon=False, loc="lower center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "probe_scoring.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    dsbs()
    views()
    probe()
