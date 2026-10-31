"""Analysis of E1 (experiments/essays_worlds.py): writes results/essays_worlds/summary.md.

Every score is in nats per question. "Expected" scores are exact expectations over outcomes in the true world
(the generator's probabilities), so they carry no outcome noise; "realized" scores use the one drawn outcome
vector, as a real scoring mechanism would.

    PYTHONPATH=. python experiments/essays_worlds_report.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from lvpm.essays import score as S
from lvpm.essays import worlds as WD

D = Path(__file__).resolve().parents[1] / "results" / "essays_worlds"
rng = np.random.default_rng(0)
EPS = 0.01


def boot(x, n=4000):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return f"{x.mean():+.3f} [{np.percentile(m, 2.5):+.3f}, {np.percentile(m, 97.5):+.3f}] (n={len(x)})"


def exp_ll(p, t):
    p = np.clip(p, EPS, 1 - EPS)
    return t * np.log(p) + (1 - t) * np.log(1 - p)


def mixture_expected(Pm, w, t, n=3000):
    """E[S_1]/n_q of a mixture (rows of Pm, weights w) when y ~ Bernoulli(t) independently."""
    Y = (rng.random((n, len(t))) < t).astype(float)
    return np.mean([S.s_tau(w, Pm, y, 1.0) for y in Y]) / len(t)


def within_spearman(df, a, b):
    rs = [spearmanr(g[a], g[b]).correlation for _, g in df.groupby("setting") if g[a].nunique() > 1]
    return np.mean(rs)


def main():
    cells = pd.read_parquet(D / "cells.parquet")
    recs = pd.DataFrame(json.loads((D / "essays.json").read_text()))
    st = json.loads((D / "settings.json").read_text())
    qs = {s: [WD.Q(**{k: (tuple(v) if k == "options" else v) for k, v in q.items()}) for q in v["questions"]]
          for s, v in st.items()}
    qidx = {s: {q.id: i for i, q in enumerate(v)} for s, v in qs.items()}
    c = cells[cells.condition != "empty"].copy()
    c["e_ll"] = exp_ll(c.p, c.p_true_model)
    c["r_ll"] = np.where(c.y == 1, np.log(c.p.clip(EPS, 1 - EPS)), np.log(1 - c.p.clip(EPS, 1 - EPS)))

    # one row per report (setting, world, profile, condition)
    R = []
    for (s, w, prof, cond), g in c[c.condition.isin(["honest_haiku", "honest_gemini", "eloquent", "overconfident",
                                                     "list", "scenarios", "truth"])].groupby(
            ["setting", "world", "profile", "condition"], dropna=False):
        z = tuple(int(ch) for ch in w)
        post = WD.signals(s, z, prof)[1] if isinstance(prof, str) else np.array(z, float)
        order = [q.id for q in qs[s]]
        hyps = sorted(g.hyp.unique())
        Pm = np.array([g[g.hyp == h].set_index("q").p.reindex(order).values for h in hyps])
        wts = np.array([g[g.hyp == h].weight.iloc[0] for h in hyps])
        t = np.array([q.p_yes(z) for q in qs[s]])
        y = g[g.hyp == hyps[0]].set_index("q").y.reindex(order).values
        seen = g[g.hyp == hyps[0]].set_index("q").seen.reindex(order).values.astype(bool)
        marg = wts @ Pm
        row = {"setting": s, "world": w, "profile": prof, "condition": cond,
               "oracle": WD.expected_log_score(qs[s], WD.predictive(qs[s], post), z),
               "S_exp": (mixture_expected(Pm, wts, t) if len(hyps) > 1 else exp_ll(Pm[0], t).mean()),
               "S_real": S.s_tau(wts, np.clip(Pm, EPS, 1 - EPS), y, 1.0) / len(y),
               "S_exp_seen": exp_ll(marg[seen], t[seen]).mean(), "S_exp_unseen": exp_ll(marg[~seen], t[~seen]).mean(),
               "belief": WD.belief_expected_log_score(qs[s], np.clip(marg, EPS, 1 - EPS), post)}
        if len(hyps) > 1:  # structured use of outcomes: update weights on SEEN outcomes, predict the rest
            row["update_gain"] = (S.held_out_score(wts, np.clip(Pm, EPS, 1 - EPS), y, ~seen, 1.0)
                                  - S.s_tau(wts, np.clip(Pm, EPS, 1 - EPS), np.where(~seen, y, np.nan), 1.0)) \
                / (~seen).sum()
        rec = recs[(recs.setting == s) & (recs.world == w) & (recs.condition == cond)
                   & ((recs.profile == prof) if isinstance(prof, str) else recs.profile.isna())]
        row["judge"] = rec.judge.dropna().iloc[0] if rec.judge.notna().any() else np.nan
        parses = [p for p in rec.parse if isinstance(p, list)]
        if len(parses) == len(hyps):
            tl = sum(wk * WD.predictive(qs[s], np.array(pp)) for wk, pp in zip(wts, parses))
            row["truelink"] = exp_ll(tl, t).mean()
        R.append(row)
    R = pd.DataFrame(R)
    best = {s: np.mean([exp_ll(np.array([q.p_yes(z) for q in qs[s]]), np.array([q.p_yes(z) for q in qs[s]]))
                        .mean() for z in WD.worlds(s)]) for s in qs}
    empty = cells[cells.condition == "empty"]
    L = ["# E1: essay scores against known latent truth (synthetic worlds)", "",
         f"{R.world.nunique()} worlds x {R.profile.dropna().nunique()} author profiles x 2 settings. Scores: nats "
         "per question; 'expected' = exact expectation over outcomes in the true world; 'realized' = one draw.",
         ""]

    # 0. the reader as a link function
    tr = R[R.condition == "truth"]
    L += ["## 0. The reader as a link function", "",
          "| | expected score |", "|---|---|",
          f"| best possible (the true probabilities) | {np.mean(list(best.values())):.3f} |",
          f"| reader given the true factors as facts (truth essay) | {tr.S_exp.mean():.3f} |",
          f"| true factors through true links (= best possible) | {tr.truelink.mean():.3f} |" if "truelink" in tr
          else "", ""]
    for s in qs:
        e = empty[empty.setting == s]
        t_avg = np.mean([np.array([q.p_yes(z) for q in qs[s]]) for z in WD.worlds(s)], axis=0)
        L.append(f"- reader alone, {s}: expected score {exp_ll(e.set_index('q').p.reindex([q.id for q in qs[s]]).values, t_avg).mean():.3f} "
                 "(averaged over worlds)")
    L.append("")

    # 1. does each score track the author's true information?
    H = R[R.condition.isin(["honest_haiku", "honest_gemini"])].copy()
    L += ["## 1. Does each score track true information? (honest essays)", "",
          "Spearman correlation with the author's oracle information (the expected score of their exact "
          "posterior), within setting, averaged over the two settings.", "",
          "| score | Spearman with oracle |", "|---|---|"]
    for col, name in [("S_real", "S2, realized (one outcome draw, all 30 questions)"),
                      ("S_exp", "S2, expected (no outcome noise)"),
                      ("truelink", "S2 with true links (parse -> generator probabilities)"),
                      ("judge", "S0: strong-model judge rating")]:
        if col in H and H[col].notna().sum() > 3:
            L.append(f"| {name} | {within_spearman(H.dropna(subset=[col]), col, 'oracle'):.2f} |")
    L += ["", "Mean by profile (expected S2 / oracle / judge):", "",
          "| profile | S2 expected | oracle | judge |", "|---|---|---|---|"]
    for p, g in H.groupby("profile"):
        L.append(f"| {p} | {g.S_exp.mean():.3f} | {g.oracle.mean():.3f} | {g.judge.mean():.1f} |")

    # 2. style, distortion, list
    def paired(a, b, col):
        A = R[R.condition == a].set_index(["setting", "world", "profile"])[col]
        B = R[R.condition == b].set_index(["setting", "world", "profile"])[col]
        return (A - B).dropna()

    L += ["", "## 2. Manipulations (paired differences, 95% bootstrap CIs)", "",
          "| comparison | S2 expected | judge |", "|---|---|---|",
          f"| eloquent restyle - honest | {boot(paired('eloquent', 'honest_haiku', 'S_exp'))} | "
          f"{boot(paired('eloquent', 'honest_haiku', 'judge'))} |",
          f"| overconfident - honest | {boot(paired('overconfident', 'honest_haiku', 'S_exp'))} | "
          f"{boot(paired('overconfident', 'honest_haiku', 'judge'))} |", "",
          "Properness: expected score *under the author's own beliefs* (worlds drawn from their posterior), "
          "overconfident - honest: " + boot(paired("overconfident", "honest_haiku", "belief")) +
          ". Proper scoring predicts < 0: an author who believes their own information loses by overstating it.",
          "", "List of predictions (for the SEEN half only) - honest essay, expected score:", "",
          f"- on SEEN questions: {boot(paired('list', 'honest_haiku', 'S_exp_seen'))}",
          f"- on UNSEEN questions: {boot(paired('list', 'honest_haiku', 'S_exp_unseen'))}"]

    # 3. holistic vs structured use of outcomes
    hol = c[c.condition.isin(["holistic", "holistic_empty"])].groupby(
        ["setting", "world", "profile", "condition"], dropna=False).e_ll.mean().reset_index()
    hs = hol[hol.condition == "holistic"].merge(H[H.condition == "honest_haiku"], on=["setting", "world", "profile"])
    st_unseen = R[R.condition == "honest_haiku"].set_index(["setting", "world", "profile"]).S_exp_unseen
    hs["structured_unseen"] = hs.set_index(["setting", "world", "profile"]).index.map(st_unseen)
    L += ["", "## 3. Letting the reader use other outcomes (holistic) vs. structured scoring", "",
          "On the UNSEEN questions. Holistic = the reader is given the essay plus the realized SEEN outcomes.", "",
          f"- reader alone + SEEN outcomes (no essay): expected {hol[hol.condition == 'holistic_empty'].e_ll.mean():.3f}",
          f"- honest essay, structured: expected {hs.structured_unseen.mean():.3f}; Spearman with oracle "
          f"{within_spearman(hs, 'structured_unseen', 'oracle'):.2f}",
          f"- honest essay + SEEN outcomes, holistic: expected {hs.e_ll.mean():.3f}; Spearman with oracle "
          f"{within_spearman(hs, 'e_ll', 'oracle'):.2f}"]
    M = R[R.condition == "scenarios"]
    if len(M):
        L += [f"- scenario mixtures, structured update on SEEN outcomes (Bayes over the writer's own scenarios): "
              f"gain on UNSEEN {boot(M.update_gain)} nats/question"]
    spend = json.loads((D / "spend.json").read_text())["usd"] if (D / "spend.json").exists() else None
    if spend:
        L += ["", f"OpenRouter spend: ${spend:.2f}."]
    R.to_csv(D / "reports_scored.csv", index=False)
    (D / "summary.md").write_text("\n".join(x for x in L if x is not None) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
