"""Sensitivity of the pre-t0 co-movement baseline to its knobs (all variants reported, none selected)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lvpm import covariation as C  # noqa: E402
from lvpm import model as M  # noqa: E402
from lvpm import scoring as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = json.loads((ROOT / "data" / "pilot_clusters.json").read_text())
T0 = D["meta"]["t0"]
res = {}
for probit in (True, False):
    for shrink in (10, 30):
        for days in (30, 60):
            tot = []
            for c, qs in D["clusters"].items():
                p = np.array([q["p_t0"] for q in qs])
                x = np.array([q["outcome"] for q in qs], float)
                G = C.daily_grid([q["history"] for q in qs], T0 - days * C.DAY, T0)
                r = C.increment_corr(G, probit=probit, shrink=shrink)
                R = M.nearest_correlation(r) if probit else C.latent_corr_from_outcome_corr(r, p)
                tot.append(S.structure_score(x, p, R=R))
            key = f"{'probit' if probit else 'prob'} scale, shrink {shrink}, {days}d window"
            res[key] = dict(total=float(np.sum(tot)), clusters_positive=int(np.sum(np.array(tot) > 0)))
            print(f"{key:40s} total {np.sum(tot):+.2f}  positive {res[key]['clusters_positive']}/8")
(ROOT / "reports" / "comovement_sensitivity.json").write_text(json.dumps(res, indent=1))
