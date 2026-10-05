"""Latent DAG models: fitting and eliciting them with the tempered log score.

A model is a DAG over hidden nodes H and observed nodes Y, each with a finite
cardinality, and one conditional table per node, ``cpt[v]`` of shape
(card of each parent, in order) + (card v,).

The tempered log score of a model on an observation y is

    S_tau(theta; y) = log sum_h  prod_{v in H} p(v | pa v)  prod_{v in Y} p(v | pa v)^tau.

At tau = 1 it is the ordinary log score of the model's prediction for Y.
Maximising its expectation over theta (tempered EM below) gives the latent
bottleneck optimum

    min_X  I(H; Y) + (D_H + tau * D_Y) / (1 - tau),

where D_H + D_Y = KL(p(h, y) || prod_v p(v | pa v)) is the DAG's violation,
split into the part among latents and the rest. Single-latent case:
I(X;Y) - tau * sum_i I(X;Y_i).

Everything is done on the full joint table, so this is for small models.
"""

import numpy as np

EPS = 1e-300


class LatentDAG:
    def __init__(self, card, parents, observed):
        """card: {node: k}; parents: {node: [parents]}; observed: [nodes] in the
        axis order of the observed joint. Every other node is hidden."""
        self.card, self.observed = dict(card), list(observed)
        self.parents = {v: list(parents.get(v, [])) for v in card}
        self.hidden = [v for v in card if v not in self.observed]
        self.nodes = self.hidden + self.observed  # axis order of the full table
        self.ax = {v: i for i, v in enumerate(self.nodes)}

    def random(self, rng, alpha=1.0):
        return {v: rng.dirichlet(np.full(self.card[v], alpha),
                                 size=tuple(self.card[p] for p in self.parents[v]))
                for v in self.nodes}

    def _factor(self, v, cpt):
        """cpt[v] broadcast to the full table."""
        order = self.parents[v] + [v]
        t = cpt[v].transpose(np.argsort([self.ax[u] for u in order]))
        shape = [1] * len(self.nodes)
        for u in order:
            shape[self.ax[u]] = self.card[u]
        return t.reshape(shape)

    def _unnormalised(self, cpt, tau):
        """log of prod_H p(v|pa) prod_Y p(v|pa)^tau, full table."""
        logt = np.zeros([self.card[v] for v in self.nodes])
        for v in self.nodes:
            w = tau if v in self.observed else 1.0
            logt = logt + w * np.log(np.maximum(self._factor(v, cpt), EPS))
        return logt

    def joint(self, cpt):
        return np.exp(self._unnormalised(cpt, 1.0))

    def predict(self, cpt):
        """The model's distribution over the observed nodes."""
        return self.joint(cpt).sum(axis=tuple(range(len(self.hidden))))

    def score(self, cpt, py, tau=1.0):
        """Expected tempered log score (bits) under the observed distribution py."""
        logt = self._unnormalised(cpt, tau)
        m = logt.max(axis=tuple(range(len(self.hidden))), keepdims=True)
        lse = np.log(np.exp(logt - m).sum(axis=tuple(range(len(self.hidden))))) + m.squeeze(
            tuple(range(len(self.hidden))))
        mask = py > 0
        return float((py[mask] * lse[mask]).sum() / np.log(2))

    def coupling(self, cpt, py, tau=1.0):
        """p(y) q(h|y) with q the tempered posterior, full table."""
        logt = self._unnormalised(cpt, tau)
        hax = tuple(range(len(self.hidden)))
        logt = logt - logt.max(axis=hax, keepdims=True)
        q = np.exp(logt)
        q /= q.sum(axis=hax, keepdims=True)
        return q * py.reshape((1,) * len(self.hidden) + py.shape)

    def m_step(self, joint):
        """Conditional tables of a full joint table."""
        cpt = {}
        for v in self.nodes:
            order = self.parents[v] + [v]
            keep = [self.ax[u] for u in order]
            marg = joint.sum(axis=tuple(i for i in range(len(self.nodes)) if i not in keep))
            marg = marg.transpose([sorted(keep).index(i) for i in keep])
            denom = marg.sum(-1, keepdims=True)
            k = self.card[v]
            cpt[v] = np.where(denom > 0, marg / np.maximum(denom, EPS), 1.0 / k)
        return cpt

    def fit(self, py, tau=1.0, rng=None, restarts=10, iters=3000, tol=1e-12, init=None):
        """Maximise the expected tempered log score by tempered EM (best of restarts)."""
        rng = np.random.default_rng() if rng is None else rng
        best = None
        inits = [init] if init is not None else []
        inits += [self.random(rng) for _ in range(restarts)]
        for cpt in inits:
            prev = -np.inf
            for t in range(iters):
                cpt = self.m_step(self.coupling(cpt, py, tau))
                if t % 25 == 0:
                    s = self.score(cpt, py, tau)
                    if s - prev < tol:
                        break
                    prev = s
            s = self.score(cpt, py, tau)
            if best is None or s > best[0]:
                best = (s, cpt)
        return best[1]

    def info(self, joint):
        """I(H;Y), D_H and D_Y (bits) of a full joint table."""
        def H(a):
            a = a[a > 0]
            return float(-(a * np.log2(a)).sum())

        nh = len(self.hidden)
        hax, yax = tuple(range(nh)), tuple(range(nh, len(self.nodes)))
        i_hy = H(joint.sum(yax)) + H(joint.sum(hax)) - H(joint)

        def cond_ent(v):
            keep = [self.ax[u] for u in self.parents[v] + [v]]
            pk = joint.sum(axis=tuple(i for i in range(len(self.nodes)) if i not in keep))
            pp = joint.sum(axis=tuple(i for i in range(len(self.nodes))
                                      if i not in keep[:-1]))
            return H(pk) - H(pp)

        d_h = sum(cond_ent(v) for v in self.hidden) - H(joint.sum(yax))
        d_y = sum(cond_ent(v) for v in self.observed) - (H(joint) - H(joint.sum(yax)))
        return dict(rate=i_hy, d_hidden=max(d_h, 0.0), d_observed=max(d_y, 0.0))
