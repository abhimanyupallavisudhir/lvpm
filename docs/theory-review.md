# Review of Task 1's theory ("What gives a latent belief meaning?")

This reviews `notes/latent-meaning.md` on branch `tavya/task_mupwxkjid2091504e8` (commit `460ab22`). I re-derived the identities and independently re-ran the most surprising numerical claim. The last section connects the theory to the latent-variable-market (LVPM) proposal in [proposal.md](proposal.md).

## Verdict

The core is right, and it is the right frame.
- A scoring rule only sees the observable joint.
- How well a latent fits is measured by the conditional total correlation it leaves, $\mathrm{TC}(Y\mid X)$.
- What counts as "the" latent among exact fits is a parsimony convention, which Wyner's common information makes precise.
- More views shrink the set of fits that a latent belief has to be chosen from.

Six things need fixing or sharpening. The most important is §C below. The note's own data show that a latent acquires observable meaning exactly when the latent model is **over-identified**, i.e. when it makes testable predictions. That is a sharper criterion than "three views", and it maps straight onto the market design.

## A. Checked and correct

| claim | check |
|---|---|
| §2 regret identity | $\mathrm{KL}\big(p(x,y)\,\|\,p(x)\prod_i p(y_i\mid x)\big)=\sum_x p(x)\,\mathrm{KL}\big(p(y\mid x)\,\|\,\prod_i p(y_i\mid x)\big)=\sum_i H(Y_i\mid X)-H(Y\mid X)$. Marginalising $X$ can only lower it, so this is an upper bound on the observable regret. |
| §3 reparametrisation | $\mathrm{TC}(Y\mid X)=\mathrm{TC}(Y)-\sum_i I(X;Y_i)+I(X;Y)$. Hence $I(X;Y)+\beta\,\mathrm{TC}(Y\mid X)\propto I(X;Y)-\tau\sum_i I(X;Y_i)+\text{const}$, with $\tau=\beta/(1+\beta)$. The objective is bounded below by $-\tau\,\mathrm{TC}(Y)$. |
| $\tau=1$ is ML / CorEx | $\min_q \mathrm{TC}_q(Y\mid X)=\min_\theta \mathrm{KL}(p_Y\,\|\,m_\theta)$, by the EM variational identity. This holds with a cardinality cap on $X$. |
| reduction to the information bottleneck (IB) | Under $X\!-\!Y_1\!-\!Y_2$: $I(X;Y)=I(X;Y_1)$ and $I(Y_1;Y_2\mid X)=I(Y_1;Y_2)-I(X;Y_2)$, which gives the IB Lagrangian. |
| tempered EM is monotone | It is alternating minimisation of the variational upper bound $E\big[\log q(x\mid y)-\log r(x)-\tau\sum_i\log s_i(y_i\mid x)\big]$. That bound is tight at $r=q(x)$, $s_i=q(y_i\mid x)$, so the objective never increases. |
| Wyner, doubly symmetric binary source with $p=0.1$ | $1+h(0.1)-2h(a)$ with $a=\tfrac{1-\sqrt{0.8}}{2}$ gives $0.8728$ bits. ✓ |
| 3-binary-view latent cheaper than truth | Re-solved independently ([script](../experiments/06_verify_task1_cheaper_latent.py)). The 3-state exact fit has $I=0.2197$ bits and constraint violation $6\times10^{-15}$; the true 2-state latent has $0.3851$ bits. ✓ |

## B. "Information bottleneck vs latent bottleneck": the Markov-chain phrasing

"The chain runs the other way" is the confusing part. Precisely:

- **Information bottleneck.** $X$ is a *representation of $Y_1$*: a (possibly noisy) function of $Y_1$ alone, $q(x\mid y_1,y_2)=q(x\mid y_1)$. So $X$ sits at the *end* of the chain $X - Y_1 - Y_2$, and $Y_2$ reaches $X$ only through $Y_1$.
- **Latent bottleneck.** $X$ may depend on all views jointly, and the aim is to put it in the *middle*, $Y_1 - X - Y_2$, so that it screens the views off from each other.

So the IB is the latent bottleneck with the extra constraint "$X$ must be readable off one view". For the doubly symmetric binary source, the cheapest exact fit under that constraint is $X=Y_1$ (1 bit). The unconstrained common cause is a fair coin $X$ with $Y_i=X\oplus\mathrm{Bern}(a)$, which costs 0.873 bits. "A cause is cheaper than a representation" is the correct takeaway.

## C. Meaning comes from over-identification, not from a number of views

**Point 1: "choosing among fits" is only exact at $\tau\to1$.**
- For $\tau<1$ the optimum generally leaves residual dependence $\mathrm{TC}(Y\mid X)>0$. It is a rate–distortion trade-off in which $\mathrm{TC}(Y\mid X)$ is the distortion.
- So two objects are being run together:
  - the **Wyner point**: first fit exactly, then minimise $I(X;Y)$;
  - **lossy latents** with $\tau<1$, such as a deliberately coarse "Russia is winning" factor.
- Only the first concerns what scoring determines. The second is a modelling choice.

**Point 2: identification is relative to a model class.**
- Kruskal's theorem makes the *rank-2* decomposition of three binary views unique. But that model has 7 parameters for 7 observable degrees of freedom, so it is **just-identified**: generic joints in an open region all fit it.
- "$|X|=2$" is then an untestable assumption, a coordinate system rather than a claim. That is exactly why a 3-state latent can be cheaper there.

**Point 3: the note's own Wyner-vs-true runs sort perfectly by over-identification** (degrees of freedom $D$ of $P(Y)$ vs parameter count $K$ of the true latent model):

| setting | $K$ vs $D$ | Wyner-minimal = true? |
|---|---|---|
| 2 views (any) | $K>D$ (under-identified) | never (10/10 cheaper) |
| 3 binary views, $\lvert X\rvert=2$ | $7=7$ (just-identified) | never (4/4 cheaper) |
| 3 ternary views, $\lvert X\rvert=2$ | $13<26$ | always (3/3) |
| 3 ternary views, $\lvert X\rvert=3$ | $20<26$ | always (3/3) |
| 4 binary views, $\lvert X\rvert=2$ | $9<15$ | always (2/2) |

**Sharper conjecture** (to replace "conventions agree under rich enough views"): *if the true latent model is generically over-identified, the Wyner-minimal exact latent is the true one, up to relabeling and splitting states.*

The heuristic: an over-identified truth lies on a proper subvariety of the observable simplex. A different, cheaper exact decomposition with more states would have to land on that subvariety, which is non-generic. Splitting a state into copies with identical conditionals leaves $I(X;Y)$ unchanged, so splits are the only "ties".

**Point 4: this is the information-theoretic version of a classical fact.** A latent hypothesis has observable content to the extent that it implies testable constraints: Ramsey-sentence content, or tetrad constraints in factor analysis.

## D. Other corrections

1. **§6 (probe scoring) is missing its key condition.** $S(q,y_p)=\log\sum_x q(x)L(y_p\mid x)$ elicits the agent's posterior over $X$ only if the probe is **conditionally independent of everything the agent conditions on, given $X$**: $Y_p\perp (y_1,y_2,\text{agent's information})\mid X$. Otherwise the agent's best report is $P(Y_p\mid \text{info})$, which is not of the form $q^\*L$. Also note that full row rank of $L$ needs $|\mathcal Y_p|\ge|\mathcal X|$.
2. **Continuous latents** (IRT/factor models, i.e. this project's markets) make injectivity on *all* distributions over $X$ impossible: finitely many binary probes cannot resolve an infinite-dimensional $q$. A belief about a continuous latent is scorable only within a parametric family, e.g. Gaussian $P(Y)$, which is another explicit convention.
3. **De Finetti.** "$H(X\mid Y_{1..n})\to0$" is not meaningful for a continuous $\theta$. Say instead that the posterior concentrates, i.e. $\theta$ is measurable with respect to the tail σ-field. The result also needs exchangeability. For non-identical conditionally independent views, the analogue is posterior consistency (Doob).
4. **Algorithms.** The note's local EM / SLSQP can miss solutions; the note itself records a rerun that did. For three or more conditionally independent views, **tensor or spectral methods** are globally consistent: Anandkumar, Ge, Hsu, Kakade & Telgarsky, *Tensor decompositions for learning latent variable models*, JMLR 2014. For hierarchical structures ($X\to Z_j\to Y_i$), latent-tree identifiability holds when every latent node has at least 3 neighbours (Choi, Tan, Anandkumar & Willsky, JMLR 2011), with consistent recovery algorithms. EM is then fine as a refinement step.

## E. What this gives us (your question 3 on Task 1)

**(a) Computing the latent structure from $P(Y_1,\dots,Y_n)$ and a DAG.**
- Yes for a single discrete latent: tempered EM, or tensor methods plus EM. It extends to latent DAGs via EM on the DAG.
- Uniqueness requires the structure to be over-identified, e.g. at least 3 neighbours per latent node in trees.
- In a market there is no population $P(Y)$, only an agent's belief over one world's outcomes. So the use is **canonicalising a belief**: given a reported joint, compute its canonical latent decomposition.

**(b) Eliciting latent beliefs according to the bottleneck criterion.** Two observations settle this.

1. **Payoffs are constant on the fiber, so the convention cannot be incentivised, and doesn't need to be.**
   - Every proper score of a latent report depends only on its implied joint $m$.
   - The mechanism should therefore elicit $m$ properly, then **canonicalise**: map the report to the Wyner point of its fiber, or apply the gauge fix.
   - Canonicalisation changes no one's payoff, so it is free. Tailcalled's fixing of $Y\sim N(0,1)$ is the continuous analogue.
   - Adding a penalty $-\lambda I_m(X;Y)$ to payoffs *would* reward compression, but for $\lambda>0$ it distorts the observable forecasts. Don't.
2. **Restricting the report family turns the log score into the (hard-constraint) bottleneck.**
   - **Proposition.** Let reports range over conditionally independent latent models with $|X|\le k$, and pay $\log m(\vec y)$. An agent with belief $p$ maximises its expected payoff at
     $$\arg\min_{m}\mathrm{KL}(p\,\|\,m)\;=\;\text{moments of }\arg\min_{q:\,|X|\le k}\mathrm{TC}_q(Y\mid X),$$
     i.e. the $\tau=1$ latent bottleneck (CorEx / maximum likelihood) under a capacity cap.
   - *Proof:* $\mathrm{KL}(p\|m)=\min_q\mathrm{KL}(p\,q\,\|\,m(x,y))\ge\min_q\min_{m'}\mathrm{KL}(p\,q\,\|\,m')=\min_q\mathrm{TC}_q(Y\mid X)$, and the moment model of the optimal $q$ attains it while staying in the family. ∎
   - Capping $I_m(X;Y)\le R$ instead of $|X|$ gives the same result at exact fit and approximately elsewhere: the report's own $I_m$ differs from the coupling's $I_q$ off the fit.
   - **So the market designer picks the capacity, the log score does the rest, and canonicalisation removes any remaining ties.**

## F. Implications for the LVPM proposal (this task)

- **At least 4 indicators per factor, not 3.** A 1-factor model on 3 binary indicators is just-identified: any $3\times3$ tetrachoric matrix with a positive product of correlations fits, up to the bounds $\lambda_i^2\le1$. The factor hypothesis becomes testable through vanishing tetrads only from $n\ge4$. This is the continuous counterpart of §C.
- **A 1-factor market is the capacity-capped bottleneck.** Log-scored bets move the market to the KL projection onto 1-factor models, which is maximum likelihood / CorEx within that family. The residual dependence the market leaves is exactly what the pilot's pairwise elicitation picked up (+1.72 vs +0.44 nats).
- **The gauge fix and label are canonicalisation.** They are free to impose because payoffs are fiber-invariant. The label test in the proposal (test T3) checks that the canonical coordinate means what its name says.
- **Held-out probes are the proposal's new-indicator test (T4)**, provided the new indicator is conditionally independent of the old ones given the latent. Measuring that independence is part of the test.
