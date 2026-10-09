# What gives a latent belief meaning?

A probability for an observable gets its meaning from a proper scoring rule
applied to the outcome. A belief about a latent variable never meets an
outcome. So what makes one latent structure for observables $Y_1,\dots,Y_n$
better than another, and how can latent beliefs be elicited or traded?

## Summary

- **A latent means what it does to predictions of observables.** A scoring rule
  can reward nothing else. Latent beliefs are pinned down exactly as far as the
  observable distribution pins them down (§1, §5).
- **Two quantities grade a latent.** *Unexplained dependence*: how far the
  observables are from being independent given the latent. *Information cost*:
  how much the latent knows about the observables. Total correlation alone
  accepts $X=Y$. The cost term is what rules that out (§2).
- **The latent bottleneck** trades the two off. The information bottleneck is
  the special case where the latent must be computable from one observable
  (§3).
- **What you get from it (§4):**
  - **An algorithm.** Tempered EM fits any latent DAG to $P(Y)$.
  - **An elicitation rule.** The *tempered log score*
    $S_\tau(\theta;y)=\log\sum_h \prod_{v\in H}p_\theta(v|\mathrm{pa}_v)\prod_{v\in Y}p_\theta(v|\mathrm{pa}_v)^{\tau}$
    is a scoring rule on one realized outcome. Its expected value is maximized
    exactly by the bottleneck-optimal latent model. At $\tau=1$ it is the
    ordinary log score.
- **$S_\tau$ is the latent bottleneck with the encoder optimized out**,
  written per observation (§4b).
- **Continuous latents** (IRT, factor analysis, mixed indicators, several
  factors) use the same score and algorithm (§4c).
- **Underdetermination comes in two kinds (§1).** *Gauge* is the same latent in
  different coordinates (relabelings, sign flips, rotations); it is harmless.
  *Genuine* underdetermination is different latents with the same predictions.
  It is real but structured, and it vanishes with enough structure or
  indicators.
- **The LVPM proposal** (log market scoring rule on the implied joint) is the
  $\tau=1$ case (§6). It is incentive-compatible, and it prices latent beliefs
  exactly when the market's model family identifies them up to gauge. It
  fails for free-form latent distributions and user-proposed classes.
  $\tau<1$ pins both, but only by compressing the latent (§5).

Code: [`lvpm/dag.py`](../lvpm/dag.py) (discrete latent DAGs),
[`lvpm/continuous.py`](../lvpm/continuous.py) (continuous latents) and
[`lvpm/latent_bottleneck.py`](../lvpm/latent_bottleneck.py). Every number below
is reproduced by `experiments/*.py`, with outputs in
[`elicitation.txt`](elicitation.txt), [`continuous.txt`](continuous.txt),
[`results.txt`](results.txt) and [`wyner_vs_true.txt`](wyner_vs_true.txt).

---

## 1. Setup: scoring sees only the implied distribution of $Y$

A latent structure is a DAG $G$ over hidden nodes $H$ and observed nodes $Y$. A
latent model $\theta$ is one conditional table $p_\theta(v\mid\mathrm{pa}_v)$ per
node. Examples:
- $X\to Y_i$: latent class analysis, or IRT when $X$ is continuous.
- $X\to Z_j\to Y_i$: a hierarchical model.

The model predicts $m_\theta(y)=\sum_h p_\theta(h,y)$. Under any proper scoring
rule, the expected score of $\theta$ depends only on $m_\theta$. Two kinds of
slack follow, and they differ in kind (question 2a).

**What a latent model says: its mixing measure.** For each latent state $x$,
write down its *profile*, the list of what each indicator would do in that
state:
$$\pi(x)=\big(P(Y_1|X{=}x),\dots,P(Y_n|X{=}x)\big).$$
With binary indicators, a profile is a point in the cube $[0,1]^n$.

A latent model is then a cloud of such points with weights: put mass $P(X{=}x)$
on the point $\pi(x)$. That weighted cloud is the *mixing measure* $\mu$, a
probability distribution over profiles. The model's prediction is the
$\mu$-weighted average of the corresponding independent distributions:
$P(Y)=\int\pi_1\otimes\dots\otimes\pi_n\,d\mu(\pi)$.

Examples:
- **Two-state latent class model:** two points in the cube, with weights
  0.4 and 0.6.
- **One-factor IRT:** as $z$ runs over $\mathbb R$, the profile
  $\big(\sigma(a_1z+b_1),\dots,\sigma(a_nz+b_n)\big)$ traces a curve in the
  cube, and $N(0,1)$ spreads the mass along it.

The *name* of a state ($x$, or the value of $z$) never appears in $\mu$; only
its profile does. That is why $\mu$ is the latent's coordinate-free content.

**Gauge: same $\mu$, different coordinates.** Relabeling discrete states only
changes which name goes with which point. $z\mapsto -z$, or any
reparametrization of $z$, slides along the same curve and carries the mass
with it. The cloud does not change.

Declaring the prior to be $N(0,I)$ with linear predictors $a\cdot z+b$ cuts the
freedom down to rotations $O(k)$, i.e. a sign flip for one factor. Check:
rotating a 2-factor IRT model by 0.7 rad changes no pattern probability by more
than $2\times10^{-8}$ (quadrature error).

Gauge is harmless and expected. Every gauge-invariant latent statement is a
statement about $\mu$, and it is *relational*. Example: "with probability 0.3
the world is in a state where indicator 1 has probability above 0.8". A
coordinate statement like "$Z>1$" means something only after the gauge is
fixed.

Your inverted-spectrum intuition fits exactly. A gauge transformation preserves
every relation to observables, including the agent's own reports. So whatever
a label carries beyond its relational role is something no score can reach.

**Genuine underdetermination: a different $\mu$ with the same average.** Two
different clouds can average to the same $P(Y)$. They are different claims
about the world, not relabelings: a relational question such as "how often is
the world in a state where indicator 1 is likely?" gets different answers.
Two examples:
- **Same curve, different mass along it.** In the free-form IRT example of
  §5, the clouds that fit put anywhere from 0.035 to 0.317 of their mass in a
  given region of the curve, yet all predict the same $P(Y)$.
- **Different curves.** Two Gaussian indicators with correlation 0.5 and one
  factor: loadings $(0.6, 0.833)$, $(0.707, 0.707)$ and $(0.9, 0.556)$ all
  reproduce $P(Y)$ exactly and score identically. They still disagree:
  $I(Z;Y)$ is 0.97, 0.79 and 1.26 bits, and $E[Z\mid y=(1,1)]$ is 0.956, 0.943
  and 0.970.

**Why "how often is indicator 1 likely?" is not the question "what is
$P(Y_1)$?"** Write $\pi_1(x)=P(Y_1{=}1\mid x)$. Then $P(Y_1{=}1)=\mathbb
E_\mu[\pi_1]$ is the *mean* of $\pi_1$, while "how often is the world in a
state where indicator 1 is likely" is the tail $\mu(\pi_1>0.8)$. Two worlds
with $P(Y_1{=}1)=0.5$:
- **(a)** $\pi_1=0.5$ always: the outcome is a genuine coin flip.
- **(b)** $\pi_1=0.9$ half the time and $0.1$ otherwise: the world is in a
  decisive state, and we just don't know which.

The tail is 0 in (a) and 0.5 in (b).

For binary indicators, $P(Y)$ is equivalent to the products over *distinct*
indicators, $\mathbb E_\mu\big[\prod_{i\in S}\pi_i\big]$ for every subset $S$.
No power of a single coordinate, such as $\mathbb E_\mu[\pi_1^2]$, ever
appears, because each indicator resolves once per world. Correlations with
other indicators reveal part of the spread of $\pi_1$, and the rest is exactly
what stays underdetermined.

This is also why more indicators help. With $n$ exchangeable indicators that
share one profile $\pi$, $P(\text{all } n=1)=\mathbb E_\mu[\pi^n]$. So the
moments up to order $n$ are observed, and as $n\to\infty$ they determine $\mu$:
de Finetti's theorem seen as a moment problem.

A physics analogy: a density matrix can be written as an ensemble of pure
states in many ways. The density matrix plays the role of $P(Y)$, the ensemble
plays $\mu$, and measurement statistics fix only the former. With infinitely
many exchangeable indicators, the decomposition becomes unique: the set of
such laws is a simplex whose extreme points are the i.i.d. laws (de Finetti).
With finitely many indicators, Kruskal-type conditions make it unique among
decompositions with few states.

How bad is this? It is not "undetermined to the point of useless". The valid
$\mu$ form a convex set, and every statement that holds across the whole set
is determined. What is left depends on the structure:

| structure | what scoring leaves free |
|---|---|
| one latent with at least 3 indicators and the fewest states; a 1-factor parametric model (IRT/FA) with at least 3 indicators; latent trees with every hidden node of degree at least 3 | gauge only |
| 2 indicators | a continuum (the free loading above) |
| latents larger than necessary | genuine alternatives; a convention picks one (fewest states or fewest bits, §5) |
| free-form latent distribution | an infinite-dimensional set that shrinks as indicators are added (§5) |

## 2. Unexplained dependence and information cost

Let a latent be coupled to the observables, $p(h,y)=p(y)\,q(h\mid y)$, where
$q$ is any "encoder". Two quantities matter.

**Unexplained dependence** is the gap to the best model with structure $G$:
$$D_G=\mathrm{KL}\Big(p(h,y)\,\Big\|\,\textstyle\prod_v p(v\mid\mathrm{pa}_v)\Big)\;\ge 0,$$
where the conditionals are those of $p(h,y)$. For one latent $X$ with
$Y_1,\dots,Y_n$ as children, this is the conditional total correlation
$\mathrm{TC}(Y|X)=\sum_i H(Y_i|X)-H(Y|X)$. It is zero iff the $Y_i$ are
independent given $X$. It equals the log-score regret, on the complete data
$(h,y)$, of the best model with structure $G$, and it upper-bounds the regret
on $Y$ alone. This is the scoring rule's half of the answer.

**Information cost** is $I(H;Y)$: how many bits the latent carries about the
observables.

On your question 1a: $D_G=0$ admits trivial latents. $X=(Y_1,\dots,Y_n)$
explains every dependence perfectly, because each $Y_i$ is a deterministic
function of it. Total correlation alone therefore cannot prefer a real
latent. The cost $I(H;Y)$ is what penalizes such a latent: it costs $H(Y)$ bits,
the maximum. A good latent explains a lot of dependence with few bits.

My earlier phrase "choosing among fits" was imprecise. It describes only the
limiting case: require $D_G=0$ exactly, then minimize cost. That is Wyner's
common information. The general criterion below is a *trade-off*. Its optimum
usually leaves a little dependence unexplained, in exchange for a much cheaper
latent. In the hierarchical example of §4, $\tau=0.8$ keeps 55% of the true
latents' information and loses only 0.076 bits of log score.

## 3. The latent bottleneck

Trade the two off with a weight $\tau\in[0,1)$. For a single latent:

$$\mathcal L_\tau(X)=I(X;Y)-\tau\sum_i I(X;Y_i)\;\;\equiv\;\;\frac{I(X;Y)+\beta\,\mathrm{TC}(Y|X)}{1+\beta},\qquad \beta=\tfrac{\tau}{1-\tau},$$

where $\equiv$ means equal up to a constant. In words: compress all the
observables jointly, while staying informative about each one separately.

For a general DAG, the objective that §4's scoring rule elicits is
$I(H;Y)+(D_H+\tau D_Y)/(1-\tau)$. Here $D_G=D_H+D_Y$ splits the violation into
the part among hidden nodes and the rest.

Known special cases:

| Special case | Known as |
|---|---|
| $\tau\to1$: exact fit ($D_G=0$), then least information | Wyner's common information |
| $\tau=1$, number of latent states fixed | maximum likelihood (EM), CorEx |
| two observables, encoder restricted to read only $Y_1$ | the information bottleneck |

### How it differs from the information bottleneck (question 1b)

In the information bottleneck you are handed $Y_1$ and want to predict $Y_2$.
You build a summary $T$ of $Y_1$: a (possibly randomized) function of $Y_1$
alone, an encoder $p(t\mid y_1)$. "$T - Y_1 - Y_2$" is shorthand for exactly
this: $T$ is computed by looking only at $Y_1$. So once you know $Y_1$, $T$
adds nothing about $Y_2$. Example: $Y_1$ is an image, $Y_2$ its label, and $T$
is a feature vector extracted from the image.

A latent is not computed from any one observable. It is a hidden common cause,
and you only ever infer it. Example: $X$ is whether a patient has a disease,
and $Y_1,Y_2$ are two noisy tests. Test 1 alone does not determine $X$, and
test 2 carries independent evidence about it. The encoder $q(x\mid y_1,y_2)$
must look at both.

The precise relation: take the latent bottleneck and allow only encoders that
ignore $Y_2$, i.e. $q(x\mid y_1,y_2)=q(x\mid y_1)$. Under that restriction,
$I(X;Y_1,Y_2)=I(X;Y_1)$ and $\mathrm{TC}(Y|X)=I(Y_1;Y_2)-I(X;Y_2)$. The objective
becomes, up to a constant, $I(Y_1;X)-\beta I(X;Y_2)$, which is exactly the
information-bottleneck objective. **The information bottleneck is the latent
bottleneck restricted to latents readable from one side.** Lifting that
restriction makes the latent strictly cheaper.

In the example below, the observables are two bits that agree 90% of the time,
each one a noisy copy of a hidden bit. The cheapest exact explanation computable
from $Y_1$ is $Y_1$ itself (1 bit). A free latent needs only 0.873 bits, which
is Wyner's closed form, reproduced numerically. The free curve is below the
information-bottleneck curve everywhere.

![bottleneck](figures/bottleneck_dsbs.png)

## 4. What this gives you

### 4a. An algorithm (question 3a)

**Input:** $P(Y)$ (or an empirical distribution from data), a DAG, and a
cardinality for each hidden node.
**Output:** all conditional tables.

The method is tempered EM. Alternate:
1. **E-step:** $q(h\mid y)\propto\prod_{v\in H}p(v\mid\mathrm{pa}_v)\prod_{v\in Y}p(v\mid\mathrm{pa}_v)^{\tau}$.
2. **M-step:** set every table to the conditionals of $p(y)\,q(h\mid y)$.

Each step increases $\mathbb E\,S_\tau$, and its fixed points are the latent
bottleneck's stationary points. At $\tau=1$ this is ordinary EM.
Implementation: `LatentDAG.fit`.

Checks:
- **Single latent:** the maximized $\mathbb E\,S_\tau$ equals
  $-\min\mathcal L_\tau-\tau\sum_i H(Y_i)$ to $5\times10^{-12}$ bits. The
  general-DAG identity holds to $5\times10^{-11}$ bits.
- **Hierarchical model,** $X\to Z_1,Z_2,Z_3$, each with two binary indicators,
  at $\tau=1$: all 20 random starts recover every table to $1.3\times10^{-7}$,
  up to relabeling.
- **Lower $\tau$ compresses:** $I(H;Y)$ falls from 1.46 (the true latents) to
  1.33 at $\tau=0.95$ and to 0.81 at $\tau=0.8$. The log-score regret on $P(Y)$
  is 0.0025 and 0.076 bits.

Limitations:
- The objective is nonconvex, so use restarts.
- `LatentDAG` enumerates the full joint table, so it handles small discrete
  models only. Continuous latents are in §4c.
- "Computes all these distributions" holds only when the structure is
  identified (§5). Otherwise, at $\tau=1$ it returns *a* model that predicts
  $P(Y)$, and at $\tau<1$ the bottleneck-optimal one.

### 4b. Where $S_\tau$ comes from, and eliciting with it (questions 3 and 3b)

Start from the latent bottleneck, written as a minimization over encoders
$q(x\mid y)$. All entropies below are under the coupling $p(y)\,q(x\mid y)$.
The single-latent case is shown; for a DAG, replace each entropy by the
corresponding $H(v\mid\mathrm{pa}_v)$.

$$\mathcal L_\tau(q)=I(X;Y)-\tau\sum_i I(X;Y_i).$$

**Step 1: write it in entropies.** Using $I(X;Y)=H(X)-H(X|Y)$ and
$I(X;Y_i)=H(Y_i)-H(Y_i|X)$:
$$\mathcal L_\tau(q)+\tau\textstyle\sum_iH(Y_i)=H(X)+\tau\sum_iH(Y_i\mid X)-H(X\mid Y).$$

**Step 2: each entropy is the best achievable log loss.** By Gibbs'
inequality, $H(X)=\min_r\mathbb E[-\log r(X)]$ and $H(Y_i|X)=\min_{c_i}\mathbb
E[-\log c_i(Y_i|X)]$. The minimizers, a prior $r$ and indicator conditionals
$c_i$, are exactly a latent model $\theta$. So
$$\mathcal L_\tau(q)+\tau\textstyle\sum_iH(Y_i)=\min_\theta\;\mathbb E\Big[\log q(x|y)-\log r(x)-\tau\sum_i\log c_i(y_i|x)\Big].$$

**Step 3: optimize the encoder out.** Minimize over $q$ as well, and swap the
order of the minimizations. For a fixed $\theta$, the bracket separates over
$y$ into $\sum_x q(x|y)\,[\log q(x|y)-\log a_\theta(x,y)]$, where
$a_\theta(x,y)=r(x)\prod_i c_i(y_i|x)^{\tau}$. This is minimized by the
*tempered posterior* $q\propto a_\theta$, with value
$-\log\sum_x a_\theta(x,y)=-S_\tau(\theta;y)$. Therefore
$$\min_q\mathcal L_\tau(q)=-\max_\theta\;\mathbb E_{y\sim p}\,S_\tau(\theta;y)\;-\;\tau\textstyle\sum_iH(Y_i).$$

So **$S_\tau$ is the latent bottleneck with the encoder optimized out, written
per observation.** It is an average over $y$ of a quantity computable from
$\theta$ and a single $y$, so it is a scoring rule. Its maximizing $\theta$ is
the bottleneck-optimal latent model, and the tempered posterior of that model
is the optimal encoder.

At $\tau=1$, $a_\theta$ is the model's joint and $\sum_xa_\theta=m_\theta(y)$.
Step 3 is then the familiar "log-likelihood = maximized ELBO" identity, and
the bottleneck reduces to maximum likelihood. The exponent $\tau$ on the
indicator factors is where compression comes from. Each indicator's evidence
counts only to the power $\tau$, so posteriors are less sharp and the latent
carries fewer bits. For a DAG the same steps give the objective quoted in
§3.

The tempered posterior $q\propto p(x)\prod_ip(y_i|x)^{\tau}$ is known in
statistics as a *power* (or *fractional*, *tempered*) posterior. That family of
methods is called **generalized Bayes**. It has nothing to do with Jeffreys
priors, which choose a reparametrization-invariant prior. Generalized Bayes
instead changes the *update*: evidence enters as $\text{likelihood}^\eta$, or
more generally as $\exp(-\eta\cdot\text{loss})$, with a learning rate $\eta$.

Its known uses differ from ours:
- **Robustness to misspecification:** Grünwald's SafeBayes picks $\eta<1$.
- **Coherent updating with losses:** Bissiri, Holmes and Walker (2016).
- **PAC-Bayes "Gibbs posteriors"**, which minimize
  $\eta\,\mathbb E_q[\text{loss}]+\mathrm{KL}(q\,\|\,\text{prior})$.

The last is exactly Step 3 above, with $\eta=\tau$. Averaging that
$\mathrm{KL}(q(\cdot|y)\,\|\,p(x))$ over $y$ gives $I(X;Y)$, the bottleneck's
information cost. So "power posterior" and "latent bottleneck" are the same
trade-off seen per observation and on average. Here $\tau$ is a compression
knob, not a robustness knob.

**Elicitation.** Pay an agent who reports a latent model $\theta$ the amount
$S_\tau(\theta;y)$ on the realized observables. If the agent believes
$Y\sim p$, their expected payment is maximized exactly by the
latent-bottleneck optimum for $p$.

Properties:
- It needs one realization and no knowledge of $p$.
- It is gauge-invariant.
- At $\tau=1$ it is the log score of $m_\theta$.
- With discrete indicators $S_\tau\le0$, so a market maker's loss is bounded
  exactly as for the log market scoring rule.

The incentives are visible in a three-binary-indicator case where the true
latent has 2 states but a 3-state latent explains the data with fewer bits.
Expected score is relative to the best report in each column; rows are reports
and columns are scoring rules:

| report | $I(X;Y)$ | $S_1$ | $S_{.999}$ | $S_{.99}$ | $S_{.9}$ |
|---|---|---|---|---|---|
| true 2-state latent | 0.385 | **0** | −0.00016 | −0.0017 | −0.023 |
| optimum at τ=0.999 | 0.229 | **0** | **0** | −0.00018 | −0.007 |
| optimum at τ=0.99 | 0.202 | −0.00007 | −0.00005 | **0** | −0.005 |
| optimum at τ=0.9 | 0.086 | −0.0058 | −0.0056 | −0.0046 | **0** |

Each rule pays most for its own optimum. The plain log score ($S_1$) cannot
tell the true latent from the cheaper 3-state one. Any $\tau<1$ breaks the tie
toward fewer bits.

Trade-offs, all real:
- **Forecasts get lossier.** At $\tau<1$ the reported model is a lossy
  forecast of $Y$ (0.0058 bits of regret at $\tau=0.9$). Its implied indicator
  probabilities are no longer the agent's best forecasts. Using $\tau$ just
  below 1 makes it a tie-breaker only.
- **$\tau$ is a choice.** The designer picks it, and nothing in the
  observables does.
- **The optimum is hard to compute.** It is a nonconvex, possibly set-valued
  property, so agents must solve a hard problem to report it.

**One unit's latent value** ("this patient has the disease") is a posterior
$q(x)$. Score it on a held-out indicator $Y_p$ with agreed likelihood
$L=P(Y_p|X)$, using $\log\sum_x q(x)L(y_p|x)$. This is strictly proper for $q$
iff the rows of $L$ are linearly independent; otherwise some reports score the
same.

![probe](figures/probe_scoring.png)

### 4c. Continuous latents: IRT, factor analysis, mixed indicators

[`lvpm/continuous.py`](../lvpm/continuous.py) implements continuous latents.
- **Latent:** $Z\in\mathbb R^k$, with prior $N(0,I)$ (the gauge choice) or a
  free distribution on a grid.
- **Indicators:** Bernoulli (logit or probit), Normal, or Categorical, mixed
  freely. Unresolved indicators are NaN.
- **Score and fitting:** integrals use Gauss–Hermite quadrature. The score is
  the same, $S_\tau(\theta;y)=\log\int p(z)\prod_ip(y_i|z)^\tau dz$, and so is
  tempered EM. The E-step runs on the grid. The M-step is a weighted maximum
  likelihood fit per indicator (logistic or probit regression, least squares,
  multinomial logit), plus reweighting of the grid when the prior is free.

Checks ([`continuous.txt`](continuous.txt)):
- **Quadrature vs closed form.** For factor analysis,
  $S_\tau=\sum_i[\tfrac{1-\tau}2\log2\pi\psi_i-\tfrac12\log\tau]+\log N(y;b,AA^\top+\Psi/\tau)$.
  Quadrature matches this to $5\times10^{-12}$ bits.
- **IRT recovery.** One-factor logit and probit IRT with 6 items: 5 random
  starts recover all parameters to $3\times10^{-7}$, up to sign. Two-factor
  IRT with 8 items: the rotation-invariant $AA^\top$ and $b$ are recovered to
  $2\times10^{-6}$.
- **Mixed indicators** (logit, probit, normal, 3-category) from 150k samples:
  the parameters come back within sampling error. The error shrinks with the
  sample size, and the fit's log score exceeds the true parameters' on the
  same sample, as a maximum-likelihood fit should.
- **Gaussian indicators, where the bottleneck latent is continuous.** At
  $\tau=1$ the loadings form a continuum (the table in §1). As $\tau\to1$, the
  tempered score selects the symmetric loadings $\sqrt\rho$ from it, and
  $I(Z;Y)$ converges to Wyner's $\tfrac12\log\frac{1+\rho}{1-\rho}=0.7925$ bits
  (0.7923 at $\tau=0.9999$). Compare the information bottleneck: the exact
  sufficient summary of a continuous $Y_1$ is $Y_1$ itself, which carries
  infinitely many bits.

**Scaling to many factors (question 1).** Nothing in the method limits the
number of factors. Only the way integrals are computed does.
- **Gaussian indicators:** everything is closed form for any $k$
  (`fit_factor_analysis`). Each EM step costs $O(Nnk+k^3)$.
  With 6 factors, 20 indicators and 50k samples, $AA^\top$ is recovered within
  sampling error (0.064), the fit beats the true parameters' log score, and
  the trace is monotone.
- **Other indicators:** for $k\le3$ the code uses tensor Gauss–Hermite nodes,
  whose count grows as $m^k$. Above that it switches to scrambled Sobol
  quasi-Monte Carlo nodes, whose cost does not grow with $k$, and builds the
  whole log-likelihood as one matrix product.
  For 4-factor IRT with 10 items, the integration error is 0.004 (total
  variation) at 1,024 nodes and 0.001 at 4,096. Fitting with 2,048 nodes
  recovers $AA^\top$ to 0.05–0.08 and $b$ to 0.004. That is limited by
  integration error and slow EM: two of three starts hit the 3,000-iteration
  cap.
- **Fixed nodes are the simplest choice, not the best.** They degrade when
  many indicators make each posterior sharp. Standard efficient alternatives,
  all of which apply to $S_\tau$ unchanged (the tempered posterior is just a
  posterior with a rescaled likelihood):
  - adaptive Gauss–Hermite or Laplace-centred importance sampling per
    observation, as used in `lme4`/`glmmTMB`;
  - stochastic-approximation EM (Metropolis–Hastings Robbins–Monro, as used
    in the `mirt` package for high-dimensional IRT);
  - amortized variational inference. A VAE is exactly this model class with a
    neural decoder, and its ELBO is Step 3 at $\tau=1$; at $\tau<1$ it is a
    β-VAE-style objective;
  - for identifiable models, polynomial-time spectral/tensor methods
    (Anandkumar et al. 2014). These exploit the same three-view structure
    that gives identifiability.
- **Hardness.** Exact inference and maximum likelihood are hard in the worst
  case for general latent DAGs, so the guarantees are for structured or
  identifiable cases.
- **Settlement.** Scoring one realized outcome needs only one log-integral,
  which is cheap to compute accurately even when fitting is expensive.
- **Remaining limitation:** EM converges slowly near $\tau=1$ with a free
  prior, and with several factors; acceleration (e.g. SQUAREM) would help.

## 5. When latent beliefs need no convention

Scoring pins a latent model exactly to the extent that $m_\theta$ determines
$\theta$.

- **Identifiable structures.** For one latent with three or more conditionally
  independent indicators, the decomposition with the fewest latent states is
  generically unique up to relabeling (Kruskal 1977; Allman, Matias & Rhodes
  2009). The same holds for latent trees in which every hidden node has at
  least three neighbours. Here $\tau=1$ already elicits the true latent, and EM
  recovers it (to $1.4\times10^{-8}$ for one latent and three binary
  indicators, and to $1.3\times10^{-7}$ in the hierarchy above).
- **Parsimony is still a convention.** Uniqueness is among the *fewest-state*
  models. Larger latents fit too, and the fewest-*bits* latent can differ.
  Exact constrained solves (local search, so "cheaper" is certified and
  "equal" means no cheaper one was found):

| setting | least-information latent vs true latent |
|---|---|
| 2 indicators (10 random models) | cheaper in 10/10: the latent is not identified |
| 3 ternary indicators (6) | equal in 6/6 |
| 3 binary indicators, 2-state truth (4) | cheaper in 4/4, using 3–4 states |
| two of those, plus a 4th indicator | equal |

- **The trend.** More indicators shrink the set of exact fits until
  conventions stop mattering. With infinitely many exchangeable indicators, the
  latent is a function of the observables (de Finetti). §6 shows the same
  trend for a continuous latent. That conventions agree in general once the
  indicators are rich enough is my conjecture from these runs, not a known
  theorem.

### Free-form latent distributions with $\tau<1$ (question 2b)

Setup: three binary IRT indicators whose true latent is $N(0,1)$. The latent
distribution is left free on 161 grid points, and I ran 3 random starts for
each setting. The table shows the spread across starts of gauge-invariant
summaries:

| links | τ | $I(Z;Y)$ | $P\big(P(Y_1{=}1\mid Z)>0.8\big)$ | forecast error (TV) |
|---|---|---|---|---|
| fixed | 1 | 0.552–0.553 | **0.18–0.22** | 0.0001 |
| fixed | 0.95 | 0.488 | 0.06–0.12 | 0.016 |
| fixed | 0.8 | 0.362–0.364 | 0.000 | 0.071 |
| fixed | 0.5 | 0.000 | 0 | 0.202 |
| free | 1 | **0.530–0.569** | **0.15–0.31** | 0.0000 |
| free | 0.95 | 0.474 | 0.001–0.015 | 0.011 |
| free | 0.8 | 0.348 | 0 | 0.065 |

With the links fixed, I also computed the exact range of $P(Z>1)$ over *all*
latent distributions that earn the optimum's score on every outcome:
$[0.035, 0.317]$ at $\tau=1$, $[0, 0.024]$ at $\tau=0.95$, and $[0, 0.002]$ at
$\tau=0.8$.

So: **$\tau<1$ removes the indeterminacy, but "fine" only in that sense.** At
$\tau=1$ the score is flat along an infinite-dimensional set. At $\tau<1$ it
strictly prefers less informative latent states, which breaks the ties. The
price is that the latent you get is the *compressed* one:
- its states' indicator profiles shrink toward uninformative (at $\tau=0.8$,
  no state makes indicator 1 likely);
- its mass collects on a few clusters;
- its implied forecast of the indicators is off (total variation 0.07 at
  $\tau=0.8$);
- at $\tau=0.5$ it disappears altogether ($I=0$).

Near $\tau=1$ the pinning is only partial. This is no accident: with discrete
indicators, the bottleneck optimum always has finite support (standard
cardinality bounds). A smooth continuous latent is never the
fewest-bits explanation of discrete indicators. It is for Gaussian
indicators (§4c).

The real choice is how to pin a free-form latent:
- **by assumption:** a parametric family, which keeps forecasts honest;
- **by data:** more indicators, which works, but slowly;
- **by compression:** $\tau<1$, which costs forecast accuracy.

The score can only distinguish what the observables, or a stated convention,
distinguish.

## 6. Latent-variable prediction markets (question 4)

[tailcalled's proposal](https://www.lesswrong.com/posts/ufW5LvcwDuL6qjdBT/latent-variables-for-prediction-markets-motivation-technical):
the market state is $\theta$, and a trader who moves it from $\phi$ to $\theta$
is paid $\log P(\vec x\mid\theta)-\log P(\vec x\mid\phi)$ on the resolved
indicators. This is Hanson's market scoring rule with $S_1$.

**Does it work?**
- **Incentives:** yes. A myopic trader moves the market to the model in its
  family that best forecasts the indicators under their beliefs, and the market
  maker's loss is bounded.
- **Latent prices:** they mean something exactly when the family identifies the
  latent. The post's own design (one Gaussian latent standardized to
  $N(0,1)$, probit links, at least three indicators) is identified up to sign.
  The loadings follow from the pairwise correlations, as in one-factor analysis.
  Logistic links behave the same in practice. That case works, and the post's "at least 3 indicators" rule is
  the right one.

**Where it does not pin latent prices.** Each of these appears among the
post's design options:
1. **Free-form latent distribution** ("let people freely drag around the
   latent variable distribution"). Payouts depend on $P(Y)$ only through the
   $2^n$ indicator probabilities. For a Gaussian latent, $P(Y>1)=0.156$, but
   every latent distribution giving the same indicator joint earns identical
   payouts. Across those, $P(Y>1)$ ranges over $[0.040, 0.317]$ with 3
   indicators, $[0.078, 0.253]$ with 6, and $[0.087, 0.237]$ with 10. Traders
   can move such beliefs with zero expected cost or reward. Keep the latent
   distribution parametric. $\tau<1$ also pins it, but only by compressing it
   (§5).
2. **Several latent factors: not actually a defect.** Rotating the loadings
   together with the latent changes nothing observable. It is gauge (§1), not
   missing information. "Which axis is *who wins*" is a coordinate choice, not
   a fact, so a market cannot price it and should not. Define such quantities
   relationally instead, e.g. "the latent direction along which
   P(Ukraine holds Crimea) rises fastest". Those are priced whenever the model
   is identified up to rotation.
3. **User-proposed latent classes.** The log score never penalizes an extra
   class, since $k+1$ classes nest $k$. A redundant class ties with the true
   model (the $S_1$ column above), and the latent drifts toward
   $X=$ "the indicators themselves". Settling with $S_\tau$ for $\tau$ slightly
   below 1 makes a proposed class pay for its bits. This answers the post's
   "who chooses the number of outcomes": the class stays iff it raises
   $S_\tau$.
4. **Misspecification.** If the indicators do not truly share one cause, the
   latent the market finds is a best-fit projection. It is meaningful as "the
   best single common cause", not as a ground truth.

**Bottom line.**
- For parametric families identified up to gauge (single- or multi-factor
  IRT/FA with enough indicators), use the post's log market scoring rule as is,
  and phrase latent questions relationally.
- For free latent structure (extra classes, free-form shapes), the plain log
  score leaves genuine latent beliefs unpriced. Either restrict the family, or
  settle with $S_\tau$ and accept the compression that comes with it.

## References

- tailcalled, *Latent variables for prediction markets: motivation, technical guide, and design considerations*, LessWrong, 2023.
- R. Hanson, *Combinatorial information market design*, 2003 (market scoring rules).
- A. Wyner, *The common information of two dependent random variables*, IEEE Trans. IT, 1975.
- N. Tishby, F. Pereira, W. Bialek, *The information bottleneck method*, 1999.
- M. Gastpar, E. Sula, *Relaxed Wyner's common information*, ITW 2019.
- G. Ver Steeg, A. Galstyan, *Discovering structure in high-dimensional data through correlation explanation*, NeurIPS 2014.
- J. Kruskal, *Three-way arrays: rank and uniqueness of trilinear decompositions*, 1977.
- E. Allman, C. Matias, J. Rhodes, *Identifiability of parameters in latent structure models with many observed variables*, Ann. Statist., 2009.
- P. Grünwald, *The safe Bayesian*, ALT 2012; P. Bissiri, C. Holmes, S. Walker, *A general framework for updating belief distributions*, JRSS-B, 2016 (power likelihoods).
- L. Cai, *High-dimensional exploratory item factor analysis by a Metropolis–Hastings Robbins–Monro algorithm*, Psychometrika, 2010.
- A. Anandkumar, R. Ge, D. Hsu, S. Kakade, M. Telgarsky, *Tensor decompositions for learning latent variable models*, JMLR, 2014.
