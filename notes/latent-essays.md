# Scoring natural-language reasoning as latent information

An essay about a war, a technology, or a scientific dispute rarely resolves.
What it does is change what readers expect about many things that do resolve.
That is exactly the position of a latent variable. This note asks what the
framework of [latent-meaning.md](latent-meaning.md) says about scoring such
essays. It works from first principles, rather than bolting a language model
onto the log score.

## Summary

- **An essay's meaning is its profile.** The profile is the conditional
  predictions the essay licenses over a space of resolvable questions, as read
  by a shared interpreter. Natural language plays the role of the link
  function, and paraphrase is the gauge group.
- **An essay is a latent *model*, not a latent value.** It names latent claims
  (scenarios, factors, mechanisms), weighs them, and connects them to
  observables. So score it as a model, with the tempered log score
  $$S_\tau(e;y)=\log\sum_h P_e(h)\prod_q P_R(y_q\mid e,h)^{\tau}.$$
  Here $h$ ranges over the essay's scenarios, $P_e(h)$ is the weight the essay
  gives each one, and $P_R(y_q\mid e,h)$ is the interpreter's reading of what
  the essay predicts for question $q$ in scenario $h$.
- **Three design choices carry the principles:**
  1. Questions are *views*: they are sampled after the essay is written, from a
     measure fixed in advance.
  2. The interpreter answers each question *separately*, given the essay and a
     scenario. The essay must name the common causes; the interpreter only
     fills in their effects.
  3. Payment is *incremental*, as in a market scoring rule against the current
     consensus.
- **What you get:**
  - Properness relative to the interpreter: the author's best essay is the
    bottleneck-optimal latent model they can express.
  - Paraphrase invariance.
  - An exact account of what cannot be scored.
- **What cannot be scored:**
  - Content that changes no resolvable prediction. This is verificationism,
    quantified.
  - Individual claims whose consequences are entangled with other claims'. That
    is Duhem–Quine, which is non-identifiability by another name. The design
    rule is to give each latent claim several conditionally independent
    resolvable consequences, ideally at least three.
- **"Observable" can be widened** beyond empirical events:
  - logical and computational resolutions, which value reasoning relative to a
    bounded reader;
  - other essays used as views (peer prediction), which identifies the
    *common cause* of the essays and not necessarily the truth.
- **The cheap answers fall short.** Conditioning a language model on the essay
  and scoring $\log P(\text{outcomes}\mid\text{essay})$ is a special case that
  omits these structures. Grading with a language-model judge is not a scoring
  rule at all.

---

## 1. Why the cheap answers are not enough

**(A) A language model as judge**, grading the essay against a rubric. This
is not a scoring rule, because nothing resolves and the payoff is the judge's
opinion. In the terms of the earlier note, it is peer prediction with one peer,
whose biases are treated as the truth. It identifies the judge, not the world.

**(B) Predictive conditioning.** Ask a language model for
$P(\text{outcomes}\mid\text{essay})$ and apply the log score. This is proper,
but it has four defects:
1. **Which questions?** If they are known when the essay is written, a list
   of answers beats an explanation.
2. **The interpreter does the latent work.** Asked for a joint forecast, the
   model supplies its own dependence structure among outcomes. The essay then
   gets credit for the effects of common causes it never named.
3. **Reading is conflated with trusting.** The model reads the essay as
   evidence about its author, not as an account to be evaluated on its own
   terms.
4. **One number per essay.** It gives no account of which claim did the work,
   or which claims were never tested.

Each defect corresponds to a piece of the formalism that (B) leaves out.
Putting those pieces back is the proposal.

## 2. Semantics: language is the link function

In a latent-variable market, a latent model is $(P(X), P(Y_i\mid X))$, and the
links are numbers. In an essay the links are sentences. "If Russia's shell
production stays ahead, Ukraine will not retake territory in the south" is a
link from a latent claim to an observable. A reader is what turns the sentence
into a probability.

So fix an **interpreter** $R$. Given an essay $e$, a latent configuration $h$
described in $e$, and a question $q$, it returns $P_R(y_q\mid e,h)$. $R$ can be
a language model with a fixed version and prompt, a human panel, or an
ensemble.

- **The profile.** The profile of $e$ in scenario $h$ is
  $\pi_e(h)=\big(P_R(y_q\mid e,h)\big)_{q\in Q}$. The essay's latent model is
  the mixing measure that puts weight $P_e(h)$ on $\pi_e(h)$. This is exactly
  the object of latent-meaning §1, with texts in place of parameter vectors.
- **Paraphrase is gauge.** Two texts with the same parse and the same profiles
  are the same latent model, and every score treats them identically. A shared
  language is what fixes the gauge between people. A latent-variable market
  has to standardize coordinates by decree; in prose, the shared meanings of
  words do it. That is why latent claims can be communicated at all.
- **Ambiguity is not gauge.** If different readings give different profiles,
  the essay is a mixture over readings. Score it as each reader would act on
  it: $\mathbb E_r[S_\tau(e;y\mid r)]$ over interpreter samples $r$. By Jensen's
  inequality this is at most the score of the pooled reading, so ambiguity
  costs and clarity pays. That is the only reward for clarity the framework
  needs.
- **Read literally; aggregate evidentially.** The interpreter must answer "if
  this account is right, what is $P(y_q)$?". It must not answer "given that
  someone wrote this, what is $P(y_q)$?". The literal reading scores the
  content. The evidential reading (discounting authors, weighing track records)
  is what a consumer does with scored content. Mixing the two would make an
  essay's score depend on its author's reputation.
- **This is the Ramsey–Lewis account of theoretical terms.** A latent concept
  means whatever plays its role in the theory's observable consequences. The
  profile is that role, made numerical.

## 3. The scoring rule

**Fixed in advance:**
- the interpreter $R$;
- a measure $\nu$ over a bank of resolvable questions. It states which
  consequences matter, which is a value judgment, like $\tau$;
- $\tau$.

**For each essay:**
1. **Parse.** $R$ extracts the essay's latent structure: latent claims or
   scenarios $h$, with the weights $P_e(h)$ the essay states or implies. If
   claims depend on claims, this is a DAG (§5). The author sees the parse and
   can revise until it says what they meant.
2. **Sample views.** After submission, draw questions $Q_e\sim\nu$.
3. **Query links.** For each $h$ and each $q\in Q_e$, ask $R$ *separately* for
   $P_R(y_q\mid e,h)$.
4. **Resolve and pay.** The score is
   $S_\tau(e;y)=\log\sum_hP_e(h)\prod_{q\in Q_e\text{ resolved}}P_R(y_q\mid e,h)^\tau$.
   In a market, the essay is a trade against the current consensus $c$ (the
   essays already accepted, read together). It pays
   $S_\tau(c\oplus e;y)-S_\tau(c;y)$.

**Why each step:**
- **Separate link queries (step 3).** This is the central choice. Within a
  scenario, questions are scored as independent, so all dependence among
  outcomes has to be carried by the essay's latent claims. By latent-meaning
  §2, the excess log loss is exactly the dependence the essay leaves
  unexplained, $\mathrm{TC}(Y\mid h)$. The essay must name the common causes;
  the interpreter only fills in their effects. A joint query would let the
  interpreter's own world model supply the common causes, and the essay would
  be credited for latent work it did not do.
- **Views sampled after the essay (step 2).** The essay is a point in profile
  space. The score should estimate its *whole* profile, not the part the author
  aimed at. A lookup table of answers to anticipated questions gains nothing,
  while generalizing structure gains everywhere. This is held-out evaluation,
  and it does the work an explicit compression penalty would otherwise do.
- **$\tau$.**
  - At $\tau=1$, essays are valued as forecasts.
  - $\tau<1$ charges for the information carried by the scenarios
    (latent-meaning §4b). That cures *scenario proliferation*: an essay that
    hedges with one scenario per answer pattern, which is the $X=Y$ latent in
    prose.
  - As shown there, $\tau<1$ also compresses honest content. So use it as a
    tie-breaker, with $\tau$ near 1, and state the value used.
- **No length penalty.**
  - Bold claims already pay their cost: the log score charges a confident wrong
    claim heavily. This is Popper's trade-off between content and risk, with a
    price on it.
  - Generalization is enforced by step 2.
  - Reader time is a real cost, but a separate one. If it matters, add it as an
    explicit exchange rate rather than hiding it in the scoring rule.
- **Incremental payment.** Credit goes to what the essay adds, not to what the
  consensus already had. The interpreter's background knowledge cancels out.

**Guarantees.** Each is an instance of a result in latent-meaning.
- **P1. Properness relative to the interpreter.** Suppose an author believes
  $Y\sim p$. They maximize expected payment by writing the essay whose parsed
  model maximizes $\mathbb E_pS_\tau$ over the expressible models
  $\Theta_R=\{\theta_e\}$.
  - If the bottleneck-optimal latent model is expressible, that essay
    expresses it (latent-meaning §4b).
  - Otherwise it expresses the best expressible model under the same
    objective.
  - At $\tau=1$ with a single scenario, this reduces to: move the interpreter's
    per-question forecasts to your own.
- **P2. Paraphrase invariance.** Scores depend on $e$ only through
  $(P_e,\pi_e)$.
- **P3. What is unpriced.** Take two essays with the same weights and the same
  links on every question that can resolve within the support of $\nu$. They
  earn identical payments on every outcome. However interesting their
  differences are, those differences have no observable meaning under $\nu$.
  This is genuine underdetermination (latent-meaning §1). It is not a defect of
  this rule: no rule can price it.
- **P4. Credit for individual claims.**
  - After resolution, each latent claim $h$ has a posterior. Its weight of
    evidence, $\log P(y\mid h)-\log P(y\mid\neg h)$ (I. J. Good), is the
    claim's own score.
  - That score is non-zero only if the scenarios the claim separates make
    different predictions on resolved questions. This is the rank condition for
    unit-level beliefs in latent-meaning §4b.
  - If two claims only ever predict through shared consequences, the outcomes
    test their conjunction, not each claim separately (Duhem–Quine).
  - Design rule: give every latent claim several conditionally independent
    resolvable consequences. Three is the threshold at which a latent's content
    is fixed by its consequences rather than by its wording (Kruskal). In a
    latent tree, every hidden node needs three neighbours.

## 4. What counts as an observable

The rule is only as meaningful as its resolutions. There are three kinds, in
decreasing strength:

1. **Empirical:** events that happen or do not.
2. **Logical and computational:** theorems later proved, computations later
   run, simulations, replications.
   - Pure reasoning (deriving consequences without new facts) is worthless to
     a logically omniscient reader and valuable to a bounded one.
   - Scoring reasoning against logical resolutions values it relative to the
     interpreter. That is exactly right if the interpreter stands for the
     audience.
   - This is the logical-induction picture: traders are paid for correcting
     the prices of mathematical statements before those statements are
     settled.
3. **Intersubjective:** other essays, or other agents' answers, used as views.
   - When questions never resolve, essays written from different information
     are conditionally independent views of the latent they address.
     Peer-prediction mechanisms (multi-task $f$-mutual-information) score each
     essay by how much information it carries about the others.
   - The latent framework says precisely what this identifies: the *common
     cause* of the views. If the authors share a bias (the same sources, or the
     same training data), the shared bias is also a latent variable, and it
     will be rewarded.
   - The truth is identified only when it is the views' *only* common cause.
     So the views must come from causally diverse channels.

Resolving questions by future expert consensus, or by a future market price,
is the self-referential limit of the third kind.

## 5. Arguments are latent DAGs

Reasoning chains claims together: A and B make C likely, and C bears on the
observables. Parse this as a latent DAG (latent-meaning §4a):
- **Nodes:** latent claims are hidden nodes, and resolvable questions are
  observed leaves.
- **Edges:** each "because" is an edge.
- **Conditionals:** the interpreter supplies every local conditional from the
  essay's text, such as $P(C\mid A,B)$ and $P(y_q\mid C)$.

$S_\tau$ for DAGs then scores the argument as a whole. Credit flows to
intermediate claims through their consequences: Quine's web of belief, with
numbers. The identifiability conditions say which claims are tested
separately. Quantities, such as "production capacity is around $X$", are
continuous latents: their links are queried at a few values and integrated as
in latent-meaning §4c.

The interpreter is never asked to judge the argument, only to read off local
conditionals. Validity is not scored separately:
- an invalid step is a wrong conditional, and it costs log score when its
  consequences resolve;
- a step with no resolvable consequences is unpriced, whether valid or not.

## 6. Mapping to the latent-variable-market formalism

| latent-variable market | essays |
|---|---|
| latent model $\theta=(P(X),P(Y_i\mid X))$ | parsed essay $(P_e(h),\pi_e(h))$ |
| link function (logistic, probit, ...) | the interpreter reading sentences |
| indicators / views | questions sampled from $\nu$ after writing |
| gauge (relabeling, rotation) | paraphrase; shared language fixes it |
| mixing measure | weights over the essay's scenario profiles; a population of essays is a measure over profiles |
| unexplained dependence $\mathrm{TC}(Y\mid X)$ | dependence the essay's named causes do not explain (charged by separate queries) |
| $\tau$ and class proliferation | $\tau$ and scenario proliferation |
| identifiability with at least 3 views | Duhem–Quine; at least 3 independent consequences per claim |
| unit-level probe scoring | claims about this one world, tested by its resolved questions |
| market scoring rule | essays as trades against the consensus |

## 7. What can still go wrong

- **The interpreter is the semantics.** A miscalibrated interpreter is a
  misspecified link function. Authors bear its errors, and they may learn to
  write to its quirks. Mitigations:
  - test the interpreter's calibration on resolved conditional questions;
  - use an ensemble of interpreters;
  - show authors the parse and the links.

  None of these removes the dependence. Meaning is always relative to a reader.
- **Temporal hygiene.** The interpreter must not know the outcomes: its
  knowledge cutoff must precede the essay, with no retrieval past it.
- **Performativity.** Essays that change the events they predict break the
  premise that outcomes are independent of reports.
- **One world.** Most essays concern a single realization. Scores have high
  variance unless an essay has many resolved consequences, and feedback is slow
  when those consequences resolve late.
- **Incentives are myopic only.** As with any market scoring rule, an author
  may hold back information to trade on it later.
- **Cost.** It takes one interpreter call per (scenario, question) pair.
  Parsing is the expensive and error-prone step.

## 8. Building it

Some pieces already exist: [`lvpm/dag.py`](../lvpm/dag.py) and
[`lvpm/continuous.py`](../lvpm/continuous.py) compute $S_\tau$ for any parsed
structure. What to build, and the experiments that would test the proposal,
are in [latent-essays-plan.md](latent-essays-plan.md).

## References

- F. P. Ramsey, *Theories*, 1929; D. Lewis, *How to define theoretical terms*, J. Phil., 1970.
- P. Duhem, *La théorie physique*, 1906; W. V. O. Quine, *Two dogmas of empiricism*, 1951.
- K. Popper, *Logik der Forschung*, 1934 (content and risk).
- I. J. Good, *Probability and the Weighing of Evidence*, 1950.
- R. Hanson, *Combinatorial information market design*, 2003.
- S. Garrabrant, T. Benson-Tilsen, A. Critch, N. Soares, J. Taylor, *Logical induction*, 2016.
- D. Prelec, *A Bayesian truth serum for subjective data*, Science, 2004.
- A. Dasgupta, A. Ghosh, *Crowdsourced judgement elicitation with endogenous proficiency*, WWW 2013.
- Y. Kong, G. Schoenebeck, *An information theoretic framework for designing information elicitation mechanisms that reward truth-telling*, ACM TEAC, 2019.
- J. Kruskal, *Three-way arrays*, 1977; E. Allman, C. Matias, J. Rhodes, Ann. Statist., 2009.
