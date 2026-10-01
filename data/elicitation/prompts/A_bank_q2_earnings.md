You are an expert forecaster building a one-factor latent variable model (item response
theory) for a cluster of related forecasting questions, topic "bank q2 earnings".

Rules: today is 2026-07-01. Do NOT use any tools (no web search, no file reading, nothing) -
answer purely from your own knowledge and judgement, reasoning as carefully as you like first.
The market probabilities are Polymarket prices at 2026-07-01 00:00 UTC; treat them as the
marginal probabilities (you are only asked about how the questions move *together*).
End your reply with a single fenced ```json block in exactly the requested format.

Questions:
[0] Will Goldman Sachs (GS) Q2 investment banking fees be above $2.85B?
    market P(YES) = 0.58
    resolution: This market will resolve to "Yes" if Goldman Sachs's investment banking fees for the upcoming second fiscal quarter, as reported in its official company earnings materials, is above the listed amount. Otherwise, this market will resolve to "No". The specified metric will be considered as reported in the company's official earnings materials. Subsequent revisions will not be considered. If the specified company's official earnings materials for th
[1] Will Morgan Stanley (MS) Q2 investment banking revenue be above $2.125B?
    market P(YES) = 0.55
    resolution: This market will resolve to "Yes" if Morgan Stanley's investment banking revenue for the upcoming second fiscal quarter, as reported in its official company earnings materials, is above the listed amount. Otherwise, this market will resolve to "No". The company trades under the ticker MS as of the creation of this market. The specified metric will be considered as reported in the company's official earnings materials. Subsequent revisions will no
[2] Will JP Morgan (JPM) Q2 investment banking fees be above $3.15B?
    market P(YES) = 0.47
    resolution: This market will resolve to "Yes" if JP Morgan's investment banking fees for the upcoming second fiscal quarter, as reported in its official company earnings materials, is above the listed amount. Otherwise, this market will resolve to "No". The specified metric will be considered as reported in the company's official earnings materials. Subsequent revisions will not be considered. If the specified company's official earnings materials for the sp
[3] Will Bank of America (BAC) Q2 provision for credit losses be above $1.6B?
    market P(YES) = 0.56
    resolution: This market will resolve to "Yes" if Bank of America's provision for credit losses for the upcoming second fiscal quarter, as reported in its official company earnings materials, is above the listed amount. Otherwise, this market will resolve to "No". The specified metric will be considered as reported in the company's official earnings materials. Subsequent revisions will not be considered. If the specified company's official earnings materials 
[4] Will Johnson & Johnson (JNJ) Q2 Innovation Medicine revenue be above $16.8B?
    market P(YES) = 0.10
    resolution: This market will resolve to "Yes" if Johnson & Johnson's Innovative Medicine revenue for the upcoming second fiscal quarter, as reported in its official company earnings materials, is above the listed amount. Otherwise, this market will resolve to "No". The specified metric will be considered as reported in the company's official earnings materials. Subsequent revisions will not be considered. If the specified company's official earnings material
[5] Will Citigroup (C) Q2 provision for credit losses be above $2.7B?
    market P(YES) = 0.55
    resolution: This market will resolve to "Yes" if Citigroup's provision for credit losses for the upcoming second fiscal quarter, as reported in its official company earnings materials, is above the listed amount. Otherwise, this market will resolve to "No". The company trades under the ticker C as of the creation of this market. The specified metric will be considered as reported in the company's official earnings materials. Subsequent revisions will not be 

Task:
1. Identify the single latent variable Y that best explains why these questions' outcomes are
   correlated (the common cause / "how things go" dimension). Give it a short label and say
   which direction is HIGH. Think of Y as standard normal across the scenarios you consider
   plausible as of today: Y = +1 is a world 1 sd towards HIGH, Y = -1 is 1 sd towards LOW.
2. For every question give P(YES | Y = +1) and P(YES | Y = -1). They should average to roughly
   the market probability. A question unrelated to Y gets equal values; one that is more
   likely when Y is LOW gets p_hi < p_lo. Stronger dependence = bigger gap.

Format:
```json
{"label": "<short name of Y>", "high_means": "<what Y = +1 looks like>", "low_means": "<what Y = -1 looks like>",
  "p_hi": [<P(YES|Y=+1) for question 0..5>], "p_lo": [<P(YES|Y=-1) for question 0..5>]}
```