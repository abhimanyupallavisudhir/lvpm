You are an expert forecaster building a one-factor latent variable model (item response
theory) for a cluster of related forecasting questions, topic "russia ukraine diplomacy".

Rules: today is 2026-07-01. Do NOT use any tools (no web search, no file reading, nothing) -
answer purely from your own knowledge and judgement, reasoning as carefully as you like first.
The market probabilities are Polymarket prices at 2026-07-01 00:00 UTC; treat them as the
marginal probabilities (you are only asked about how the questions move *together*).
End your reply with a single fenced ```json block in exactly the requested format.

Questions:
[0] Russia x Ukraine ceasefire agreement by August 31, 2026?
    market P(YES) = 0.12
    resolution: This market will resolve to “Yes” if there is a ceasefire agreement between Russia and Ukraine by the specified date, 11:59 PM ET. Otherwise, this market will resolve to “No”. A ceasefire agreement refers to any mutually-agreed suspension of direct military engagement between Russia and Ukraine, which is either officially announced by both countries or confirmed by a consensus of credible reporting to have been mutually agreed by both countries. 
[1] Ukraine signs peace deal with Russia by August 31?
    market P(YES) = 0.07
    resolution: This market will resolve to “Yes” if, Ukraine signs any written instrument (e.g., treaty, ceasefire/armistice, framework/“roadmap,” exchange of letters, or mediated agreement text) that: (i) includes both Ukraine and the Russian Federation as parties, and (ii) either ends hostilities/establishes a ceasefire or commits both sides to a defined process toward ending the war (i.e., stated objective of peace/normalization plus principles, steps, and/o
[2] Ukraine election called by August 31, 2026?
    market P(YES) = 0.05
    resolution: This market will resolve to "Yes" if the next Ukrainian presidential election is scheduled by August 31, 2026, 11:59 PM ET. Otherwise, this market will resolve to "No". This market is about whether a date for the next Ukrainian election is announced between market creation and the specified date, 11:59 PM ET. Whether the election is supposed to take place during this timeframe or later will have no effect on the resolution of this market. The pri
[3] Will Putin meet with Zelenskyy by August 31, 2026?
    market P(YES) = 0.06
    resolution: This market will resolve to "Yes" if Vladimir Putin meets with Volodymyr Zelenskyy between this market's creation and the specified date, 11:59 PM ET. Otherwise, this market will resolve to "No". An exchange of words, handshake, direct conversation, or other clear personal interaction between the named individuals will qualify as a meeting. Merely standing in proximity, making eye contact, or being present in the same room or event without direct
[4] Putin and Zelenskyy shake hands by August 31?
    market P(YES) = 0.02
    resolution: This market will resolve to "Yes" if Vladimir Putin and Volodymyr Zelenskyy shake hands by August 31, 2026, 11:59 PM ET. Otherwise, this market will resolve to "No". For the handshake to qualify toward a "Yes" resolution, there must be video or photographic evidence. The resolution source for this market will be photographic or digital evidence verified by a consensus of credible reporting.
[5] Will Zelenskyy talk to Putin by August 31?
    market P(YES) = 0.11
    resolution: This market will resolve to "Yes" if Volodymyr Zelenskyy and Vladimir Putin talk by August 31, 2026, 11:59 PM ET. Otherwise, it will resolve to "No". A talk is defined as any interaction between Volodymyr Zelenskyy and Vladimir Putin, occurring either in person or through verbal communication by phone or video call. Events that occurred prior to the start of this market will not qualify toward its resolution. The resolution source will be a conse
[6] Will Ukraine agree to cede territory to Russia by August 31, 2026?
    market P(YES) = 0.06
    resolution: This market will resolve to "Yes" if a deal in which Ukraine agrees to cede any territory under its control at the time of the agreement is reached between the Russian Federation and Ukraine by August 31, 2026, 11:59 PM ET. Otherwise, this market will resolve to “No”. The deal must be a publicly announced mutual agreement or otherwise confirmed by an overwhelming consensus of credible reporting. If such an agreement is officially reached before t
[7] Ukraine agrees not to join NATO by August 31?
    market P(YES) = 0.11
    resolution: This market will resolve to "Yes" if Ukraine publicly agrees not to join NATO by August 31, 2026, 11:59 PM ET. Otherwise, this market will resolve to “No”. An official pledge by Ukraine not to join NATO will qualify for a “Yes” resolution whether as a unilateral announcement or part of an agreement with the Russian Federation. Any agreement or pledge made before the resolution date of this market will qualify, regardless of if/when the agreement 
[8] Russia x Ukraine diplomatic meeting by September 30, 2026?
    market P(YES) = 0.80
    resolution: This market will resolve to "Yes" if there is a diplomatic meeting between representatives of Russia and Ukraine by the listed date, 11:59 PM ET. Otherwise, this market will resolve to “No”. A diplomatic meeting refers to a deliberate meeting between representatives of the listed countries who are acting in an official capacity and are authorized to engage in negotiation or diplomacy regarding Russia-Ukraine relations on behalf of their governmen

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
  "p_hi": [<P(YES|Y=+1) for question 0..8>], "p_lo": [<P(YES|Y=-1) for question 0..8>]}
```