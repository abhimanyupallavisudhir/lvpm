You are an expert forecaster building a one-factor latent variable model (item response
theory) for a cluster of related forecasting questions, topic "world cup".

Rules: today is 2026-07-01. Do NOT use any tools (no web search, no file reading, nothing) -
answer purely from your own knowledge and judgement, reasoning as carefully as you like first.
The market probabilities are Polymarket prices at 2026-07-01 00:00 UTC; treat them as the
marginal probabilities (you are only asked about how the questions move *together*).
End your reply with a single fenced ```json block in exactly the requested format.

Questions:
[0] Will Trump be in the WC Champions Photo?
    market P(YES) = 0.50
    resolution: This market will resolve to “Yes” if Donald Trump is visibly in the frame for the winning team's official team photo following the 2026 FIFA World Cup Final. Otherwise, this market will resolve to “No”. If Trump does not attend the 2026 FIFA World Cup Final, this market will resolve to “No”. If the 2026 FIFA World Cup Final is cancelled or postponed beyond August 2, 2026, 11:59 PM ET, this market will resolve to “No”. The resolution source for th
[1] World Cup: Unbeaten Champion?
    market P(YES) = 0.97
    resolution: This market will resolve to “Yes” if the 2026 FIFA World Cup champion goes unbeaten in every match during the 2026 FIFA World Cup competition. Otherwise, this market will resolve to “No”. For the purpose of this market, “unbeaten” is defined as having not recorded a loss during any match in any stage of the 2026 FIFA World Cup. If the 2026 FIFA World Cup competition is cancelled, postponed after August 2, 2026, 11:59 PM ET, or there is no confirm
[2] Will France reach the 2026 FIFA World Cup final?
    market P(YES) = 0.46
    resolution: This market will resolve “Yes” if the listed team reaches the 2026 FIFA World Cup final. If at any point it becomes impossible for the listed nation to advance to the 2026 FIFA World Cup final (e.g. they are mathematically eliminated), the associated market will resolve to "No". If the 2026 FIFA World Cup is cancelled, postponed after August 2, 2026, 11:59 PM ET or the 2026 FIFA World Cup final matchup has not been declared within that timeframe,
[3] Will A Nation That Has Never Won the World Cup Win in 2026?
    market P(YES) = 0.23
    resolution: This market will resolve to “Yes” if the winner of the 2026 FIFA World Cup is a nation that has never previously won a FIFA World Cup. Otherwise, this market will resolve to “No”. For the purpose of this market, the nations that have previously won a FIFA World Cup are: Uruguay, Italy, Germany, Brazil, England, Argentina, France, and Spain. If the 2026 champion is any nation other than these eight, this market resolves “Yes”. If the 2026 FIFA Wor
[4] President Trump to Attend World Cup Final?
    market P(YES) = 0.88
    resolution: This market will resolve to “Yes” if Donald Trump attends the 2026 FIFA World Cup Final. Otherwise, this market will resolve to “No”. Attending the match is defined as being in physical attendance during any part of the match. If the 2026 FIFA World Cup Final is cancelled or postponed beyond August 2, 2026, 11:59 PM ET, this market will resolve to “No”. The resolution source for this market will be a consensus of credible reporting.
[5] World Cup: Goalkeeper to Score?
    market P(YES) = 0.04
    resolution: This market will resolve to “Yes” if any goalkeeper scores a goal in any match of the 2026 FIFA World Cup. Otherwise, this market will resolve to “No”. Goals scored (including penalties) during regular time, stoppage time, or extra time count. Goals scored by a goalkeeper from a penalty shootout do not count. Goals ruled as an own goal will not count. The scorer must be a goalkeeper as officially recorded by FIFA. If the 2026 FIFA World Cup is ca
[6] World Cup: Most Player Goals Record Broken?
    market P(YES) = 0.07
    resolution: Note: Current record 13 goals (Just Fontaine, France, 1958). This market will resolve “Yes” if any player scores 14 or more goals across the entire 2026 FIFA World Cup, including extra time. Otherwise, this market will resolve to “No”. Penalty shootout goals do not count toward a player's total. If the 2026 FIFA World Cup is cancelled, postponed after August 2, 2026, 11:59 PM ET, or it cannot be determined whether the record was broken within tha
[7] Will Spain reach the Semifinals at the 2026 FIFA World Cup?
    market P(YES) = 0.43
    resolution: This market will resolve “Yes” if the listed team reaches the 2026 FIFA World Cup Semifinals. If at any point it becomes impossible for the listed club to advance to the 2026 FIFA World Cup Semifinals (e.g. they are mathematically eliminated), the associated market will resolve to "No". If the 2026 FIFA World Cup is cancelled, postponed after July 25, 2026, 11:59 PM ET or the 2026 FIFA World Cup Semifinals matchup has not been declared within tha
[8] Will Messi and Ronaldo Shake Hands During World Cup?
    market P(YES) = 0.07
    resolution: This market will resolve “Yes” if Lionel Messi and Cristiano Ronaldo are observed physically shaking hands with one another at any point during the 2026 FIFA World Cup window. Otherwise, this market will resolve to “No”. The 'handshake' must occur while the listed players are on the pitch or the bench area; handshakes in the locker room or other areas will not count. This can occur before, during, or after the match. Otherwise, this market will r
[9] World Cup: Ronaldo to Score a Free Kick?
    market P(YES) = 0.07
    resolution: This market will resolve “Yes” if Cristiano Ronaldo scores a goal directly from a free kick during any match of the 2026 FIFA World Cup. Otherwise, this market will resolve to “No”. A direct free kick goal is one scored directly from the free kick and officially credited to Cristiano Ronaldo by FIFA. A goal that deflects off the wall or a defender and still enters the net counts, provided FIFA credits the goal to Ronaldo. A free kick first touche
[10] Will Australia reach the Round of 16 at the 2026 FIFA World Cup?
    market P(YES) = 0.44
    resolution: This market will resolve “Yes” if the listed team reaches the 2026 FIFA World Cup Round of 16. If at any point it becomes impossible for the listed team to advance to the 2026 FIFA World Cup Round of 16 (e.g. they are mathematically eliminated), the associated market will resolve to "No". If the 2026 FIFA World Cup is cancelled, postponed after July 17, 2026, 11:59 PM ET or the 2026 FIFA World Cup Round of 16 matchup has not been declared within 
[11] Will 7+ matches be decided by penalty shootout during the 2026 FIFA World Cup?
    market P(YES) = 0.59
    resolution: This market will resolve to “Yes” if the total number of matches decided by a penalty shootout during the 2026 FIFA World Cup equals or exceeds the listed number. Otherwise, this market will resolve to “No”. A match is counted if it proceeds to a penalty shootout to determine the result. Only knockout-stage matches can be decided by shootout. If the 2026 FIFA World Cup is shortened, truncated, or ends early for any reason, the outcome shall be de

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
  "p_hi": [<P(YES|Y=+1) for question 0..11>], "p_lo": [<P(YES|Y=-1) for question 0..11>]}
```