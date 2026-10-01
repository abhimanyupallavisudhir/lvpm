You are an expert forecaster building a one-factor latent variable model (item response
theory) for a cluster of related forecasting questions, topic "iran israel".

Rules: today is 2026-07-01. Do NOT use any tools (no web search, no file reading, nothing) -
answer purely from your own knowledge and judgement, reasoning as carefully as you like first.
The market probabilities are Polymarket prices at 2026-07-01 00:00 UTC; treat them as the
marginal probabilities (you are only asked about how the questions move *together*).
End your reply with a single fenced ```json block in exactly the requested format.

Questions:
[0] Strait of Hormuz traffic returns to normal by July 31?
    market P(YES) = 0.38
    resolution: This market will resolve to “Yes” if IMF Portwatch publishes a 7-day moving average of transit calls (“Arrivals of Ships”) for the Strait of Hormuz equal to or above 60 for any date between market creation and July 31, 2026. Otherwise, this market will resolve to “No”. Daily transit calls include container, dry bulk, roll-on/roll-off, general cargo, and tanker ships. Ships not reported by IMF Portwatch will not be considered. This market will res
[1] Strait of Hormuz traffic returns to normal by July 15?
    market P(YES) = 0.15
    resolution: This market will resolve to “Yes” if IMF Portwatch publishes a 7-day moving average of transit calls (“Arrivals of Ships”) for the Strait of Hormuz equal to or above 60 for any date between market creation and July 15, 2026. Otherwise, this market will resolve to “No”. Daily transit calls include container, dry bulk, roll-on/roll-off, general cargo, and tanker ships. Ships not reported by IMF Portwatch will not be considered. This market will res
[2] Kharg Island no longer under Iranian control by August 31?
    market P(YES) = 0.04
    resolution: This market will resolve to "Yes" if Kharg Island is no longer under Iranian control by August 31, 2026, 11:59 PM ET. Otherwise, this market will resolve to "No". “No longer under the control of Iran” means that Iran no longer exercises primary governmental or military control over Kharg Island, and another state, occupying force, or internationally backed authority has established control over the island. Temporary raids, isolated landings, spec
[3] Israel closes its airspace by July 31?
    market P(YES) = 0.12
    resolution: This market will resolve to “Yes” if Israel initiates a major closure of its airspace by the specified date, 11:59 PM ET. Otherwise, this market will resolve to “No”. A “major closure” is defined as a broad closure, cancellation, or complete suspension of commercial aviation across the entirety of Israeli civilian airspace or a region encompassing a majority of Israeli civilian airspace, including commercial flights transiting, arriving in, and d
[4] Israel withdraws from Lebanon by July 31, 2026?
    market P(YES) = 0.02
    resolution: This market will resolve to "Yes" if Israel announces it has withdrawn all ground forces from Lebanon by the specified date, 11:59 PM ET. Otherwise, this market will resolve to "No". For this market to resolve to "Yes" it is sufficient that Israel announces its ground forces have withdrawn from all Lebanese territory, regardless of if some Lebanese territory remains under their control or ground incursions by Israeli forces continue. However, an 
[5] Mojtaba Khamenei seen in public by July 31?
    market P(YES) = 0.23
    resolution: This market will resolve to "Yes" if Supreme Leader of Iran, Mojtaba Khamenei, is photographed or videotaped between market creation and the listed date (ET). Otherwise, this market will resolve to "No". For this market to resolve to "Yes," the images or video must be taken and released within this market's timeframe. The images or video must be authentic. Archival material, previously recorded footage, reused images, digitally altered content, A
[6] Israel x Hezbollah permanent peace deal by July 31, 2026?
    market P(YES) = 0.04
    resolution: This market will resolve to “Yes” if Israel and Hezbollah agree to a permanent peace deal by the specified date, 11:59 PM ET. Otherwise, this market will resolve to “No”. A permanent peace deal refers to any agreement which explicitly indicates that military hostilities between Israel and Hezbollah have ended or will permanently cease, or uses equivalent language clearly signaling a lasting end to military hostilities between Israel and Hezbollah
[7] Iran agrees to surrender enriched uranium stockpile by July 31, 2026?
    market P(YES) = 0.03
    resolution: This market will resolve to "Yes" if Iran publicly agrees to surrender its enriched uranium stockpile by July 31, 2026, 11:59 PM ET. Otherwise, this market will resolve to “No”. An official pledge by Iran to surrender its enriched uranium stockpile will qualify for a “Yes” resolution whether as a unilateral announcement or part of an agreement with the U.S. or Israel. An agreement by Iran to surrender any amount of its enriched uranium stockpile 
[8] Iran agrees to end enrichment of uranium by July 31?
    market P(YES) = 0.04
    resolution: This market will resolve to "Yes" if Iran publicly agrees to end all enrichment of uranium by July 31, 2026, 11:59 PM ET. Otherwise, this market will resolve to “No”. An official pledge by Iran to end all enrichment of Uranium will qualify for a “Yes” resolution, whether as a unilateral announcement or part of an agreement with the U.S. or Israel. Any agreement or pledge made before the resolution date of this market will qualify, regardless of i
[9] Israel x Iran permanent peace deal by July 31, 2026?
    market P(YES) = 0.04
    resolution: This market will resolve to “Yes” if Israel and Iran agree to a permanent peace deal by the specified date, 11:59 PM ET. Otherwise, this market will resolve to “No”. A permanent peace deal refers to any agreement which explicitly indicates that military hostilities between Israel and Iran have ended or will permanently cease, or uses equivalent language clearly signaling a lasting end to military hostilities between Israel and Iran. Agreements th
[10] US-Iran nuclear deal by July 31?
    market P(YES) = 0.84
    resolution: This market will resolve to "Yes" if an official agreement over Iranian nuclear research and/or nuclear weapon development, defined as a publicly announced mutual agreement, is reached between the United States and Iran by July 31, 2026, 11:59 PM ET. Otherwise, this market will resolve to “No”. If such an agreement is officially reached before the resolution date, this market will resolve to "Yes", regardless of if/when the agreement goes into ef
[11] Israeli parliament dissolved by July 31?
    market P(YES) = 0.61
    resolution: This market will resolve to "Yes" if the sitting Israeli Knesset (Israel's parliament), as of this market's inception, is dissolved between January 27 and July 31, 2026, 11:59 PM ET. Otherwise, this market will resolve to "No". The primary resolution source for this market is official information from the government of Israel; however, a consensus of credible reporting will also be used.

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