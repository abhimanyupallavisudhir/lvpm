You are an expert forecaster estimating how a cluster of related forecasting questions
(topic "world politics") depend on each other.

Rules: today is 2026-07-01. Do NOT use any tools (no web search, no file reading, nothing) -
answer purely from your own knowledge and judgement, reasoning as carefully as you like first.
The market probabilities are Polymarket prices at 2026-07-01 00:00 UTC; treat them as the
marginal probabilities (you are only asked about how the questions move *together*).
End your reply with a single fenced ```json block in exactly the requested format.

Questions:
[0] Will AfD win an absolute majority of seats in Sachsen-Anhalt?
    market P(YES) = 0.41
    resolution: Parliamentary elections to elect the Landtag of Sachsen-Anhalt are scheduled to take place on September 6, 2026. This market will resolve to "Yes" if the party Alternative for Germany (AfD) wins an absolute majority of seats in the state parliament of Sachsen-Anhalt (Landtag) as a result of this election. Otherwise, this market will resolve to "No". If voting in the Sachsen-Anhalt election for the Landtag does not occur or the results are not kno
[1] Starmer out by July 31, 2026?
    market P(YES) = 0.95
    resolution: This market will resolve to “Yes” if Keir Starmer ceases to be the Prime Minister of the United Kingdom for any period of time between market creation and July 31, 2026, 11:59 PM ET. Otherwise, this market will resolve to “No”. An announcement of Keir Starmer's resignation/removal before this market's end date will immediately resolve this market to "Yes", regardless of when the announced resignation/removal goes into effect. The resolution sourc
[2] Will the Workers' Party nominate Lula for President of Brazil by August 15?
    market P(YES) = 0.93
    resolution: The 2026 Brazil presidential election is currently scheduled to be held on October 4, 2026. Political parties must file candidate registration applications with the Superior Electoral Court (Tribunal Superior Eleitoral, TSE) for the 2026 Brazil presidential election by August 15, 2026, 7:00 PM Brasília time (UTC−03:00). This market will resolve to "Yes" if the Workers’ Party (Partido dos Trabalhadores, PT) files a candidate registration applicati
[3] Romanian parliament dissolved by July 31?
    market P(YES) = 0.04
    resolution: This market will resolve to "Yes" if the sitting Romanian Chamber of Deputies and Senate are dissolved by the specified date, 11:59 PM ET. Otherwise, this market will resolve to "No". The primary resolution source for this market is official information from the government of Romania, however a consensus of credible reporting will also be used.
[4] Will Marco Rubio attend NATO Summit?
    market P(YES) = 0.94
    resolution: This market will resolve to "Yes" if Marco Rubio attends the NATO Summit event scheduled for July 7, 2026 – July 8, 2026 in Ankara, Türkiye. Otherwise, this market will resolve to "No". If the event is canceled or postponed beyond August 30, 2026, 11:59 PM ET, this market will resolve to "No". Attending the NATO Summit is defined as being in physical attendance in the event at any point between the start and end of the Summit. The resolution sour
[5] Will Tidö parties win a majority in the 2026 Swedish parliamentary elections?
    market P(YES) = 0.09
    resolution: In October 2022, the Swedish political parties Sweden Democrats (SD), Moderate Party (M), Christian Democrats (KD), and Liberals (L) signed the Tidö Agreement (Tidöavtalet) to form a government after the 2022 Swedish parliamentary elections. Parliamentary elections are scheduled to be held in Sweden on September 13, 2026. This market will resolve to “Yes” if the member parties of the Tidö Agreement (SD, M, KD, and L) win a combined total of 175 s
[6] Italy Senate approves Nuclear Power Bill by August 31?
    market P(YES) = 0.36
    resolution: On June 4, 2026, the Italian Chamber of Deputies approved the nuclear power enabling law with 155 votes in favor, 8 abstentions, and 86 against. The bill now goes to the Senate for final approval. This market will resolve to "Yes" if the Italian Senate finally approves the Nuclear Power Delegation Bill by August 31, 2026, 11:59 PM CET. Otherwise, this market will resolve to "No". The primary resolution source for this market will be information f
[7] Haiti elections delayed again?
    market P(YES) = 0.69
    resolution: After earlier delays, the first round of the 2026 Haitian legislative and presidential elections is currently scheduled to be held on August 30, 2026, with a potential second round set for December 6, 2026. This market will resolve to "Yes" if the first round of voting for the presidency or the national legislative bodies, including the Senate (Sénat) and the Chamber of Deputies (Chambre des Députés), is rescheduled to take place after August 30,

Task: for every pair (i, j) with i < j, give P(question j resolves YES | question i resolves YES)
and P(question j resolves YES | question i resolves NO). If you think i and j are unrelated,
both equal the market probability of j. Consider what a world in which i resolves YES (or NO)
looks like and how j fares in it. There are 28 pairs.

Format (one entry per pair, all pairs, i < j):
```json
{"pairs": [[i, j, P(j YES | i YES), P(j YES | i NO)], ...]}
```