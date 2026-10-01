You are an expert forecaster building a one-factor latent variable model (item response
theory) for a cluster of related forecasting questions, topic "us politics".

Rules: today is 2026-07-01. Do NOT use any tools (no web search, no file reading, nothing) -
answer purely from your own knowledge and judgement, reasoning as carefully as you like first.
The market probabilities are Polymarket prices at 2026-07-01 00:00 UTC; treat them as the
marginal probabilities (you are only asked about how the questions move *together*).
End your reply with a single fenced ```json block in exactly the requested format.

Questions:
[0] U.S. anti-cartel operation outside of the U.S. by July 31?
    market P(YES) = 0.51
    resolution: This market will resolve to "Yes" if U.S. government personnel (military, DEA, CIA, or any other agency) directly participate on the ground in an anti-cartel operation or conduct a kinetic strike directed against a cartel on foreign soil between market creation and the specified date, 11:59 PM ET. Otherwise, this market will resolve to “No”. U.S. personnel must directly participate to qualify. U.S. personnel involved in intelligence, surveillance
[1] SCOTUS accepts sports event contract case by July 31, 2026?
    market P(YES) = 0.04
    resolution: This market will resolve to "Yes" if the Supreme Court of the United States grants certiorari in a case explicitly concerning the legality, regulation, or jurisdictional authority over sports event contracts by July 31, 2026, 11:59 PM ET. Otherwise, this market will resolve to "No." A case qualifies if it addresses at least one of the following: (1) whether contracts based on sporting event outcomes constitute regulated derivatives under the Comm
[2] SCOTUS bars counting mail ballots after election day?
    market P(YES) = 0.72
    resolution: This market will resolve to “Yes” if the United States Supreme Court, in Watson v. Republican National Committee, rules that the federal election-day statutes preempt a state law that allows ballots cast by federal Election Day to be received and counted after election day, by August 1, 2026, 11:59 PM ET. Otherwise, this market will resolve to “No”. This market will resolve based on the Supreme Court’s decision in this case. Other related cases o
[3] SCOTUS strikes down Trump's Birthright Citizenship EO?
    market P(YES) = 0.95
    resolution: The Supreme Court is hearing arguments on the merits of Donald Trump’s Executive Order “Protecting the Meaning and Value of American Citizenship” (https://www.whitehouse.gov/presidential-actions/2025/01/protecting-the-meaning-and-value-of-american-citizenship/) which directed that children born in the United States would not automatically receive U.S. citizenship if their parents were in the country unlawfully or on certain temporary visas. This 
[4] Pete Hegseth out as Secretary of Defense by July 31?
    market P(YES) = 0.04
    resolution: This market will resolve to “Yes” if Pete Hegseth ceases to be U.S. Secretary of Defense for any period of time between market creation and the specified date, 11:59 PM ET. Otherwise, this market will resolve to “No”. An announcement of Pete Hegseth's resignation/removal by this market's end date will immediately resolve this market to "Yes", regardless of when the announced resignation/removal goes into effect. The resolution source for this mar
[5] US x Cuba economic deal by July 31, 2026?
    market P(YES) = 0.10
    resolution: This market will resolve to "Yes" if an official agreement over trade, tariffs, sanctions, or the US embargo on Cuba, defined as a publicly announced mutual agreement, is reached between the United States and Cuba by the specified date, 11:59 PM ET. Otherwise, this market will resolve to “No”. A qualifying agreement may include, but is not limited to, US sanctions relief for Cuba or other easing of U.S. restrictions on Cuban imports, exports, shi
[6] Kristi Noem divorce by August 31?
    market P(YES) = 0.17
    resolution: This market will resolve to "Yes" if Kristi Noem and/or her spouse announce their intention to divorce by August 31, 2026, 11:59 PM ET. Otherwise, this market will resolve to "No". An announcement by the specified date of their intention to divorce will be sufficient for a "Yes" resolution, regardless of whether the divorce later actually occurs, or whether the actual divorce occurs outside of the timeframe of this market. The resolution source w
[7] Will Brent Hennrich advance to the general election for WA-03?
    market P(YES) = 0.08
    resolution: The non-partisan primary election for the WA-03 congressional district seat in the U.S. House of Representatives is scheduled to take place on August 4, 2026. The two candidates who receive the most votes in this primary election will qualify for the General Election. This market will resolve to “Yes” If the listed candidate advances from the primary to the general election for the WA-03 congressional district seat in the U.S. House of Representa
[8] US x Cuba diplomatic meeting by July 31?
    market P(YES) = 0.64
    resolution: This market will resolve to "Yes" if there is a diplomatic meeting between representatives of the United States and Cuba between market creation and the listed date, 11:59 PM ET. Otherwise, this market will resolve to “No”. A diplomatic meeting refers to a deliberate meeting between representatives of the listed countries who are acting in an official capacity and are authorized to engage in negotiation or diplomacy regarding US-Cuba relations on
[9] Will Jonathan Kreiss-Tomkins advance from the 2026 Alaska Governor primary election?
    market P(YES) = 0.47
    resolution: The non-partisan primary election for Governor of Alaska is scheduled to take place on August 18, 2026. The top four candidates in this election by number of votes won will advance to the general election for Governor of Alaska. This market will resolve to “Yes” If the listed candidate advances from the primary to the general election for Governor of Alaska. Otherwise this market will resolve to “No”. If no 2026 Alaska gubernatorial primary takes
[10] U.S. issues passport with Trump's face on it by July 31?
    market P(YES) = 0.82
    resolution: This market will resolve to “Yes” if any United States passports containing images or representations of Donald Trump’s face are issued by the United States government by July 31, 2026, 11:59 PM ET. Otherwise, this market will resolve to “No”. The passports must actually be issued to U.S. citizens. Proposals, approvals, announcements that qualifying passports will be issued, or unreleased sample passports will not qualify. The primary resolution 
[11] James Comey charges dropped by July 31?
    market P(YES) = 0.10
    resolution: Former FBI director James Comey was indicted on April 28 (see: https://edition.cnn.com/2026/04/28/politics/justice-department-indicts-ex-fbi-director-james-comey-again). This market will resolve to "Yes" if all charges against James Comey stemming from this indictment are officially dropped, dismissed, withdrawn, or reduced to a non-felony charge by July 31, 2026, 11:59 PM ET. Otherwise, this market will resolve to "No". The primary resolution so

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