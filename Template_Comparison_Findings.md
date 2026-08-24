# Template comparison findings

Results from 4,252 production searches run against default, max-clause-test and reduced-tiers.

## Multiple identifiers

- 161 searches carry both an A-number and a receipt number, 83 in first-a and 78 in first-b
- Only FIRST sends them, about 9.5% of their traffic
- CRIS and UIPATH send receipt only, GLOBAL and BHUB send A-number only
- 314 searches carry no identifier at all, 239 from FIRST and 75 from BHUB

## Reduced-tiers

- Returns 99.3 identities per search against 99.9 for default
- Surfaces 186,000 distinct identities against 185,000 for default
- 95.5% of results overlap with default in staging, and the top hit never changed
- No search returned fewer than 10 identities

## Max-clause-test

- Returns 92.1 identities per search, close to the other two
- But only 70,663 distinct identities across the whole run against roughly 186,000, about a third
- It returns similar counts each time, but keeps returning the same people
- 82 searches returned a single identity, 237 returned fewer than 10, and 4 returned none

## Where max-clause-test breaks down

- Searches with a name and nothing else, no date of birth and no identifier
- On first and last name it returns about 6 identities where the others return 100
- On first, middle and last it returns 1
- On first, middle, last and date of birth it returns none
- Anywhere a date of birth or an identifier is present it performs the same as the other two
- This matches the edge cases in the PDF, and those searches are about 7% of the sample and mostly FIRST

## What production traffic looks like

- Receipt number alone is the most common search, 1,939, about 46%
- A-number alone, 692, about 16%
- Name plus date of birth plus A-number plus country of birth, 560, about 13%
- Overall, 62% of searches carry an identifier and no name

## Caveats

- Staging had 3,024 failed calls and completed only 1,228 of the 4,252 searches, and it caps results at 10, so treat staging figures as indicative
- Production shows about 12% of searches where the top hit differs between templates, but staging shows none for the same pair, so this looks like measurement rather than the templates. Still being checked

## Bottom line

- Reduced-tiers looks safe to ship on this evidence
- Max-clause-test is identical to the others wherever a date of birth or identifier is present, which is nearly all real traffic. The distinct-identity gap and the name-only behaviour are the two things worth understanding before it goes out
- The first template only behaves differently for those 161 searches from one consumer, so it is a small and contained question
