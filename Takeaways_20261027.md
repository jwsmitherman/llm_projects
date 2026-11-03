# Cold Start Spike (IDEN-46479): Staging PSS Run, October 27

Source: ColdStart_raw_20261027_170316.csv. Staging PSS, 35 calls, all status 200.

## Key Takeaways

| # | Takeaway | Detail | Evidence (raw CSV rows) |
|---|---|---|---|
| 1 | The delay is real and large | Calls reached 6.85 seconds, close to the roughly 8 seconds NovaSonic reported. The slowness reproduces in staging without NovaSonic's Lambda or the external Apigee endpoint. | Row 10: 6,851 ms; Row 5: 6,270 ms |
| 2 | Time is spent waiting on PSS or OpenSearch | In every slow call, almost all of the time is waiting for PSS to respond. The ICAM OAuth token took 30 to 130 ms and connection setup about 10 to 15 ms, far below the 2 seconds estimated for authentication. | ttfb_ms vs auth_total_ms and conn_ms, all rows |
| 3 | Slowness came early, not after idle time | All slow calls happened in the first 7 minutes or so of the run (17:03 to 17:10). After that, every call took under 0.75 seconds, including after a 120 second idle gap. | Rows 2 to 10 slow; rows 11 to 36 under 0.75 s |
| 4 | The 30 second OpenSearch idle setting does not explain the pattern | The slowest call came one second after another search, with no idle time before it. In the fast period, calls after idle gaps were only about 0.15 seconds slower, whether the gap was 5 or 120 seconds, with no change at the 30 second mark. | Row 10 (warm call); cold vs warm, rows 11 to 36 |
| 5 | Most likely cause is a warm-up after a quiet period | Possible sources are PSS pods or the JVM starting up, caches filling, or OpenSearch loading data into memory. Load from other traffic on staging is the other possibility. This run cannot tell these apart. | Pattern across rows 2 to 36 |
| 6 | Limits of this run | Only the PSS layer was measured; OpenSearch on its own and Apigee were not. The run stopped after 8 of 12 planned trials; the 20 second gap and the second repeat of the 120 and 300 second gaps did not run. | trial_id column |

## Next Steps

| # | Next step | Purpose | Who | Status |
|---|---|---|---|---|
| 1 | Add the staging OpenSearch `_search` URL and token to the script and re-run | Split PSS time from OpenSearch time; show whether slow calls triggered refreshes or overlapped other traffic | Josh | Open |
| 2 | Run first thing in the morning after staging is idle overnight | Confirm the warm-up pattern and fill in the missing trials | Josh | Open |
| 3 | Confirm whether Gatling or other load ran against staging 17:03 to 17:10 on Oct 27 | Rule out other traffic as the cause of the early slowness | Arvind | Open |
| 4 | Check whether PSS pods scale down or restart when idle; pull PSS logs for 17:03 to 17:10 on Oct 27 | See whether a pod start or scale-up lines up with the slow calls | PSS infrastructure owner | Open |
| 5 | Add the internal Apigee layer once credentials arrive | Answer the ticket question on whether Apigee has its own cold start | Josh, Arvind | Blocked on credentials |

## Slowest Calls

| CSV row | Call | Search wait (ms) | Search wait (s) | Token call (ms) | Time on Oct 27 |
|---|---|---|---|---|---|
| 10 | Warm 1 after 300 s gap | 6,851 | 6.85 | 131 | 17:10:03 |
| 5 | Cold after 60 s gap | 6,270 | 6.27 | 108 | 17:04:43 |
| 9 | Cold after 300 s gap | 3,785 | 3.78 | 106 | 17:09:58 |
| 2 | Startup call 1 | 3,185 | 3.19 | 31 | 17:03:36 |
| 3 | Startup call 2 | 3,064 | 3.06 | 32 | 17:03:39 |
| 7 | Warm 2 after 60 s gap | 2,803 | 2.80 | 103 | 17:04:53 |
| 6 | Warm 1 after 60 s gap | 1,483 | 1.48 | 36 | 17:04:50 |
| 4 | Startup call 3 | 953 | 0.95 | 33 | 17:03:42 |

Search wait is the `ttfb_ms` column; token call is `auth_total_ms`.
