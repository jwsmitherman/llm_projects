# OpenSearch Snapshot: Production → Stage (Run Date 2026-11-04)

Step-by-step checklist based on the Confluence runbook *"Runbook: Opensearch Snapshot Production Data to Stage"*, plus the stage cleanup steps Paul walked through.
Use the date **`20261104`** everywhere. If you finish after midnight, change it to the new date.

> **Before you start**
> - **The OpenSearch Admin login can break production.** On the prod clusters, run only the exact `GET` and `PUT _snapshot` commands in this document.
> - Do the snapshot work (Parts 3–8) **near the end of the workday or after hours**, because the load can cause errors for other users.
> - **STOP** wherever a step says so, and check with Paul.

---

## Part 1: Get logged in

### 1.1 Get the prod admin password from Vault
1. In Vault, click your user icon (top right) and choose **Sign out**. Vault only allows a session in prod or nonprod, not both.
2. Open the Confluence runbook and click the **Vault credentials** link in the Quick Links table.
3. Sign in. Check that the namespace dropdown at the top left shows **PCIS_prod**.
4. Scroll down to **OpenSearch Admin**.
5. Note the username. Copy the password with the copy icon rather than revealing it on screen.

### 1.2 Open four browser tabs and log in to each one with the OpenSearch Admin credentials

| Tab | Cluster | URL |
|---|---|---|
| A | Partyrecord Prod | `https://opensearch-partyrecord-prod.pcis.uscis.dhs.gov/_dashboards` |
| B | Partyrecord Stage | `https://opensearch-partyrecord-staging.pcis.uscis.dhs.gov/_dashboards` |
| C | Identity Prod | `https://opensearch-identity-prod.pcis.uscis.dhs.gov/_dashboards` |
| D | Identity Stage | `https://opensearch-identity-staging.pcis.uscis.dhs.gov/_dashboards` |

When the tenant popup appears, choose **Private** and click **Confirm**.

### 1.3 Open Dev Tools in every tab
- Click **Dev tools** at the top right of the home page and dismiss any popup.
- Click in the left pane, press **Cmd+A**, then **Delete** to clear it.
- **To run a command:** click on its line and press **Cmd+Enter**. The result appears on the right.

### 1.4 Notepad template (fill this in as you go)
```
PROD INDEXES (Part 3)
  PR  aim-dedup-parties           ->
  PR  iis-history-api-alias       ->
  PR  dedup-party-match           ->
  PR  irq-index-alias             ->
  ID  iis-identity-api-alias      ->
  ID  iis-identity-consumer-alias ->

STAGE OLD INDEXES (Part 6)
  PR  aim-dedup-parties           ->
  PR  iis-history-api-alias       ->
  PR  dedup-party-match           ->
  PR  irq-index-alias             ->
  ID  iis-identity-api-alias      ->
  ID  iis-identity-consumer-alias ->

RESTORED DOC COUNTS (Part 5.4)
  ->
```

---

## Part 2: Free up disk space on stage (can be done in the morning)

### 2.1 Check free space in AWS
1. Open the AWS console and go to **OpenSearch Service → Dashboard**.
2. Find the rows `opensearch-partyrecord-stage` and `opensearch-identity-stage`.
3. Scroll right to the **minimum free storage** column.
4. Rule: each stage cluster needs **at least 1 TB** in that column.
   - Partyrecord stage showed **901 GB**, so it needs cleanup. Do steps 2.2–2.5 in tab B.
   - If identity stage also shows less than 1 TB, do steps 2.2–2.5 in tab D as well.

### 2.2 List the indexes (tab B)
```
GET _cat/indices?v&s=i
```
Find the two indexes whose names end in **`_restored-prod-20260807`** and copy their exact names.

### 2.3 Confirm nothing points to them (tab B)
```
GET _cat/aliases?v&s=alias
```
Read down the `index` column. If any row lists a name ending in `20260807`, **STOP** and message Paul.

### 2.4 Delete them, one at a time (tab B)
```
DELETE <exact index name #1>
```
```
DELETE <exact index name #2>
```
Each one should return `{"acknowledged": true}`.
- **Never use `*` in a DELETE command.**
- **Do not delete** anything ending in `20260914` (the aliases point to these) or `20260918` (Paul is using these).

### 2.5 Recheck the space
Refresh the AWS dashboard. The partyrecord-stage minimum should now be well over 1 TB.

---

## Part 3: Record the prod indexes (end of day)

### 3.1 Tab A: Partyrecord Prod
```
GET _alias/aim-dedup-parties
GET _alias/iis-history-api-alias
GET _alias/dedup-party-match
GET _alias/irq-index-alias
```
The index name is the first key in each response: `{ "<index name>": { "aliases": ... } }`. Write each one in your notepad.

### 3.2 Tab C: Identity Prod
```
GET _alias/iis-identity-api-alias
GET _alias/iis-identity-consumer-alias
```

---

## Part 4: Create the snapshots (tab A, then tab C immediately after)

### 4.1 Tab A: Partyrecord Prod
```
PUT _snapshot/prod-partyrecord-IDEN-28978-repo/prod-partyrecord-20261104-snapshot
{
  "indices": "aim-dedup-parties-20250519,dedup-party-match-20250519,irq-index",
  "ignore_unavailable": false,
  "include_global_state": false,
  "metadata": { "taken_by": "pcis-admin", "taken_because": "To restore to staging." }
}
```
- Confirm the index list matches the distinct names in your notepad. The list above is the one Paul used.
- Expected response: `{"accepted": true}`.
- Put a single pair of quotes around the index list. The Confluence example has doubled quotes, which is invalid JSON.

### 4.2 Tab C: Identity Prod
```
PUT _snapshot/prod-identity-08282024-repo/prod-identity-20261104-snapshot
{
  "indices": "<identity index name(s) from notepad, comma-separated>",
  "ignore_unavailable": false,
  "include_global_state": false,
  "metadata": { "taken_by": "pcis-admin", "taken_because": "To restore to staging." }
}
```
If both identity aliases point to the same index, list it only once.

---

## Part 5: Wait for the snapshots and restore into stage

### 5.1 Check snapshot status (re-run every 10 minutes; this can take up to an hour)
Tab A:
```
GET _snapshot/prod-partyrecord-IDEN-28978-repo/prod-partyrecord-20261104-snapshot
```
Tab C:
```
GET _snapshot/prod-identity-08282024-repo/prod-identity-20261104-snapshot
```
- `"state": "IN_PROGRESS"`: keep waiting.
- `"state": "SUCCESS"`: continue.
- `"state": "FAILED"` or `"PARTIAL"`: **STOP** and take a screenshot.

### 5.2 Restore into Partyrecord Stage (tab B)
First confirm the snapshot is visible from stage by running the tab A command from 5.1 here. Then run:
```
POST _snapshot/prod-partyrecord-IDEN-28978-repo/prod-partyrecord-20261104-snapshot/_restore
{
  "indices": "aim-dedup-parties-20250519,dedup-party-match-20250519,irq-index",
  "ignore_unavailable": false,
  "rename_pattern": "(.+)",
  "rename_replacement": "$1_restored-prod-20261104",
  "include_global_state": false,
  "include_aliases": false
}
```

### 5.3 Restore into Identity Stage (tab D)
```
POST _snapshot/prod-identity-08282024-repo/prod-identity-20261104-snapshot/_restore
{
  "indices": "<same identity list as 4.2>",
  "ignore_unavailable": false,
  "rename_pattern": "(.+)",
  "rename_replacement": "$1_restored-prod-20261104",
  "include_global_state": false,
  "include_aliases": false
}
```

### 5.4 Wait for green (tabs B and D, re-run every 10 minutes)
```
GET _cat/indices/*restored-prod-20261104*?v
```
- Wait until every row shows `green` in the `health` column. Yellow means it is still copying.
- Write down the `docs.count` for each index. Part 10 compares against these.

---

## Part 6: Record the current stage indexes

Tab B:
```
GET _alias/aim-dedup-parties
GET _alias/iis-history-api-alias
GET _alias/dedup-party-match
GET _alias/irq-index-alias
```
Tab D:
```
GET _alias/iis-identity-api-alias
GET _alias/iis-identity-consumer-alias
```
Write these under **STAGE OLD** in your notepad. They should currently end in `_restored-prod-20260914`.

---

## Part 7: Switch the aliases to the new indexes

> **Ask Paul first:** should `irq-index-alias` be moved too? It is not in the runbook's alias step, but `irq-index` is restored. If he says yes, add a remove/add pair for it in 7.1.

### 7.1 Tab B: Partyrecord Stage
- In each `remove` line, use the **STAGE OLD** name from your notepad.
- In each `add` line, use the **PROD** name followed by `_restored-prod-20261104`.
```
POST _aliases
{ "actions": [
  { "remove": { "index": "<STAGE OLD dedup-party-match>",     "alias": "dedup-party-match" } },
  { "add":    { "index": "<PROD dedup-party-match>_restored-prod-20261104",     "alias": "dedup-party-match" } },
  { "remove": { "index": "<STAGE OLD iis-history-api-alias>", "alias": "iis-history-api-alias" } },
  { "add":    { "index": "<PROD iis-history-api-alias>_restored-prod-20261104", "alias": "iis-history-api-alias" } },
  { "remove": { "index": "<STAGE OLD aim-dedup-parties>",     "alias": "aim-dedup-parties" } },
  { "add":    { "index": "<PROD aim-dedup-parties>_restored-prod-20261104",     "alias": "aim-dedup-parties" } }
]}
```
Expected response: `{"acknowledged": true}`.

### 7.2 Tab D: Identity Stage
```
POST _aliases
{ "actions": [
  { "remove": { "index": "<STAGE OLD iis-identity-consumer-alias>", "alias": "iis-identity-consumer-alias" } },
  { "add":    { "index": "<PROD iis-identity-consumer-alias>_restored-prod-20261104", "alias": "iis-identity-consumer-alias" } },
  { "remove": { "index": "<STAGE OLD iis-identity-api-alias>",      "alias": "iis-identity-api-alias" } },
  { "add":    { "index": "<PROD iis-identity-api-alias>_restored-prod-20261104",      "alias": "iis-identity-api-alias" } }
]}
```

### 7.3 Verify
Re-run the `GET _alias/...` commands from Part 6. Every result should now end in **`_restored-prod-20261104`**.

> The old `_restored-prod-20260914` indexes are still on stage. Do **not** delete them. Ask Paul when they can be removed.

---

## Part 8: Copy the snapshot to S3 (Harness)
1. On the runbook page, click the **Harness pipeline** link (`pcis_elasticsearch_dump_ci`).
2. Click **Run**.
3. Set the branch to **develop** (not main) and fill in any other inputs.
4. Wait for the run to show **Success**.
5. Open the logs and write down the **S3 output folder**. Part 9 needs it.

---

## Part 9: Load into Databricks

> **Check before running this part.** `pcis_metadata.elasticdump_identities` is a **view**, not a table, so the runbook's `ALTER TABLE ... RENAME` steps may not work as written. Run this first to see which table the view reads from:
> ```sql
> SHOW CREATE TABLE eciscor_prod.pcis_metadata.elasticdump_identities;
> ```

The runbook pattern, repeated for each of `_identities`, `_parties` and `_party_match`:
```sql
-- 1. Load the new export into a temp table
CREATE OR REPLACE TABLE pcis_metadata.elasticdump_identities_temp AS
SELECT * FROM read_files(
  '/Volumes/eciscor_prod/pcis_metadata/pcis_elasticdump/for-comparison-<value>/<value>/',
  format => 'json',
  schema => '_id string, _index string, _score bigint, _source string, _type string');

-- 2. Check the count BEFORE renaming. It must match the docs.count from Part 5.4.
SELECT COUNT(*) FROM pcis_metadata.elasticdump_identities_temp;

-- 3. Only if the counts match: archive the current table, then promote the temp table
ALTER TABLE pcis_metadata.elasticdump_identities
  RENAME TO pcis_metadata.elasticdump_identities_archive_20261104;
ALTER TABLE pcis_metadata.elasticdump_identities_temp
  RENAME TO pcis_metadata.elasticdump_identities;
```
Adjust the table names to match what `SHOW CREATE TABLE` returns.

---

## Part 10: Finish
1. Check that each table's record count matches the `docs.count` of its index.
2. Run `OPTIMIZE` on each table.
3. Update the views so they point to the latest tables.
4. Check the view counts to confirm each view points to the right table.
5. Refresh all materialized views built on the three main tables.
6. Post in **ESIS chat** and the **Dark Saturn DSU chat**:
   > "OpenSearch prod→stage snapshot and Databricks elasticdump refresh complete for 11/04."

---

## Side task: Michele's missing identity
Once you are logged in to tab C (Identity Prod), run:
```
GET iis-identity-api-alias/_doc/9fd2c8a5c20c4c99b7bf0c73f295d7aa
```
- `"found": true`: the identity exists, so her search terms did not match it. Compare them with the identity's name, DOB and A-number.
- `"found": false`: the identity no longer exists in prod, most likely because it was merged into another identity or deleted.

---

## Useful read-only commands
```
GET _cat/indices?v&s=i          # all indexes, sorted by name
GET _cat/aliases?v&s=alias      # all aliases, sorted
GET _cat/allocation?v           # disk usage per node
GET _cluster/health             # overall cluster health
GET _cat/shards?v&s=state       # shard status (look for UNASSIGNED)
```
