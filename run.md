# OpenSearch Snapshot: Production → Stage

**Run date: Thursday 2026-11-05.** Every command below already uses the date code **`20261105`**.

> **Running this on a different day?** In VS Code, press **Cmd+Option+F** (Find and Replace), replace `20261105` with that day's date in `YYYYMMDD` format, and click **Replace All**. Do this before you start.

This is the Confluence runbook *"Runbook: Opensearch Snapshot Production Data to Stage"* broken into one action per step, plus the stage cleanup Paul walked through.

---

## Read before you start

1. **You will be logged in as OpenSearch Admin, which can break production.** On the two PROD tabs, run only the commands in this document, exactly as written.
2. **Do one step at a time.** Each step tells you what you should see when it worked. If you see something different, **STOP** and take a screenshot.
3. **Parts 1–2 can be done any time. Do Parts 3–10 late in the afternoon or after hours.** The snapshot load can cause errors for other users during the day.
4. **How Dev Tools works.** You will use it in almost every step:
   - The **left pane** is where you type or paste commands.
   - To run a command, **click anywhere on its first line**, then press **Cmd+Enter** (or click the small ▶ that appears at the right of that line).
   - The result appears in the **right pane**.
   - A green **`200 - OK`** badge at the top right of the result pane means the command succeeded. A yellow or red badge (for example `403` or `404`) means it failed.
   - **To clear the left pane:** click inside it, press **Cmd+A**, then press **Delete**.

---

## Part 1: Setup

### 1.1 Create your notepad file
You will write index names into this file as you go. Later steps copy names out of it.

1. Open **VS Code**.
2. Press **Cmd+N** to open a new, empty file.
3. Copy the gray block below and paste it into that empty file. To copy it, hover over the block and click the copy icon at its top right, or select all the text inside it and press **Cmd+C**. Then click in the empty file and press **Cmd+V**.
   ```
   ===== OPENSEARCH SNAPSHOT NOTEPAD 20261105 =====

   --- STAGE CLEANUP (Part 2) ---
   Partyrecord stage min free BEFORE: 
   Identity stage min free BEFORE:    
   Deleted index 1: 
   Deleted index 2: 
   Partyrecord stage min free AFTER:  

   --- PROD INDEXES (Part 3) ---
   PR aim-dedup-parties           -> 
   PR iis-history-api-alias       -> 
   PR dedup-party-match           -> 
   PR irq-index-alias             -> 
   ID iis-identity-api-alias      -> 
   ID iis-identity-consumer-alias -> 

   PR SNAPSHOT LIST (distinct PR names, comma-separated, no spaces):
   ID SNAPSHOT LIST (distinct ID names, comma-separated, no spaces):

   --- SNAPSHOT STATUS (Part 5) ---
   PR snapshot state: 
   ID snapshot state: 

   --- RESTORED INDEXES + DOC COUNTS (Part 5) ---
   

   --- STAGE OLD INDEXES (Part 6) ---
   PR aim-dedup-parties           -> 
   PR iis-history-api-alias       -> 
   PR dedup-party-match           -> 
   PR irq-index-alias             -> 
   ID iis-identity-api-alias      -> 
   ID iis-identity-consumer-alias -> 

   --- HARNESS (Part 8) ---
   S3 output folder: 
   ```
4. Press **Cmd+S**.
5. In the Save dialog, set **Save As** to `opensearch_notepad_20261105.txt`.
6. Set **Where** to **Desktop**.
7. Click **Save**.
8. Leave this file open in VS Code for the whole run. **Press Cmd+S every time you add something to it.**

### 1.2 Get the OpenSearch Admin password from Vault
1. Go to the **Vault** tab in Chrome.
2. Click your user icon at the top right, then click **Sign out**. Vault can only be signed in to prod or nonprod at one time, so you have to sign out first.
3. Go to the Confluence runbook tab (*Runbook: Opensearch Snapshot Production Data to Stage*). Near the top of the page, find the **Quick Links** table. Click the link in the **Vault credentials** row (it ends in `namespace=PCIS_prod&version=7`).
4. A Vault sign-in page opens. Sign in the same way you normally sign in to Vault (same method and password as nonprod).
5. **Check:** the dropdown at the top left of Vault must say **PCIS_prod**. If it says PCIS_nonprod, go back to step 2.
6. Scroll down until you see **OpenSearch Admin**.
7. Find the username and write it on a sticky note or keep it in your head. Do **not** put it in the notepad file.
8. Click the **copy icon** next to the password. Do not click the eye icon; keep the password hidden on screen.

### 1.3 Open the four OpenSearch tabs
Do steps a–f once for each row of the table, in order A, B, C, D.

| Tab | Cluster | URL to paste in the address bar |
|---|---|---|
| **A** | Partyrecord **PROD** | `https://opensearch-partyrecord-prod.pcis.uscis.dhs.gov/_dashboards` |
| **B** | Partyrecord **STAGE** | `https://opensearch-partyrecord-staging.pcis.uscis.dhs.gov/_dashboards` |
| **C** | Identity **PROD** | `https://opensearch-identity-prod.pcis.uscis.dhs.gov/_dashboards` |
| **D** | Identity **STAGE** | `https://opensearch-identity-staging.pcis.uscis.dhs.gov/_dashboards` |

- a. Press **Cmd+T** to open a new Chrome tab.
- b. Paste the URL for that row and press **Enter**.
- c. Enter the OpenSearch Admin **username**. In the password box, press **Cmd+V** to paste the password, then click **Log in**. If the password is no longer on your clipboard, copy it from Vault again (step 1.2.8).
- d. If a **"Select your tenant"** popup appears, click **Private**, then **Confirm**.
- e. Click **Dev tools** at the top right of the home page. If a welcome popup appears, click **Dismiss**.
- f. Clear the sample command: click in the left pane, press **Cmd+A**, then **Delete**. **Do not run the sample `GET _search` command.**

**Check:** you have four Dev Tools tabs. Read the address bar to tell them apart: `partyrecord-prod`, `partyrecord-staging`, `identity-prod`, `identity-staging`.

> **Tip:** before you run anything, look at the address bar and confirm you are in the right tab. The PROD and STAGE tabs look identical.

---

## Part 2: Free up disk space on stage

### 2.1 Check free space in AWS
1. Open a new Chrome tab (**Cmd+T**).
2. Sign in to the AWS console the same way you did with Paul. You need the **PCS prod** AWS account. Your earlier screenshot showed `aws-dhs-uscis-eis-pcs-prod` with the role `EC2PowerUser` at the top right of the console. If you are not sure how to get there, use your **ICAM myAccess** bookmark, or ask Paul for the sign-in link.
3. **Check:** the account name at the top right of the console contains **`pcs-prod`**, and the region next to it says **United States (N. Virginia)**. If the region is different, click it and choose **US East (N. Virginia) us-east-1**.
4. Click in the address bar, paste this URL, and press **Enter**:
   `https://us-east-1.console.aws.amazon.com/aos/home?region=us-east-1#opensearch/dashboard`
5. **Expected:** a page titled **Dashboard** with a **Domain health** box and a table of 6 domains below it.
6. In the **Name** column, find the row **`opensearch-partyrecord-stage`**.
7. Scroll the table to the right, using the scroll bar under the table or Shift + mouse wheel. Go past **Total free space** until you see the column that starts with **Minim...** (minimum free storage). If the column name is cut off, drag the right edge of its header to widen it.
8. Write that row's value in the notepad next to `Partyrecord stage min free BEFORE:` (for example `901.3 GiB`).
9. Do the same for **`opensearch-identity-stage`** and write its value next to `Identity stage min free BEFORE:`.
10. Press **Cmd+S** to save the notepad.
11. **Rule: each stage cluster needs at least 1 TB of minimum free storage.** 1 TB is about 1,000 GiB, so a value shown in GiB that is under 1,000 is below the limit.
   - Partyrecord stage below 1 TB: do steps 2.2–2.6 in **Tab B**.
   - Identity stage below 1 TB: also do steps 2.2–2.6 in **Tab D**.
   - Both at 1 TB or more: skip to Part 3.

### 2.2 List all indexes (Tab B, Partyrecord STAGE)
1. Go to **Tab B**. **Check that the address bar says `partyrecord-staging`.**
2. Clear the left pane (**Cmd+A**, **Delete**).
3. Paste:
   ```
   GET _cat/indices?v&s=i
   ```
4. Click on the line and press **Cmd+Enter**.
5. **Expected:** the right pane shows a table with columns `health status index uuid pri rep docs.count ... store.size`.
6. Look down the `index` column for names ending in **`_restored-prod-20260807`**. There should be two, one starting with `aim-dedup-parties-` and one starting with `dedup-party-match-`.
7. For each one, select the full index name in the right pane, copy it (**Cmd+C**), and paste it into the notepad next to `Deleted index 1:` and `Deleted index 2:`.
8. Press **Cmd+S** to save the notepad.

### 2.3 Confirm no alias points to those indexes (Tab B)
1. Clear the left pane.
2. Paste:
   ```
   GET _cat/aliases?v&s=alias
   ```
3. Click on the line and press **Cmd+Enter**.
4. **Expected:** a table with columns `alias index filter routing.index ...`.
5. Read every row of the `index` column.
6. **If any row shows a name ending in `20260807`, STOP and message Paul.** Do not delete anything.
7. If none do, continue.

### 2.4 Delete the first old index (Tab B)
1. Clear the left pane.
2. Type `DELETE ` (the word DELETE followed by one space).
3. Copy **Deleted index 1** from the notepad and paste it right after the space. The line should look like:
   ```
   DELETE aim-dedup-parties-20250519_restored-prod-20260807
   ```
   (Your name may differ slightly. Use exactly what is in your notepad.)
4. **Check the line three times:**
   - It ends in `20260807`.
   - It has **no `*`** in it.
   - The address bar still says `partyrecord-staging`.
5. Click on the line and press **Cmd+Enter**.
6. **Expected:** `{"acknowledged": true}` with a `200 - OK` badge.

### 2.5 Delete the second old index (Tab B)
1. Repeat 2.4 using **Deleted index 2** from the notepad.

> **Never delete:**
> - anything ending in `20260914`, because stage is currently using these;
> - anything ending in `20260918`, because Paul is using these;
> - anything without `restored-prod` in its name;
> - anything with a `*` in the command.

### 2.6 Confirm the space was freed
1. Go back to the AWS tab and press **Cmd+R** to refresh.
2. Scroll right to the **minimum free storage** column for `opensearch-partyrecord-stage`.
3. Write the value in the notepad next to `Partyrecord stage min free AFTER:` and press **Cmd+S**.
4. **Expected:** well over 1 TB. If it is still under 1 TB, **STOP** and message Paul.

---

## Part 3: Record the prod index names (late afternoon)

### 3.1 Partyrecord prod (Tab A)
1. Go to **Tab A**. **Check that the address bar says `partyrecord-prod`.**
2. Clear the left pane.
3. Paste:
   ```
   GET _alias/aim-dedup-parties
   ```
4. Click on the line and press **Cmd+Enter**.
5. **Expected result** (your index name may differ):
   ```
   {
     "aim-dedup-parties-20250519": {
       "aliases": {
         "aim-dedup-parties": {}
       }
     }
   }
   ```
6. **How to read it:** the text in quotes on the **second line** is the index name. In this example it is `aim-dedup-parties-20250519`.
7. Copy that index name, without the quotes, into the notepad next to `PR aim-dedup-parties ->`. Press **Cmd+S**.
8. Clear the left pane and repeat steps 3–7 for each of these, writing each result on its matching notepad line:
   ```
   GET _alias/iis-history-api-alias
   ```
   ```
   GET _alias/dedup-party-match
   ```
   ```
   GET _alias/irq-index-alias
   ```

### 3.2 Identity prod (Tab C)
1. Go to **Tab C**. **Check that the address bar says `identity-prod`.**
2. Repeat the same process for each of these, writing each result on its matching `ID` line:
   ```
   GET _alias/iis-identity-api-alias
   ```
   ```
   GET _alias/iis-identity-consumer-alias
   ```

### 3.3 Build the snapshot lists in the notepad
1. Look at your four **PR** index names. Some aliases may point to the same index; for example, `iis-history-api-alias` often points to the same index as `aim-dedup-parties`.
2. Write each **distinct** PR index name once, separated by commas with **no spaces**, next to `PR SNAPSHOT LIST:`. It will most likely be:
   ```
   aim-dedup-parties-20250519,dedup-party-match-20250519,irq-index
   ```
3. Do the same for the two **ID** names next to `ID SNAPSHOT LIST:`. If both are the same name, write it once.
4. Press **Cmd+S**.
5. **Optional check:** send a screenshot of your notepad before moving to Part 4.

---

## Part 4: Create the snapshots (Tab A, then Tab C right away)

### 4.1 Partyrecord prod snapshot (Tab A)
1. Go to **Tab A**. **Check that the address bar says `partyrecord-prod`.**
2. Clear the left pane.
3. Paste this whole block:
   ```
   PUT _snapshot/prod-partyrecord-IDEN-28978-repo/prod-partyrecord-20261105-snapshot
   {
     "indices": "PASTE_PR_SNAPSHOT_LIST_HERE",
     "ignore_unavailable": false,
     "include_global_state": false,
     "metadata": { "taken_by": "pcis-admin", "taken_because": "To restore to staging." }
   }
   ```
4. Double-click `PASTE_PR_SNAPSHOT_LIST_HERE` to select it, then paste your **PR SNAPSHOT LIST** from the notepad in its place.
5. **Check:**
   - There is exactly **one** `"` before and **one** `"` after the list.
   - There are no spaces in the list.
6. Click on the `PUT` line and press **Cmd+Enter**.
7. **Expected:** `{"accepted": true}`.

### 4.2 Identity prod snapshot (Tab C), immediately after 4.1
1. Go to **Tab C**. **Check that the address bar says `identity-prod`.**
2. Clear the left pane.
3. Paste:
   ```
   PUT _snapshot/prod-identity-08282024-repo/prod-identity-20261105-snapshot
   {
     "indices": "PASTE_ID_SNAPSHOT_LIST_HERE",
     "ignore_unavailable": false,
     "include_global_state": false,
     "metadata": { "taken_by": "pcis-admin", "taken_because": "To restore to staging." }
   }
   ```
4. Replace `PASTE_ID_SNAPSHOT_LIST_HERE` with your **ID SNAPSHOT LIST**.
5. Click on the `PUT` line and press **Cmd+Enter**.
6. **Expected:** `{"accepted": true}`.

---

## Part 5: Wait for the snapshots, then restore to stage

### 5.1 Wait for the partyrecord snapshot (Tab A)
1. In **Tab A**, clear the left pane and paste:
   ```
   GET _snapshot/prod-partyrecord-IDEN-28978-repo/prod-partyrecord-20261105-snapshot
   ```
2. Click on the line and press **Cmd+Enter**.
3. In the right pane, find `"state":`.
   - `"IN_PROGRESS"`: wait 10 minutes and run it again.
   - `"SUCCESS"`: write `SUCCESS` next to `PR snapshot state:` and press **Cmd+S**.
   - `"FAILED"` or `"PARTIAL"`: **STOP** and take a screenshot.
4. This can take up to 1 hour.

### 5.2 Wait for the identity snapshot (Tab C)
1. In **Tab C**, do the same with:
   ```
   GET _snapshot/prod-identity-08282024-repo/prod-identity-20261105-snapshot
   ```
2. When it says `"SUCCESS"`, write `SUCCESS` next to `ID snapshot state:` and press **Cmd+S**.

> **Do not continue until both snapshots say `SUCCESS`.**

### 5.3 Restore into partyrecord stage (Tab B)
1. Go to **Tab B**. **Check that the address bar says `partyrecord-staging`.**
2. Clear the left pane and paste:
   ```
   GET _snapshot/prod-partyrecord-IDEN-28978-repo/prod-partyrecord-20261105-snapshot
   ```
3. Run it. **Expected:** the same `"SUCCESS"` result as in Tab A. This confirms stage can see the snapshot.
4. Clear the left pane and paste:
   ```
   POST _snapshot/prod-partyrecord-IDEN-28978-repo/prod-partyrecord-20261105-snapshot/_restore
   {
     "indices": "PASTE_PR_SNAPSHOT_LIST_HERE",
     "ignore_unavailable": false,
     "rename_pattern": "(.+)",
     "rename_replacement": "$1_restored-prod-20261105",
     "include_global_state": false,
     "include_aliases": false
   }
   ```
5. Replace `PASTE_PR_SNAPSHOT_LIST_HERE` with your **PR SNAPSHOT LIST**.
6. Click on the `POST` line and press **Cmd+Enter**.
7. **Expected:** `{"accepted": true}`.

### 5.4 Restore into identity stage (Tab D)
1. Go to **Tab D**. **Check that the address bar says `identity-staging`.**
2. Clear the left pane and paste:
   ```
   POST _snapshot/prod-identity-08282024-repo/prod-identity-20261105-snapshot/_restore
   {
     "indices": "PASTE_ID_SNAPSHOT_LIST_HERE",
     "ignore_unavailable": false,
     "rename_pattern": "(.+)",
     "rename_replacement": "$1_restored-prod-20261105",
     "include_global_state": false,
     "include_aliases": false
   }
   ```
3. Replace `PASTE_ID_SNAPSHOT_LIST_HERE` with your **ID SNAPSHOT LIST**.
4. Click on the `POST` line and press **Cmd+Enter**.
5. **Expected:** `{"accepted": true}`.

### 5.5 Wait for the restored indexes to turn green (Tabs B and D)
1. In **Tab B**, clear the left pane and paste:
   ```
   GET _cat/indices/*restored-prod-20261105*?v
   ```
2. Run it. You will see one row per restored index. Look at the first column, `health`:
   - `yellow` or `red`: still copying. Wait 10 minutes and run it again.
   - **all rows `green`**: done.
3. When every row is green, copy each row's `index` name and `docs.count` number into the notepad under `RESTORED INDEXES + DOC COUNTS`. Press **Cmd+S**.
4. Repeat steps 1–3 in **Tab D**.

> **Do not continue until every restored index in both tabs is `green`.**

---

## Part 6: Record the current (old) stage index names

### 6.1 Partyrecord stage (Tab B)
1. In **Tab B**, clear the left pane and run these one at a time, the same way as Part 3:
   ```
   GET _alias/aim-dedup-parties
   ```
   ```
   GET _alias/iis-history-api-alias
   ```
   ```
   GET _alias/dedup-party-match
   ```
   ```
   GET _alias/irq-index-alias
   ```
2. Write each index name on the matching line under **STAGE OLD INDEXES** in the notepad.
3. **Expected:** they end in `_restored-prod-20260914`.
4. Press **Cmd+S**.

### 6.2 Identity stage (Tab D)
1. In **Tab D**, run these one at a time:
   ```
   GET _alias/iis-identity-api-alias
   ```
   ```
   GET _alias/iis-identity-consumer-alias
   ```
2. Write them under **STAGE OLD INDEXES** and press **Cmd+S**.

---

## Part 7: Move the aliases to the new indexes

> **Before this part, message Paul:** "Should `irq-index-alias` also be moved to the new restored irq-index? It's not in the runbook's alias step." Wait for his answer.

### 7.1 Build the partyrecord alias command
1. In VS Code, scroll to the bottom of the notepad and paste this block:
   ```
   POST _aliases
   { "actions": [
     { "remove": { "index": "OLD_DPM",  "alias": "dedup-party-match" } },
     { "add":    { "index": "NEW_DPM",  "alias": "dedup-party-match" } },
     { "remove": { "index": "OLD_HIST", "alias": "iis-history-api-alias" } },
     { "add":    { "index": "NEW_HIST", "alias": "iis-history-api-alias" } },
     { "remove": { "index": "OLD_AIM",  "alias": "aim-dedup-parties" } },
     { "add":    { "index": "NEW_AIM",  "alias": "aim-dedup-parties" } }
   ]}
   ```
2. Replace each placeholder:

   | Placeholder | Replace with |
   |---|---|
   | `OLD_DPM` | STAGE OLD `dedup-party-match` value |
   | `NEW_DPM` | PROD `dedup-party-match` value + `_restored-prod-20261105` |
   | `OLD_HIST` | STAGE OLD `iis-history-api-alias` value |
   | `NEW_HIST` | PROD `iis-history-api-alias` value + `_restored-prod-20261105` |
   | `OLD_AIM` | STAGE OLD `aim-dedup-parties` value |
   | `NEW_AIM` | PROD `aim-dedup-parties` value + `_restored-prod-20261105` |

   **Example:** if PROD `dedup-party-match` is `dedup-party-match-20250519`, then `NEW_DPM` becomes `dedup-party-match-20250519_restored-prod-20261105`.
3. **Check:** every `NEW_` value must exactly match a name in your `RESTORED INDEXES` list from Part 5.5.
4. If Paul said yes to irq: add two more lines before the `]}`, and add a comma at the end of the line before them:
   ```
     { "remove": { "index": "OLD_IRQ", "alias": "irq-index-alias" } },
     { "add":    { "index": "NEW_IRQ", "alias": "irq-index-alias" } }
   ```
5. Press **Cmd+S**.

### 7.2 Run it (Tab B)
1. Go to **Tab B**. **Check that the address bar says `partyrecord-staging`.**
2. Clear the left pane.
3. Copy the completed block from your notepad and paste it.
4. Click on the `POST _aliases` line and press **Cmd+Enter**.
5. **Expected:** `{"acknowledged": true}`.

### 7.3 Build and run the identity alias command (Tab D)
1. In the notepad, paste:
   ```
   POST _aliases
   { "actions": [
     { "remove": { "index": "OLD_CONSUMER", "alias": "iis-identity-consumer-alias" } },
     { "add":    { "index": "NEW_CONSUMER", "alias": "iis-identity-consumer-alias" } },
     { "remove": { "index": "OLD_API",      "alias": "iis-identity-api-alias" } },
     { "add":    { "index": "NEW_API",      "alias": "iis-identity-api-alias" } }
   ]}
   ```
2. Replace the placeholders the same way as 7.1:
   - `OLD_` values come from STAGE OLD.
   - `NEW_` values are the PROD value + `_restored-prod-20261105`.
3. Go to **Tab D**. **Check that the address bar says `identity-staging`.**
4. Clear the left pane, paste the block, click on the `POST` line, and press **Cmd+Enter**.
5. **Expected:** `{"acknowledged": true}`.

### 7.4 Verify
1. In **Tab B**, re-run each `GET _alias/...` command from Part 6.1.
2. In **Tab D**, re-run each `GET _alias/...` command from Part 6.2.
3. **Expected:** every index name now ends in **`_restored-prod-20261105`**.
4. If any still ends in `20260914`, **STOP** and take a screenshot.

> **Leave the old `_restored-prod-20260914` indexes alone.** Ask Paul when they can be deleted.

---

## Part 8: Copy the snapshot to S3 (Harness)

This pipeline copies the data from OpenSearch into S3, where Databricks can read it in Part 9.

### 8.1 Open the pipeline
1. Go to the Confluence runbook tab and scroll to the section **Copy Snapshot to S3**.
2. That section has two links:
   - the GitHub repo `https://git.uscis.dhs.gov/USCIS/pcis-elasticsearch-dump`;
   - a long Harness link starting with `https://harness.tools.k8s.uscis.dhs.gov/ng/account/...`.
3. Click the **Harness** link (the second one). If Harness asks you to sign in, sign in with your usual USCIS sign-in.
4. **Expected:** a page for the pipeline **`pcis_elasticsearch_dump_ci`** that lists past executions (recent runs), newest first.

### 8.2 Look at the last successful run first
This shows you what inputs were used before, so you can copy them.
1. In the executions list, find the most recent run with a green **Success** status.
2. Click it to open it.
3. Find the inputs that run used. Look for a tab or button named **Inputs** or **Input Set** near the top of the execution page.
4. Take a screenshot of those inputs, or write them down.
5. Go back to the pipeline page with your browser's **Back** button.

### 8.3 Start a new run
1. Click the blue **Run** button at the top right of the pipeline page.
2. A **Run Pipeline** panel opens.
3. Find the **Git Branch** field (it may be labeled **Branch**) and set it to **`develop`**. **Do not use `main`.** The runbook says so explicitly.
4. Fill in any other required inputs so they match the last successful run from 8.2.
   - If an input looks like it needs **today's date** or a **snapshot name**, use `20261105` or `prod-partyrecord-20261105-snapshot` / `prod-identity-20261105-snapshot`.
   - **If any input is unclear, STOP.** Take a screenshot of the panel and ask Paul before running.
5. Click **Run Pipeline** at the bottom of the panel.

### 8.4 Wait for it to finish
1. You are taken to the execution page. It shows the pipeline stages as boxes.
2. Each stage turns **blue** (running), then **green** (success) or **red** (failed).
3. Wait until the whole run shows **Success** at the top. Refresh the page with **Cmd+R** every few minutes.
4. **If any stage turns red:** click it, take a screenshot of its log, and **STOP**.

### 8.5 Record the S3 output folder
1. On the finished execution page, click the last green step.
2. The log for that step opens on the right.
3. Press **Cmd+F** and search for `pcis_elasticdump` (or `s3://`).
4. Find the path the data was written to. It looks similar to:
   `.../pcis_elasticdump/for-comparison-<something>/<something>/`
5. Copy the full path into the notepad next to `S3 output folder:` and press **Cmd+S**.
6. **If you can't find the path,** you can find it in Databricks in Part 9.2 instead.

---

## Part 9: Load into Databricks

The runbook loads three datasets: **identities**, **parties** and **party_match**. In Databricks, `elasticdump_identities` turned out to be a **view** (a saved query pointing at a table), not a table. Parts 9.1–9.3 work out how the objects are set up before anything is changed.

> **Do not run any `ALTER`, `DROP` or `CREATE OR REPLACE` command in Part 9 until you finish 9.3 and know which path (A or B) applies.**

### 9.1 Open a notebook
1. Open Databricks (your **Home - Databricks** bookmark).
2. In the left sidebar, click **Workspace**, then open your user folder (`joshua.w.smitherman@uscis.dhs.gov`).
3. Click **Create** (top right), then **Notebook**. Name it `elasticdump_refresh_20261105`.
4. At the top right of the notebook, click the compute dropdown and pick a running SQL warehouse or cluster. Your existing notebooks use one starting with `dbis-sql-...`.
5. Each step below says "new cell". To add one, hover below the last cell and click **+ Code**. Start every cell with `%sql` on its own line, then the SQL.
6. To run a cell, click inside it and press **Shift+Enter**.

### 9.2 Find the new S3 folder
1. New cell:
   ```sql
   %sql
   LIST '/Volumes/eciscor_prod/pcis_metadata/pcis_elasticdump/'
   ```
2. Run it. **Expected:** a list of folders whose names start with `for-comparison-...`.
3. Click the `modification_time` column header to sort it, so the newest folder is at the top.
4. The newest folder should match what you wrote for `S3 output folder:` in Part 8.5. If you left that line blank, write this folder name there now.
5. Look inside it. New cell, replacing `<FOLDER>` with the folder name:
   ```sql
   %sql
   LIST '/Volumes/eciscor_prod/pcis_metadata/pcis_elasticdump/<FOLDER>/'
   ```
6. **Expected:** subfolders, probably one per dataset (identities, parties, party_match), or one per index.
7. Write each subfolder's full path into the notepad on a new line, labeled with the dataset it belongs to. For example:
   ```
   IDENTITIES path:  /Volumes/eciscor_prod/pcis_metadata/pcis_elasticdump/<FOLDER>/<identities subfolder>/
   PARTIES path:     /Volumes/eciscor_prod/pcis_metadata/pcis_elasticdump/<FOLDER>/<parties subfolder>/
   PARTY_MATCH path: /Volumes/eciscor_prod/pcis_metadata/pcis_elasticdump/<FOLDER>/<party_match subfolder>/
   ```
8. Press **Cmd+S**. **If you can't tell which subfolder is which, STOP** and take a screenshot.

### 9.3 Find out how the current objects are set up
1. New cell:
   ```sql
   %sql
   SHOW TABLES IN eciscor_prod.pcis_metadata LIKE 'elasticdump*'
   ```
2. Run it. **Expected:** a list of names such as `elasticdump_identities`, `elasticdump_parties`, `elasticdump_party_match`, plus possibly `..._archive_<date>` and other dated versions.
3. Take a screenshot of the full list.
4. New cell:
   ```sql
   %sql
   SHOW CREATE TABLE eciscor_prod.pcis_metadata.elasticdump_identities
   ```
5. Run it. Click the result cell to expand it, and copy the full text into the notepad under a new heading `VIEW DEFINITION - identities`.
6. Repeat steps 4–5 for `elasticdump_parties` and `elasticdump_party_match`.
7. Press **Cmd+S**.
8. **Decide which path applies.** Look at each definition:
   - **Path A:** it starts with `CREATE VIEW` and ends with `FROM eciscor_prod.pcis_metadata.<some table name>`. The object is a view pointing at a table. Write that table name in the notepad (for example `identities current table: elasticdump_identities_20260914`).
   - **Path B:** it starts with `CREATE TABLE`. The object is a real table, and the runbook's rename steps apply as written.
9. **STOP and send a screenshot of the three definitions** before going on, so the path can be confirmed.

---

### 9.4 Path A: the object is a view pointing at a table
Do 9.4.1–9.4.5 once for **identities**, then repeat for **parties**, then for **party_match**. The steps use identities as the example.

#### 9.4.1 Load the new data into a new dated table
New cell. Replace `<IDENTITIES path>` with the path from your notepad and keep the quotes:
```sql
%sql
CREATE TABLE eciscor_prod.pcis_metadata.elasticdump_identities_20261105 AS
SELECT * FROM read_files(
  '<IDENTITIES path>',
  format => 'json',
  schema => '_id string, _index string, _score bigint, _source string, _type string'
)
```
Run it. **Expected:** it finishes with no error. This can take several minutes.

#### 9.4.2 Check the row count
New cell:
```sql
%sql
SELECT COUNT(*) AS new_rows FROM eciscor_prod.pcis_metadata.elasticdump_identities_20261105
```
- Compare `new_rows` with the `docs.count` of the matching restored index in your notepad (Part 5.5). For identities, use the identity index count.
- **They must be equal, or very close.** If they differ by more than a few, **STOP**. The old view is untouched, so nothing is broken.
- Write the number in the notepad: `identities new_rows: <number>`. Press **Cmd+S**.

#### 9.4.3 Optimize the new table
New cell:
```sql
%sql
OPTIMIZE eciscor_prod.pcis_metadata.elasticdump_identities_20261105
```
Run it. **Expected:** a result table of file statistics, with no error.

#### 9.4.4 Point the view at the new table
1. Copy the **VIEW DEFINITION - identities** text from your notepad into a new cell, and put `%sql` on the first line.
2. Change the first words from `CREATE VIEW` to `CREATE OR REPLACE VIEW`.
3. In the `FROM` line, replace the **old table name** with `elasticdump_identities_20261105`.
4. Leave everything else exactly as it was.
5. **Check:** only those two things changed.
6. Run it. **Expected:** no error.

#### 9.4.5 Confirm the view now uses the new data
New cell:
```sql
%sql
SELECT COUNT(*) AS view_rows FROM eciscor_prod.pcis_metadata.elasticdump_identities
```
**Expected:** `view_rows` equals `new_rows` from 9.4.2.

#### 9.4.6 Repeat for parties and party_match
Repeat 9.4.1–9.4.5, changing:
- `elasticdump_identities` → `elasticdump_parties`, and use the **PARTIES path**;
- then `elasticdump_identities` → `elasticdump_party_match`, and use the **PARTY_MATCH path**.

For the row counts, compare parties and party_match with the **partyrecord** restored index counts in your notepad (aim-dedup-parties for parties, dedup-party-match for party_match).

> **Do not drop the old dated tables.** Leave them in place. Ask Paul when they can be removed.

---

### 9.5 Path B: the object is a real table (runbook as written)
Do 9.5.1–9.5.4 for **identities**, then **parties**, then **party_match**.

#### 9.5.1 Load into a temp table
```sql
%sql
CREATE OR REPLACE TABLE eciscor_prod.pcis_metadata.elasticdump_identities_temp AS
SELECT * FROM read_files(
  '<IDENTITIES path>',
  format => 'json',
  schema => '_id string, _index string, _score bigint, _source string, _type string'
)
```

#### 9.5.2 Check the count
```sql
%sql
SELECT COUNT(*) AS new_rows FROM eciscor_prod.pcis_metadata.elasticdump_identities_temp
```
It must match the notepad `docs.count`. **If it doesn't, STOP.**

#### 9.5.3 Archive the old table, then promote the temp table
Run these as **two separate cells**, in this order:
```sql
%sql
ALTER TABLE eciscor_prod.pcis_metadata.elasticdump_identities
  RENAME TO eciscor_prod.pcis_metadata.elasticdump_identities_archive_20261105
```
```sql
%sql
ALTER TABLE eciscor_prod.pcis_metadata.elasticdump_identities_temp
  RENAME TO eciscor_prod.pcis_metadata.elasticdump_identities
```

#### 9.5.4 Optimize
```sql
%sql
OPTIMIZE eciscor_prod.pcis_metadata.elasticdump_identities
```

---

## Part 10: Finish

### 10.1 Refresh the materialized views
1. New cell:
   ```sql
   %sql
   SHOW VIEWS IN eciscor_prod.pcis_metadata
   ```
2. Run it. In the result, look at the `isMaterialized` column. Every row with `true` is a materialized view. One of them should be `mv_es_mlaas_scores`.
3. Write the names of all materialized views into the notepad and press **Cmd+S**.
4. The runbook says to refresh all materialized views that use the three main tables. To check which ones do, run `SHOW CREATE TABLE eciscor_prod.pcis_metadata.<mv name>` for each and look for `elasticdump_identities`, `elasticdump_parties` or `elasticdump_party_match` in the text.
5. For each materialized view that uses them, new cell:
   ```sql
   %sql
   REFRESH MATERIALIZED VIEW eciscor_prod.pcis_metadata.<mv name>
   ```
6. Run it. **Expected:** it completes with no error. This can take a while.
7. **If you are unsure whether to refresh one, check the runbook's linked notebook "PCIS Opensearch Databricks Notebook (Production)".** It contains the views and materialized views the runbook expects. Open it from the link in the **Copy S3 Data into Databricks** section of the Confluence page and compare.

### 10.2 Final count check
New cell:
```sql
%sql
SELECT 'identities'  AS dataset, COUNT(*) AS row_count FROM eciscor_prod.pcis_metadata.elasticdump_identities
UNION ALL
SELECT 'parties',     COUNT(*) FROM eciscor_prod.pcis_metadata.elasticdump_parties
UNION ALL
SELECT 'party_match', COUNT(*) FROM eciscor_prod.pcis_metadata.elasticdump_party_match
```
**Expected:** each count matches the `docs.count` in your notepad from Part 5.5. Take a screenshot and keep it.

### 10.3 Post completion
1. Open Teams.
2. Go to **ESIS chat** and post:
   > OpenSearch prod→stage snapshot complete for 11/05. Stage aliases now point to `_restored-prod-20261105` indexes. Databricks elasticdump tables refreshed and counts verified.
3. Go to **Dark Saturn DSU chat** and post the same message.

### 10.4 Close out
1. Save the notepad one last time (**Cmd+S**). Keep it as the record of this run.
2. Close the four OpenSearch tabs. Admin sessions should not be left open.
3. Sign out of Vault (user icon, then **Sign out**).

---

## Side task: Michele's missing identity (any time after Part 1)
1. Go to **Tab C**. **Check that the address bar says `identity-prod`.**
2. Clear the left pane and paste:
   ```
   GET iis-identity-api-alias/_doc/9fd2c8a5c20c4c99b7bf0c73f295d7aa
   ```
3. Click on the line and press **Cmd+Enter**.
4. Read `"found":` in the result:
   - `"found": true`: the identity exists, so her search terms did not match it. Screenshot the result with names, DOB and A-number covered, and compare them with what she searched.
   - `"found": false`: the identity no longer exists in prod, most likely because it was merged into another identity or deleted.

---

## Reference: read-only commands (safe on any tab)
```
GET _cat/indices?v&s=i          # all indexes, sorted by name
GET _cat/aliases?v&s=alias      # all aliases, sorted by alias
GET _cat/allocation?v           # disk used/free per node
GET _cluster/health             # overall cluster health (green/yellow/red)
GET _cat/shards?v&s=state       # shard status (look for UNASSIGNED)
```
