# stg_plan_mapping_min.py
import os
import pyodbc
import pandas as pd
from typing import Optional, Callable

# ---------------------------------------------------------------------
# Connection (matches your manhattan_mapping.py env-driven style)
# ---------------------------------------------------------------------
def _build_conn_str() -> str:
    server  = os.getenv("SQL_SERVER", "QWVIDBSQLB401.ngquotit.com")
    database= os.getenv("SQL_DATABASE", "NGCS")
    driver  = os.getenv("SQL_DRIVER", "ODBC Driver 17 for SQL Server")
    encrypt = os.getenv("SQL_ENCRYPT", "no").lower() in {"1", "true", "yes", "y"}
    trust   = os.getenv("SQL_TRUST_SERVER_CERT", "yes").lower() in {"1", "true", "yes", "y"}
    return (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"Trusted_Connection=yes;"
        f"Encrypt={'yes' if encrypt else 'no'};"
        f"TrustServerCertificate={'yes' if trust else 'no'};"
    )

def _open_conn(conn_str: Optional[str]) -> pyodbc.Connection:
    return pyodbc.connect(conn_str or _build_conn_str(), autocommit=False)

# ---------------------------------------------------------------------
# Public API: insert only LoadTaskId, PolicyNumber, PlanId
#   - No updates (ever)
#   - If skip_duplicates=True, only inserts rows not already present for (LoadTaskId, PolicyNumber)
# ---------------------------------------------------------------------
def insert_stg_plan_mapping_min(
    df: pd.DataFrame,
    load_task_id: int,
    *,
    skip_duplicates: bool = True,
    conn: Optional[pyodbc.Connection] = None,
    conn_str: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> int:
    """
    Insert only (LoadTaskId, PolicyNumber, PlanId) into dbo.STGPlanMappingFactors.

    Args:
        df: DataFrame with columns (at least) PolicyNumber, PlanId
        load_task_id: int to populate LoadTaskId
        skip_duplicates: if True, don't insert when (LoadTaskId, PolicyNumber) already exists
        conn / conn_str: DB connection or connection string
        log: logger

    Returns:
        int: number of rows inserted this call
    """
    if df is None or df.empty:
        log("[STG-MIN] Nothing to insert (input DataFrame empty).")
        return 0

    # Normalize minimal payload
    src = df.copy()
    if "PolicyNumber" not in src.columns:
        src["PolicyNumber"] = ""
    if "PlanId" not in src.columns:
        src["PlanId"] = ""

    src["PolicyNumber"] = src["PolicyNumber"].astype(str).str.strip()
    src["PlanId"]       = src["PlanId"].astype(str).str.strip()
    src = src[src["PolicyNumber"] != ""].reset_index(drop=True)
    if src.empty:
        log("[STG-MIN] No non-empty PolicyNumber rows to insert.")
        return 0

    src["LoadTaskId"]   = int(load_task_id)
    src = src[["LoadTaskId", "PolicyNumber", "PlanId"]]

    # Open connection
    created_conn = False
    if conn is None:
        conn = _open_conn(conn_str)
        created_conn = True

    cur = conn.cursor()
    cur.fast_executemany = True

    log(f"[STG-MIN] Preparing to insert {len(src)} row(s) into dbo.STGPlanMappingFactors "
        f"({'skip dups' if skip_duplicates else 'allow dups'}).")

    # Stage to temp table
    cur.execute("""
        IF OBJECT_ID('tempdb..#stg_min') IS NOT NULL DROP TABLE #stg_min;
        CREATE TABLE #stg_min (
            LoadTaskId   INT           NOT NULL,
            PolicyNumber NVARCHAR(200) NOT NULL,
            PlanId       NVARCHAR(100)     NULL
        );
    """)

    cur.executemany("""
        INSERT INTO #stg_min (LoadTaskId, PolicyNumber, PlanId)
        VALUES (?, ?, ?)
    """, [(int(r.LoadTaskId), str(r.PolicyNumber), (None if r.PlanId == "" else str(r.PlanId)))
          for _, r in src.iterrows()])

    inserted = 0

    if skip_duplicates:
        # Insert only new rows; capture count via OUTPUT
        cur.execute("""
            IF OBJECT_ID('tempdb..#ins') IS NOT NULL DROP TABLE #ins;
            CREATE TABLE #ins (PolicyNumber NVARCHAR(200));

            INSERT INTO dbo.STGPlanMappingFactors (LoadTaskId, PolicyNumber, PlanId)
            OUTPUT inserted.PolicyNumber INTO #ins
            SELECT s.LoadTaskId, s.PolicyNumber, s.PlanId
            FROM #stg_min AS s
            WHERE NOT EXISTS (
                SELECT 1
                FROM dbo.STGPlanMappingFactors t
                WHERE t.LoadTaskId = s.LoadTaskId
                  AND t.PolicyNumber = s.PolicyNumber
            );

            SELECT COUNT(*) FROM #ins;
        """)
        row = cur.fetchone()
        inserted = int(row[0]) if row and row[0] is not None else 0
    else:
        # Blind insert, allow duplicates
        cur.execute("""
            INSERT INTO dbo.STGPlanMappingFactors (LoadTaskId, PolicyNumber, PlanId)
            SELECT LoadTaskId, PolicyNumber, PlanId
            FROM #stg_min;
        """)
        # pyodbc rowcount can be -1 if not available; fallback to len(src)
        inserted = cur.rowcount if cur.rowcount and cur.rowcount >= 0 else len(src)

    conn.commit()
    if created_conn:
        conn.close()

    log(f"[STG-MIN] Insert complete. Rows inserted: {inserted}.")
    return inserted


# Optional: simple CLI usage for ad-hoc testing
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Insert (LoadTaskId, PolicyNumber, PlanId) into STGPlanMappingFactors")
    ap.add_argument("--csv", required=False, help="CSV with PolicyNumber,PlanId columns")
    ap.add_argument("--load_task_id", required=True, type=int)
    ap.add_argument("--skip_dups", action="store_true", default=False)
    args = ap.parse_args()

    def _log(s: str): print(s, flush=True)

    if args.csv:
        df_in = pd.read_csv(args.csv, dtype=str).fillna("")
    else:
        # demo payload
        df_in = pd.DataFrame({
            "PolicyNumber": ["1102686", "1102942"],
            "PlanId":       ["DVH15K23", "AK7010EIL"],
        })

    insert_stg_plan_mapping_min(
        df=df_in,
        load_task_id=args.load_task_id,
        skip_duplicates=bool(args.skip_dups),
        log=_log
    )


### 

from stg_plan_mapping_min import insert_stg_plan_mapping_min

# payload must have PolicyNumber and PlanId
payload = joined[["PolicyNumber", "PlanId"]].copy()  # or whichever df holds those two

inserted = insert_stg_plan_mapping_min(
    df=payload,
    load_task_id=int(load_task_id),
    skip_duplicates=True,  # only insert new (no updates)
    log=log
)
log(f"[ManhattanLife] STG insert-min rows: {inserted}")