# stg_plan_mapping_min.py
import pyodbc
import pandas as pd
from typing import Callable


def build_conn_str(server: str, database: str) -> str:
    return (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={server};"
        f"DATABASE={database};"
        "Trusted_Connection=yes;"
        "Encrypt=no;"
        "TrustServerCertificate=yes;"
    )


def stg_plan_mapping_min(
    df: pd.DataFrame,
    *,
    server: str,
    database: str,
    log: Callable[[str], None] = print
) -> int:
    """
    Blind insert using f-string SQL.
    df MUST contain: LoadTaskId, PolicyNumber, PlanId.
    """

    required = {"LoadTaskId", "PolicyNumber", "PlanId"}
    missing = required - set(df.columns)
    if missing:
        log(f"[STG] ERROR: Missing required columns: {missing}")
        return 0

    if df.empty:
        log("[STG] No rows to insert.")
        return 0

    conn_str = build_conn_str(server, database)

    try:
        conn = pyodbc.connect(conn_str, autocommit=False)
    except Exception as e:
        log(f"[STG] ERROR: Could not connect to SQL Server: {e}")
        return 0

    cur = conn.cursor()
    insert_count = 0

    try:
        for _, r in df.iterrows():

            load_task_id = int(r["LoadTaskId"])
            policy = str(r["PolicyNumber"]).replace("'", "''")
            planid = str(r["PlanId"]).replace("'", "''") if r["PlanId"] else ""

            sql = f"""
                INSERT INTO dbo.STGPlanMappingFactors
                    (LoadTaskId, PolicyNumber, PlanId)
                VALUES
                    ({load_task_id}, '{policy}', '{planid}');
            """

            cur.execute(sql)
            insert_count += 1

        conn.commit()
        log(f"[STG] Insert complete. Rows inserted: {insert_count}")
        return insert_count

    except Exception as e:
        conn.rollback()
        log(f"[STG] ERROR during insert; rolled back: {e}")
        return 0

    finally:
        try:
            conn.close()
        except:
            pass
