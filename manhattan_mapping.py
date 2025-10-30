"""
manhattan_mapping.py
--------------------
Helper to populate ProductType and PlanName from SQL Server
ONLY for Manhattan Life. Keeps DB logic separate from the main LLM script.
"""

import os
from typing import Optional
import pandas as pd
import pyodbc


def is_manhattan_issuer(name: str) -> bool:
    """Return True if issuer name looks like 'Manhattan Life' (case/space tolerant)."""
    if not name:
        return False
    key = "".join(name.lower().split())
    return key in {"manhattanlife", "manhattan_life", "manhattan-life"}


def apply_manhattan_mapping(
    out_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    plan_code_col: str = "PlanCode",
    *,
    load_task_id: int,
    company_issuer_id: int,
    server: str,
    database: str,
    driver: str = "ODBC Driver 18 for SQL Server",
    encrypt: bool = False,
    trust_server_certificate: bool = True,
) -> pd.DataFrame:
    """Fetch SQL mapping and apply ProductType <- PolicyNumber, PlanName <- ProductName."""
    if plan_code_col not in raw_df.columns:
        print("[ManhattanLife] PlanCode column missing — skipping mapping.")
        return out_df

    conn_str = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        "Trusted_Connection=yes;"
        f"Encrypt={'yes' if encrypt else 'no'};"
        f"TrustServerCertificate={'yes' if trust_server_certificate else 'no'};"
    )

    with pyodbc.connect(conn_str) as conn:
        sql = r"""
        SELECT DISTINCT
            p.IssuerPlanId   AS PlanCode,
            s.PolicyNumber,
            cpp.ProductName
        FROM dbo.STGPlanMappingFactors s (NOLOCK)
        OUTER APPLY (
            SELECT TOP 1 p.*
            FROM dbo.PlanMappingIssuerDetail p (NOLOCK)
            LEFT JOIN dbo.PlanMappingIssuerDetailState ps (NOLOCK)
                ON p.PlanMappingIssuerDetailId = ps.PlanMappingIssuerDetailId
            WHERE
                (p.IssuerPlanName = s.PlanName OR p.IssuerPlanName IS NULL OR p.IssuerPlanName = '')
                AND (p.IssuerPlanId = s.PlanId OR p.IssuerPlanId IS NULL OR p.IssuerPlanId = '')
                AND (p.[Year] = s.[Year] OR p.[Year] IS NULL)
                AND (p.PercentRate = s.RatePercent OR p.PercentRate IS NULL)
                AND (p.CurrencyRate = s.RateDollar OR p.CurrencyRate IS NULL)
                AND (ps.StateCode = s.StateCode OR ps.StateCode IS NULL)
            ORDER BY
                CASE WHEN p.IssuerPlanName = s.PlanName THEN 1 ELSE 0 END +
                CASE WHEN p.IssuerPlanId   = s.PlanId   THEN 1 ELSE 0 END +
                CASE WHEN p.PercentRate    = s.RatePercent THEN 1 ELSE 0 END +
                CASE WHEN p.[Year]         = s.[Year]   THEN 1 ELSE 0 END +
                CASE WHEN p.CurrencyRate   = s.RateDollar THEN 1 ELSE 0 END +
                CASE WHEN ps.StateCode     = s.StateCode THEN 1 ELSE 0 END
                DESC
        ) p
        JOIN dbo.CompanyProduct cp WITH (NOLOCK)
            ON cp.Deleted = 0
           AND cp.CompanyIssuerId = ?
        JOIN dbo.CompanyProductPlan cpp WITH (NOLOCK)
            ON cpp.Deleted = 0
           AND cpp.CompanyProductId = cp.CompanyProductId
           AND p.CompanyProductPlanId = cpp.CompanyProductPlanId
        WHERE s.LoadTaskId = ?;
        """

        map_df = pd.read_sql(sql, conn, params=[company_issuer_id, load_task_id]).fillna("")
        if map_df.empty:
            print("[ManhattanLife] SQL mapping returned 0 rows — skipping.")
            return out_df

    src_key = raw_df[plan_code_col].astype(str).str.strip()
    map_df = map_df.drop_duplicates(subset=["PlanCode"]).copy()
    map_df.index = map_df["PlanCode"].astype(str).str.strip()

    mapped_policy = src_key.map(map_df["PolicyNumber"]).fillna("")
    mapped_name   = src_key.map(map_df["ProductName"]).fillna("")

    if "ProductType" not in out_df.columns:
        out_df["ProductType"] = ""
    if "PlanName" not in out_df.columns:
        out_df["PlanName"] = ""

    out_df["ProductType"] = mapped_policy.where(mapped_policy.ne(""), out_df["ProductType"])
    out_df["PlanName"]    = mapped_name.where(mapped_name.ne(""), out_df["PlanName"])

    print(f"[ManhattanLife] Populated ProductType/PlanName for {sum(mapped_policy.ne(''))} records.")
    return out_df
