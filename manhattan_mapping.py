# manhattan_mapping.py
# --------------------------------
# Handles SQL Server connection and retrieval of Manhattan Life plan mapping data.

from __future__ import annotations

import os
import pandas as pd
import pyodbc


def get_manhattan_mapping(
    load_task_id: int,
    company_issuer_id: int,
    log=print
) -> pd.DataFrame:
    """
    Connects to SQL Server via Windows Authentication and returns a mapping DataFrame
    with columns: PlanCode, PolicyNumber, ProductName.

    Reads connection settings from environment variables (with safe defaults):
      - SQL_SERVER (default: QWVIDBSQLB401.ngquotit.com)
      - SQL_DATABASE (default: NGCS)
      - SQL_DRIVER (default: ODBC Driver 18 for SQL Server)
      - SQL_ENCRYPT (default: no)
      - SQL_TRUST_SERVER_CERT (default: yes)
    """

    server = os.getenv("SQL_SERVER", "QWVIDBSQLB401.ngquotit.com")
    database = os.getenv("SQL_DATABASE", "NGCS")
    driver = os.getenv("SQL_DRIVER", "ODBC Driver 18 for SQL Server")  # use "ODBC Driver 17 for SQL Server" if needed
    encrypt = os.getenv("SQL_ENCRYPT", "no").lower() in {"1", "true", "yes", "y"}
    trust   = os.getenv("SQL_TRUST_SERVER_CERT", "yes").lower() in {"1", "true", "yes", "y"}

    conn_str = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        "Trusted_Connection=yes;"
        f"Encrypt={'yes' if encrypt else 'no'};"
        f"TrustServerCertificate={'yes' if trust else 'no'};"
    )

    # NOTE: f-string is used per your requirement (no parameterized query here).
    sql = f"""
    SELECT DISTINCT
        p.IssuerPlanId AS PlanCode,
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
       AND cp.CompanyIssuerId = {company_issuer_id}
    JOIN dbo.CompanyProductPlan cpp WITH (NOLOCK)
        ON cpp.Deleted = 0
       AND cpp.CompanyProductId = cp.CompanyProductId
       AND p.CompanyProductPlanId = cpp.CompanyProductPlanId
    WHERE s.LoadTaskId = {load_task_id};
    """

    log(f"[DB] Connecting to SQL Server → {server} | DB={database}")
    with pyodbc.connect(conn_str) as conn:
        df = pd.read_sql(sql, conn).fillna("")

    # Enforce expected columns (use empty if missing)
    for col in ("PlanCode", "PolicyNumber", "ProductName"):
        if col not in df.columns:
            df[col] = ""

    log(f"[DB] Retrieved {len(df):,} Manhattan Life mapping rows.")
    return df[["PlanCode", "PolicyNumber", "ProductName"]]
