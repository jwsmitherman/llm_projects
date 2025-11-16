import pyodbc
import pandas as pd


def build_conn_str(server: str, database: str) -> str:
    return (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={server};"
        f"DATABASE={database};"
        "Trusted_Connection=yes;"
        "Encrypt=no;"
        "TrustServerCertificate=yes;"
    )


def get_manhattan_mapping(
    load_task_id: int,
    company_issuer_id: int,
    *,
    server: str,
    database: str,
    log=print
) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
        PlanCode, PolicyNumber, ProductName

    Uses a simple f-string SQL query and SQL Server connection.
    """

    # Build connection string
    conn_str = build_conn_str(server, database)

    # Build SQL with required IDs
    sql = f"""
        SELECT DISTINCT
            p.IssuerPlanId AS PlanCode,
            s.PolicyNumber,
            cpp.ProductName
        FROM dbo.STGPlanMappingFactors s WITH (NOLOCK)
        OUTER APPLY (
            SELECT TOP 1 p.*
            FROM dbo.PlanMappingIssuerDetail p WITH (NOLOCK)
            LEFT JOIN dbo.PlanMappingIssuerDetailState ps WITH (NOLOCK)
                ON p.PlanMappingIssuerDetailId = ps.PlanMappingIssuerDetailId
            WHERE
                (p.IssuerPlanName = s.PlanName OR p.IssuerPlanName IS NULL OR p.IssuerPlanName = '')
                AND (p.IssuerPlanId   = s.PlanId   OR p.IssuerPlanId IS NULL   OR p.IssuerPlanId = '')
                AND (p.[Year]         = s.[Year]   OR p.[Year] IS NULL)
                AND (p.PercentRate    = s.RatePercent OR p.PercentRate IS NULL)
                AND (p.CurrencyRate   = s.RateDollar  OR p.CurrencyRate IS NULL)
                AND (ps.StateCode     = s.StateCode   OR ps.StateCode IS NULL)
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

    log(f"[ManhattanLife] Connecting to SQL Server {server} / DB {database}")

    try:
        with pyodbc.connect(conn_str) as conn:
            df = pd.read_sql(sql, conn).fillna("")
    except Exception as e:
        log(f"[ManhattanLife] ERROR connecting or querying SQL: {e}")
        return pd.DataFrame(columns=["PlanCode", "PolicyNumber", "ProductName"])

    # Enforce expected columns
    for col in ("PlanCode", "PolicyNumber", "ProductName"):
        if col not in df.columns:
            df[col] = ""

    log(f"[ManhattanLife] Retrieved {len(df)} mapped rows.")
    return df[["PlanCode", "PolicyNumber", "ProductName"]]
