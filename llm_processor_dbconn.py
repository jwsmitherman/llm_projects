# ======================================================
# llm_processor.py  (simplified for brevity)
# ======================================================

import os
import pandas as pd
# ... your other imports (LangChain, OpenAI, etc.)

# Add this import
from manhattan_mapping import is_manhattan_issuer, apply_manhattan_mapping

# ======================================================
# 1. Your existing LLM pipeline logic
# ======================================================

def run_llm_pipeline(df: pd.DataFrame, issuer: str):
    # --- existing setup ---
    print(f"[INFO] Running pipeline for issuer: {issuer}")

    # all your existing logic, transformations, and LLM processing
    # producing the final out_df
    out_df = your_llm_processing_function(df, issuer)

    # ======================================================
    # 2. Post-processing: Manhattan Life mapping
    # ======================================================
    if is_manhattan_issuer(issuer):
        print("[INFO] Manhattan Life detected. Fetching SQL mapping...")
        try:
            out_df = apply_manhattan_mapping(
                out_df=out_df,
                raw_df=df,  # original raw input with PlanCode
                plan_code_col="PlanCode",
                load_task_id=int(os.getenv("MANHATTAN_LOAD_TASK_ID", "13449")),
                company_issuer_id=int(os.getenv("MANHATTAN_COMPANY_ISSUER_ID", "2204")),
                server=os.getenv("SQL_SERVER", "QWVIDBSQLB401.ngquotit.com"),
                database=os.getenv("SQL_DATABASE", "NGCS"),
                driver=os.getenv("SQL_DRIVER", "ODBC Driver 18 for SQL Server"),
                encrypt=os.getenv("SQL_ENCRYPT", "no").lower() in ("1","true","yes","y"),
                trust_server_certificate=os.getenv("SQL_TRUST_SERVER_CERT", "yes").lower() in ("1","true","yes","y"),
            )
        except Exception as e:
            print(f"[WARN] Manhattan Life mapping failed: {e}")

    # ======================================================
    # 3. Return or export results (existing behavior)
    # ======================================================
    return out_df


# Example usage:
if __name__ == "__main__":
    # Pretend we have a loaded dataframe and issuer
    df = pd.read_csv("input_file.csv")
    issuer = "Manhattan Life"
    final_df = run_llm_pipeline(df, issuer)
    final_df.to_csv("output_results.csv", index=False)
    print("[DONE] Processing complete.")
