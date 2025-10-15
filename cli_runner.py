# cli_runner.py
import sys
import argparse

# If your file is named processor.py (as referenced by app.py), use this:
from llm_processor import run_llm_pipeline
# If your file is instead named llm_processor.py, comment the line above and uncomment below:
# from llm_processor import run_llm_pipeline


def main():
    p = argparse.ArgumentParser(
        description="Run LLM mapping pipeline via CLI (no Flask)."
    )
    p.add_argument("--issuer", required=True, help="Carrier key, e.g. molina")
    p.add_argument("--paycode", required=True, help="e.g. FromApp/Monthly/Rebill/etc.")
    p.add_argument("--trandate", required=True, help="YYYY-MM-DD (will be written to output)")
    p.add_argument("--csv_path", required=True, help="Path to input CSV")
    p.add_argument("--template_dir", required=True, help="Folder with <issuer>_prompt.txt and <issuer>_rules.json")
    args = p.parse_args()

    def _log(line: str):
        # stream logs immediately to console
        print(line, flush=True)

    try:
        out_path = run_llm_pipeline(
            issuer=args.issuer,
            paycode=args.paycode,
            trandate=args.trandate,
            csv_path=args.csv_path,
            template_dir=args.template_dir,
            log=_log,
        )
        print(f"OUTPUT_FILE={out_path}", flush=True)
        return 0
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
