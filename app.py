import os, time
from pathlib import Path
import argparse
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import pandas as pd

# ---- constants / paths ----
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"          # <-- use this path for both input and results
OUTPUTS = BASE / "outputs"    # download route points here, we’ll mirror results file into it
OUTPUTS.mkdir(exist_ok=True)

PREVIEW_ROWS = 10  # <= show only 10 rows per table

def read_any_csv(path: Path) -> pd.DataFrame:
    # we only need CSV per your screenshot; easy to extend to xlsx if you add later
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")

@app.route("/", methods=["GET", "POST"])
def index():
    logs = []
    input_html = None
    results_html = None
    result_download = None

    if request.method == "POST":
        # ---- Simulated LLM pipeline logs (~3s total) ----
        steps = [
            "1/10: Locating input at data/parse_pbp_files.csv…",
            "2/10: Loading PARSED_PBP into DataFrame…",
            "3/10: Normalizing headers (snake_case)…",
            "4/10: Sampling rows to profile field patterns…",
            "5/10: Initializing AzureChatOpenAI client…",
            "6/10: Building mapping prompt (source→target schema)…",
            "7/10: Getting LLM mapping JSON; validating & repairing…",
            "8/10: Applying synonyms + fuzzy fallback for missing fields…",
            "9/10: Coercing numeric/percent cost-sharing fields…",
            "10/10: Finalizing output & loading translated results…",
        ]
        for line in steps:
            logs.append(line)
            time.sleep(0.3)  # 10 * 0.3s ~= 3 seconds

        # ---- Input preview (always from data/parse_pbp_files.csv) ----
        parsed_path = DATA / "parse_pbp_files.csv"
        try:
            df_in = read_any_csv(parsed_path)
            input_html = df_in.head(PREVIEW_ROWS).to_html(
                classes="table table-sm table-striped table-hover",
                index=False,
                border=0
            )
            logs.append(f"Loaded PARSED_PBP: {parsed_path.name} ({len(df_in)} rows).")
        except Exception as e:
            input_html = f"<div class='text-danger'>Error reading {parsed_path}: {e}</div>"
            logs.append(f"Error reading input: {e}")

        # ---- Results preview (always from data/benefits_translated_llm.csv) ----
        results_path = DATA / "benefits_translated_llm.csv"
        try:
            df_res = read_any_csv(results_path)
            results_html = df_res.head(PREVIEW_ROWS).to_html(
                classes="table table-sm table-striped table-hover",
                index=False,
                border=0
            )
            # save a copy into outputs/ so the Download button works
            out_copy = OUTPUTS / results_path.name
            df_res.to_csv(out_copy, index=False)
            result_download = out_copy.name
            logs.append(f"Loaded results: {results_path.name} ({len(df_res)} rows).")
        except Exception as e:
            results_html = f"<div class='text-danger'>Error reading {results_path}: {e}</div>"
            logs.append(f"Error reading results: {e}")

    return render_template(
        "index.html",
        logs=logs,
        input_table_html=input_html,
        results_table_html=results_html,
        download_name=result_download,
    )

@app.route("/download/<path:filename>")
def download(filename):
    return send_from_directory((BASE / "outputs"), filename, as_attachment=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "5000")))
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, debug=True)
