import os, time
from pathlib import Path
import argparse
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import pandas as pd
from werkzeug.utils import secure_filename

from config import PREVIEW_ROWS  # optional, if you kept config.py; else set PREVIEW_ROWS = 1000

BASE = Path(__file__).resolve().parent
UPLOADS = BASE / "uploads"
OUTPUTS = BASE / "outputs"
UPLOADS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

ALLOWED = {"csv", "xlsx", "xls"}

def allowed(fname: str) -> bool:
    return "." in fname and fname.rsplit(".", 1)[1].lower() in ALLOWED

def read_any_table(path: Path, nrows: int | None = None) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path, nrows=nrows)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path, nrows=nrows)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")

@app.route("/", methods=["GET", "POST"])
def index():
    logs = []
    input_html = None
    result_html = None
    result_download = None

    if request.method == "POST":
        parsed = request.files.get("parsed_pbp")  # <-- only this input now
        if not parsed or parsed.filename == "":
            flash("Please upload PARSED_PBP (CSV/XLSX).")
            return redirect(url_for("index"))

        if not allowed(parsed.filename):
            flash("Only CSV/XLSX files are allowed for PARSED_PBP.")
            return redirect(url_for("index"))

        parsed_path = UPLOADS / secure_filename(parsed.filename)
        parsed.save(parsed_path)

        # --- Simulated LLM pipeline logs with ~3s total delay (10 steps × 0.3s)
        steps = [
            "1/10: Upload received; validating file type & size…",
            "2/10: Loading PARSED_PBP into DataFrame…",
            "3/10: Normalizing headers (snake_case)…",
            "4/10: Sampling rows to profile field patterns…",
            "5/10: Initializing AzureChatOpenAI client…",
            "6/10: Building mapping prompt (source→target schema)…",
            "7/10: Getting LLM mapping JSON; validating & repairing…",
            "8/10: Applying synonyms + fuzzy fallback for missing fields…",
            "9/10: Coercing numeric/percent cost-sharing fields…",
            "10/10: Finalizing output & saving translated results…",
        ]
        for line in steps:
            logs.append(line)
            time.sleep(0.3)  # ~3.0s total

        # --- Input preview (top table, always from uploaded parsed file)
        try:
            df_in = read_any_table(parsed_path, nrows=None)
            input_html = df_in.head(PREVIEW_ROWS).to_html(
                classes="table table-sm table-striped table-hover", index=False
            )
        except Exception as e:
            logs.append(f"Error reading PARSED_PBP: {e}")
            input_html = f"<div class='text-danger'>Error reading PARSED_PBP: {e}</div>"

        # --- Results table (always from benefits_translated_llm.* if present)
        # Priority 1: if user also uploaded a results file under the SAME name? (not part of this flow)
        # Priority 2: look for outputs/benefits_translated_llm.(csv|xlsx|xls)
        found = None
        for candidate in [
            OUTPUTS / "benefits_translated_llm.csv",
            OUTPUTS / "benefits_translated_llm.xlsx",
            OUTPUTS / "benefits_translated_llm.xls",
        ]:
            if candidate.exists():
                found = candidate
                break

        if found is None:
            # As a convenience, also check if a file with that name was placed in uploads/
            for candidate in [
                UPLOADS / "benefits_translated_llm.csv",
                UPLOADS / "benefits_translated_llm.xlsx",
                UPLOADS / "benefits_translated_llm.xls",
            ]:
                if candidate.exists():
                    found = candidate
                    break

        if found is not None:
            try:
                df_res = read_any_table(found, nrows=None)
                result_html = df_res.head(PREVIEW_ROWS).to_html(
                    classes="table table-sm table-striped table-hover", index=False
                )
                # Copy to outputs if it lives in uploads so the download route works
                if found.parent != OUTPUTS:
                    out_path = OUTPUTS / found.name
                    df_res.to_csv(out_path, index=False) if found.suffix.lower() == ".csv" else df_res.to_excel(out_path, index=False)
                    found = out_path
                result_download = found.name
                logs.append(f"Loaded results from {found.name}.")
            except Exception as e:
                logs.append(f"Error reading benefits_translated_llm: {e}")
                result_html = f"<div class='text-danger'>Error reading results: {e}</div>"
        else:
            logs.append("No benefits_translated_llm file found in outputs/ or uploads/.")
            result_html = "<div class='text-warning'>No results file found. Place <b>benefits_translated_llm.csv</b> in the <i>outputs/</i> folder and refresh.</div>"

    return render_template(
        "index.html",
        logs=logs,
        input_table_html=input_html,
        results_table_html=result_html,
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
