import os
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import pandas as pd
from werkzeug.utils import secure_filename

from mapping import translate

BASE = Path(__file__).resolve().parent
UPLOADS = BASE / "uploads"
OUTPUTS = BASE / "outputs"
OUTPUTS.mkdir(exist_ok=True)
UPLOADS.mkdir(exist_ok=True)

ALLOWED = {"csv"}

def allowed(fname):
    return "." in fname and fname.rsplit(".", 1)[1].lower() in ALLOWED

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")

@app.route("/", methods=["GET", "POST"])
def index():
    df_html = None
    logs = []
    out_name = None

    if request.method == "POST":
        parsed = request.files.get("parsed_pbp")
        target = request.files.get("target_benchmark")
        if not parsed or not target or parsed.filename == "" or target.filename == "":
            flash("Please upload both PARSED_PBP and TARGET_BENCHMARK CSV files.")
            return redirect(url_for("index"))

        if not (allowed(parsed.filename) and allowed(target.filename)):
            flash("Only CSV files are allowed.")
            return redirect(url_for("index"))

        parsed_path = UPLOADS / secure_filename(parsed.filename)
        target_path = UPLOADS / secure_filename(target.filename)
        parsed.save(parsed_path)
        target.save(target_path)

        try:
            df_out, logs = translate(str(parsed_path), str(target_path))
            out_name = "benefits_translated_llm.csv"
            out_path = OUTPUTS / out_name
            df_out.to_csv(out_path, index=False)
            df_html = df_out.head(1000).to_html(
                classes="table table-sm table-striped table-hover", index=False
            )
        except Exception as e:
            logs = [f"Error during translation: {e}"]

    return render_template("index.html", table_html=df_html, logs=logs, download_name=out_name)

@app.route("/download/<path:filename>")
def download(filename):
    return send_from_directory((BASE / "outputs"), filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
