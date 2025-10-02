import os
from pathlib import Path
from datetime import datetime
import subprocess
from flask import Flask, render_template_string, request, send_from_directory

app = Flask(__name__)

# --- Paths ---
BASE = Path(__file__).resolve().parent
INBOUND_DIR = BASE / "inbound"
OUTBOUND_DIR = BASE / "outbound"
BAT_FILE = BASE / "run_process_refactored_llm.bat"  # <-- your .bat

INBOUND_DIR.mkdir(parents=True, exist_ok=True)
OUTBOUND_DIR.mkdir(parents=True, exist_ok=True)

# --- Page (single-file template) ---
PAGE = r"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>CSV Processor (.bat)</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 32px; background:#f7f8fb; }
    .wrap { max-width: 980px; margin: 0 auto; }
    .card { background: #fff; border: 1px solid #e6e6e6; border-radius: 12px; padding: 18px 22px; box-shadow: 0 6px 20px rgba(0,0,0,0.06); }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    label { display:block; margin:.6rem 0 .25rem; font-weight:600; }
    input, select { width:100%; padding:.55rem .65rem; border:1px solid #d0d7de; border-radius:8px; }
    button { padding:.6rem 1rem; border:0; border-radius:8px; background:#2563eb; color:#fff; font-weight:600; margin-top:14px; }
    button:hover { background:#1e50c6; cursor:pointer; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .note { color:#566; font-size:.92rem; margin-top:8px }
    pre { background:#0f172a; color:#e2e8f0; padding:14px; border-radius:10px; overflow:auto; max-height:380px; }
    .ok { color:#0a7a22; font-weight:700; }
    .bad { color:#a21616; font-weight:700; }
    .dl { display:inline-block; margin:8px 0 0; padding:.45rem .75rem; border:1px solid #1f883d; color:#1f883d; text-decoration:none; border-radius:8px; }
    .dl:hover { background:#e9f6ec; }
    .mt { margin-top:18px }
  </style>
</head>
<body>
<div class="wrap">
  <h2>Upload CSV & Run Process (.bat)</h2>
  <div class="row">
    <div class="card">
      <form method="post" enctype="multipart/form-data">
        <label>CSV File</label>
        <input type="file" name="file" accept=".csv" required>

        <label>File Location (sent to .bat)</label>
        <input type="text" name="file_location" value="{{ inbound_dir }}" required>

        <label>File Name (sent to .bat)</label>
        <input type="text" name="file_name" value="manhattan_life_raw_data.csv" required>

        <label>TranDate</label>
        <input type="text" name="trandate" value="{{ default_trandate }}" required>

        <label>PayCode</label>
        <input type="text" name="paycode" value="AWM01" required>

        <label>Issuer</label>
        <select name="issuer" required>
          {% for i in issuers %}
            <option value="{{ i }}">{{ i }}</option>
          {% endfor %}
        </select>

        <div class="note">
          Note: after you click <b>Run Process</b>, your selections are shown below and the
          batch file is executed. Command and logs will appear in the results area.
        </div>

        <button type="submit">Run Process</button>
      </form>
    </div>

    <div class="card">
      <h3>Results</h3>

      {% if selections %}
        <div class="mt">
          <div><b>Selections</b></div>
          <pre class="mono">{{ selections }}</pre>
        </div>
      {% endif %}

      {% if cmd %}
        <div class="mt">
          <div><b>Command</b></div>
          <pre class="mono">{{ cmd }}</pre>
        </div>
      {% endif %}

      {% if ran %}
        <div class="mt">
          <div><b>Status</b>:
            {% if success %}
              <span class="ok">Success</span>
            {% else %}
              <span class="bad">Failed</span>
            {% endif %}
            (exit code {{ exit_code }})
          </div>
        </div>
      {% endif %}

      {% if out_name and success %}
        <div class="mt">
          <a class="dl" href="/download/{{ out_name }}">Download {{ out_name }}</a>
        </div>
      {% endif %}

      {% if stdout or stderr %}
        <div class="mt"><b>Logs</b></div>
        <pre class="mono">--- STDOUT ---{{ '\n' + stdout if stdout else '' }}{% if stderr %}\n\n--- STDERR ---\n{{ stderr }}{% endif %}</pre>
      {% endif %}
    </div>
  </div>
</div>

</body>
</html>
"""

ISSUERS = ["manhattan_life", "ameritas", "molina"]

@app.route("/", methods=["GET", "POST"])
def index():
    # Defaults for page render
    context = {
        "inbound_dir": str(INBOUND_DIR.resolve()),
        "default_trandate": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "issuers": ISSUERS,
        "selections": "",
        "cmd": "",
        "ran": False,
        "success": False,
        "exit_code": None,
        "stdout": "",
        "stderr": "",
        "out_name": "",
    }

    if request.method == "POST":
        # --- Save uploaded file ---
        up = request.files.get("file")
        saved_path = ""
        if up and up.filename:
            saved_path = INBOUND_DIR / up.filename
            up.save(saved_path.as_posix())

        # --- Collect form args ---
        file_location = request.form.get("file_location", str(INBOUND_DIR.resolve())).strip()
        file_name     = request.form.get("file_name", up.filename if up else "").strip()
        trandate      = request.form.get("trandate", "").strip()
        paycode       = request.form.get("paycode", "").strip()
        issuer        = request.form.get("issuer", "").strip()

        # Show the selections (and where file was saved)
        context["selections"] = (
            f"Saved file: {saved_path}\n"
            f"file_location: {file_location}\n"
            f"file_name:     {file_name}\n"
            f"trandate:      {trandate}\n"
            f"paycode:       {paycode}\n"
            f"issuer:        {issuer}\n"
        )

        # --- Build and run .bat ---
        # cmd /c <bat> "<inbound>" "<filename>" "<trandate>" "<paycode>" "<issuer>"
        cmd = [
            "cmd", "/c",
            str(BAT_FILE),
            file_location,
            file_name,
            trandate,
            paycode,
            issuer
        ]
        context["cmd"] = " ".join(f'"{c}"' if " " in c else c for c in cmd)

        # Run in BASE so any relative paths inside the .bat/.py work
        proc = subprocess.run(
            cmd,
            cwd=BASE,
            capture_output=True,
            text=True,
            shell=False
        )

        context["ran"] = True
        context["exit_code"] = proc.returncode
        context["stdout"] = proc.stdout or ""
        context["stderr"] = proc.stderr or ""

        # Check for the expected output produced by your pipeline
        out_path = OUTBOUND_DIR / f"{issuer}_standard_template.csv"
        context["success"] = (proc.returncode == 0) and out_path.exists()
        if context["success"]:
            context["out_name"] = out_path.name

    return render_template_string(PAGE, **context)

@app.route("/download/<path:filename>")
def download(filename: str):
    return send_from_directory(OUTBOUND_DIR.as_posix(), filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
