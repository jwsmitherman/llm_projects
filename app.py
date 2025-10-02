from flask import Flask, request, render_template_string, send_from_directory
from pathlib import Path
from datetime import datetime
import subprocess
import os

app = Flask(__name__)

# -----------------------
# Config (edit if needed)
# -----------------------
BASE_DIR = Path(__file__).resolve().parent
INBOUND_DIR = BASE_DIR / "inbound"
OUTBOUND_DIR = BASE_DIR / "outbound"
BAT_FILE = BASE_DIR / "run_process_refactored_llm.bat"  # your .bat file

INBOUND_DIR.mkdir(parents=True, exist_ok=True)
OUTBOUND_DIR.mkdir(parents=True, exist_ok=True)

ISSUERS = ["manhattan_life", "ameritas", "molina"]  # adjust to your carriers

# -----------------------
# HTML (Bootstrap + a bit of JS)
# -----------------------
PAGE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Carrier CSV Processor (.bat)</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <!-- Bootstrap CSS (no auth, no secret key needed) -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body { padding: 2rem; background: #f7f8fb; }
    .card { border-radius: 16px; box-shadow: 0 6px 20px rgba(0,0,0,0.06); }
    .form-label { font-weight: 600; }
    .help { color:#6c757d; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    pre { background:#0f172a; color:#e2e8f0; padding:1rem; border-radius:12px; max-height: 360px; overflow:auto; }
    .spinner { display:none; }
  </style>
</head>
<body>
<div class="container-lg">
  <div class="row g-4">
    <div class="col-12 col-xl-7">
      <div class="card p-4">
        <h3 class="mb-3">Upload & Run (.bat)</h3>
        <form id="proc-form" method="post" enctype="multipart/form-data" class="row gy-3">
          <div class="col-12">
            <label class="form-label">CSV File</label>
            <input class="form-control" type="file" name="file" id="file" accept=".csv" required>
            <div class="form-text">File will be saved to <span class="mono">{{ inbound_dir }}</span></div>
          </div>

          <div class="col-12 col-md-6">
            <label class="form-label">File Name (auto-filled from upload)</label>
            <input class="form-control" type="text" name="file_name" id="file_name" placeholder="manhattan_life_raw_data.csv" required>
          </div>

          <div class="col-12 col-md-6">
            <label class="form-label">Tran Date (ISO)</label>
            <input class="form-control" type="text" name="trandate" id="trandate" value="{{ default_trandate }}" placeholder="YYYY-MM-DDTHH:MM:SS" required>
          </div>

          <div class="col-12 col-md-6">
            <label class="form-label">Pay Code</label>
            <input class="form-control" type="text" name="paycode" id="paycode" value="AWM01" required>
          </div>

          <div class="col-12 col-md-6">
            <label class="form-label">Issuer</label>
            <select class="form-select" name="issuer" id="issuer" required>
              {% for i in issuers %}
                <option value="{{ i }}">{{ i }}</option>
              {% endfor %}
            </select>
          </div>

          <!-- Hidden: file-location derived from server config -->
          <input type="hidden" name="file_location" value="{{ inbound_dir }}">

          <div class="col-12 d-flex align-items-center gap-3">
            <button class="btn btn-primary px-4" type="submit" id="run-btn">
              <span class="spinner-border spinner-border-sm me-2 spinner" id="spinner" role="status" aria-hidden="true"></span>
              Run Process
            </button>
            <span class="help">Batch invoked as:<br><span class="mono">run_process_refactored_llm.bat "&lt;inbound&gt;" "&lt;filename&gt;" "&lt;trandate&gt;" "&lt;paycode&gt;" "&lt;issuer&gt;"</span></span>
          </div>
        </form>
      </div>
    </div>

    <div class="col-12 col-xl-5">
      <div class="card p-4">
        <h4 class="mb-3">Result</h4>

        {% if ran %}
        <div class="alert {{ 'alert-success' if success else 'alert-danger' }}" role="alert">
          {{ 'Success — output found.' if success else 'Failed — see logs below.' }}
        </div>
        {% endif %}

        {% if out_name and success %}
          <div class="mb-3">
            <a class="btn btn-outline-success" href="/download/{{ out_name }}">Download {{ out_name }}</a>
          </div>
        {% endif %}

        {% if shown_cmd %}
          <div class="mb-2"><strong>Command</strong></div>
          <pre class="mono">{{ shown_cmd }}</pre>
        {% endif %}

        {% if stdout or stderr %}
          <div class="mb-2"><strong>Logs</strong></div>
          <pre class="mono">--- STDOUT ---{{ '\n' + stdout if stdout else '' }}{% if stderr %}\n\n--- STDERR ---\n{{ stderr }}{% endif %}</pre>
        {% endif %}
      </div>
    </div>
  </div>
</div>

<script>
  // Auto-fill file_name from chosen file
  const fileInput = document.getElementById('file');
  const fileName = document.getElementById('file_name');
  fileInput?.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      fileName.value = e.target.files[0].name;
    }
  });

  // Show spinner and disable button on submit
  const form = document.getElementById('proc-form');
  const btn = document.getElementById('run-btn');
  const spn = document.getElementById('spinner');
  form?.addEventListener('submit', () => {
    btn.disabled = true;
    spn.style.display = 'inline-block';
  });
</script>
</body>
</html>
"""

# -----------------------
# Routes
# -----------------------
@app.route("/", methods=["GET", "POST"])
def main():
    ran = False
    success = False
    stdout = ""
    stderr = ""
    shown_cmd = ""
    out_name = ""

    if request.method == "POST":
        # 1) Save uploaded file
        up = request.files.get("file")
        if up and up.filename:
            save_path = INBOUND_DIR / up.filename
            up.save(save_path.as_posix())

        # 2) Collect params
        file_location = request.form.get("file_location", str(INBOUND_DIR.resolve())).strip()
        file_name     = request.form.get("file_name", up.filename if up else "").strip()
        trandate      = request.form.get("trandate", "").strip()
        paycode       = request.form.get("paycode", "").strip()
        issuer        = request.form.get("issuer", "").strip()

        # 3) Build .bat command:
        # cmd /c run_process_refactored_llm.bat "<inbound>" "<filename>" "<trandate>" "<paycode>" "<issuer>"
        cmd = [
            "cmd", "/c",
            str(BAT_FILE),
            file_location,
            file_name,
            trandate,
            paycode,
            issuer
        ]
        shown_cmd = " ".join(f'"{c}"' if " " in c else c for c in cmd)

        # 4) Run in project directory so all relative paths work
        completed = subprocess.run(
            cmd,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            shell=False
        )
        ran = True
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""

        # 5) Expected output path: ./outbound/{issuer}_standard_template.csv
        out_path = OUTBOUND_DIR / f"{issuer}_standard_template.csv"
        success = (completed.returncode == 0) and out_path.exists()
        if success:
            out_name = out_path.name

    return render_template_string(
        PAGE,
        inbound_dir=str(INBOUND_DIR.resolve()),
        default_trandate=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        issuers=ISSUERS,
        ran=ran,
        success=success,
        stdout=stdout,
        stderr=stderr,
        shown_cmd=shown_cmd,
        out_name=out_name,
    )

@app.route("/download/<path:filename>")
def download(filename: str):
    return send_from_directory(OUTBOUND_DIR.as_posix(), filename, as_attachment=True)

if __name__ == "__main__":
    # No secret key needed—no sessions/flash used.
    app.run(host="127.0.0.1", port=5000, debug=True)
