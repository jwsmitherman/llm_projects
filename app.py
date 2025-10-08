# app.py
import os, uuid, threading
from collections import deque
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename

from processor import run_llm_pipeline  # <-- our separate LLM module

UPLOAD_DIR   = Path("./uploads")
OUTBOUND_DIR = Path("./outbound")
ALLOWED_EXTS = {"csv"}
MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200MB

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTBOUND_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR.as_posix()

# In-memory job store
jobs = {}  # job_id -> {"logs": deque, "status": "running|done|error", "output_path": str|None}

def log(job_id: str, msg: str):
    now = datetime.now().strftime("%H:%M:%S")
    jobs[job_id]["logs"].append(f"[{now}] {msg}")

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>LLM Mapping Runner</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #f7f7f8; }
    .wrap { display: grid; grid-template-columns: 420px 1fr; gap: 12px; padding: 16px; }
    .card { background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; }
    .title { font-weight: 700; margin-bottom: 8px; }
    label { display:block; font-size: 14px; margin-top: 12px; }
    input[type="text"], input[type="date"], input[type="file"] {
      width: 100%; padding: 10px; border: 1px solid #d1d5db; border-radius: 8px; background: #fff;
    }
    .btn { margin-top: 16px; padding: 10px 14px; background: #111827; color:#fff; border: none; border-radius: 8px; cursor: pointer; }
    .btn:disabled { opacity: .5; cursor: not-allowed; }
    pre { white-space: pre-wrap; background:#0b1020; color:#d9e1f2; padding: 12px; border-radius:8px; height: 70vh; overflow:auto; }
    .hint { color:#6b7280; font-size:12px; margin-top:6px; }
    .ok { color: #065f46; }
    .err { color: #991b1b; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="title">Run LLM Mapping</div>
      <form id="form">
        <label>Issuer
          <input type="text" name="issuer" placeholder="molina / ameritas / manhattan_life" required />
        </label>
        <label>Pay Code
          <input type="text" name="paycode" placeholder="FromApp" required />
        </label>
        <label>Tran Date
          <input type="date" name="trandate" required />
        </label>
        <label>Template Folder (contains &lt;issuer&gt;_prompt.txt and &lt;issuer&gt;_rules.json)
          <input type="text" name="template_dir" placeholder="./carrier_prompts" required />
        </label>
        <label>Upload CSV
          <input type="file" name="file" accept=".csv" required />
        </label>
        <button class="btn" id="startBtn" type="submit">Start</button>
      </form>
      <div id="result" class="hint"></div>
    </div>

    <div class="card">
      <div class="title">Logs</div>
      <pre id="logs">Waiting to start…</pre>
    </div>
  </div>

  <script>
    const form = document.getElementById('form');
    const startBtn = document.getElementById('startBtn');
    const logsEl = document.getElementById('logs');
    const resultEl = document.getElementById('result');
    let pollTimer = null;
    let currentJob = null;

    function appendLogs(lines) {
      if (logsEl.textContent === 'Waiting to start…') logsEl.textContent = '';
      logsEl.textContent += (Array.isArray(lines) ? lines.join('\\n') : lines) + '\\n';
      logsEl.scrollTop = logsEl.scrollHeight;
    }

    async function poll(jobId) {
      try {
        const r = await fetch('/logs?job_id=' + jobId);
        const data = await r.json();
        if (data.logs && data.logs.length) appendLogs(data.logs);
        if (data.status === 'done') {
          clearInterval(pollTimer); pollTimer = null;
          resultEl.innerHTML = '<span class="ok">Completed.</span> Output: ' + data.output_path;
          startBtn.disabled = false;
        } else if (data.status === 'error') {
          clearInterval(pollTimer); pollTimer = null;
          resultEl.innerHTML = '<span class="err">Failed.</span> See logs above.';
          startBtn.disabled = false;
        }
      } catch (e) {
        clearInterval(pollTimer); pollTimer = null;
        resultEl.innerHTML = '<span class="err">Error polling logs.</span>';
        startBtn.disabled = false;
      }
    }

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      logsEl.textContent = 'Starting…\\n';
      resultEl.textContent = '';
      startBtn.disabled = true;

      const fd = new FormData(form);
      const r = await fetch('/start', { method: 'POST', body: fd });
      if (!r.ok) {
        startBtn.disabled = false;
        appendLogs('Server error starting job.');
        return;
      }
      const data = await r.json();
      currentJob = data.job_id;
      pollTimer = setInterval(() => poll(currentJob), 800);
    });
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/start", methods=["POST"])
def start():
    issuer = (request.form.get("issuer") or "").strip()
    paycode = (request.form.get("paycode") or "").strip()
    trandate = (request.form.get("trandate") or "").strip()
    template_dir = (request.form.get("template_dir") or "").strip()

    if not (issuer and paycode and trandate and template_dir):
        return jsonify({"error": "Missing required fields"}), 400
    if "file" not in request.files:
        return jsonify({"error": "CSV file missing"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    filename = secure_filename(f.filename)
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext != "csv":
        return jsonify({"error": "Only .csv files are allowed"}), 400

    save_path = UPLOAD_DIR / filename
    f.save(save_path)

    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"logs": deque(maxlen=2000), "status": "running", "output_path": None}
    log(job_id, f"Job created: {job_id}")
    log(job_id, f"Issuer={issuer}  PayCode={paycode}  TranDate={trandate}")
    log(job_id, f"Template dir={template_dir}")
    log(job_id, f"CSV saved: {save_path.as_posix()}")

    def _run():
        try:
            out_path = run_llm_pipeline(
                issuer=issuer,
                paycode=paycode,
                trandate=trandate,
                csv_path=save_path.as_posix(),
                template_dir=template_dir,
                log=lambda line: log(job_id, line),
            )
            jobs[job_id]["output_path"] = out_path
            jobs[job_id]["status"] = "done"
        except Exception as e:
            jobs[job_id]["status"] = "error"
            log(job_id, f"ERROR: {type(e).__name__}: {e}")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({"job_id": job_id})

@app.route("/logs")
def get_logs():
    job_id = request.args.get("job_id", "")
    if job_id not in jobs:
        return jsonify({"error": "Unknown job_id"}), 404
    lines = []
    dq = jobs[job_id]["logs"]
    while dq:
        lines.append(dq.popleft())
    return jsonify({
        "status": jobs[job_id]["status"],
        "output_path": jobs[job_id]["output_path"],
        "logs": lines
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
