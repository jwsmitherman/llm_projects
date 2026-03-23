import os, io, json
from datetime import datetime, timezone
from typing import Any, Dict, List, Set

from flask import Flask, jsonify, request
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.exceptions import AzureError

# build_benefits (langchain/openai/pandas) is imported lazily inside /save
# so Flask boots successfully before heavy packages finish installing.

print("DEBUG - BLOB_CONNECTION_STRING =", os.getenv("BLOB_CONNECTION_STRING"))

app = Flask(__name__)

# ── Blob helpers ──────────────────────────────────────────────────────────────

def _get_service_client():
    conn_str = os.getenv("BLOB_CONNECTION_STRING")
    if not conn_str:
        raise ValueError("BLOB_CONNECTION_STRING is not set")
    return BlobServiceClient.from_connection_string(conn_str)

def _get_or_create_container(name):
    svc = _get_service_client()
    cc  = svc.get_container_client(name)
    try: cc.create_container()
    except AzureError: pass
    return cc

def save_json_to_blob(cc, blob_name, data):
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
    cc.upload_blob(name=blob_name, data=io.BytesIO(payload), overwrite=True,
                   content_settings=ContentSettings(content_type="application/json; charset=utf-8"))

def read_json_from_blob(cc, blob_name):
    return json.loads(cc.download_blob(blob_name).readall().decode("utf-8"))

def read_text_from_blob(cc, blob_name):
    return cc.download_blob(blob_name).readall().decode("utf-8")

def utc_now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# ── Prompt loader ─────────────────────────────────────────────────────────────

PROMPT_FILES = {
    "system_prompt":     "system_prompt.txt",
    "few_shot_examples": "few_shot_examples.txt",
    "human_template":    "human_template.txt",
}

def _load_prompts_from_blob():
    cc = _get_or_create_container(os.getenv("BLOB_PROMPTS_CONTAINER", "prompts"))
    return {key: read_text_from_blob(cc, fname) for key, fname in PROMPT_FILES.items()}

# ── Required PBP columns ──────────────────────────────────────────────────────

REQUIRED_COLS = ["LoadID", "FileName", "ID", "header", "category", "field", "value", "DT"]

# ── POST /save ────────────────────────────────────────────────────────────────

@app.route("/save", methods=["POST"])
def save_json_payload():
    try:
        if not request.is_json:
            return jsonify({"status": "error", "detail": "Request must be JSON"}), 400
        body = request.get_json(silent=True)
        if body is None:
            return jsonify({"status": "error", "detail": "Invalid JSON body"}), 400
        if not isinstance(body, list):
            return jsonify({"status": "error", "detail": "Body must be a JSON array"}), 400

        # Validate rows
        normalized: List[Dict[str, Any]] = []
        for idx, row in enumerate(body):
            if not isinstance(row, dict):
                return jsonify({"status": "error", "detail": f"Row {idx} is not an object"}), 400
            missing = [c for c in REQUIRED_COLS if c not in row]
            if missing:
                return jsonify({"status": "error", "detail": f"Row {idx} missing: {missing}"}), 400
            normalized.append({c: row.get(c) for c in REQUIRED_COLS})

        # Extract LoadID
        load_ids: Set[str] = {str(r["LoadID"]).strip() for r in normalized if r.get("LoadID")}
        if not load_ids:
            return jsonify({"status": "error", "detail": "No LoadID found"}), 400
        if len(load_ids) > 1:
            return jsonify({"status": "error", "detail": f"Multiple LoadIDs: {sorted(load_ids)}"}), 400

        load_id   = list(load_ids)[0]
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        blob_name = f"{load_id}-{timestamp}.json"

        inbound_container  = os.getenv("BLOB_INBOUND_CONTAINER",  "payloads")
        outbound_container = os.getenv("BLOB_OUTBOUND_CONTAINER", "outbound")
        inbound_cc         = _get_or_create_container(inbound_container)
        outbound_cc        = _get_or_create_container(outbound_container)

        # Step 1: Save inbound PBP payload to blob
        save_json_to_blob(inbound_cc, blob_name, normalized)

        # Step 2: Load prompts from blob (system_prompt, few_shot_examples, human_template)
        prompts = _load_prompts_from_blob()

        # Step 3: Call LLM via build_benefits.py
        # THIS IS WHERE THE LLM IS CALLED.
        # build_benefits.run_benefit_processing() assembles prompts,
        # calls Azure OpenAI with the PBP data, and returns benefit rows.
        # Imported lazily so startup stays lightweight.
        from build_benefits import run_benefit_processing
        processed_results = run_benefit_processing(normalized, prompts)

        # Step 4: Save results to outbound blob
        save_json_to_blob(outbound_cc, blob_name, processed_results)

        return jsonify({
            "status":  "success",
            "load_id": load_id,
            "message": "Processed. Retrieve via /results/<load_id>.",
            "inbound_location":  {"container": inbound_container,  "blob": blob_name},
            "outbound_location": {"container": outbound_container, "blob": blob_name},
        }), 201

    except AzureError as e:
        return jsonify({"status": "error", "detail": f"Azure error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500

# ── GET /results/<load_id> ────────────────────────────────────────────────────

@app.route("/results/<load_id>", methods=["GET"])
def get_results(load_id):
    try:
        cc    = _get_or_create_container(os.getenv("BLOB_OUTBOUND_CONTAINER", "outbound"))
        blobs = list(cc.list_blobs(name_starts_with=f"{load_id}-"))
        if not blobs:
            return jsonify({"status": "error", "detail": "No results found"}), 404
        blob = sorted(blobs, key=lambda b: b.last_modified)[-1]
        data = json.loads(cc.get_blob_client(blob.name).download_blob().readall())
        return jsonify({"status": "success", "load_id": load_id, "blob": blob.name,
                        "result_count": len(data), "results": data}), 200
    except AzureError as e:
        return jsonify({"status": "error", "detail": f"Azure error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500

# ── GET / — health ────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "timestamp": utc_now_iso()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
