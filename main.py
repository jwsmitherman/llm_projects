import os, io, json, threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Set

from flask import Flask, jsonify, request
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.exceptions import AzureError

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

def blob_exists(cc, blob_name) -> bool:
    try:
        cc.get_blob_client(blob_name).get_blob_properties()
        return True
    except Exception:
        return False

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

# ── Background worker ─────────────────────────────────────────────────────────

def _process_in_background(normalized: list, blob_name: str,
                            inbound_container: str, outbound_container: str):
    """
    Runs in a daemon thread so /save can return 202 immediately.
    Writes a 'status' blob so /results can report pending vs success vs error.
    """
    status_blob  = blob_name.replace(".json", "-status.json")
    outbound_cc  = _get_or_create_container(outbound_container)

    # Mark as processing
    save_json_to_blob(outbound_cc, status_blob,
                      {"status": "processing", "started_at": utc_now_iso()})
    try:
        prompts = _load_prompts_from_blob()

        from build_benefits import run_benefit_processing
        processed_results = run_benefit_processing(normalized, prompts)

        save_json_to_blob(outbound_cc, blob_name, processed_results)
        save_json_to_blob(outbound_cc, status_blob, {
            "status":       "success",
            "completed_at": utc_now_iso(),
            "result_count": len(processed_results),
            "blob":         blob_name,
        })

    except Exception as e:
        save_json_to_blob(outbound_cc, status_blob, {
            "status":   "error",
            "error":    str(e),
            "blob":     blob_name,
        })

# ── POST /save  — accepts payload, returns 202 immediately ───────────────────

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

        # Save inbound payload to blob synchronously
        inbound_cc = _get_or_create_container(inbound_container)
        save_json_to_blob(inbound_cc, blob_name, normalized)

        # Launch LLM processing in background thread — do NOT block HTTP response
        t = threading.Thread(
            target=_process_in_background,
            args=(normalized, blob_name, inbound_container, outbound_container),
            daemon=True,
        )
        t.start()

        # Return 202 Accepted immediately — client polls /results/<load_id>
        return jsonify({
            "status":    "accepted",
            "load_id":   load_id,
            "blob_name": blob_name,
            "message":   "Processing started. Poll GET /results/<load_id> for status.",
            "poll_url":  f"/results/{load_id}",
            "inbound_location": {"container": inbound_container, "blob": blob_name},
        }), 202

    except AzureError as e:
        return jsonify({"status": "error", "detail": f"Azure error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


# ── GET /results/<load_id>  — returns status or final results ─────────────────
#
# Response shapes:
#   {"status": "processing"}                          — LLM still running
#   {"status": "success", "result_count": N, "results": [...]}  — done
#   {"status": "error",   "error": "..."}             — something failed
#   {"status": "not_found"}                           — unknown load_id

@app.route("/results/<load_id>", methods=["GET"])
def get_results(load_id):
    try:
        outbound_container = os.getenv("BLOB_OUTBOUND_CONTAINER", "outbound")
        cc = _get_or_create_container(outbound_container)

        # Find the most recent blob for this load_id
        blobs = sorted(
            [b for b in cc.list_blobs(name_starts_with=f"{load_id}-")
             if not b.name.endswith("-status.json")],
            key=lambda b: b.last_modified,
        )
        status_blobs = sorted(
            [b for b in cc.list_blobs(name_starts_with=f"{load_id}-")
             if b.name.endswith("-status.json")],
            key=lambda b: b.last_modified,
        )

        if not status_blobs and not blobs:
            return jsonify({"status": "not_found", "load_id": load_id}), 404

        # Read the status blob
        if status_blobs:
            status_info = read_json_from_blob(cc, status_blobs[-1].name)
            job_status  = status_info.get("status")

            if job_status == "processing":
                return jsonify({
                    "status":     "processing",
                    "load_id":    load_id,
                    "started_at": status_info.get("started_at"),
                    "message":    "LLM is still processing. Try again in 15 seconds.",
                }), 202

            if job_status == "error":
                return jsonify({
                    "status":  "error",
                    "load_id": load_id,
                    "error":   status_info.get("error"),
                }), 500

            if job_status == "success" and blobs:
                data = read_json_from_blob(cc, blobs[-1].name)
                return jsonify({
                    "status":       "success",
                    "load_id":      load_id,
                    "blob":         blobs[-1].name,
                    "result_count": len(data),
                    "results":      data,
                }), 200

        # Fallback: status blob missing but result blob exists (legacy runs)
        if blobs:
            data = read_json_from_blob(cc, blobs[-1].name)
            return jsonify({
                "status":       "success",
                "load_id":      load_id,
                "blob":         blobs[-1].name,
                "result_count": len(data),
                "results":      data,
            }), 200

        return jsonify({"status": "not_found", "load_id": load_id}), 404

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
