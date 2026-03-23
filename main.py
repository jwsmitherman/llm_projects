import os
import io
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Set

from flask import Flask, jsonify, request
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.exceptions import AzureError

# NOTE: build_benefits (langchain/openai/pandas) is imported lazily inside
# the /save route — NOT at module level. This keeps startup lightweight so
# Flask boots successfully regardless of whether heavy packages are installed,
# matching the behaviour of the working backup version.

# Debug print to verify env vars in Azure and local
print("DEBUG - BLOB_CONNECTION_STRING =", os.getenv("BLOB_CONNECTION_STRING"))

app = Flask(__name__)

# ============================================================
# Azure blob helpers
# ============================================================

def _get_service_client() -> BlobServiceClient:
    conn_str = os.getenv("BLOB_CONNECTION_STRING")
    if not conn_str:
        raise ValueError("BLOB_CONNECTION_STRING is not set")
    return BlobServiceClient.from_connection_string(conn_str)


def _get_or_create_container(container_name: str):
    svc = _get_service_client()
    cc = svc.get_container_client(container_name)
    try:
        cc.create_container()
    except AzureError:
        pass  # already exists
    return cc


def save_json_to_blob(container_client, blob_name: str, data_obj: Any) -> None:
    payload_bytes = json.dumps(data_obj, ensure_ascii=False).encode("utf-8")
    container_client.upload_blob(
        name=blob_name,
        data=io.BytesIO(payload_bytes),
        overwrite=True,
        content_settings=ContentSettings(content_type="application/json; charset=utf-8"),
    )


def read_json_from_blob(container_client, blob_name: str) -> Any:
    raw = container_client.download_blob(blob_name).readall()
    return json.loads(raw.decode("utf-8"))


def read_text_from_blob(container_client, blob_name: str) -> str:
    raw = container_client.download_blob(blob_name).readall()
    return raw.decode("utf-8")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ============================================================
# Prompt loader — reads from the "prompts" blob container
# ============================================================

PROMPT_FILES = {
    "system_prompt":    "system_prompt.txt",
    "few_shot_examples": "few_shot_examples.txt",
    "human_template":   "human_template.txt",
}


def _load_prompts_from_blob() -> Dict[str, str]:
    """
    Read all three prompt files from the 'prompts' blob container.
    Returns a dict with keys: system_prompt, few_shot_examples, human_template.
    """
    prompts_container = os.getenv("BLOB_PROMPTS_CONTAINER", "prompts")
    cc = _get_or_create_container(prompts_container)
    return {key: read_text_from_blob(cc, filename) for key, filename in PROMPT_FILES.items()}


# ============================================================
# Benefit rules loader — reads from the "payloads" blob container
# ============================================================

def _load_benefit_rules_from_blob(load_id: str) -> List[Dict]:
    """
    Load the ParseBenefitFileRule rows for the given load_id from blob.
    Expected blob name: benefit_rules_{load_id}.json
    """
    rules_container = os.getenv("BLOB_RULES_CONTAINER", "payloads")
    blob_name = f"benefit_rules_{load_id}.json"
    cc = _get_or_create_container(rules_container)
    return read_json_from_blob(cc, blob_name)


# ============================================================
# Required PBP columns
# ============================================================

REQUIRED_COLS = ["LoadID", "FileName", "ID", "header", "category", "field", "value", "DT"]


# ============================================================
# POST /save  — ingest PBP payload, process, save outbound
# ============================================================

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

        # ── Validate rows ──────────────────────────────────────────────────────
        normalized_rows: List[Dict[str, Any]] = []
        for idx, row in enumerate(body):
            if not isinstance(row, dict):
                return jsonify({"status": "error", "detail": f"Row {idx} is not an object"}), 400
            missing = [col for col in REQUIRED_COLS if col not in row]
            if missing:
                return jsonify({
                    "status": "error",
                    "detail": f"Row {idx} missing required columns: {missing}",
                }), 400
            normalized_rows.append({col: row.get(col) for col in REQUIRED_COLS})

        # ── Extract LoadID ─────────────────────────────────────────────────────
        load_ids: Set[str] = {
            str(r["LoadID"]).strip() for r in normalized_rows if r.get("LoadID")
        }
        if len(load_ids) == 0:
            return jsonify({"status": "error", "detail": "No LoadID found in payload"}), 400
        if len(load_ids) > 1:
            return jsonify({
                "status": "error",
                "detail": f"Multiple LoadIDs detected: {sorted(load_ids)}",
            }), 400

        load_id = list(load_ids)[0]
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        blob_name = f"{load_id}-{timestamp}.json"

        inbound_container  = os.getenv("BLOB_INBOUND_CONTAINER",  "payloads")
        outbound_container = os.getenv("BLOB_OUTBOUND_CONTAINER", "outbound")

        inbound_cc  = _get_or_create_container(inbound_container)
        outbound_cc = _get_or_create_container(outbound_container)

        # ── Step 1: Save inbound PBP payload ──────────────────────────────────
        save_json_to_blob(inbound_cc, blob_name, normalized_rows)

        # ── Step 2: Load benefit_rules for this LoadID from blob ───────────────
        benefit_rules = _load_benefit_rules_from_blob(load_id)

        # ── Step 3: Build the input_json the processing function expects ───────
        input_json = {
            "pbp":           normalized_rows,
            "benefit_rules": benefit_rules,
        }

        # ── Step 4: Load prompts from the "prompts" container ──────────────────
        prompts = _load_prompts_from_blob()

        # ── Step 5: Run benefit processing ────────────────────────────────────
        # Lazy import — only pulled in here so module-level startup stays fast
        from build_benefits import run_benefit_processing
        processed_results = run_benefit_processing(input_json, prompts)

        # ── Step 6: Save outbound results ─────────────────────────────────────
        save_json_to_blob(outbound_cc, blob_name, processed_results)

        # ── Step 7: Return metadata ────────────────────────────────────────────
        return jsonify({
            "status":   "success",
            "load_id":  load_id,
            "message":  "Payload saved and processed. Retrieve results via /results/<load_id>.",
            "inbound_location": {
                "container": inbound_container,
                "blob":      blob_name,
            },
            "outbound_location": {
                "container": outbound_container,
                "blob":      blob_name,
            },
        }), 201

    except AzureError as e:
        return jsonify({"status": "error", "detail": f"Azure error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


# ============================================================
# GET /results/<load_id>  — retrieve processed outbound results
# ============================================================

@app.route("/results/<load_id>", methods=["GET"])
def get_results(load_id: str):
    try:
        outbound_container = os.getenv("BLOB_OUTBOUND_CONTAINER", "outbound")
        svc = BlobServiceClient.from_connection_string(os.environ["BLOB_CONNECTION_STRING"])
        cc  = svc.get_container_client(outbound_container)

        blobs = list(cc.list_blobs(name_starts_with=f"{load_id}-"))
        if not blobs:
            return jsonify({"status": "error", "detail": "No processed results found"}), 404

        # Pick the most recently modified blob for this load_id
        blob        = sorted(blobs, key=lambda b: b.last_modified)[-1]
        blob_client = cc.get_blob_client(blob.name)
        data        = json.loads(blob_client.download_blob().readall())

        return jsonify({
            "status":       "success",
            "load_id":      load_id,
            "blob":         blob.name,
            "result_count": len(data),
            "results":      data,
        }), 200

    except AzureError as e:
        return jsonify({"status": "error", "detail": f"Azure error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


# ============================================================
# GET /  — health check
# ============================================================

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "timestamp": utc_now_iso()})


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
