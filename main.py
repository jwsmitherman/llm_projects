"""
main.py - Flask API for desert/DESR Medicare benefits pipeline

Endpoints
---------
  GET  /                      health (returns build_benefits version)
  POST /save                  accept PBP payload, kick off checkpointed processing
  GET  /results/<load_id>     poll status / fetch final results
  GET  /status/<load_id>      same as /results but returns ONLY status (no rows)

How it works
------------
/save validates the payload, saves it as an inbound blob, then launches a
background thread that calls checkpoint_runner.process_load_with_checkpoints.
That function processes plans one at a time, writing each plan's output to a
checkpoint blob immediately. Progress is tracked in a status blob.

/results reads the status blob to report progress. If the run is complete,
it reads the combined output blob and returns the rows. If incomplete, it
returns counts.

If Flask is restarted mid-run, the worker thread is lost - BUT the partial
progress is durable in blob storage. A subsequent run (Flask or batch script)
for the same LOAD_ID will resume from the last completed plan.
"""

import io
import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Set

from flask import Flask, jsonify, request
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.exceptions import AzureError

from checkpoint_runner import (
    process_load_with_checkpoints,
    read_status,
    output_blob_name,
    outbound_container_name,
)


app = Flask(__name__)


# -----------------------------------------------------------------------------
# Blob helpers (only for inbound payload save + final result read)
# -----------------------------------------------------------------------------

def _svc():
    conn = os.getenv("BLOB_CONNECTION_STRING")
    if not conn:
        raise ValueError("BLOB_CONNECTION_STRING is not set")
    return BlobServiceClient.from_connection_string(conn)


def _container(name: str):
    cc = _svc().get_container_client(name)
    try:
        cc.create_container()
    except AzureError:
        pass
    return cc


def _save_json(container: str, blob_name: str, data) -> None:
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
    _container(container).upload_blob(
        name=blob_name,
        data=io.BytesIO(payload),
        overwrite=True,
        content_settings=ContentSettings(content_type="application/json; charset=utf-8"),
    )


def _read_json(container: str, blob_name: str):
    raw = _container(container).download_blob(blob_name).readall()
    return json.loads(raw.decode("utf-8"))


def _blob_exists(container: str, blob_name: str) -> bool:
    try:
        _container(container).get_blob_client(blob_name).get_blob_properties()
        return True
    except Exception:
        return False


def _read_text(container: str, blob_name: str) -> str:
    raw = _container(container).download_blob(blob_name).readall()
    return raw.decode("utf-8")


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def utc_now_compact():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# -----------------------------------------------------------------------------
# Prompt loader (shared with batch script - same blob source)
# -----------------------------------------------------------------------------

PROMPT_FILES = {
    "system_prompt":     "system_prompt.txt",
    "few_shot_examples": "few_shot_examples.txt",
    "human_template":    "human_template.txt",
}


def _load_prompts() -> Dict[str, str]:
    prompts_container = os.getenv("BLOB_PROMPTS_CONTAINER", "prompts")
    return {key: _read_text(prompts_container, fname)
            for key, fname in PROMPT_FILES.items()}


# -----------------------------------------------------------------------------
# Payload validation
# -----------------------------------------------------------------------------

REQUIRED_COLS = ["LoadID", "FileName", "ID", "header", "category", "field", "value", "DT"]


def _validate_and_normalize(body: list) -> tuple:
    """Returns (normalized_rows, load_id, plan_filenames) or raises ValueError."""
    normalized: List[Dict[str, Any]] = []
    for idx, row in enumerate(body):
        if not isinstance(row, dict):
            raise ValueError(f"Row {idx} is not an object")
        missing = [c for c in REQUIRED_COLS if c not in row]
        if missing:
            raise ValueError(f"Row {idx} missing required columns: {missing}")
        normalized.append({c: row.get(c) for c in REQUIRED_COLS})

    load_ids: Set[str] = {str(r["LoadID"]).strip() for r in normalized if r.get("LoadID")}
    if not load_ids:
        raise ValueError("No LoadID found in payload")
    if len(load_ids) > 1:
        raise ValueError(f"Multiple LoadIDs in single payload: {sorted(load_ids)}")

    plans: Set[str] = {str(r["FileName"]).strip() for r in normalized if r.get("FileName")}
    return normalized, list(load_ids)[0], sorted(plans)


# -----------------------------------------------------------------------------
# Background worker
# -----------------------------------------------------------------------------

def _process_in_background(load_id: str, normalized_rows: list):
    """
    Runs in a daemon thread so /save can return 202 immediately. Uses the
    shared checkpoint_runner - same code path as the batch script.
    """
    try:
        prompts = _load_prompts()
        summary = process_load_with_checkpoints(
            load_id=load_id,
            pbp_rows=normalized_rows,
            prompts=prompts,
            force_reprocess=False,    # Flask never force-reprocesses; use batch for that
            max_plans_this_run=None,  # do them all
        )
        print(f"[/save background] load_id={load_id} status={summary['status']} "
              f"rows={summary['combined_rows_count']}")
    except Exception as e:
        # Status blob is already updated by the runner. Just log here.
        import traceback
        print(f"[/save background] FATAL for load_id={load_id}: {e}")
        traceback.print_exc()


# -----------------------------------------------------------------------------
# POST /save
# -----------------------------------------------------------------------------

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

        try:
            normalized, load_id, plans = _validate_and_normalize(body)
        except ValueError as ve:
            return jsonify({"status": "error", "detail": str(ve)}), 400

        print(f"[/save] LoadID={load_id} | {len(normalized):,} rows | {len(plans)} plan(s)")

        # Save inbound payload to blob synchronously so we have a record of
        # what was submitted, even if processing is interrupted
        inbound_container = os.getenv("BLOB_INBOUND_CONTAINER", "payloads")
        inbound_blob = f"medicare_input_loadid{load_id}.json"
        _save_json(inbound_container, inbound_blob, normalized)

        # Launch processing in background thread
        t = threading.Thread(
            target=_process_in_background,
            args=(load_id, normalized),
            daemon=True,
        )
        t.start()

        return jsonify({
            "status":     "accepted",
            "load_id":    load_id,
            "plan_count": len(plans),
            "inbound_blob": inbound_blob,
            "message":    "Processing started. Poll GET /results/<load_id> for status.",
            "poll_url":   f"/results/{load_id}",
        }), 202

    except AzureError as e:
        return jsonify({"status": "error", "detail": f"Azure error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


# -----------------------------------------------------------------------------
# GET /results/<load_id>
# -----------------------------------------------------------------------------

@app.route("/results/<load_id>", methods=["GET"])
def get_results(load_id):
    """
    Reads progress from the status blob. If complete, returns combined results.
    If partial/processing, returns status only.
    """
    try:
        status = read_status(load_id)
        if not status:
            return jsonify({"status": "not_found", "load_id": load_id}), 404

        # Always return progress info
        base_response = {
            "status":         status.get("status"),
            "load_id":        load_id,
            "build_version":  status.get("build_version"),
            "started_at":     status.get("started_at"),
            "updated_at":     status.get("updated_at"),
            "completed_at":   status.get("completed_at"),
            "n_plans_total":  status.get("n_plans_total"),
            "n_plans_done":   status.get("n_plans_done"),
            "n_plans_failed": status.get("n_plans_failed"),
        }

        if status.get("status") == "processing":
            return jsonify({**base_response,
                            "message": "Processing in progress. Poll again in 30s."}), 202

        if status.get("status") in ("success", "partial"):
            # Read the combined output blob
            out_blob = status.get("output_blob") or output_blob_name(load_id)
            outbound_container = outbound_container_name()
            if _blob_exists(outbound_container, out_blob):
                data = _read_json(outbound_container, out_blob)
                plan_ids = sorted({r.get("planid") for r in data if r.get("planid")})
                return jsonify({
                    **base_response,
                    "result_count": len(data),
                    "plan_count":   len(plan_ids),
                    "plan_ids":     plan_ids,
                    "output_blob":  out_blob,
                    "results":      data,
                }), 200
            else:
                return jsonify({**base_response,
                                "detail": "Output blob missing"}), 500

        if status.get("status") == "error":
            return jsonify({**base_response,
                            "error": status.get("error")}), 500

        return jsonify(base_response), 200

    except AzureError as e:
        return jsonify({"status": "error", "detail": f"Azure error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


# -----------------------------------------------------------------------------
# GET /status/<load_id>  - same info as /results but never returns the rows
# -----------------------------------------------------------------------------

@app.route("/status/<load_id>", methods=["GET"])
def get_status_only(load_id):
    """
    Lightweight progress check that returns plan-by-plan status without the
    potentially-huge results array. Useful for monitoring or building progress
    UIs.
    """
    try:
        status = read_status(load_id)
        if not status:
            return jsonify({"status": "not_found", "load_id": load_id}), 404
        return jsonify(status), 200
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


# -----------------------------------------------------------------------------
# GET / - health
# -----------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def health():
    try:
        import build_benefits as _bb
        bb_version = getattr(_bb, "__BUILD_VERSION__", "UNKNOWN")
    except Exception as e:
        bb_version = f"IMPORT_ERROR: {e}"

    return jsonify({
        "status":             "ok",
        "timestamp":          utc_now_iso(),
        "build_benefits_ver": bb_version,
    })


if __name__ == "__main__":
    try:
        import build_benefits as _bb
        print(f"[main] build_benefits version: {getattr(_bb, '__BUILD_VERSION__', 'UNKNOWN')}")
    except Exception as e:
        print(f"[main] could not import build_benefits: {e}")

    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        threaded=True,
    )
