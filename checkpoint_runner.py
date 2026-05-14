"""
checkpoint_runner.py - shared plan-by-plan checkpoint processing

Used by both run_benefits_creation.py (batch) and main.py (Flask /save) so the
two pathways share one code path. Each plan is processed independently, its
output written to a checkpoint blob immediately on completion. If anything
crashes, completed plans are reused on the next run.

Blob layout
-----------
  checkpoints/checkpoint_{load_id}_{plan_filename}.json    per-plan results
  checkpoints/status_{load_id}.json                        run-wide status
  outbound/output_benefits_{load_id}.json                  combined output

The status blob is a single JSON object updated as the run progresses:
  {
    "status":        "processing" | "success" | "error" | "partial",
    "load_id":       "210",
    "started_at":    "2026-05-12T...",
    "updated_at":    "2026-05-12T...",
    "completed_at":  "2026-05-12T..." or null,
    "build_version": "2026-05-12-sweeper-and-retry-v5",
    "n_plans_total": 67,
    "n_plans_done":  43,
    "n_plans_failed": 1,
    "plans": {
      "Foo_H001-001-000": {"status": "done", "rows": 47, "elapsed_s": 52.3},
      "Foo_H001-002-000": {"status": "failed", "error": "...", "elapsed_s": 12.1},
      ...
    }
  }
"""

import io
import json
import os
import re
import time
import traceback
from datetime import datetime, timezone
from typing import Optional

from azure.storage.blob import BlobServiceClient, ContentSettings

from build_benefits import (
    group_rows_by_plan,
    run_one_plan_processing,
    __BUILD_VERSION__ as BUILD_VERSION,
)


# -----------------------------------------------------------------------------
# Blob helpers - internal; callers should use the public API at the bottom
# -----------------------------------------------------------------------------

def _svc() -> BlobServiceClient:
    conn = os.environ["BLOB_CONNECTION_STRING"]
    return BlobServiceClient.from_connection_string(conn)


def _container(name: str):
    return _svc().get_container_client(name)


def _ensure_container(name: str) -> None:
    try:
        _container(name).create_container()
    except Exception:
        pass  # already exists


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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# -----------------------------------------------------------------------------
# Container names - single source of truth, override via env
# -----------------------------------------------------------------------------

def _checkpoints_container() -> str:
    return os.getenv("BLOB_CHECKPOINTS_CONTAINER", "checkpoints")


def outbound_container_name() -> str:
    """Public - name of the outbound container where combined outputs are written."""
    return os.getenv("BLOB_OUTBOUND_CONTAINER", "outbound")


# Backward-compat alias for internal callers
_outbound_container = outbound_container_name


# -----------------------------------------------------------------------------
# Naming
# -----------------------------------------------------------------------------

def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def checkpoint_blob_name(load_id: str, plan_filename: str) -> str:
    return f"checkpoint_{load_id}_{_safe_filename(plan_filename)}.json"


def status_blob_name(load_id: str) -> str:
    return f"status_{load_id}.json"


def output_blob_name(load_id: str) -> str:
    return f"output_benefits_{load_id}.json"


# -----------------------------------------------------------------------------
# Status helpers
# -----------------------------------------------------------------------------

def _init_status(load_id: str, plan_filenames: list) -> dict:
    return {
        "status":         "processing",
        "load_id":        load_id,
        "started_at":     _utc_now_iso(),
        "updated_at":     _utc_now_iso(),
        "completed_at":   None,
        "build_version":  BUILD_VERSION,
        "n_plans_total":  len(plan_filenames),
        "n_plans_done":   0,
        "n_plans_failed": 0,
        "plans":          {fn: {"status": "pending"} for fn in plan_filenames},
    }


def _write_status(load_id: str, status: dict) -> None:
    status["updated_at"] = _utc_now_iso()
    _save_json(_checkpoints_container(), status_blob_name(load_id), status)


def read_status(load_id: str) -> Optional[dict]:
    """Public - used by /results endpoint to report progress."""
    if not _blob_exists(_checkpoints_container(), status_blob_name(load_id)):
        return None
    return _read_json(_checkpoints_container(), status_blob_name(load_id))


# -----------------------------------------------------------------------------
# Combined output assembly
# -----------------------------------------------------------------------------

def assemble_combined_output(load_id: str, plan_filenames: list) -> tuple:
    """
    Read every plan's checkpoint and concatenate into one list.
    Returns (combined_rows, missing_plan_filenames).
    """
    combined: list = []
    missing: list = []
    cp_container = _checkpoints_container()
    for fn in plan_filenames:
        cp = checkpoint_blob_name(load_id, fn)
        if _blob_exists(cp_container, cp):
            combined.extend(_read_json(cp_container, cp))
        else:
            missing.append(fn)
    return combined, missing


def write_combined_output(load_id: str, combined_rows: list) -> str:
    """Write the combined output blob. Returns the blob name."""
    out_blob = output_blob_name(load_id)
    _save_json(_outbound_container(), out_blob, combined_rows)
    return out_blob


# -----------------------------------------------------------------------------
# THE MAIN ENTRY POINT - process one load end-to-end with checkpointing
# -----------------------------------------------------------------------------

def process_load_with_checkpoints(
    load_id: str,
    pbp_rows: list,
    prompts: dict,
    *,
    force_reprocess: bool = False,
    max_plans_this_run: Optional[int] = None,
    on_plan_complete=None,  # optional callback (plan_filename, rows_out) for live progress
) -> dict:
    """
    Process a payload plan-by-plan with checkpointing. Resumable.

    Parameters
    ----------
    load_id            : load identifier (used in blob names)
    pbp_rows           : list of PBP rows (already extracted from payload dict)
    prompts            : dict with system_prompt, few_shot_examples, human_template
    force_reprocess    : if True, ignore existing checkpoints and reprocess everything
    max_plans_this_run : if set, stop after processing this many plans this invocation
                         (remaining plans picked up on next call - useful for time-budgeting)
    on_plan_complete   : optional callable(plan_filename, n_rows) called after each plan

    Returns
    -------
    dict with keys:
      load_id, n_plans_total, n_plans_done, n_plans_failed, n_plans_pending,
      combined_rows_count, output_blob, status ('success'|'partial'|'error')
    """
    _ensure_container(_checkpoints_container())
    _ensure_container(_outbound_container())

    plan_groups = group_rows_by_plan(pbp_rows)
    plan_filenames = list(plan_groups.keys())

    print(f"[checkpoint_runner] load_id={load_id} build={BUILD_VERSION}")
    print(f"[checkpoint_runner] discovered {len(plan_filenames)} plan(s) in payload")

    # Initialize or resume status
    existing_status = read_status(load_id)
    if existing_status and not force_reprocess:
        # Resume - preserve completed plans, reset failed/pending
        status = existing_status
        for fn in plan_filenames:
            if fn not in status["plans"]:
                status["plans"][fn] = {"status": "pending"}
        status["status"] = "processing"
        print(f"[checkpoint_runner] resuming - {status['n_plans_done']} plan(s) already done")
    else:
        status = _init_status(load_id, plan_filenames)

    _write_status(load_id, status)

    # Process each plan
    cp_container = _checkpoints_container()
    n_processed_this_call = 0

    for i, fn in enumerate(plan_filenames, 1):
        cp_name = checkpoint_blob_name(load_id, fn)
        already_done = _blob_exists(cp_container, cp_name)

        if already_done and not force_reprocess:
            if status["plans"][fn].get("status") != "done":
                # Sync status with reality
                rows = _read_json(cp_container, cp_name)
                status["plans"][fn] = {"status": "done", "rows": len(rows)}
                status["n_plans_done"] = sum(1 for p in status["plans"].values() if p.get("status") == "done")
                _write_status(load_id, status)
            print(f"[{i:>3}/{len(plan_filenames)}] SKIP  {fn}  (checkpoint exists)")
            continue

        if max_plans_this_run and n_processed_this_call >= max_plans_this_run:
            print(f"[{i:>3}/{len(plan_filenames)}] DEFER {fn}  (hit max_plans_this_run cap)")
            continue

        plan_rows = plan_groups[fn]
        print(f"[{i:>3}/{len(plan_filenames)}] PROCESS  {fn}  ({len(plan_rows):,} rows)")

        status["plans"][fn] = {"status": "processing", "started_at": _utc_now_iso()}
        _write_status(load_id, status)

        t0 = time.monotonic()
        try:
            plan_output = run_one_plan_processing(plan_rows, prompts)
            elapsed = time.monotonic() - t0

            # Persist checkpoint FIRST (durability) THEN update status
            _save_json(cp_container, cp_name, plan_output)

            status["plans"][fn] = {
                "status":    "done",
                "rows":      len(plan_output),
                "elapsed_s": round(elapsed, 1),
            }
            status["n_plans_done"] = sum(1 for p in status["plans"].values() if p.get("status") == "done")
            _write_status(load_id, status)
            n_processed_this_call += 1

            print(f"           DONE in {elapsed:.1f}s - {len(plan_output)} rows -> '{cp_name}'")
            if on_plan_complete:
                try:
                    on_plan_complete(fn, len(plan_output))
                except Exception as cb_err:
                    print(f"           (on_plan_complete callback raised: {cb_err})")

        except Exception as e:
            elapsed = time.monotonic() - t0
            tb = traceback.format_exc()
            status["plans"][fn] = {
                "status":    "failed",
                "error":     str(e),
                "elapsed_s": round(elapsed, 1),
            }
            status["n_plans_failed"] = sum(1 for p in status["plans"].values() if p.get("status") == "failed")
            _write_status(load_id, status)
            print(f"           FAILED after {elapsed:.1f}s: {e}")
            print(tb)
            # Continue with the next plan - DON'T crash the whole run
            n_processed_this_call += 1
            continue

    # Assemble combined output - even if partial, downstream may want what we have
    combined, missing = assemble_combined_output(load_id, plan_filenames)
    out_blob = write_combined_output(load_id, combined)

    # Final status
    if not missing and status["n_plans_failed"] == 0:
        status["status"] = "success"
    elif missing or status["n_plans_failed"] > 0:
        status["status"] = "partial"
    status["completed_at"] = _utc_now_iso()
    status["combined_rows_count"] = len(combined)
    status["output_blob"]    = out_blob
    _write_status(load_id, status)

    summary = {
        "load_id":              load_id,
        "n_plans_total":        len(plan_filenames),
        "n_plans_done":         status["n_plans_done"],
        "n_plans_failed":       status["n_plans_failed"],
        "n_plans_pending":      len(missing) - status["n_plans_failed"],
        "combined_rows_count":  len(combined),
        "output_blob":          out_blob,
        "status":               status["status"],
        "missing_plans":        missing,
    }

    print(f"\n[checkpoint_runner] {summary['status'].upper()}")
    print(f"  plans: {summary['n_plans_done']}/{summary['n_plans_total']} done | "
          f"{summary['n_plans_failed']} failed | {summary['n_plans_pending']} pending")
    print(f"  combined rows: {summary['combined_rows_count']}")
    print(f"  output blob: '{_outbound_container()}/{out_blob}'")

    return summary
