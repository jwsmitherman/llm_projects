@app.route("/save", methods=["POST"])
def save_json_payload():
    try:
        if not request.is_json:
            return jsonify({"status": "error", "detail": "Request must be JSON"}), 400

        body = request.get_json(silent=True)
        if body is None:
            return jsonify({"status": "error", "detail": "Invalid JSON"}), 400

        if not isinstance(body, list):
            return jsonify({"status": "error", "detail": "Payload must be a list of objects"}), 400

        normalized_rows = []
        for idx, row in enumerate(body):
            if not isinstance(row, dict):
                return jsonify({
                    "status": "error",
                    "detail": f"Row {idx} is not an object"
                }), 400

            missing = [col for col in REQUIRED_COLS if col not in row]
            if missing:
                return jsonify({
                    "status": "error",
                    "detail": f"Row {idx} missing required columns: {missing}"
                }), 400

            normalized_rows.append({col: row.get(col) for col in REQUIRED_COLS})

        payload_loadids: Set[str] = set(
            str(r["LoadID"]).strip() for r in normalized_rows if r.get("LoadID")
        )

        if len(payload_loadids) == 0:
            return jsonify({"status": "error", "detail": "No LoadID values found in payload"}), 400

        if len(payload_loadids) > 1:
            return jsonify({
                "status": "error",
                "detail": f"Multiple LoadIDs detected in payload: {sorted(payload_loadids)}"
            }), 400

        load_id = list(payload_loadids)[0]

        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        blob_name = f"{load_id}-{timestamp}.json"

        inbound_container_name = os.getenv("BLOB_INBOUND_CONTAINER", os.getenv("BLOB_CONTAINER", "payloads"))
        outbound_container_name = os.getenv("BLOB_OUTBOUND_CONTAINER", "outbound")

        inbound_cc = get_or_create_container(inbound_container_name)
        outbound_cc = get_or_create_container(outbound_container_name)

        # Save inbound payload
        save_json_to_blob(inbound_cc, blob_name, normalized_rows)

        # Process
        processed_results = run_benefit_processing(normalized_rows)

        # Save processed results to outbound
        save_json_to_blob(outbound_cc, blob_name, processed_results)

        # Read saved outbound results back
        results_json = read_json_from_blob(outbound_cc, blob_name)

        return jsonify({
            "status": "success",
            "status_code": 200,
            "load_id": load_id,
            "message": "Payload ingested, processed, and results returned from outbound blob",
            "inbound_location": {
                "container": inbound_container_name,
                "blob": blob_name
            },
            "outbound_location": {
                "container": outbound_container_name,
                "blob": blob_name
            },
            "rows_ingested": len(normalized_rows),
            "result": results_json
        }), 200

    except AzureError as e:
        return jsonify({
            "status": "error",
            "detail": f"Azure error: {str(e)}"
        }), 500

    except Exception as e:
        return jsonify({
            "status": "error",
            "detail": str(e)
        }), 500