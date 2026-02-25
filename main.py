import os
from datetime import datetime
from flask import Flask, jsonify
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import AzureError

app = Flask(__name__)

def get_blob_clients():
    conn_str = os.getenv("BLOB_CONNECTION_STRING")
    container_name = os.getenv("BLOB_CONTAINER", "payloads")

    if not conn_str:
        raise ValueError("BLOB_CONNECTION_STRING is not set")

    service_client = BlobServiceClient.from_connection_string(conn_str)
    container_client = service_client.get_container_client(container_name)
    return container_client, container_name

@app.get("/")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat() + "Z"})

@app.get("/save/<loadid>")
def save_load_id(loadid: str):
    try:
        container_client, container_name = get_blob_clients()

        content = f"LOADID: {loadid}\nSaved at {datetime.utcnow().isoformat()}Z"
        blob_name = f"{loadid}.txt"

        container_client.upload_blob(
            name=blob_name,
            data=content.encode("utf-8"),
            overwrite=True
        )

        return jsonify({
            "status": "success",
            "container": container_name,
            "blob": blob_name,
            "loadid": loadid
        })

    except ValueError as e:
        return jsonify({"status": "error", "detail": str(e)}), 500
    except AzureError as e:
        return jsonify({"status": "error", "detail": f"Azure error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500

if __name__ == "__main__":
    # local-only. App Service uses gunicorn startup command.
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)

#https://desrapi-e9g6awabhmwcc0f5.westcentralus-01.azurewebsites.net/
#https://desrapi-e9g6awabhmwcc0f5.westcentralus-01.azurewebsites.net/save/12345
