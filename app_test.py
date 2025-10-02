from flask import Flask, render_template_string, request
import os
import shutil

app = Flask(__name__)

# Make sure inbound folder exists
INBOUND_DIR = os.path.join(os.getcwd(), "inbound")
os.makedirs(INBOUND_DIR, exist_ok=True)

# HTML template embedded into Python
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>CSV Processor</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 40px; background: #f9f9f9; }
    h2 { color: #333; }
    form { background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); width: 500px; }
    label { display: block; margin-top: 10px; font-weight: bold; }
    input { width: 100%; padding: 8px; margin-top: 4px; border: 1px solid #ccc; border-radius: 4px; }
    button { margin-top: 15px; padding: 10px; background: #007BFF; color: #fff; border: none; border-radius: 4px; }
    button:hover { background: #0056b3; cursor: pointer; }
    .results { margin-top: 30px; padding: 15px; background: #e9ffe9; border: 1px solid #b2d8b2; border-radius: 6px; }
  </style>
</head>
<body>
  <h2>Upload CSV and Run Process</h2>
  <form action="/process" method="post" enctype="multipart/form-data">
    <label>Upload CSV:</label>
    <input type="file" name="file" required>

    <label>File Location:</label>
    <input type="text" name="file_location" value="./inbound/" required>

    <label>File Name:</label>
    <input type="text" name="file_name" value="manhattan_life_raw_data.csv" required>

    <label>TranDate:</label>
    <input type="text" name="trandate" value="2025-09-17T14:06:00" required>

    <label>PayCode:</label>
    <input type="text" name="paycode" value="AWM01" required>

    <label>Issuer:</label>
    <input type="text" name="issuer" value="manhattan_life" required>

    <button type="submit">Run Process</button>
  </form>

  {% if results %}
  <div class="results">
    <h3>Selections:</h3>
    <pre>{{ results }}</pre>
  </div>
  {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/process", methods=["POST"])
def process():
    file = request.files["file"]
    file_location = request.form["file_location"]
    file_name = request.form["file_name"]
    trandate = request.form["trandate"]
    paycode = request.form["paycode"]
    issuer = request.form["issuer"]

    # Save file to inbound folder
    saved_path = os.path.join(INBOUND_DIR, file.filename)
    file.save(saved_path)

    results = f"""
    File saved to inbound: {saved_path}
    File Location: {file_location}
    File Name: {file_name}
    TranDate: {trandate}
    PayCode: {paycode}
    Issuer: {issuer}
    """

    return render_template_string(HTML_TEMPLATE, results=results)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
