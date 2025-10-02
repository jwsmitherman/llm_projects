from flask import Flask, request, render_template_string
import os
from pathlib import Path
import shutil

app = Flask(__name__)

INBOUND_DIR = Path("./inbound")
INBOUND_DIR.mkdir(exist_ok=True)

# Simple HTML template
HTML_FORM = """
<!doctype html>
<title>Test Upload & Form</title>
<h1>Upload CSV & Enter Parameters</h1>
<form method="post" enctype="multipart/form-data">
  <label>CSV File:</label><br>
  <input type="file" name="file" required><br><br>

  <label>File Location:</label><br>
  <input type="text" name="file_location" value="./inbound/"><br><br>

  <label>File Name:</label><br>
  <input type="text" name="file_name" value="manhattan_life_raw_data.csv"><br><br>

  <label>TranDate:</label><br>
  <input type="text" name="trandate" value="2025-09-17T14:06:00"><br><br>

  <label>PayCode:</label><br>
  <input type="text" name="paycode" value="AWM01"><br><br>

  <label>Issuer:</label><br>
  <input type="text" name="issuer" value="manhattan_life"><br><br>

  <input type="submit" value="Run Process">
</form>

{% if results %}
<h2>Results</h2>
<pre>{{ results }}</pre>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def upload_and_echo():
    results = None
    if request.method == "POST":
        # Save file
        file = request.files.get("file")
        if file and file.filename:
            saved_path = INBOUND_DIR / file.filename
            file.save(saved_path)

        # Collect inputs
        file_location = request.form.get("file_location", "")
        file_name = request.form.get("file_name", "")
        trandate = request.form.get("trandate", "")
        paycode = request.form.get("paycode", "")
        issuer = request.form.get("issuer", "")

        # Prepare output string
        results = f"""
        ✅ File saved to inbound: {saved_path}
        📂 File Location: {file_location}
        📄 File Name: {file_name}
        🗓️ TranDate: {trandate}
        💰 PayCode: {paycode}
        🏢 Issuer: {issuer}
        """

    return render_template_string(HTML_FORM, results=results)


if __name__ == "__main__":
    app.run(debug=True)
