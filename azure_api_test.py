import requests
import sys

ENDPOINT = "https://joshu-meub0vlp-swedencentral.cognitiveservices.azure.com"
API_KEY = "YOUR_KEY"
URL = f"{ENDPOINT}/openai/deployments?api-version=2024-12-01-preview"

def test_auth():
    print("Testing authenticated request...")
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(URL, headers=headers, timeout=10)
        print(f"\nStatus Code: {response.status_code}")
        print(response.text)

        if response.ok:
            print("✅ SUCCESS: Authenticated call worked.")
        else:
            print("⚠️ Reached endpoint but authentication failed.")

    except Exception as e:
        print("❌ ERROR: Could NOT connect.\n")
        print(str(e))
        sys.exit(1)

if __name__ == "__main__":
    test_auth()
