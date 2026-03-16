import os
import sys

import requests

port = os.getenv("STREAMLIT_SERVER_PORT", "8501")
url = f"http://127.0.0.1:{port}/"

try:
    response = requests.get(url, timeout=5)
    if response.status_code == 200:
        print("ok")
        sys.exit(0)
except Exception:
    pass

print("unhealthy")
sys.exit(1)
