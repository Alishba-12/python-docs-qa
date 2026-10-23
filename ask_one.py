import sys
from fastapi.testclient import TestClient
from app import app

q = " ".join(sys.argv[1:])
with TestClient(app) as client:
    data = client.post("/ask", json={"question": q}).json()
print("grounded:", data["grounded"])
print("A:", data["answer"])
print("sources:", [s["section_id"] for s in data["sources"]])
