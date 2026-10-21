from fastapi.testclient import TestClient
from app import app

questions = [
    "How do I write an if statement?",
    "What is a list comprehension?",
    "What is the capital of France?",
    "How do I train a neural network in PyTorch?",
    "How do I download a webpage with the requests library?",
]

with TestClient(app) as client:
    for q in questions:
        r = client.post("/ask", json={"question": q})
        print("\nQ:", q)
        if r.status_code != 200:
            print("ERROR", r.status_code, r.text)
            continue
        data = r.json()
        print("grounded:", data["grounded"])
        print("A:", data["answer"])
        print("sources:", [s["section_id"] for s in data["sources"]])
