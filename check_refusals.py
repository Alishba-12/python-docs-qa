from fastapi.testclient import TestClient
from app import app

questions = [
    "How do I sort a dictionary by its values?",
    "How do I write rows to a CSV file using the csv module?",
    "How does Python's garbage collector decide when to free memory?",
    "How do I measure how long a piece of code takes to run?",
]

with TestClient(app) as client:
    for q in questions:
        best = max(h["cosine"] for h in app.state.retriever.search(q, k=5))
        r = client.post("/ask", json={"question": q})
        data = r.json()
        print("\nQ:", q)
        print("best cosine:", round(best, 2), "| grounded:", data["grounded"])
        print("A:", data["answer"][:300])
        print("sources:", [s["section_id"] for s in data["sources"]])
