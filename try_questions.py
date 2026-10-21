import json
import time

from fastapi.testclient import TestClient
from app import app

questions = [
    "What happens if I change a list while I am looping over it?",
    "How do I make a variable private inside a class?",
    "What is the difference between a tuple and a list?",
    "How do I reverse a string?",
    "What does the walrus operator do?",
    "How do I sort a list?",
    "Why does 0.1 + 0.2 not equal 0.3?",
    "What is the difference between is and ==?",
    "How do I run a Python script from the command line?",
    "How do I read a CSV file?",
    "How do I use asyncio?",
    "What are decorators?",
    "How do I connect to a MySQL database?",
    "What is the GIL?",
]

log = []
with TestClient(app) as client:
    for q in questions:
        r = client.post("/ask", json={"question": q})
        if r.status_code != 200:
            print("\nQ:", q, "\nERROR", r.status_code, r.text[:200])
            continue
        data = r.json()
        sources = list(dict.fromkeys(s["section_id"] for s in data["sources"]))
        print("\nQ:", q)
        print("grounded:", data["grounded"])
        print("A:", data["answer"][:500])
        print("sources:", sources)
        log.append({"question": q, "grounded": data["grounded"],
                    "answer": data["answer"], "sources": sources})
        time.sleep(2)

import os
os.makedirs("results", exist_ok=True)
with open("results/qa_log.json", "w", encoding="utf-8") as f:
    json.dump(log, f, ensure_ascii=False, indent=2)
print("\nSaved results/qa_log.json")
