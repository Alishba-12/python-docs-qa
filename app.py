"""FastAPI service: ask a question about the Python Tutorial, get a grounded answer."""
import os
import re
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from index import Retriever

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MIN_COSINE = float(os.getenv("MIN_COSINE", "0.25"))
TOP_K = 10
NOT_FOUND_MESSAGE = (
    "The Python Tutorial documentation does not contain an answer to this question."
)

SYSTEM_PROMPT = (
    "You answer questions about the Python 3 Tutorial. "
    "Use ONLY the numbered passages given to you. "
    "Do not use outside knowledge, even if you know the answer. "
    "Answer only if at least one passage directly answers the question. "
    "If the passages only mention related words or topics but do not actually "
    "answer the question, reply with exactly NOT_IN_DOCS and nothing else. "
    "Do not stretch an example to fit a different question. "
    "If a passage states a limit or caveat that matters for the question, "
    "include it. "
    "Keep the answer short and add passage numbers in square brackets like [1] "
    "after the claims they support."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.retriever = Retriever()
    yield


app = FastAPI(title="Python Tutorial Q&A", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class Source(BaseModel):
    section_id: str
    url: str
    title: str
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    sources: list[Source]


def ask_llm(question, passages):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set")
    from groq import Groq

    client = Groq(api_key=api_key)
    context = "\n\n".join(
        f"[{i}] ({p['path']})\n{p['text']}" for i, p in enumerate(passages, 1)
    )
    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Passages:\n\n{context}\n\nQuestion: {question}",
                },
            ],
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}")
    return completion.choices[0].message.content.strip()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    hits = app.state.retriever.search(req.question, k=TOP_K, mode="hybrid")
    best_cosine = max(h["cosine"] for h in hits)
    if best_cosine < MIN_COSINE:
        return AskResponse(answer=NOT_FOUND_MESSAGE, grounded=False, sources=[])

    answer = ask_llm(req.question, hits)
    if "NOT_IN_DOCS" in answer:
        return AskResponse(answer=NOT_FOUND_MESSAGE, grounded=False, sources=[])

    cited = {int(n) for n in re.findall(r"[\[【](\d+)[\]】]", answer)}
    used = [h for i, h in enumerate(hits, 1) if i in cited] or hits
    sources = [
        Source(
            section_id=h["section_id"],
            url=h["url"],
            title=h["title"],
            text=h["text"],
            score=h["score"],
        )
        for h in used
    ]
    return AskResponse(answer=answer, grounded=True, sources=sources)
