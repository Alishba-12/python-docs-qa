"""Build the search index (chunks and embeddings) and provide a hybrid retriever."""
import argparse
import json
import os
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

SECTIONS_PATH = os.path.join("data", "sections.json")
DEFAULT_OUT_DIR = "data"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 150
DEFAULT_OVERLAP = 30
RRF_K = 60


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def split_words(text, chunk_size, overlap):
    """Split text into pieces of about chunk_size words, keeping line breaks."""
    words = re.findall(r"\S+\s*", text)
    if len(words) <= chunk_size:
        return [text]
    step = max(chunk_size - overlap, 1)
    pieces = []
    for start in range(0, len(words), step):
        pieces.append("".join(words[start:start + chunk_size]).strip())
        if start + chunk_size >= len(words):
            break
    return pieces


def make_chunks(sections, chunk_size, overlap):
    chunks = []
    for section in sections:
        for i, piece in enumerate(split_words(section["text"], chunk_size, overlap)):
            chunks.append({
                "chunk_id": f'{section["id"]}::{i}',
                "section_id": section["id"],
                "url": section["url"],
                "title": section["title"],
                "path": section["path"],
                "text": piece,
            })
    return chunks


def chunk_search_text(chunk):
    """The title path is added so the heading also helps the search."""
    return f'{chunk["path"]}\n{chunk["text"]}'


def build_index(chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP,
                model_name=DEFAULT_MODEL, out_dir=DEFAULT_OUT_DIR):
    with open(SECTIONS_PATH, encoding="utf-8") as f:
        sections = json.load(f)
    chunks = make_chunks(sections, chunk_size, overlap)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        [chunk_search_text(c) for c in chunks],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    np.save(os.path.join(out_dir, "embeddings.npy"), embeddings.astype(np.float32))
    meta = {
        "model_name": model_name,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "num_sections": len(sections),
        "num_chunks": len(chunks),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Indexed {len(sections)} sections into {len(chunks)} chunks -> {out_dir}/")


class Retriever:
    """Loads the saved index and searches it (vector, bm25 or hybrid)."""

    def __init__(self, index_dir=DEFAULT_OUT_DIR):
        with open(os.path.join(index_dir, "chunks.json"), encoding="utf-8") as f:
            self.chunks = json.load(f)
        with open(os.path.join(index_dir, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.embeddings = np.load(os.path.join(index_dir, "embeddings.npy"))
        self.model = SentenceTransformer(self.meta["model_name"])
        self.bm25 = BM25Okapi([tokenize(chunk_search_text(c)) for c in self.chunks])

    def search(self, query, k=5, mode="hybrid"):
        q_vec = self.model.encode([query], normalize_embeddings=True)[0]
        cosine = self.embeddings @ q_vec
        bm25_scores = self.bm25.get_scores(tokenize(query))
        vector_rank = np.argsort(-cosine)
        bm25_rank = np.argsort(-bm25_scores)

        if mode == "vector":
            ranked = [(int(i), float(cosine[i])) for i in vector_rank[:k]]
        elif mode == "bm25":
            ranked = [(int(i), float(bm25_scores[i])) for i in bm25_rank[:k]]
        else:  # hybrid: Reciprocal Rank Fusion of both rankings
            scores = {}
            for rank_list in (vector_rank[:50], bm25_rank[:50]):
                for rank, i in enumerate(rank_list):
                    scores[int(i)] = scores.get(int(i), 0.0) + 1.0 / (RRF_K + rank + 1)
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for i, score in ranked:
            chunk = dict(self.chunks[i])
            chunk["score"] = score
            chunk["cosine"] = float(cosine[i])
            results.append(chunk)
        return results


def main():
    parser = argparse.ArgumentParser(description="Build the search index.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    build_index(args.chunk_size, args.overlap, args.model, args.out_dir)


if __name__ == "__main__":
    main()
