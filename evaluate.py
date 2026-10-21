"""Evaluate retrieval: Hit Rate@k on a set of questions with known source sections."""
import argparse
import csv
import json
import os

from index import Retriever


def main():
    parser = argparse.ArgumentParser(description="Compute Hit Rate@k.")
    parser.add_argument("--questions", default="questions.json")
    parser.add_argument("--index-dir", default="data")
    parser.add_argument("--mode", default="hybrid", choices=["vector", "bm25", "hybrid"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--label", default="baseline", help="name for this run in results.csv")
    args = parser.parse_args()

    with open(args.questions, encoding="utf-8") as f:
        questions = json.load(f)

    retriever = Retriever(args.index_dir)
    known_ids = {c["section_id"] for c in retriever.chunks}

    hits_at_k = 0
    hits_at_1 = 0
    misses = []
    for item in questions:
        expected = item["section_id"]
        expected = set(expected) if isinstance(expected, list) else {expected}
        unknown = expected - known_ids
        if unknown:
            print(f"WARNING: unknown section id(s) {unknown} in question: {item['question']}")

        results = retriever.search(item["question"], k=args.k, mode=args.mode)
        ranked_ids = [r["section_id"] for r in results]
        rank = next((i + 1 for i, sid in enumerate(ranked_ids) if sid in expected), None)

        if rank is not None:
            hits_at_k += 1
        if rank == 1:
            hits_at_1 += 1
        status = f"HIT (rank {rank})" if rank else "MISS"
        print(f"{status:12} {item['question']}")
        if rank is None:
            misses.append({"question": item["question"], "expected": sorted(expected), "got": ranked_ids})

    n = len(questions)
    rate = hits_at_k / n
    print(f"\nMode: {args.mode} | label: {args.label}")
    print(f"Hit Rate@{args.k}: {hits_at_k}/{n} = {rate:.2%}")
    print(f"Hit Rate@1: {hits_at_1}/{n} = {hits_at_1 / n:.2%}")

    if misses:
        print("\nMisses:")
        for m in misses:
            print(f"- {m['question']}\n    expected: {m['expected']}\n    got:      {m['got']}")

    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", "results.csv")
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["label", "mode", "index_dir", "k", "hits", "total", "hit_rate", "hit_rate_at_1"])
        writer.writerow([args.label, args.mode, args.index_dir, args.k, hits_at_k, n,
                         f"{rate:.4f}", f"{hits_at_1 / n:.4f}"])


if __name__ == "__main__":
    main()
