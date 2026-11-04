import json
import os

from load_data import load_scifact
from evaluate import evaluate
from bm25_search import build_bm25_search
from dense_search import build_dense_search
from hybrid_search import build_hybrid_search
from rerank_search import build_rerank_search

if __name__ == "__main__":
    corpus, queries, qrels = load_scifact()
    bm25 = build_bm25_search(corpus)
    dense = build_dense_search(corpus)
    hybrid = build_hybrid_search(bm25, dense)

    systems = {
        "bm25": bm25,
        "dense": dense,
        "hybrid": hybrid,
        "hybrid+minilm@20": build_rerank_search(
            hybrid, corpus, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", depth=20),
        "hybrid+bge@20": build_rerank_search(
            hybrid, corpus, model_name="BAAI/bge-reranker-base", depth=20),
    }

    results = {}
    for name, search in systems.items():
        print("running", name)
        results[name] = evaluate(search, queries, qrels)

    os.makedirs("results", exist_ok=True)
    with open("results/results.json", "w") as f:
        json.dump(results, f)