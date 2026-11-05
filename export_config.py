import json

config = {
    "winner": "hybrid",
    "why": "only system clearly above bm25 (bootstrap 95% range on nDCG@10 excludes 0); "
           "rerankers added latency without a clear gain",
    "dataset": "BEIR SciFact (test split, 300 queries, 5183 documents)",
    "bm25": {"library": "bm25s", "stopwords": "en", "stemmer": "english"},
    "dense": {
        "model": "BAAI/bge-small-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "normalize_embeddings": True,
    },
    "fusion": {"method": "reciprocal_rank_fusion", "rrf_k": 60},
    "first_stage_k": 100,
    "results": {"ndcg@10": 0.723, "recall@10": 0.859, "recall@20": 0.889,
                "median_ms_approx": 10},
    "hardware_note": "latency measured on an RTX 4050 laptop GPU, approximate",
}

with open("config.json", "w") as f:
    json.dump(config, f, indent=2)
print("wrote config.json")