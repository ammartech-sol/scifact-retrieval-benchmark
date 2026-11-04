import torch
from sentence_transformers import CrossEncoder

from load_data import load_scifact
from evaluate import evaluate
from bm25_search import build_bm25_search
from dense_search import build_dense_search
from hybrid_search import build_hybrid_search


def build_rerank_search(first_stage, corpus,
                        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", depth=50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(model_name, device=device, max_length=512)

    def search(query_text):
        candidates = first_stage(query_text)
        top = candidates[:depth]
        rest = candidates[depth:]

        pairs = [(query_text, corpus[d]) for d in top]
        scores = model.predict(pairs, batch_size=32, show_progress_bar=False)

        reranked = [d for s, d in sorted(zip(scores, top), reverse=True)]
        return reranked + rest

    return search


if __name__ == "__main__":
    corpus, queries, qrels = load_scifact()
    bm25 = build_bm25_search(corpus)
    dense = build_dense_search(corpus)
    hybrid = build_hybrid_search(bm25, dense)

    for depth in [20, 50]:
        search = build_rerank_search(hybrid, corpus,
                                     model_name="BAAI/bge-reranker-base", depth=depth)
        print("bge-reranker-base depth", depth, evaluate(search, queries, qrels)["summary"])