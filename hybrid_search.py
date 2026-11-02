from load_data import load_scifact
from evaluate import evaluate
from bm25_search import build_bm25_search
from dense_search import build_dense_search


def build_hybrid_search(search_a, search_b, rrf_k=60, k=100):
    def search(query_text):
        scores = {}
        for ranked in [search_a(query_text), search_b(query_text)]:
            position = 1
            for doc_id in ranked:
                scores[doc_id] = scores.get(doc_id, 0) + 1 / (rrf_k + position)
                position = position + 1
        best = sorted(scores, key=scores.get, reverse=True)
        return best[:k]

    return search


if __name__ == "__main__":
    corpus, queries, qrels = load_scifact()
    bm25 = build_bm25_search(corpus)
    dense = build_dense_search(corpus)
    hybrid = build_hybrid_search(bm25, dense)
    print(evaluate(hybrid, queries, qrels)["summary"])