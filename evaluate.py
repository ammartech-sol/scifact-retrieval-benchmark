import math
import time
import numpy as np

def credit(position):
    return 1 / math.log2(position + 1)

def recall_at_k(ranked, correct, k):
    top_k = ranked[:k]          # blank 1: the first k results of ranked
    found = 0
    for doc_id in top_k:
        if doc_id in correct:          # blank 2: doc_id is in the correct set
            found = found + 1
    return found / len(correct)           # blank 3: found divided by how many correct docs there are

def reciprocal_rank(ranked, correct):
    position = 1
    for doc_id in ranked:
        if doc_id in correct:          # blank 1: is this doc_id correct?
            return 1 / position   # blank 2: 1 divided by the position
        position = position + 1
    return 0           # blank 3: the score when nothing correct was found

def ndcg_at_k(ranked, correct, k):
    dcg = 0
    position = 1
    for doc_id in ranked[:k]:
        if doc_id in correct:                                   # blank 1
            dcg = dcg + credit(position)
        position = position + 1

    ideal = 0
    for slot in range(1, min(len(correct), k) + 1):
        ideal = ideal + credit(slot)

    return dcg / ideal                                    # blank 2

def evaluate(search, queries, qrels):
    names = ["recall@5", "recall@10", "recall@20", "recall@50", "recall@100", "mrr@100", "ndcg@10"]
    per_query = {name: {} for name in names}
    times_ms = []

    for query_id in queries:
        start = time.perf_counter()
        ranked = search(queries[query_id])
        times_ms.append((time.perf_counter() - start) * 1000)

        correct = qrels[query_id]
        per_query["recall@5"][query_id] = recall_at_k(ranked, correct, 5)
        per_query["recall@10"][query_id] = recall_at_k(ranked, correct, 10)
        per_query["recall@20"][query_id] = recall_at_k(ranked, correct, 20)
        per_query["recall@50"][query_id] = recall_at_k(ranked, correct, 50)
        per_query["recall@100"][query_id] = recall_at_k(ranked, correct, 100)
        per_query["mrr@100"][query_id] = reciprocal_rank(ranked, correct)
        per_query["ndcg@10"][query_id] = ndcg_at_k(ranked, correct, 10)

    summary = {name: sum(s.values()) / len(s) for name, s in per_query.items()}
    summary["median_ms"] = float(np.median(times_ms))
    summary["p95_ms"] = float(np.percentile(times_ms, 95))
    return {"summary": summary, "per_query": per_query}


if __name__ == "__main__":
    from load_data import load_scifact
    import random

    corpus, queries, qrels = load_scifact()
    all_ids = list(corpus)

    def random_search(query_text):
        ids = all_ids.copy()
        random.shuffle(ids)
        return ids[:100]

    print(evaluate(random_search, queries, qrels)["summary"])

    text_to_id = {text: qid for qid, text in queries.items()}

    def cheating_search(query_text):
        qid = text_to_id[query_text]
        return list(qrels[qid])

    print(evaluate(cheating_search, queries, qrels)["summary"])