import json
from load_data import load_scifact

with open("results/results.json") as f:
    results = json.load(f)
corpus, queries, qrels = load_scifact()

bm25_r = results["bm25"]["per_query"]["recall@10"]
dense_r = results["dense"]["per_query"]["recall@10"]

only_dense = [q for q in queries if bm25_r[q] == 0 and dense_r[q] > 0]
only_bm25 = [q for q in queries if bm25_r[q] > 0 and dense_r[q] == 0]        # blank: dense found nothing, BM25 found something
both_fail = [q for q in queries if bm25_r[q] == 0 and dense_r[q] == 0]

print("only dense finds it:", len(only_dense))
print("only BM25 finds it:", len(only_bm25))
print("both fail:", len(both_fail))


def show(group, title, n=5):
    print("\n=====", title, "=====")
    for q in group[:n]:
        print("CLAIM:", queries[q])
        for doc_id in qrels[q]:
            print("   CORRECT ABSTRACT:", corpus[doc_id][:250])


show(only_dense, "ONLY DENSE FINDS IT")
show(only_bm25, "ONLY BM25 FINDS IT")
show(both_fail, "BOTH FAIL")