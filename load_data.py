import ir_datasets


def load_scifact():
    dataset = ir_datasets.load("beir/scifact/test")

    corpus = {}
    for doc in dataset.docs_iter():
        corpus[doc.doc_id] = doc.title + ". " + doc.text

    queries = {}
    for q in dataset.queries_iter():
        queries[q.query_id] = q.text

    qrels = {}
    for row in dataset.qrels_iter():
        if row.query_id not in qrels:
            qrels[row.query_id] = set()
        qrels[row.query_id].add(row.doc_id)

    return corpus, queries, qrels


if __name__ == "__main__":
    corpus, queries, qrels = load_scifact()
    print("documents:", len(corpus))
    print("queries:", len(queries))
    print("queries with answers:", len(qrels))