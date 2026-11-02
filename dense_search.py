import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from load_data import load_scifact
from evaluate import evaluate

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def build_dense_search(corpus, model_name="BAAI/bge-small-en-v1.5", k=100):
    doc_ids = list(corpus)
    texts = [corpus[d] for d in doc_ids]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    doc_vectors = model.encode(texts, batch_size=64,
                               normalize_embeddings=True, show_progress_bar=True)

    def search(query_text):
        q = model.encode(QUERY_PREFIX + query_text, normalize_embeddings=True)
        scores = doc_vectors @ q
        best = np.argsort(-scores)[:k]
        return [doc_ids[i] for i in best]

    return search


if __name__ == "__main__":
    corpus, queries, qrels = load_scifact()
    search = build_dense_search(corpus)
    print(evaluate(search, queries, qrels)["summary"])