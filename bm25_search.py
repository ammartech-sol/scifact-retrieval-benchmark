import bm25s
import Stemmer
from load_data import load_scifact
from evaluate import evaluate

def build_bm25_search(corpus, use_stopwords=True, use_stemmer=True, k=100):
    doc_ids = list(corpus)
    texts = [corpus[doc_id] for doc_id in doc_ids]

    stemmer = Stemmer.Stemmer("english") if use_stemmer else None
    stopwords = "en" if use_stopwords else None

    corpus_tokens = bm25s.tokenize(texts, stopwords=stopwords, stemmer=stemmer)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    def search(query_text):
        query_tokens = bm25s.tokenize(query_text, stopwords=stopwords, stemmer=stemmer)
        positions, scores = retriever.retrieve(query_tokens, k=k, show_progress=False)
        return [doc_ids[p] for p in positions[0]]

    return search

if __name__ == "__main__":
    corpus, queries, qrels = load_scifact()
    for use_stopwords in [True, False]:
        for use_stemmer in [True, False]:
            search = build_bm25_search(corpus, use_stopwords, use_stemmer)
            result = evaluate(search, queries, qrels)
            print("stopwords:", use_stopwords, "stemmer:", use_stemmer, result["summary"])