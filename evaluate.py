import math

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


if __name__ == "__main__":
    ranked1 = ["X", "A", "Y", "Z", "B", "W", "V", "U", "C", "T"]
    correct = {"A", "B", "C"}
    print(recall_at_k(ranked1, correct, 5))    # expect 0.666..., that is 0.667
    print(recall_at_k(ranked1, correct, 10))   # expect 1.0
    print(recall_at_k(ranked1, correct, 1))    # expect 0.0
    ranked2 = ["A", "P", "Q", "B", "R", "C"]
    print(reciprocal_rank(ranked1, correct))     # expect 0.5  (A is at rank 2)
    print(reciprocal_rank(ranked2, correct))     # expect 1.0  (A is at rank 1)
    print(reciprocal_rank(["P", "Q"], correct))  # expect 0   (nothing correct)
    print(ndcg_at_k(ranked1, correct, 10))   # expect about 0.619
    print(ndcg_at_k(ranked2, correct, 10))   # expect about 0.839