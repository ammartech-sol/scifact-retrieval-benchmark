import json
import random


def bootstrap_diff(scores_a, scores_b, n=1000):
    ids = list(scores_a)
    diffs = [scores_a[i] - scores_b[i] for i in ids]
    means = []
    for _ in range(n):
        sample = [random.choice(diffs) for _ in range(len(diffs))]
        means.append(sum(sample) / len(sample))
    means.sort()
    return sum(diffs) / len(diffs), means[int(n * 0.025)], means[int(n * 0.975) - 1]


if __name__ == "__main__":
    random.seed(0)
    with open("results/results.json") as f:
        results = json.load(f)

    pairs = [("dense", "bm25"), ("hybrid", "dense"), ("hybrid", "bm25"),
             ("hybrid+bge@20", "hybrid"), ("hybrid+minilm@20", "hybrid")]
    for a, b in pairs:
        mean, low, high = bootstrap_diff(results[a]["per_query"]["ndcg@10"],
                                         results[b]["per_query"]["ndcg@10"])
        print(f"{a} minus {b}: {mean:+.3f}   95% range [{low:+.3f}, {high:+.3f}]")