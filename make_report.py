import json
import matplotlib.pyplot as plt

with open("results/results.json") as f:
    results = json.load(f)

columns = ["recall@5", "recall@10", "recall@20", "ndcg@10", "median_ms"]
lines = ["| system | " + " | ".join(columns) + " |",
         "|---|" + "---|" * len(columns)]
for name, r in results.items():
    s = r["summary"]
    cells = [f"{s[c]:.1f}" if c == "median_ms" else f"{s[c]:.3f}" for c in columns]
    lines.append("| " + name + " | " + " | ".join(cells) + " |")

with open("results/results_table.md", "w") as f:
    f.write("\n".join(lines))
print("\n".join(lines))

names = list(results)
ndcg = [results[n]["summary"]["ndcg@10"] for n in names]
plt.figure(figsize=(8, 4))
plt.bar(names, ndcg)
plt.ylim(0, 1)
plt.ylabel("nDCG@10")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("results/ndcg_chart.png", dpi=150)

latency = [results[n]["summary"]["median_ms"] for n in names]
plt.figure(figsize=(7, 4.5))
plt.scatter(latency, ndcg)
for n, x, y in zip(names, latency, ndcg):
    plt.annotate(n, (x, y), textcoords="offset points", xytext=(5, 5))
plt.xscale("log")
plt.xlabel("median latency per query (ms, log scale)")
plt.ylabel("nDCG@10")
plt.ylim(0.6, 0.8)
plt.tight_layout()
plt.savefig("results/quality_vs_speed.png", dpi=150)