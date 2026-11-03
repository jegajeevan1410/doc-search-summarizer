import pandas as pd
from search import bm25_search, semantic_search, hybrid_search

test = pd.read_csv("data/test_set.csv")
methods = {"BM25": bm25_search, "Semantic": semantic_search, "Hybrid": hybrid_search}

rows = []
for name, search_fn in methods.items():
    hit1 = hit5 = mrr = 0
    for _, r in test.iterrows():
        ids = search_fn(r["query"], top_n=10)["doc_id"].tolist()
        if r["doc_id"] in ids:
            rank = ids.index(r["doc_id"]) + 1
            mrr += 1 / rank
            hit1 += rank == 1
            hit5 += rank <= 5

    n = len(test)
    rows.append({"method": name, "Hit@1": hit1 / n, "Hit@5": hit5 / n, "MRR@10": mrr / n})

results = pd.DataFrame(rows).round(3)
print(results)
results.to_csv("data/search_eval.csv", index=False)