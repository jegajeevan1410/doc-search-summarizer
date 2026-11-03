import pandas as pd

summaries = pd.read_csv("data/summary_eval.csv")
test = pd.read_csv("data/test_set.csv")[["doc_id", "body"]]

sample = summaries.sample(n=10, random_state=42)[["doc_id", "title", "query", "llm_summary"]]
sample = sample.merge(test, on="doc_id")   # add the article so you can check facts

for col in ["relevance", "accuracy", "coherence", "conciseness", "notes"]:
    sample[col] = ""

sample.to_csv("data/human_eval.csv", index=False, encoding="utf-8-sig")
print("Created data/human_eval.csv with", len(sample), "summaries to rate")