import re
import time
import pandas as pd
from rouge_score import rouge_scorer
from summarize import summarize

test = pd.read_csv("data/test_set.csv")
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def rouge_f1(prediction, reference):
    scores = scorer.score(reference, prediction)
    return {k: v.fmeasure for k, v in scores.items()}

rows = []
for i, r in test.iterrows():
    one_doc = test.iloc[[i]][["doc_id", "title"]]
    llm_summary = summarize(r["query"], one_doc, "medium")
    llm_summary = re.sub(r"\[\d+\]", "", llm_summary).strip()   # remove [1] markers

    lead3 = " ".join(re.split(r"(?<=[.!?])\s+", r["body"])[:3])

    llm_scores = rouge_f1(llm_summary, r["summary"])
    lead_scores = rouge_f1(lead3, r["summary"])

    rows.append({
        "doc_id": r["doc_id"], "title": r["title"], "query": r["query"],
        "llm_summary": llm_summary, "reference": r["summary"],
        **{f"llm_{k}": v for k, v in llm_scores.items()},
        **{f"lead3_{k}": v for k, v in lead_scores.items()},
    })
    print(f"{i + 1}/50 done")
    time.sleep(1)

out = pd.DataFrame(rows)
out.to_csv("data/summary_eval.csv", index=False)

print("\nAverage ROUGE (F1):")
for m in ["rouge1", "rouge2", "rougeL"]:
    print(f"  {m}:  LLM = {out[f'llm_{m}'].mean():.3f}   Lead-3 = {out[f'lead3_{m}'].mean():.3f}")