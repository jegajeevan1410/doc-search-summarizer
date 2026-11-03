import pandas as pd

df = pd.read_csv("data/human_eval.csv", encoding="cp1252")
criteria = ["relevance", "accuracy", "coherence", "conciseness"]

df[criteria] = df[criteria].apply(pd.to_numeric, errors="coerce")

print("Human evaluation (1-5), 10 summaries:\n")
for c in criteria:
    print(f"  {c:<12} {df[c].mean():.2f}")
print(f"\n  {'overall':<12} {df[criteria].mean().mean():.2f}")

notes = df[df["notes"].notna()][["title", "notes"]]
if len(notes):
    print("\nNotes:")
    for _, r in notes.iterrows():
        print(f"  - {r['title']}: {r['notes']}")

df[criteria].mean().round(2).to_csv("data/human_eval_scores.csv")