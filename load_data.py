import re
import os
import pandas as pd

DATA_DIR = "data/BBC News Summary"
ARTICLES_DIR = os.path.join(DATA_DIR, "News Articles")
SUMMARIES_DIR = os.path.join(DATA_DIR, "Summaries")

rows = []

for category in os.listdir(ARTICLES_DIR):
    article_folder = os.path.join(ARTICLES_DIR, category)
    summary_folder = os.path.join(SUMMARIES_DIR, category)

    for filename in os.listdir(article_folder):
        with open(os.path.join(article_folder, filename), encoding="latin-1") as f:
            article = f.read()
        with open(os.path.join(summary_folder, filename), encoding="latin-1") as f:
            summary = f.read()

        rows.append({
            "category": category,
            "filename": filename,
            "article": article,
            "summary": summary,
        })

df = pd.DataFrame(rows)

def clean_text(text):
    text = re.sub(r"\s+", " ", text)          # collapse newlines/extra spaces
    text = re.sub(r"\.(?=[A-Z])", ". ", text)  # add space after "." if missing
    return text.strip()

# First line = title, rest = body
df["title"] = df["article"].apply(lambda x: x.split("\n")[0].strip())
df["body"] = df["article"].apply(lambda x: "\n".join(x.split("\n")[1:]))

df["body"] = df["body"].apply(clean_text)
df["summary"] = df["summary"].apply(clean_text)

# Remove duplicates and empty articles
before = len(df)
df = df.drop_duplicates(subset="body")
df = df[df["body"].str.len() > 0]
print("Removed duplicates/empty:", before - len(df))

# Useful extra columns
df["word_count"] = df["body"].str.split().str.len()
df["doc_id"] = df["category"] + "_" + df["filename"].str.replace(".txt", "", regex=False)

df = df[["doc_id", "category", "title", "body", "summary", "word_count"]]
df.to_csv("data/bbc_clean.csv", index=False)

print("Clean articles saved:", len(df))
print(df["word_count"].describe())