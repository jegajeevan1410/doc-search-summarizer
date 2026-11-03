import os
import time
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL")

df = pd.read_csv("data/bbc_clean.csv")

# 10 random articles per category = 50 total (random_state keeps it repeatable)
test_df = df.groupby("category").sample(n=10, random_state=42).reset_index(drop=True)

def make_query(title, body):
    text = " ".join(body.split()[:300])
    prompt = f"""Here is a news article.

Title: {title}
Article: {text}

Write ONE short search query (3 to 8 words) that a person might type to find this article.
- Do NOT copy the title word for word.
- Return only the query, nothing else."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip().strip('"')

queries = []
for i, row in test_df.iterrows():
    q = make_query(row["title"], row["body"])
    queries.append(q)
    print(f"{i + 1}/50  {row['title']}  →  {q}")
    time.sleep(1)   # stay within Groq free-tier rate limits

test_df["query"] = queries
test_df[["doc_id", "category", "title", "query", "summary", "body"]].to_csv(
    "data/test_set.csv", index=False
)
print("\nSaved data/test_set.csv")