import re
import pandas as pd
from rank_bm25 import BM25Okapi
import os
import numpy as np
from sentence_transformers import SentenceTransformer

df = pd.read_csv("data/bbc_clean.csv")

def tokenize(text):
    return re.findall(r"\w+", text.lower())

# Search over title + body
corpus = (df["title"] + " " + df["body"]).tolist()
bm25 = BM25Okapi([tokenize(doc) for doc in corpus])

def bm25_search(query, top_n=5):
    scores = bm25.get_scores(tokenize(query))
    top_idx = scores.argsort()[::-1][:top_n]
    results = df.iloc[top_idx][["doc_id", "category", "title"]].copy()
    results["score"] = scores[top_idx].round(2)
    return results

# ---------- Embedding search ----------
model = SentenceTransformer("all-MiniLM-L6-v2")
EMB_PATH = "data/embeddings.npy"

# Create embeddings once, then reuse the saved file
if os.path.exists(EMB_PATH):
    doc_embeddings = np.load(EMB_PATH)
else:
    doc_embeddings = model.encode(corpus, normalize_embeddings=True, show_progress_bar=True)
    np.save(EMB_PATH, doc_embeddings)

def semantic_search(query, top_n=5):
    query_emb = model.encode(query, normalize_embeddings=True)
    scores = doc_embeddings @ query_emb          # cosine similarity
    top_idx = scores.argsort()[::-1][:top_n]
    results = df.iloc[top_idx][["doc_id", "category", "title"]].copy()
    results["score"] = scores[top_idx].round(3)
    return results

# ---------- Hybrid search (RRF) ----------
def hybrid_search(query, top_n=5, k=60):
    bm25_scores = bm25.get_scores(tokenize(query))
    sem_scores = doc_embeddings @ model.encode(query, normalize_embeddings=True)

    rrf = np.zeros(len(df))
    for scores in (bm25_scores, sem_scores):
        ranked = scores.argsort()[::-1]
        for rank, idx in enumerate(ranked):
            rrf[idx] += 1 / (k + rank + 1)

    top_idx = rrf.argsort()[::-1][:top_n]
    results = df.iloc[top_idx][["doc_id", "category", "title"]].copy()
    results["score"] = rrf[top_idx].round(4)
    return results

# ---------- Final search with excerpts ----------
def best_excerpt(query_emb, text, window=3):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    passages = [" ".join(sentences[i:i + window]) for i in range(0, len(sentences), window)]
    passage_embs = model.encode(passages, normalize_embeddings=True)
    return passages[int((passage_embs @ query_emb).argmax())]

def search(query, top_n=5):
    results = hybrid_search(query, top_n)
    query_emb = model.encode(query, normalize_embeddings=True)
    bodies = df.set_index("doc_id").loc[results["doc_id"], "body"]
    results["excerpt"] = [best_excerpt(query_emb, body) for body in bodies]
    return results

if __name__ == "__main__":
    query = input("Enter your search: ")
    top_n = int(input("How many results? ") or 5)

    for i, row in enumerate(search(query, top_n).itertuples(), 1):
        print(f"\n{i}. [{row.category}] {row.title}  (score: {row.score})")
        print(f"   ...{row.excerpt}...")