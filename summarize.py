import os
from dotenv import load_dotenv
from groq import Groq
import pandas as pd

df = pd.read_csv("data/bbc_clean.csv")

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL")

LENGTHS = {
    "short": "2-3 sentences (about 50 words)",
    "medium": "one paragraph (about 120 words)",
    "long": "2-3 paragraphs (about 250 words)",
}

def summarize(query, results, length="medium"):
    bodies = df.set_index("doc_id").loc[results["doc_id"], "body"]

    # Build the article text (cap very long articles at 800 words)
    docs_text = ""
    for i, (title, body) in enumerate(zip(results["title"], bodies), 1):
        docs_text += f"\n[{i}] {title}\n{' '.join(body.split()[:400])}\n"

    prompt = f"""User question: {query}

Articles:
{docs_text}

Write a summary of these articles that answers the user's question.
- Length: {LENGTHS[length]}
- Use ONLY information from the articles. Do not add outside facts.
- Write one connected summary, not a list of separate article summaries.
- Mark sources like [1], [2].
- Skip any article that is not relevant to the question."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You summarize news articles clearly and accurately."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        reasoning_effort="low",
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    from search import search
    query = input("Enter your search: ")
    top_n = int(input("How many articles? ") or 5)
    length = input("Summary length (short/medium/long): ").strip().lower() or "medium"

    results = search(query, top_n)

    print("\nArticles used:")
    for i, title in enumerate(results["title"], 1):
        print(f"  [{i}] {title}")

    print(f"\n--- Summary ({length}) ---")
    print(summarize(query, results, length))