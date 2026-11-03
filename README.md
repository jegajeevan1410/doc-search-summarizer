# Document Search and Summarization Using LLMs

Search about 2,000 BBC news articles with a question or topic and get a short, cited summary of the most relevant ones.

The system combines **keyword search (BM25)** with **meaning-based search (sentence embeddings)**, merges them with **Reciprocal Rank Fusion**, and uses an **LLM (via Groq)** to summarize the top results. A **Streamlit** web app provides the interface.

> The full methodology, evaluation and challenges are in the project report.

---

## Features

- **Hybrid search:** BM25 + `all-MiniLM-L6-v2` embeddings, combined with Reciprocal Rank Fusion (RRF)
- **Excerpts:** the most relevant 3-sentence passage from each result
- **LLM summaries:** one connected summary with source markers like [1], [2]
- **Adjustable summary length:** short / medium / long
- **Web app:** query auto-suggestions, paginated results (5 per page), choice of how many articles to summarize (3–10)
- **Evaluation scripts:** search accuracy (Hit@1, Hit@5, MRR), ROUGE scores, manual review

---

## Results at a glance

| Search method | Hit@1 | Hit@5 | MRR@10 |
|---|---|---|---|
| BM25 | 0.82 | 0.98 | 0.895 |
| Embeddings | 0.80 | 0.98 | 0.884 |
| **Hybrid (RRF)** | **0.82** | **1.00** | **0.907** |

Summary quality (manual review of 10 summaries, 1–5): relevance 4.9, coherence 4.1, conciseness 3.7, accuracy 3.6.

---

## Requirements

- Python 3.10 or newer (built and tested with Python 3.13 on Windows)
- About 2 GB of free disk space (PyTorch is installed with Sentence-Transformers)
- A free **Groq API key**: https://console.groq.com
- A free **Kaggle** account to download the dataset

---

## Setup

### 1. Get the code

```bash
git clone https://github.com/jegajeevan1410/doc-search-summarizer.git
cd doc-search-summarizer
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

- Windows: `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

You should see `(venv)` at the start of the terminal line.

### 3. Install the packages

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

1. On Kaggle, search for **"BBC News Summary"** (by Pariza Sharif) and download it.
2. Unzip it and place it so the folders look like this:

```
data/
└── BBC News Summary/
    ├── News Articles/
    │   ├── business/
    │   ├── entertainment/
    │   ├── politics/
    │   ├── sport/
    │   └── tech/
    └── Summaries/
        ├── business/
        └── ... (same 5 folders)
```

> The zip sometimes contains the folder twice (`BBC News Summary/BBC News Summary/...`). Keep only one copy so the path matches the one above.

### 5. Add your API key

Create a file called `.env` in the project folder:

```
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

- No quotes and no spaces around `=`.
- `.env` is listed in `.gitignore`, so your key is never uploaded.
- If the model is ever unavailable, pick another from https://console.groq.com/docs/models and change only `GROQ_MODEL` (for example `openai/gpt-oss-20b`).

---

## Run

### Step 1: Prepare the data (once)

```bash
python load_data.py
```

This cleans the articles, removes duplicates and saves `data/bbc_clean.csv` (2,089 articles).

### Step 2: Start the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**.
The first run builds the embeddings (1–2 minutes) and saves them to `data/embeddings.npy`; later runs start much faster.

### Using the app

1. Type a query (or pick a suggested article title), e.g. `oil prices`
2. Choose how many articles to summarize and the summary length
3. Read the cited summary, then browse all results with **Previous / Next**
4. Expand a result to see its most relevant excerpt (📝 = used in the summary)

### Command-line versions (optional)

```bash
python search.py       # search only: prints top results with excerpts
python summarize.py    # search + summary in the terminal
```

---

## Reproduce the evaluation (optional)

| Command | What it does | Output |
|---|---|---|
| `python make_test_set.py` | Picks 50 articles (10 per category) and generates a search query for each | `data/test_set.csv` |
| `python evaluate_search.py` | Hit@1, Hit@5 and MRR@10 for BM25, embeddings and hybrid | `data/search_eval.csv` |
| `python evaluate_summary.py` | ROUGE-1/2/L for LLM summaries vs the Lead-3 baseline | `data/summary_eval.csv` |
| `python make_human_eval.py` | Creates a sheet of 10 summaries to rate by hand | `data/human_eval.csv` |
| `python score_human_eval.py` | Averages the manual ratings | printed scores |

> These scripts call the LLM many times. On Groq's free tier (8,000 tokens per minute, 200,000 per day) they can hit rate limits; the scripts pause between calls, but you may need to run them at different times.

---

## Project structure

```
doc-search-summarizer/
├── app.py                  # Streamlit interface
├── load_data.py            # Load, clean and de-duplicate the dataset
├── search.py               # BM25, embedding and hybrid (RRF) search + excerpts
├── summarize.py            # LLM summarization via Groq
├── make_test_set.py        # Build the evaluation test set
├── evaluate_search.py      # Search accuracy metrics
├── evaluate_summary.py     # ROUGE evaluation
├── make_human_eval.py      # Manual review sheet
├── score_human_eval.py     # Manual review scores
├── requirements.txt
├── .env                    # API key (not in Git)
├── .gitignore
└── data/
    ├── BBC News Summary/   # raw dataset (download separately)
    ├── bbc_clean.csv       # created by load_data.py
    └── embeddings.npy      # created on first search (not in Git)
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError` for the dataset | Check the folder path matches step 4 exactly |
| `OSError: The paging file is too small` (Windows) | Close other heavy apps (and any other running Streamlit or Python), then retry. Or increase virtual memory: *View advanced system settings → Performance → Advanced → Virtual memory* |
| `RateLimitError: 429` from Groq | Free-tier limit reached. Wait a minute (per-minute limit) or try later (daily limit), or switch `GROQ_MODEL` to another model |
| `model not found` | Update `GROQ_MODEL` in `.env` from https://console.groq.com/docs/models |
| `accept_new_options` error in the app | Update Streamlit: `pip install --upgrade streamlit` |
| Search results look outdated after changing the data | Delete `data/embeddings.npy` so it is rebuilt |
| Warning about `HF_TOKEN` | Harmless; it can be ignored |

---

## Tech stack

Python · pandas · rank_bm25 · Sentence-Transformers (`all-MiniLM-L6-v2`) · NumPy · Groq API (`openai/gpt-oss-120b`) · rouge-score · Streamlit

## Author

V. Jegajeevan · [GitHub](https://github.com/jegajeevan1410) · [LinkedIn](https://linkedin.com/in/jegajeevan1410)
