import streamlit as st
from search import search, hybrid_search, best_excerpt, model, df
from summarize import summarize

st.set_page_config(page_title="News Search & Summarizer", page_icon="📰", layout="wide")
st.title("📰 BBC News Search & Summarizer")

# Cache results so changing nothing doesn't call the LLM again
@st.cache_data(show_spinner=False)
def run(query, top_n, length):
    results = search(query, top_n)
    summary = summarize(query, results, length)
    return results, summary

@st.cache_data(show_spinner=False)
def get_all_results(query, max_results=50):
    return hybrid_search(query, max_results)

PER_PAGE = 5

@st.cache_data
def get_suggestions():
    return sorted(df["title"].unique().tolist())

# ---------- Inputs ----------
query = st.selectbox(
    "Search the news",
    options=get_suggestions(),
    index=None,
    placeholder="Start typing, e.g. oil prices, people losing their jobs",
    accept_new_options=True,   # lets the user type their own query too
)

col1, col2 = st.columns(2)
top_n = col1.slider("Articles to summarize", 3, 10, 5)
length = col2.radio("Summary length", ["short", "medium", "long"], index=1, horizontal=True)

# ---------- Output ----------
if query:
    with st.spinner("Searching and summarizing..."):
        results, summary = run(query, top_n, length)

    st.subheader("Summary")
    st.markdown(summary.replace("$", "\\$"))

    st.subheader("Search results")
    all_results = get_all_results(query)
    total_pages = (len(all_results) - 1) // PER_PAGE + 1

    # Go back to page 1 when the query changes
    if st.session_state.get("last_query") != query:
        st.session_state.page = 1
        st.session_state.last_query = query

    page = st.session_state.page
    start = (page - 1) * PER_PAGE
    page_results = all_results.iloc[start:start + PER_PAGE]

    query_emb = model.encode(query, normalize_embeddings=True)
    bodies = df.set_index("doc_id")["body"]

    for rank, row in enumerate(page_results.itertuples(), start + 1):
        tag = "📝 " if rank <= top_n else ""
        with st.expander(f"{tag}[{rank}] {row.title}  ·  {row.category}"):
            st.caption(f"Relevance score: {row.score}")
            st.write(best_excerpt(query_emb, bodies[row.doc_id]))

    st.caption("📝 = used in the summary")

    prev_col, info_col, next_col = st.columns([1, 2, 1])
    if prev_col.button("⬅ Previous", disabled=page == 1):
        st.session_state.page -= 1
        st.rerun()
    info_col.markdown(f"<div style='text-align:center'>Page {page} of {total_pages}</div>",
                      unsafe_allow_html=True)
    if next_col.button("Next ➡", disabled=page == total_pages):
        st.session_state.page += 1
        st.rerun()