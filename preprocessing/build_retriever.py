
import json
import time
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer # <-- MODIFIED: Import WordNetLemmatizer
import os

# --- CONFIG ---
CHUNKS_PATH = "D:/college/sem 7/CL&NLP/PersonalTutor/Project/extractor/chunks_class_9_English_Beehive.json"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FAISS_INDEX_PATH = "faiss_index.bin"
EMBEDDINGS_PATH = "embeddings.npy"
METADATA_PATH = "metadata.json"

# weights for fusion
ALPHA_DENSE = 0.6  # weight for dense sim
BETA_SPARSE = 0.4  # weight for BM25

# set random seed for reproducibility
np.random.seed(42)

# --- PREPROCESS HELPERS ---
# NOTE: NLTK resources are downloaded lazily during init_retriever to avoid
# network activity at module import time (useful for tests/CI).
EN_STOPWORDS = None
lemmatizer = None

def preprocess_for_bm25(text: str):
    # <-- MODIFIED: The function now includes a lemmatization step
    # Tokenize, lowercase, and keep only alphabetic tokens
    tokens = [t.lower() for t in word_tokenize(text) if t.isalpha()]
    
    # Lemmatize and remove stopwords/short tokens in one pass
    lemmatized_tokens = [
        lemmatizer.lemmatize(t) for t in tokens 
        if t not in EN_STOPWORDS and len(t) > 1
    ]
    return lemmatized_tokens

# --- LOAD DATA / BUILD INDEXES ---
# Module-level placeholders for lazy initialization
_initialized = False
bm25 = None
embed_model = None
index = None
embeddings = None
metadata = None
corpus = None
reranker = None


def init_retriever(force_rebuild: bool = False):
    """Initialize the retriever components (BM25, embeddings, FAISS, reranker).

    This function is idempotent and will do the heavy work only once unless
    `force_rebuild=True` is passed.
    """
    global _initialized, bm25, embed_model, index, embeddings, metadata, corpus, reranker
    if _initialized and not force_rebuild:
        return

    # import heavy ML libraries lazily to avoid costly module import
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import faiss

    global EN_STOPWORDS, lemmatizer

    # Ensure NLTK resources are available (download lazily)
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except Exception:
        # Best-effort: proceed if downloads fail (e.g., offline CI)
        pass

    EN_STOPWORDS = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    print("Loading chunks...")
    chunks = pd.read_json(CHUNKS_PATH, orient='records')
    corpus = chunks['text'].tolist()

    # metadata list (one entry per chunk)
    metadata = []
    for i, row in chunks.iterrows():
        metadata.append({
            "doc_id": i,
            "book": row.get("book", ""),
            "chapter": row.get("chapter", ""),
            "page_number": row.get("page_number", None),
            "chunk_id": row.get("chunk_id", None)
        })

    # ---------- BM25 ----------
    print("Building BM25...")
    tokenized_corpus = [preprocess_for_bm25(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    # ---------- Dense embeddings & FAISS ----------
    print("Loading embedding model:", EMBED_MODEL_NAME)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    # load or compute embeddings
    if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH) and not force_rebuild:
        print("Loading persisted embeddings and FAISS index...")
        embeddings = np.load(EMBEDDINGS_PATH)
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        print("Encoding corpus (this may take a moment)...")
        start = time.time()
        embeddings = embed_model.encode(corpus, convert_to_numpy=True, show_progress_bar=True)
        duration = time.time() - start
        print(f"Embeddings created in {duration:.1f}s. Shape: {embeddings.shape}")

        # normalize vectors for cosine similarity
        print("Normalizing embeddings for cosine search...")
        faiss.normalize_L2(embeddings)

        # create FAISS index for inner product (cosine on normalized vectors)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # persist
        np.save(EMBEDDINGS_PATH, embeddings)
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print("Saved embeddings, index, metadata.")

    # ---------- Cross-Encoder (reranker) ----------
    print("Loading cross-encoder reranker:", RERANKER_NAME)
    reranker = CrossEncoder(RERANKER_NAME)

    _initialized = True

# ---------- UTILS: NORMALIZERS ----------
def _min_max_normalize(arr):
    arr = np.array(arr, dtype=float)
    if arr.size == 0:
        return arr
    minv, maxv = float(arr.min()), float(arr.max())
    if maxv - minv < 1e-9:
        return np.ones_like(arr)
    return (arr - minv) / (maxv - minv)

# ---------- HYBRID SEARCH (fusion + rerank) ----------
def hybrid_search(query: str, k=3, top_k=5, rerank_top_k=10, alpha=ALPHA_DENSE, beta=BETA_SPARSE):
    """
    Returns re-ranked list of candidate chunks with metadata and scores.
    Steps:
      1) BM25 top_k
      2) FAISS top_k (cosine via inner-product)
      3) Create candidate union and compute normalized scores
      4) Weighted fusion of dense & sparse scores
      5) Cross-encoder rerank top `rerank_top_k`
    """
    # Ensure the retriever is initialized (lazy init)
    if not _initialized:
        init_retriever()

    # -- BM25
    q_tokens = preprocess_for_bm25(query)
    bm25_scores = bm25.get_scores(q_tokens)
    bm25_topk_idx = np.argsort(bm25_scores)[::-1][:top_k]

    # -- Dense (optional: FAISS/embeddings may be unavailable in some CI/test envs)
    dense_idx = []
    dense_scores = np.array([])
    try:
        if embed_model is not None and index is not None and 'faiss' in globals():
            q_vec = embed_model.encode([query], convert_to_numpy=True)
            # normalize safely if faiss available
            try:
                faiss.normalize_L2(q_vec)
            except Exception:
                pass
            distances, dense_idx = index.search(q_vec, top_k)
            dense_idx = dense_idx.flatten()
            dense_scores = distances.flatten()  # inner product (cosine because normalized)
        else:
            # dense components unavailable; leave dense_idx empty
            dense_idx = np.array([], dtype=int)
            dense_scores = np.array([])
    except Exception as e:
        # best-effort fallback: proceed with BM25-only
        print(f"⚠️ Retrieval error (dense search): {e}")
        dense_idx = np.array([], dtype=int)
        dense_scores = np.array([])

    # -- Build candidate set
    candidate_set = set(int(i) for i in bm25_topk_idx) | set(int(i) for i in (dense_idx if len(dense_idx) else []))
    candidates = list(candidate_set)

    # Collect raw scores (default 0 if missing)
    bm25_vals = np.array([bm25_scores[i] if i < len(bm25_scores) else 0.0 for i in candidates])
    # Build dense_vals aligned with candidates; if no dense results, use zeros
    if dense_scores is None or len(dense_scores) == 0:
        dense_vals = np.zeros(len(candidates))
    else:
        dense_vals = np.array([ (distances[0][list(dense_idx).index(i)] if i in dense_idx else 0.0) for i in candidates ])

    # Normalize scores to [0,1]
    bm25_norm = _min_max_normalize(bm25_vals)
    dense_norm = _min_max_normalize(dense_vals)

    # Weighted fusion (simple)
    fused = alpha * dense_norm + beta * bm25_norm

    # Prepare list of candidates with fused scores
    candidate_info = []
    for i, doc_idx in enumerate(candidates):
        candidate_info.append({
            "doc_id": int(doc_idx),
            "text": corpus[doc_idx],
            "bm25_score": float(bm25_vals[i]),
            "dense_score": float(dense_vals[i]),
            "fused_score": float(fused[i]),
            "meta": metadata[doc_idx] if doc_idx < len(metadata) else {}
        })

    # Sort by fused_score desc
    candidate_info.sort(key=lambda x: x["fused_score"], reverse=True)

    # Rerank top rerank_top_k with cross-encoder
    top_for_rerank = candidate_info[:rerank_top_k]
    if len(top_for_rerank) > 0:
        pairs = [(query, c["text"]) for c in top_for_rerank]
        rerank_scores = reranker.predict(pairs)  # array of floats
        for i, s in enumerate(rerank_scores):
            top_for_rerank[i]["rerank_score"] = float(s)
        # final sort by rerank_score
        top_for_rerank.sort(key=lambda x: x["rerank_score"], reverse=True)
        # combine back: put reranked top at front, then the rest
        final_list = top_for_rerank + candidate_info[rerank_top_k:]
    else:
        final_list = candidate_info

    # return top_k results (final)

    
    return final_list[:top_k]

# ---------- Simple CLI for manual testing ----------
if __name__ == "__main__":
    print("Retriever ready. Type 'exit' to quit.")
    while True:
        q = input("\nQuestion: ")
        if q.strip().lower() == "exit":
            break
        results = hybrid_search(q, top_k=5, rerank_top_k=10)
        print(f"\nTop {len(results)} results:")
        for rank, r in enumerate(results, start=1):
            print(f"\n--- Rank {rank} ---")
            print(f"doc_id: {r['doc_id']} fused: {r['fused_score']:.4f} rerank: {r.get('rerank_score', 'N/A')}")
            print("meta:", r.get("meta", {}))
            print("text:", r["text"][:300].replace("\n", " ") + ("..." if len(r["text"])>300 else ""))