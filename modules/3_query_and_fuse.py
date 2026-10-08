import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. SETUP (From previous steps) ---
raw_document = """
Machine learning models require significant computational resources. 
The training phase often utilizes clusters of GPUs to process large datasets efficiently. 
In contrast, the inference phase, where the model makes predictions, can sometimes be run on CPUs, depending on latency requirements.
Historically, recurrent neural networks (RNNs) were used for sequence tasks. 
However, the Transformer architecture introduced in 2017 replaced RNNs because self-attention mechanisms allow for highly parallelized training.
"""

text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20, length_function=len)
raw_chunks = text_splitter.split_text(raw_document)

processed_database = [{"chunk_id": f"doc_1_chunk_{i}", "text": chunk} for i, chunk in enumerate(raw_chunks)]
chunk_texts = [row["text"] for row in processed_database]

# Build BM25
bm25_index = BM25Okapi([text.lower().split() for text in chunk_texts])
# Build FAISS
embedder = SentenceTransformer('all-MiniLM-L6-v2')
faiss_index = faiss.IndexFlatL2(384)
faiss_index.add(embedder.encode(chunk_texts))

# --- 2. NEW LOGIC: Dual Querying ---
query = "What hardware is typically used for inference?"
print(f"🔍 Query: '{query}'\n")

# A. Keyword Search (BM25)
tokenized_query = query.lower().split()
bm25_scores = bm25_index.get_scores(tokenized_query)
bm25_ranks = np.argsort(bm25_scores)[::-1] # Sort highest to lowest

# B. Semantic Search (FAISS)
query_vector = embedder.encode([query])
distances, faiss_ranks = faiss_index.search(query_vector, len(processed_database))
faiss_ranks = faiss_ranks[0]

# --- 3. NEW LOGIC: Reciprocal Rank Fusion (RRF) ---
k = 60
rrf_scores = {i: 0.0 for i in range(len(processed_database))}

# Add BM25 rank scores
for rank, doc_id in enumerate(bm25_ranks):
    rrf_scores[doc_id] += 1.0 / (k + rank + 1) # rank + 1 because ranks shouldn't be 0

# Add FAISS rank scores
for rank, doc_id in enumerate(faiss_ranks):
    rrf_scores[doc_id] += 1.0 / (k + rank + 1)

# Sort documents by their final fused RRF score
fused_results = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

# --- 4. OUTPUT RESULTS ---
print("🏆 TOP FUSED RESULT:")
top_doc_id = fused_results[0]
print(f"ID: {processed_database[top_doc_id]['chunk_id']}")
print(f"Text: {processed_database[top_doc_id]['text']}")