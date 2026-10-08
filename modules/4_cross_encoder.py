import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. SETUP ---
raw_document = """
Machine learning models require significant computational resources. 
The training phase often utilizes clusters of GPUs to process large datasets efficiently. 
In contrast, the inference phase, where the model makes predictions, can sometimes be run on CPUs, depending on latency requirements.
Historically, recurrent neural networks (RNNs) were used for sequence tasks. 
However, the Transformer architecture introduced in 2017 replaced RNNs because self-attention mechanisms allow for highly parallelized training.
"""

text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20, length_function=len)
processed_database = [{"chunk_id": f"doc_1_chunk_{i}", "text": chunk} for i, chunk in enumerate(text_splitter.split_text(raw_document))]
chunk_texts = [row["text"] for row in processed_database]

bm25_index = BM25Okapi([text.lower().split() for text in chunk_texts])
embedder = SentenceTransformer('all-MiniLM-L6-v2')
faiss_index = faiss.IndexFlatL2(384)
faiss_index.add(embedder.encode(chunk_texts))

# --- 2. DUAL QUERY & RRF ---
query = "What hardware is typically used for inference?"
print(f"🔍 Query: '{query}'\n")

bm25_ranks = np.argsort(bm25_index.get_scores(query.lower().split()))[::-1]
_, faiss_ranks = faiss_index.search(embedder.encode([query]), len(processed_database))

rrf_scores = {i: 0.0 for i in range(len(processed_database))}
for rank, doc_id in enumerate(bm25_ranks): rrf_scores[doc_id] += 1.0 / (60 + rank + 1)
for rank, doc_id in enumerate(faiss_ranks[0]): rrf_scores[doc_id] += 1.0 / (60 + rank + 1)
fused_results = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

print("❌ RRF RESULT (Fast but imprecise):")
print(f"Text: {processed_database[fused_results[0]]['text']}\n")

# --- 3. NEW LOGIC: CROSS-ENCODER RERANKING ---
# We load a model specifically trained by Microsoft on search engine queries
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# We pair the exact query with the text of each document in our fused list
rerank_pairs = [[query, processed_database[doc_id]["text"]] for doc_id in fused_results]

# The model scores the actual relevance of the pair
rerank_scores = reranker.predict(rerank_pairs)

# We resort our final list based on the Cross-Encoder's high-precision scores
final_ranked_indices = [doc_id for _, doc_id in sorted(zip(rerank_scores, fused_results), reverse=True)]

print("🎯 RERANKED RESULT (Slow but highly precise):")
best_doc_id = final_ranked_indices[0]
print(f"ID: {processed_database[best_doc_id]['chunk_id']}")
print(f"Text: {processed_database[best_doc_id]['text']}")