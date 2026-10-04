# Enterprise Hybrid-Search RAG Pipeline

A production-grade Retrieval-Augmented Generation (RAG) pipeline built to solve semantic blind spots and eliminate LLM hallucinations. 

## Architecture
1. **Dual Retrieval:** Combines sparse keyword search (BM25) for exact terminology with dense semantic search (FAISS + all-MiniLM-L6-v2) for conceptual matching.
2. **Reciprocal Rank Fusion (RRF):** Merges bounded vector distances and unbounded BM25 scores algorithmically based on rank position.
3. **Cross-Encoder Reranking:** Passes the top fused candidates through `ms-marco-MiniLM-L-6-v2` for high-precision attention-based scoring.
4. **Citation Guardrails:** Utilizes strictly typed Pydantic schemas to force the LLM (Llama 3.2) to return boolean verification flags and explicit source chunk IDs, guaranteeing zero hallucinations.