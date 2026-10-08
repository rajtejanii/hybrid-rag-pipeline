# Hybrid Retrieval-Augmented Generation (RAG) Pipeline

An advanced, end-to-end Retrieval-Augmented Generation (RAG) pipeline designed for high-accuracy data extraction and analytical querying. This system combines dense vector retrieval with sparse keyword search (Hybrid Search), utilizes cross-encoder reranking for precision, and implements strict citation guardrails to prevent AI hallucinations and ensure data reliability.

Built to process complex, unstructured datasets and translate them into actionable, verifiable insights.

##  Key Features

*   **Hybrid Search Architecture:** Merges semantic vector search with BM25 keyword indexing to capture both contextual meaning and exact-match terminology.
*   **Reciprocal Rank Fusion (RRF):** Intelligently fuses search scores from both sparse and dense retrievers to surface the most relevant documents.
*   **Cross-Encoder Reranking:** Applies a secondary machine learning model to score and re-order the fused results, maximizing the relevance of the context passed to the LLM.
*   **Citation Guardrails:** Enforces deterministic output mapping by requiring the generation model to explicitly cite the source chunks used for its claims, preventing hallucinations.
*   **Modular Pipeline:** Developed iteratively with decoupled stages (ingestion, indexing, querying, reranking, and validation) consolidated into a streamlined execution script.

##  System Architecture

The pipeline follows a strict data flow to ensure maximum accuracy:

*   **Ingestion & Chunking:** Raw documents are loaded using LangChain and Pandas/NumPy, then processed via semantic chunking to maintain logical boundaries.
*   **Dual-Indexing:** Chunks are simultaneously embedded into a Vector Store and a sparse Keyword Index.
*   **Query Processing:** User queries are vectorized and processed through both indexes simultaneously.
*   **Fusion & Reranking:** Results are fused using RRF, then a Cross-Encoder evaluates the absolute relevance of the query-document pairs.
*   **Generation & Guardrails:** The highly curated context is passed to the LLM, wrapped in a strict validation prompt that forces in-line citations mapped to the original dataset.

##  Repository Structure

The project was built using an iterative modular approach, moving from isolated pipeline steps to a unified execution script:

```text
├── modules/
│   ├── 1_ingest_and_chunk.py       # Data ingestion and semantic splitting
│   ├── 2_build_indexes.py          # Vector embedding and keyword indexing
│   ├── 3_query_and_fuse.py         # Reciprocal Rank Fusion implementation
│   ├── 4_cross_encoder.py          # Cross-encoder reranking logic
│   └── 5_citation_guardrail.py     # Output validation and hallucination prevention
├── hybrid_rag.py                   # Main consolidated execution pipeline
├── requirements.txt                # Project dependencies
└── README.md                       # System documentation
```

##  Installation & Setup

**Clone the repository:**
```bash
git clone [https://github.com/rajtejanii/hybrid-rag-pipeline.git](https://github.com/rajtejanii/hybrid-rag-pipeline.git)
cd hybrid-rag-pipeline
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Configure Environment Variables:**
Ensure your LLM backend (e.g., Ollama or relevant API keys) is running and accessible. 

##  Usage

The entire pipeline is consolidated for easy execution. Run the main script to initialize the data processing, build the indexes, and execute a query with full guardrails:

```bash
python hybrid_rag.py
```
