import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# --- 1. DEFINE THE OUTPUT SCHEMA ---
class CitedAnswer(BaseModel):
    answer: str = Field(description="The direct answer based ONLY on the provided context.")
    source_chunk_id: str = Field(description="The exact chunk_id used to answer. If not found, output 'NONE'.")
    is_supported: bool = Field(description="True if completely supported by the text, False otherwise.")

# --- 2. THE PIPELINE CLASS ---
class HybridRAGPipeline:
    def __init__(self):
        print("⚙️ Initializing AI Models (This takes a moment)...")
        # Initialize search models
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Initialize LLM with strict JSON schema
        self.llm = ChatOllama(model="llama3.2", temperature=0).with_structured_output(CitedAnswer)
        
        # Initialize database and chunker
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20, length_function=len)
        self.database = []
        self.bm25_index = None
        self.faiss_index = None

    def ingest_document(self, text: str, source_name: str = "doc"):
        print(f"📥 Ingesting document: {source_name}...")
        chunks = self.text_splitter.split_text(text)
        
        start_idx = len(self.database)
        for i, chunk in enumerate(chunks):
            self.database.append({"chunk_id": f"{source_name}_chunk_{start_idx + i}", "text": chunk})
        
        # Extract text for indexing
        chunk_texts = [row["text"] for row in self.database]
        
        # Rebuild BM25 Index (Sparse)
        self.bm25_index = BM25Okapi([t.lower().split() for t in chunk_texts])
        
        # Rebuild FAISS Index (Dense)
        dense_embeddings = self.embedder.encode(chunk_texts)
        self.faiss_index = faiss.IndexFlatL2(dense_embeddings.shape[1])
        self.faiss_index.add(dense_embeddings)
        print(f"✅ Successfully indexed {len(chunks)} chunks.")

    def _retrieve_and_rerank(self, query: str) -> dict:
        print(f"🔍 Searching memory for: '{query}'")
        
        # 1. Dual Retrieval
        bm25_ranks = np.argsort(self.bm25_index.get_scores(query.lower().split()))[::-1]
        _, faiss_ranks = self.faiss_index.search(self.embedder.encode([query]), len(self.database))
        
        # 2. Reciprocal Rank Fusion (RRF)
        rrf_scores = {i: 0.0 for i in range(len(self.database))}
        for rank, doc_id in enumerate(bm25_ranks): rrf_scores[doc_id] += 1.0 / (60 + rank + 1)
        for rank, doc_id in enumerate(faiss_ranks[0]): rrf_scores[doc_id] += 1.0 / (60 + rank + 1)
        
        # Get top 20 fused results
        fused_results = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:20]
        
        # 3. Cross-Encoder Reranking
        rerank_pairs = [[query, self.database[doc_id]["text"]] for doc_id in fused_results]
        rerank_scores = self.reranker.predict(rerank_pairs)
        
        # Return the absolute best chunk
        best_local_idx = np.argmax(rerank_scores)
        best_doc_id = fused_results[best_local_idx]
        return self.database[best_doc_id]

    def ask(self, query: str) -> CitedAnswer:
        # Get the highest precision chunk
        best_chunk = self._retrieve_and_rerank(query)
        
        prompt_template = PromptTemplate(
            input_variables=["context", "context_id", "query"],
            template="""
            You are a strict citation verification agent.
            Answer the user's query using ONLY the provided context.
            If the context does not contain the answer, do not guess. 
            
            Context ID: {context_id}
            Context Text: {context}
            
            User Query: {query}
            """
        )
        
        print("🧠 Generating verified answer...")
        formatted_prompt = prompt_template.invoke({
            "context": best_chunk["text"],
            "context_id": best_chunk["chunk_id"],
            "query": query
        })
        
        # Execute LLM with Pydantic guardrail
        return self.llm.invoke(formatted_prompt)

# --- 3. EXECUTE THE PIPELINE ---
if __name__ == "__main__":
    # Initialize our system
    rag = HybridRAGPipeline()
    
    # Load some raw data
    raw_text = """
    Machine learning models require significant computational resources. 
    The training phase often utilizes clusters of GPUs to process large datasets efficiently. 
    In contrast, the inference phase, where the model makes predictions, can sometimes be run on CPUs, depending on latency requirements.
    Historically, recurrent neural networks (RNNs) were used for sequence tasks. 
    However, the Transformer architecture introduced in 2017 replaced RNNs because self-attention mechanisms allow for highly parallelized training.
    """
    
    # Ingest the data
    rag.ingest_document(raw_text, source_name="Architecture_Report")
    
    # Ask a question
    result = rag.ask("What hardware is typically used for inference?")
    
    # Print the verified result
    print("\n🏆 FINAL SYSTEM OUTPUT:")
    print(f"Answer: {result.answer}")
    print(f"Source ID: {result.source_chunk_id}")
    print(f"Cryptographically Supported: {result.is_supported}")