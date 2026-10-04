import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. PREVIOUS STEP: Chunking the Data ---
raw_document = """
Machine learning models require significant computational resources. 
The training phase often utilizes clusters of GPUs to process large datasets efficiently. 
In contrast, the inference phase, where the model makes predictions, can sometimes be run on CPUs, depending on latency requirements.
Historically, recurrent neural networks (RNNs) were used for sequence tasks. 
However, the Transformer architecture introduced in 2017 replaced RNNs because self-attention mechanisms allow for highly parallelized training.
"""

text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20, length_function=len)
raw_chunks = text_splitter.split_text(raw_document)

processed_database = []
for index, chunk_text in enumerate(raw_chunks):
    processed_database.append({
        "chunk_id": f"doc_1_chunk_{index}",
        "text": chunk_text
    })

# --- 2. NEW STEP: Building the BM25 Index ---
# We extract just the text strings from our database of dictionaries
chunk_texts = [row["text"] for row in processed_database]

# BM25 requires tokenized text (a list of words) to calculate term frequencies
tokenized_corpus = [text.lower().split() for text in chunk_texts]
bm25_index = BM25Okapi(tokenized_corpus)
print("✅ BM25 Index Built Successfully!")

# --- 3. NEW STEP: Building the FAISS Vector Index ---
# We load a lightweight, highly efficient open-source embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Convert the text chunks into mathematical arrays
dense_embeddings = embedder.encode(chunk_texts)

# Determine the dimensions of our vectors (this model uses 384 dimensions)
dimension = dense_embeddings.shape[1]

# Initialize FAISS to calculate the Euclidean (L2) distance between vectors
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(dense_embeddings)
print(f"✅ FAISS Index Built! Total vectors stored: {faiss_index.ntotal}")