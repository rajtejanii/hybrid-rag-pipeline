from langchain_text_splitters import RecursiveCharacterTextSplitter
# 1. The Raw Data (Imagine this is a messy PDF we just extracted text from)
raw_document = """
Machine learning models require significant computational resources. 
The training phase often utilizes clusters of GPUs to process large datasets efficiently. 
In contrast, the inference phase, where the model makes predictions, can sometimes be run on CPUs, depending on latency requirements.
Historically, recurrent neural networks (RNNs) were used for sequence tasks. 
However, the Transformer architecture introduced in 2017 replaced RNNs because self-attention mechanisms allow for highly parallelized training.
"""

# 2. Configure the Splitter
# We use 'RecursiveCharacterTextSplitter' because it tries to split on paragraphs first, 
# then sentences, keeping natural language thoughts together rather than cutting words in half.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,       # The max number of characters per chunk
    chunk_overlap=20,     # Overlap prevents cutting a crucial sentence in half
    length_function=len
)

# 3. Process the Document
raw_chunks = text_splitter.split_text(raw_document)

# 4. Attach Metadata (Crucial for Citations)
processed_database = []

for index, chunk_text in enumerate(raw_chunks):
    document_row = {
        "chunk_id": f"doc_1_chunk_{index}",
        "source": "Architecture_Report.pdf",
        "text": chunk_text
    }
    processed_database.append(document_row)

# Let's see what we built
for row in processed_database:
    print(f"ID: {row['chunk_id']}")
    print(f"Text: {row['text']}\n")