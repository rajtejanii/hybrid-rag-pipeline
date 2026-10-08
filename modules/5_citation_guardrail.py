from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# --- 1. THE WINNING CHUNK (From Step 6) ---
# In a real app, these variables come directly from the cross-encoder output
retrieved_chunk_id = "doc_1_chunk_2"
retrieved_text = "In contrast, the inference phase, where the model makes predictions, can sometimes be run on CPUs, depending on latency requirements."
user_query = "What hardware is typically used for inference?"

# --- 2. DEFINE THE STRICT SCHEMA ---
# We force the LLM to conform to this exact structure. 
class CitedAnswer(BaseModel):
    answer: str = Field(description="The direct answer to the user's question based ONLY on the provided context.")
    source_chunk_id: str = Field(description="The exact chunk_id of the document used to answer. If the context does not contain the answer, output 'NONE'.")
    is_supported: bool = Field(description="True if the answer is completely supported by the provided text, False otherwise.")

# --- 3. CONFIGURE THE LLM ---
# We use the local Ollama model you used for your previous agents
llm = ChatOllama(model="llama3.2", temperature=0)

# Bind the Pydantic schema to the model so it outputs structured data, not free text
structured_llm = llm.with_structured_output(CitedAnswer)

# --- 4. BUILD THE PROMPT ---
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

# --- 5. EXECUTE THE GUARDRAIL ---
print("🧠 Formatting prompt and querying LLM...")
formatted_prompt = prompt_template.invoke({
    "context": retrieved_text,
    "context_id": retrieved_chunk_id,
    "query": user_query
})

result = structured_llm.invoke(formatted_prompt)

# --- 6. VERIFY OUTPUT ---
print("\n✅ SECURE RAG RESPONSE GENERATED:")
print(f"Answer: {result.answer}")
print(f"Cited Source: {result.source_chunk_id}")
print(f"Is fully supported by text? {result.is_supported}")

# The application can now safely reject the answer if `is_supported` is False