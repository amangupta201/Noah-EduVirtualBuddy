from pdf_loader import process_pdfs, chunk_text
from embedding import EmbeddingModel, ChromaCompatibleEmbeddingFunction
from vector_store import VectorStore
import numpy as np

# Function to simulate the QA process
def ask_question(query: str, vs: VectorStore, embedding_model: EmbeddingModel):
    # Step 1: Generate embedding for the question
    query_embedding = embedding_model.encode([query])

    # Step 2: Search the vector store for the most relevant chunk
    most_relevant_chunk = vs.search(query_embedding[0])

    if most_relevant_chunk:
        return most_relevant_chunk
    else:
        return "Sorry, I couldn't find an answer to your question."

# Load the PDF text
pdf_paths = ["sample_test.pdf"]
pdf_data = process_pdfs(pdf_paths)

# Initialize embedder
embedding_model = EmbeddingModel()
chroma_embedding_fn = ChromaCompatibleEmbeddingFunction(embedding_model)

# ✅ New VectorStore with updated client
vs = VectorStore(embedding_dim=384, embedding_function=chroma_embedding_fn)

# Chunk and embed
for path, text in pdf_data.items():
    chunks = chunk_text(text)
    if not chunks:
        print(f"No chunks created for {path}")
        continue

    embeddings = embedding_model.encode(chunks)
    if embeddings.size == 0:
        print(f"No embeddings generated for {path}")
        continue

    vs.add_embeddings(embeddings, chunks)
    print(f"✅ Stored {len(chunks)} chunks for {path}")

# Example usage: Ask a question to the bot
question = "What is the main topic of the PDF?"
answer = ask_question(question, vs, embedding_model)
print(f"Answer: {answer}")
