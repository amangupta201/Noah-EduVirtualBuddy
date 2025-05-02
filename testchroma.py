from pdf_loader import process_pdfs, chunk_text
from embedding import EmbeddingModel, ChromaCompatibleEmbeddingFunction
from vector_store import VectorStore
import numpy as np

# Load PDF text
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
