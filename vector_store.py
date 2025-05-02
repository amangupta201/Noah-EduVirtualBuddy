# vector_store.py
import chromadb
import uuid
from datetime import datetime
import numpy as np

class VectorStore:
    def __init__(self, embedding_dim: int, embedding_function, persist_directory: str = "./chroma_db"):
        self.embedding_dim = embedding_dim
        self.embedding_function = embedding_function

        # ✅ Use new Chroma client format
        self.client = chromadb.PersistentClient(path=persist_directory)

        self.collection = self.client.get_or_create_collection(
            name="rag_collection",
            embedding_function=self.embedding_function,
            metadata={
                "description": "Collection for RAG pipeline",
                "created": str(datetime.now())
            }
        )

    def add_embeddings(self, embeddings, chunks):
        embeddings_list = embeddings.tolist()
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

        self.collection.add(
            embeddings=embeddings_list,
            documents=chunks,
            ids=ids
        )

    def search(self, query_embedding, k=5):
        query_embedding_list = query_embedding.tolist()

        results = self.collection.query(
            query_embeddings=query_embedding_list,
            n_results=k
        )
        return results['documents'][0] if results['documents'] else []

    def list_all_ids(self):
        all_data = self.collection.get()
        return all_data.get("ids", [])
