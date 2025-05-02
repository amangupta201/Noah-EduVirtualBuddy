import requests
from config import OPENAI_API_KEY, OPENAI_MODEL

class RAGPipeline:
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store

    def ask(self, question):
        """
        Generate a response to a question using RAG.

        Args:
            question (str): User question.

        Returns:
            str: Model-generated answer.
        """
        query_embedding = self.embedder.encode([question])
        relevant_chunks = self.vector_store.search(query_embedding, k=5)
        context = "\n\n".join(relevant_chunks)

        prompt = f"Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"

        payload = {
            "model": OPENAI_MODEL,
            "messages": [{"role": "user", "content": prompt}]
        }

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
