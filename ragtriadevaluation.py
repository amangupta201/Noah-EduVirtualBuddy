import logging
import numpy as np

from config import OPENAI_API_KEY
from embedding import EmbeddingModel, ChromaCompatibleEmbeddingFunction
from vector_store import VectorStore

# ---------------------------
# Logging setup
# ---------------------------
logging.basicConfig(filename='trulens.logs',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------
# Initialize Vector Store and Embedding Model
# ---------------------------
try:
    embedding_model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
    embedding_fn = ChromaCompatibleEmbeddingFunction(embedding_model)

    vector_store = VectorStore(
        embedding_dim=384,  # embedding size of MiniLM-L6-v2
        embedding_function=embedding_fn,
        persist_directory="./chroma_db"  # Make sure this matches your Flask setup
    )
    print("✅ VectorStore loaded from Flask's Chroma DB.")
except Exception as e:
    logging.error("❌ Failed to load VectorStore", exc_info=True)

# ---------------------------
# TruLens-compatible RAG wrapper
# ---------------------------
from trulens.apps.app import TruApp, instrument
from trulens.core import TruSession
from openai import OpenAI

session = TruSession()
session.reset_database()

oai_client = OpenAI(api_key=OPENAI_API_KEY)

class RAG:
    @instrument
    def retrieve(self, query: str) -> list:
        try:
            query_embedding = embedding_model.encode([query])[0]
            return vector_store.search(query_embedding, k=4)
        except Exception as e:
            logging.error("❌ Error during retrieve", exc_info=True)
            return []

    @instrument
    def generate_completion(self, query: str, context_str: list) -> str:
        try:
            if not context_str:
                return "Sorry, I couldn't find relevant information."
            completion = oai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0,
                messages=[
                    {"role": "user", "content": f"We have provided context below.\n"
                                                f"---------------------\n{context_str}\n"
                                                f"---------------------\n"
                                                f"Based on this, answer the question: {query}"}
                ]
            ).choices[0].message.content
            return completion or "No response generated."
        except Exception as e:
            logging.error("❌ Error during completion", exc_info=True)
            return "Error generating completion."

    @instrument
    def query(self, query: str) -> str:
        try:
            context_str = self.retrieve(query=query)
            return self.generate_completion(query=query, context_str=context_str)
        except Exception as e:
            logging.error("❌ Error during query", exc_info=True)
            return "Error processing the query."

rag = RAG()

# ---------------------------
# TruLens Feedback Configuration
# ---------------------------
from trulens.core import Feedback, Select
from trulens.providers.openai import OpenAI as OpenAIFeedback

provider = OpenAIFeedback(model_engine="gpt-4.1-mini", api_key=OPENAI_API_KEY)

# ---------------------------
# Feedbacks Setup
# ---------------------------
f_groundedness = Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness") \
    .on(Select.RecordCalls.retrieve.rets.collect())  # Collect feedback for retrieval
    .on_output()  # Groundedness measure on the generated output

f_answer_relevance = Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance") \
    .on_input().on_output()  # Relevance measure for both input and output

f_context_relevance = Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance") \
    .on_input().on(Select.RecordCalls.retrieve.rets[:])  # Context relevance feedback based on retrieved context
    .aggregate(np.mean)  # Optionally, aggregate if you are dealing with multiple sources of context


# Add feedbacks to TruApp
tru_rag = TruApp(
    rag,
    app_name="RAG",
    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance],
)

# ---------------------------
# Run queries and record feedback
# ---------------------------
try:
    with tru_rag as recording:
        response_1 = rag.query("What is the proper procedure for requesting time off, and how much notice is required?")
        response_2 = rag.query("Who should I talk to if I experience or witness harassment in the workplace?")
        
        print(response_1)
        print(response_2)
except Exception as e:
    logging.error("❌ Error during TruLens evaluation run", exc_info=True)

# ---------------------------
# Launch TruLens dashboard
# ---------------------------
from trulens.dashboard import run_dashboard

try:
    run_dashboard(session, port=8502)
except Exception as e:
    logging.error("❌ Dashboard launch failed", exc_info=True)
