# -------------------------------------
# Imports & Setup
# -------------------------------------
import logging
import os
import joblib
import numpy as np

from config import OPENAI_API_KEY
from pdf_loader import load_pdf_text, process_pdfs, chunk_text
from embedding import EmbeddingModel, ChromaCompatibleEmbeddingFunction

import chromadb
from chromadb.config import Settings

from trulens_eval import Tru
from trulens_eval.tru_app import App as TruApp
from trulens_eval.feedback import Feedback
from trulens_eval.feedback.provider.openai import OpenAI as OpenAIFeedback
from trulens_eval.tru_custom_app import instrument
from trulens_eval.guardrails.context import context_filter
from trulens_eval.dashboard import run_dashboard

# -------------------------------------
# Logging
# -------------------------------------
logging.basicConfig(filename='trulens.logs',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------------------------
# Embedding and Vector Store Setup
# -------------------------------------
try:
    custom_model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
    embedding_function = ChromaCompatibleEmbeddingFunction(custom_model)

    chroma_client = chromadb.PersistentClient(path="./trulens_chroma_db")
    vector_store = chroma_client.get_or_create_collection(
        name="Vehicles", embedding_function=embedding_function
    )

    pdfs = ["uploads/EST03.pdf"]
    CACHE_FILE = "embedding_cache.pkl"
    USE_CACHE = True

    texts_to_add = []
    ids_to_add = []

    if USE_CACHE and os.path.exists(CACHE_FILE):
        print("🔁 Loading cached embeddings...")
        cache = joblib.load(CACHE_FILE)
        texts_to_add = cache["texts"]
        ids_to_add = cache["ids"]
    else:
        print("⚙️ Processing PDFs and generating embeddings...")
        processed_texts = process_pdfs(pdfs)
        documents = []

        for text in processed_texts.values():
            if text.strip():
                documents.extend(chunk_text(text))

        texts = [doc if isinstance(doc, str) else doc.page_content for doc in documents]
        existing_ids = set(vector_store.get()["ids"])

        for i, text in enumerate(texts):
            doc_id = f"doc_{i}"
            if doc_id not in existing_ids:
                texts_to_add.append(text)
                ids_to_add.append(doc_id)

        joblib.dump({"texts": texts_to_add, "ids": ids_to_add}, CACHE_FILE)
        print("✅ Embedding data cached.")

    if texts_to_add:
        vector_store.add(documents=texts_to_add, ids=ids_to_add)

except Exception as e:
    logging.error(f"Error during initialization or document processing: {e}", exc_info=True)

# -------------------------------------
# TruLens Session
# -------------------------------------
session = Tru().start_session()
session.reset_database()

# -------------------------------------
# RAG Class
# -------------------------------------
from openai import OpenAI

oai_client = OpenAI(api_key=OPENAI_API_KEY)

class RAG:
    @instrument
    def retrieve(self, query: str) -> list:
        try:
            results = vector_store.query(query_texts=query, n_results=4)
            return [doc for sublist in results["documents"] for doc in sublist]
        except Exception as e:
            logging.error(f"Error during retrieve: {e}", exc_info=True)
            return []

    @instrument
    def generate_completion(self, query: str, context_str: list) -> str:
        try:
            if not context_str:
                return "Sorry, I couldn't find an answer to your question."
            completion = oai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": f"We have provided context information below.\n"
                               f"---------------------\n{context_str}\n"
                               f"---------------------\nThen, given this information, please answer the question: {query}",
                }]
            ).choices[0].message.content
            return completion or "Did not find an answer."
        except Exception as e:
            logging.error(f"Error generating completion: {e}", exc_info=True)
            return "Error generating response."

    @instrument
    def query(self, query: str) -> str:
        try:
            context_str = self.retrieve(query=query)
            return self.generate_completion(query=query, context_str=context_str)
        except Exception as e:
            logging.error(f"Error during full query: {e}", exc_info=True)
            return "Error processing the query."

# -------------------------------------
# Feedback Functions
# -------------------------------------
provider = OpenAIFeedback(model_engine="gpt-4.1-mini", api_key=OPENAI_API_KEY)
guardrail_provider = OpenAIFeedback(model_engine="gpt-4.1-nano", api_key=OPENAI_API_KEY)

f_groundedness = Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness") \
    .on(TruApp.select().RecordCalls.retrieve.rets.collect()) \
    .on_output()

f_answer_relevance = Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance") \
    .on_input().on_output()

f_context_relevance = Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance") \
    .on_input().on(TruApp.select().RecordCalls.retrieve.rets[:]) \
    .aggregate(np.mean)

# -------------------------------------
# TruApp Evaluation (Base RAG)
# -------------------------------------
rag = RAG()

tru_rag = TruApp(
    rag,
    app_name="RAG",
    app_version="base",
    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance],
)

try:
    with tru_rag as recording:
        rag.query("What is the proper procedure for requesting time off, and how much notice is required?")
        rag.query("Who should I talk to if I experience or witness harassment in the workplace?")
        session.get_leaderboard()
except Exception as e:
    logging.error("Error during TruApp base execution", exc_info=True)

# -------------------------------------
# Guardrail Filtering and Evaluation
# -------------------------------------
f_context_relevance_score = Feedback(
    guardrail_provider.context_relevance, name="Context Relevance"
)

class FilteredRAG(RAG):
    @instrument
    @context_filter(feedback=f_context_relevance_score, threshold=0.75, keyword_for_prompt="query")
    def retrieve(self, query: str) -> list:
        try:
            results = vector_store.query(query_texts=query, n_results=4)
            return [doc for sublist in results["documents"] for doc in sublist] if "documents" in results else []
        except Exception as e:
            logging.error(f"Error in filtered retrieve: {e}", exc_info=True)
            return []

filtered_rag = FilteredRAG()
filtered_tru_rag = TruApp(
    filtered_rag,
    app_name="RAG",
    app_version="filtered",
    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance],
)

try:
    with filtered_tru_rag as recording:
        filtered_rag.query("What is the company policy on personal phone use or social media during work hours?")
        session.get_leaderboard()
except Exception as e:
    logging.error("Error during Filtered TruApp execution", exc_info=True)

# -------------------------------------
# Start Dashboard
# -------------------------------------
try:
    run_dashboard(session, port=8501)
except Exception as e:
    logging.error("Error starting dashboard", exc_info=True)
