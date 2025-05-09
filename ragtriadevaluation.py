import logging
import os
import pickle
from config import OPENAI_API_KEY
from pdf_loader import load_pdf_text, process_pdfs, chunk_text
from embedding import EmbeddingModel, ChromaCompatibleEmbeddingFunction
import chromadb
from chromadb.config import Settings
from trulens.apps.app import TruApp, instrument
from trulens.core import TruSession, Feedback, Select # Add Feedback and Select here
from openai import OpenAI
from trulens.providers.openai import OpenAI  # Use OpenAI instead of OpenAIFeedback
import numpy as np

# Setup basic error logging to file (trulens.log)
logging.basicConfig(filename='trulens.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Cache file for embeddings
CACHE_FILE = "cached_embeddings.pkl"
USE_CACHE = True

# Initialize embedding model and Chroma
custom_model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
embedding_function = ChromaCompatibleEmbeddingFunction(custom_model)

# Use PersistentClient for Chroma
chroma_client = chromadb.PersistentClient(path="./trulens_chroma_db")
vector_store = chroma_client.get_or_create_collection(
    name="Vehicles", embedding_function=embedding_function
)

# Load and process PDFs in batches
def process_pdfs_in_batches(pdf_paths, batch_size=10):
    all_texts = []
    all_ids = []
    batch = []

    for i, pdf_path in enumerate(pdf_paths):
        batch.append(pdf_path)
        if len(batch) == batch_size or i == len(pdf_paths) - 1:
            processed_texts = process_pdfs(batch)
            documents = []
            for text in processed_texts.values():
                if text.strip():
                    documents.extend(chunk_text(text))

            texts = [doc if isinstance(doc, str) else doc.page_content for doc in documents]
            all_texts.extend(texts)
            all_ids.extend([f"doc_{i}" for i in range(len(all_texts) - len(batch), len(all_texts))])

            batch = []  # Reset batch after processing

    return all_texts, all_ids

# Check for cached embeddings
def load_cached_embeddings():
    if USE_CACHE and os.path.exists(CACHE_FILE):
        print("🔁 Loading cached embeddings...")
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
        return cache["texts"], cache["ids"]
    return [], []

# Save embeddings to cache
def save_embeddings_to_cache(texts, ids):
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"texts": texts, "ids": ids}, f)
    print("✅ Embedding data cached.")

# Process and cache embeddings
def process_and_cache_embeddings(pdf_paths):
    existing_ids = set(vector_store.get()["ids"])
    texts_to_add, ids_to_add = load_cached_embeddings()

    # Process PDFs in batches of 10
    texts, ids = process_pdfs_in_batches(pdf_paths)

    for i, text in enumerate(texts):
        doc_id = ids[i]
        if doc_id not in existing_ids:
            texts_to_add.append(text)
            ids_to_add.append(doc_id)

    if texts_to_add:
        # Add new embeddings to Chroma DB
        vector_store.add(documents=texts_to_add, ids=ids_to_add)
        # Cache the new embeddings
        save_embeddings_to_cache(texts_to_add, ids_to_add)

# -------------------------------------
# TruLens Integration
# -------------------------------------
session = TruSession()
session.reset_database()

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
                               f"---------------------\nThen, given this information, please answer the question: {query}"}]
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

rag = RAG()

# -------------------------------------
# Feedback Functions and Evaluation
# -------------------------------------
provider = OpenAI(model_engine="gpt-4.1-mini", api_key=OPENAI_API_KEY)
guardrail_provider = OpenAI(model_engine="gpt-4.1-nano", api_key=OPENAI_API_KEY)

f_groundedness = Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness") \
    .on(Select.RecordCalls.retrieve.rets.collect()) \
    .on_output()

f_answer_relevance = Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance") \
    .on_input().on_output()

f_context_relevance = Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance") \
    .on_input().on(Select.RecordCalls.retrieve.rets[:]) \
    .aggregate(np.mean)

# TruApp for RAG integration
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
# Run TruLens Dashboard on 0.0.0.0:8502
# -------------------------------------
from trulens.dashboard import run_dashboard

try:
    run_dashboard(session, port=8502)
    print("TruLens dashboard started successfully!")
except Exception as e:
    logging.error("Error starting dashboard", exc_info=True)
    print(f"Error starting dashboard: {e}")

