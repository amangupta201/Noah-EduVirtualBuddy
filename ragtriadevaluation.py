import logging
import os
import pickle
import numpy as np

from config import OPENAI_API_KEY
from pdf_loader import load_pdf_text, process_pdfs, chunk_text
from embedding import EmbeddingModel, ChromaCompatibleEmbeddingFunction

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from trulens.apps.app import TruApp, instrument
from trulens.core import TruSession, Feedback, Select
from trulens.providers.openai import OpenAI as OpenAIFeedback
from trulens.core.guardrails.base import context_filter
from trulens.dashboard import run_dashboard

# Setup error logging
logging.basicConfig(filename='trulens.logs',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # Initialize embedding
    custom_model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
    embedding_function = ChromaCompatibleEmbeddingFunction(custom_model)

    # Initialize ChromaDB with persistent storage
    chroma_client = chromadb.PersistentClient(path="./trulens_chroma_db")
    vector_store = chroma_client.get_or_create_collection(
        name="Vehicles", embedding_function=embedding_function
    )

    cache_file = "cached_embeddings.pkl"

    if os.path.exists(cache_file):
        logging.info("Loading cached embeddings...")
        with open(cache_file, "rb") as f:
            texts, embeddings = pickle.load(f)
        vector_store.add_embeddings(embeddings, texts)
        logging.info("Cached embeddings loaded successfully.")
    else:
        logging.info("Processing PDFs and generating embeddings...")
        pdfs = ["uploads/Advertising.pdf"]
        processed_texts = process_pdfs(pdfs)

        documents = []
        for text in processed_texts.values():
            if text.strip():
                documents.extend(chunk_text(text))

        texts = [doc if isinstance(doc, str) else doc.page_content for doc in documents]
        embeddings = embedding_function.embed_documents(texts)

        vector_store.add_embeddings(embeddings, texts)

        with open(cache_file, "wb") as f:
            pickle.dump((texts, embeddings), f)
        logging.info("Embeddings generated and cached.")

    # TruLens session init
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

    rag = RAG()

    # Feedbacks
    provider = OpenAIFeedback(model_engine="gpt-4.1-mini", api_key=OPENAI_API_KEY)
    guardrail_provider = OpenAIFeedback(model_engine="gpt-4.1-nano", api_key=OPENAI_API_KEY)

    f_groundedness = Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness") \
        .on(Select.RecordCalls.retrieve.rets.collect()) \
        .on_output()

    f_answer_relevance = Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance") \
        .on_input().on_output()

    f_context_relevance = Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance") \
        .on_input().on(Select.RecordCalls.retrieve.rets[:]) \
        .aggregate(np.mean)

    tru_rag = TruApp(
        rag,
        app_name="RAG",
        app_version="base",
        feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance],
    )

    with tru_rag as recording:
        rag.query("What is the proper procedure for requesting time off, and how much notice is required?")
        rag.query("Who should I talk to if I experience or witness harassment in the workplace?")
        session.get_leaderboard()

    # Guardrail Filtering
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

    with filtered_tru_rag as recording:
        filtered_rag.query("What is the company policy on personal phone use or social media during work hours?")
        session.get_leaderboard()

    # Run dashboard
    run_dashboard(session, port=8501)

except Exception as e:
    logging.error("Error in full pipeline execution", exc_info=True)
