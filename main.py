import os
import logging
import datetime
import boto3
import uuid
from flask import Flask, request, jsonify, send_from_directory
from pdf_loader import load_pdf_text, chunk_text, process_pdfs
from embedding import EmbeddingModel, ChromaCompatibleEmbeddingFunction
from vector_store import VectorStore
from rag_pipeline import RAGPipeline

# Initialize the Flask app
app = Flask(__name__, static_folder='frontend')

# Initialize the Embedding Model and Vector Store
embedder = EmbeddingModel()
embedding_function = ChromaCompatibleEmbeddingFunction(embedder)
vector_store = VectorStore(embedding_dim=768, embedding_function=embedding_function)
rag_pipeline = RAGPipeline(embedder, vector_store)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

dynamodb = boto3.resource('dynamodb', region_name='eu-north-1')
feedback_table = dynamodb.Table('RAGFeedback')

# Setting up logging to log questions and answers
log_filename = 'qa_bot_logs.log'
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s - %(message)s')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def file_exists(filename):
    return os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename))


@app.route('/admin')
def serve_admin_index():
    logging.info("First touchpoint with backend: Admin index served")
    return send_from_directory(app.static_folder, 'admin.html')


@app.route('/user')
def serve_user_index():
    logging.info("First touchpoint with backend: User index served")
    return send_from_directory(app.static_folder, 'user.html')


@app.route('/upload', methods=['POST'])
def upload_pdf():
    logging.info("First touchpoint with backend: PDF upload initiated")
    files = request.files.getlist('files')
    if not files:
        logging.error("No files part in the request")
        return jsonify({"error": "No files part"}), 400

    pdf_paths = []
    existing_files = []
    for file in files:
        if file and allowed_file(file.filename):
            if file_exists(file.filename):
                existing_files.append(file.filename)
            else:
                filename = f"{file.filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                pdf_paths.append(filepath)

    if existing_files:
        logging.error(f"The following files already exist: {', '.join(existing_files)}")
        return jsonify({"error": f"The following files already exist: {', '.join(existing_files)}"}), 409

    batch_size = 10
    try:
        pdf_data = process_pdfs(pdf_paths)
        for path, text in pdf_data.items():
            chunks = chunk_text(text)
            logging.info("Embedding creation: Starting embedding process")
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_embeddings = embedder.encode(batch_chunks)
                vector_store.add_embeddings(batch_embeddings, batch_chunks)
                logging.info("Embedding creation: Embeddings added to vector store")

        return jsonify({"message": "PDFs uploaded and processed successfully"}), 200

    except Exception as e:
        logging.exception("Error during PDF processing")
        return jsonify({"error": f"Failed to process PDFs: {str(e)}"}), 500


@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')
    if not question:
        logging.error("Missing question in the request")
        return jsonify({"error": "Missing question"}), 400

    try:
        logging.info("Search: Initiating search for the question")
        answer = rag_pipeline.ask(question)
        logging.info("Search: Question processed and answer generated")
        # Log the question and answer
        logging.info(f"Response from bot: Question: {question}\nAnswer: {answer}")

        # Log the question and answer
        logging.info(f"Question: {question}\nAnswer: {answer}")

        return jsonify({"answer": answer}), 200
    except Exception as e:
        logging.error(f"Error answering question: {e}")
        return jsonify({"error": "Failed to generate an answer."}), 500


@app.route('/feedback', methods=['POST'])
def save_feedback():
    data = request.get_json()
    question = data.get('question')
    answer = data.get('answer')
    feedback = data.get('feedback')
    timestamp = datetime.datetime.utcnow().isoformat()

    if not all([question, answer, feedback]):
        logging.error("Missing feedback data in the request")
        return jsonify({"error": "Missing feedback data"}), 400

    try:
        logging.info("DynamoDB feedback addition: Saving feedback to DynamoDB")
        feedback_table.put_item(Item={
            'id': str(uuid.uuid4()),
            'timestamp': timestamp,
            'question': question,
            'answer': answer,
            'feedback': feedback
        })
        logging.info("DynamoDB feedback addition: Feedback saved successfully")
        return jsonify({"message": "Feedback saved successfully"}), 200
    except Exception as e:
        logging.error(f"Error saving feedback: {e}")
        return jsonify({"error": "Failed to save feedback"}), 500


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
        logging.info("First touchpoint with backend: Flask app starting")
    app.run(debug=True, host='0.0.0.0', port=8080)
