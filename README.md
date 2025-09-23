# 🎓 Noah - Your Virtual Educational Buddy

> Transform your static study notes into an interactive learning companion powered by AI

Ever wished your study notes could talk back? Noah makes it possible! Built with Flask and ChromaDB, this project implements a custom RAG (Retrieval-Augmented Generation) pipeline that turns any PDF document into an intelligent, conversational study partner.

## ✨ Features

- 📄 **PDF Upload & Processing** - Upload any study material (notes, textbooks, papers)
- 💬 **Interactive Q&A** - Ask questions directly about your uploaded content
- 🤖 **Smart Responses** - Powered by OpenAI's language models with context-aware answers
- ✅ **Feedback System** - Mark answers as correct, wrong, or unclear to improve responses
- 🔍 **Semantic Search** - Advanced retrieval using ChromaDB vector database
- 📚 **Multi-Document Support** - Upload multiple PDFs and query across all content

## 🚀 How It Works

Noah uses a sophisticated RAG pipeline to provide accurate, contextual answers:

1. **Document Ingestion**: PDFs are processed and chunked into meaningful segments
2. **Vector Storage**: Text chunks are embedded and stored in ChromaDB for fast retrieval
3. **Query Processing**: User questions trigger semantic search across the document corpus
4. **Response Generation**: OpenAI's LLM generates answers using retrieved context
5. **Continuous Learning**: User feedback helps refine future responses

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **Vector Database**: ChromaDB
- **LLM**: OpenAI GPT API
- **PDF Processing**: PyPDF2/pdfplumber
- **Text Embeddings**: OpenAI Embeddings API
- **Frontend**: HTML, CSS, JavaScript

## 📦 Installation

### Prerequisites

- Python 3.8+
- OpenAI API key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/noah-educational-buddy.git
cd noah-educational-buddy
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
OPENAI_API_KEY=your_api_key_here
```

5. Run the application:
```bash
python app.py
```

6. Open your browser and navigate to `http://localhost:5000`

## 💡 Usage

### Getting Started

1. **Upload a PDF**: Click "Upload Document" and select your study material
2. **Wait for Processing**: Noah will chunk and embed your document
3. **Start Asking**: Type questions about your content in the chat interface
4. **Provide Feedback**: Rate responses to help Noah learn your preferences

### Example Interactions

```
You: "What are the main types of chemical bonds?"

Noah: "Based on your chemistry notes, there are three main types of chemical bonds:

1. **Ionic Bonds** - Formed between metals and non-metals through electron transfer
2. **Covalent Bonds** - Created when atoms share electrons
3. **Metallic Bonds** - Found in metals where electrons form a 'sea' around metal cations

Would you like me to explain any of these in more detail?"

[✅ Correct] [❌ Wrong] [❓ Unclear]
```

## 📁 Project Structure

```
noah-educational-buddy/
├── app.py                 # Flask application
├── models/
│   ├── rag_pipeline.py   # RAG implementation
│   ├── document_processor.py
│   └── feedback_handler.py
├── static/
│   ├── css/
│   ├── js/
│   └── uploads/          # Uploaded PDFs
├── templates/
│   ├── index.html
│   └── chat.html
├── requirements.txt
├── .env.example
└── README.md
```

## 🔧 Configuration

Key configuration options in `config.py`:

```python
# OpenAI Settings
OPENAI_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"

# ChromaDB Settings
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "documents"

# Processing Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_TOKENS = 4000
```

## 🤝 Contributing

Contributions are welcome! Here are some ways to get involved:

- 🐛 Report bugs or suggest features
- 📝 Improve documentation
- 🔧 Submit pull requests
- 💡 Share ideas for new educational features

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and test thoroughly
4. Commit: `git commit -m 'Add amazing feature'`
5. Push: `git push origin feature/amazing-feature`
6. Open a Pull Request

## 📊 Roadmap

- [ ] **Multi-format Support** - Word docs, PowerPoint, images
- [ ] **Study Session Tracking** - Analytics and progress monitoring  
- [ ] **Collaborative Learning** - Share documents with classmates
- [ ] **Mobile App** - iOS and Android versions
- [ ] **Advanced Feedback** - Machine learning from user interactions
- [ ] **Integration APIs** - Connect with popular study platforms

## ⚠️ Limitations

- Requires OpenAI API key (paid service)
- Processing time depends on document size
- Currently supports PDF format only
- Answers quality depends on source material clarity

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for providing powerful language models
- ChromaDB team for the excellent vector database
- Flask community for the robust web framework
- All contributors and beta testers

## 📧 Contact

Have questions or suggestions? Feel free to reach out:

- GitHub Issues: [Create an issue](https://github.com/amangupta201/PDF-RAG-Bot/issues)
- Email: your.email@example.com

---

**Made with ❤️ for students everywhere**

*Noah represents the future of personalized education - where AI doesn't replace learning, but enhances it.*
