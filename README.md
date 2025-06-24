# RAG Pipeline - Complete Setup Guide

A production-ready Retrieval-Augmented Generation (RAG) pipeline for PDF documents with semantic search, hybrid retrieval, and local LLM integration.

## 🚀 Features

- **PDF Processing**: Extract text while preserving document structure
- **Semantic Chunking**: Intelligent text splitting with metadata preservation
- **Hybrid Retrieval**: Combines semantic similarity + keyword matching
- **Local LLM Integration**: Uses Ollama for answer generation
- **Modern UI**: React + Tailwind CSS interface
- **Structured Responses**: JSON output with source references
- **Robust Error Handling**: Handles edge cases and malformed PDFs

## 📋 Prerequisites

1. **Python 3.8+**
2. **Node.js 16+** (for frontend)
3. **Ollama** (for local LLM)

## 🛠️ Installation

### 1. Backend Setup

```bash
# Create virtual environment
python -m venv rag_env
source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir uploads chroma_db
```

### 2. Install and Configure Ollama

```bash
# Install Ollama (visit https://ollama.ai for platform-specific instructions)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (choose one)
ollama pull llama2        # 7B model (recommended)
ollama pull mistral       # Alternative
ollama pull codellama     # For code-related documents

# Start Ollama server
ollama serve
```

### 3. Frontend Setup

The frontend is included as a React component artifact. To run it locally:

```bash
# Create React app
npx create-react-app rag-frontend
cd rag-frontend

# Install additional dependencies
npm install lucide-react

# Replace src/App.js with the React component from the artifact
# Make sure to update the API_BASE URL if needed
```

## 🚀 Running the Application

### 1. Start the Backend

```bash
# Activate virtual environment
source rag_env/bin/activate

# Run FastAPI server
python main.py
```

The API will be available at `http://localhost:8000`

### 2. Start the Frontend

```bash
# In the frontend directory
npm start
```

The frontend will be available at `http://localhost:3000`

### 3. Start Ollama (if not already running)

```bash
ollama serve
```

## 📡 API Endpoints

### Upload PDF
```http
POST /upload_pdf
Content-Type: multipart/form-data

Body: PDF file
```

### Ask Question
```http
POST /ask_question
Content-Type: application/json

{
  "question": "What is the main topic of this document?",
  "pdf_id": "optional-pdf-id",
  "max_chunks": 5
}
```

### Health Check
```http
GET /health
```

### Get Statistics
```http
GET /stats
```

## 🔧 Configuration

Edit the `Config` class in `main.py` to customize:

```python
class Config:
    UPLOAD_DIR = "uploads"
    CHROMA_DB_PATH = "./chroma_db"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Change embedding model
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL = "llama2"  # Change LLM model
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    MAX_RETRIEVAL_CHUNKS = 5
```

## 📊 Response Format

The system returns structured JSON responses:

```json
{
  "answer": "The document discusses...",
  "source_chunks": [
    {
      "chunk_id": "uuid_page_1_chunk_0",
      "page_number": 1,
      "chapter_title": "Introduction",
      "section_title": "Overview",
      "content": "This document provides...",
      "relevance_score": 0.892
    }
  ]
}
```

## 🎯 Usage Examples

### Query Types Supported

1. **Specific Page Questions**
   - "What's on page 5?"
   - "Summarize page 3"

2. **Conceptual Questions**
   - "What is the main argument?"
   - "How does X relate to Y?"

3. **Factual Queries**
   - "What are the key findings?"
   - "List the recommendations"

4. **Comparison Questions**
   - "Compare approach A vs B"
   - "What are the differences between X and Y?"

## 🔍 How It Works

1. **PDF Upload**: Extract text while preserving structure (chapters, sections, pages)
2. **Semantic Chunking**: Split text into meaningful chunks with metadata
3. **Embedding**: Convert chunks to vector representations using Sentence Transformers
4. **Storage**: Store in ChromaDB for fast semantic search
5. **Hybrid Retrieval**: Combine semantic similarity + TF-IDF keyword matching
6. **LLM Generation**: Send context to Ollama for answer generation
7. **Response**: Return structured JSON with sources

## 🛡️ Error Handling

The system handles:
- Malformed PDFs
- Multi-column layouts
- Missing table of contents
- Empty pages
- Network errors
- LLM service downtime

## 🔧 Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   - Ensure Ollama is running: `ollama serve`
   - Check model is pulled: `ollama list`

2. **PDF Processing Errors**
   - Verify PDF is not corrupted
   - Check file permissions

3. **Memory Issues**
   - Reduce `CHUNK_SIZE` in config
   - Use smaller embedding model

4. **Slow Performance**
   - Reduce `MAX_RETRIEVAL_CHUNKS`
   - Use faster embedding model
   - Optimize chunk size

## 📈 Performance Optimization

1. **Embedding Model**: Use smaller models for speed
2. **Chunk Size**: Balance between context and performance
3. **Retrieval Count**: Limit chunks for faster responses
4. **Caching**: ChromaDB provides built-in caching

## 🚀 Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

### Environment Variables

```bash
export OLLAMA_BASE_URL=http://ollama-service:11434
export CHROMA_DB_PATH=/data/chroma_db
export UPLOAD_DIR=/data/uploads
```

## 🧪 Testing

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test PDF upload
curl -X POST -F "file=@sample.pdf" http://localhost:8000/upload_pdf

# Test question answering
curl -X POST -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}' \
  http://localhost:8000/ask_question
```

## 📚 Dependencies

- **FastAPI**: Web framework
- **PyMuPDF**: PDF processing
- **ChromaDB**: Vector database
- **Sentence Transformers**: Embeddings
- **Scikit-learn**: TF-IDF vectorization
- **Ollama**: Local LLM inference

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Submit pull request

## 📄 License

MIT License - see LICENSE file for details

## 🔗 Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [ChromaDB Documentation](https://docs.trychroma.com)
- [Sentence Transformers](https://www.sbert.net)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
