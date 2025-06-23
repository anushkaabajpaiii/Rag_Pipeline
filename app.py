from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import os
import shutil
from typing import Dict, Any, Optional
import logging
from pathlib import Path

# Import your RAG pipeline
from rag_pipeline import AdvancedRAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Advanced RAG Pipeline API",
    description="API for processing PDFs and querying content using an advanced RAG pipeline.",
    version="1.0.0"
)

# Directory to store uploaded PDFs
UPLOAD_DIR = Path("data")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize the RAG pipeline (singleton instance)
try:
    rag_pipeline = AdvancedRAGPipeline(
        ollama_model="llama3.1:8b",
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        db_path="./chromadb",
        collection_name="advanced_rag"
    )
    logger.info("RAG pipeline initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize RAG pipeline: {e}")
    raise RuntimeError(f"RAG pipeline initialization failed: {e}")

# Background task for processing PDFs
def process_pdf_in_background(pdf_path: str, clear_existing: bool = False):
    """Process the PDF in the background."""
    try:
        chunk_count = rag_pipeline.process_pdf(pdf_path, clear_existing=clear_existing)
        logger.info(f"Processed PDF: {pdf_path}, {chunk_count} chunks added.")
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}")
        raise

@app.post("/upload-pdf/")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    clear_existing: bool = False
) -> Dict[str, Any]:
    """
    Upload a PDF file and process it in the background.
    
    Args:
        file: The PDF file to upload.
        clear_existing: Whether to clear the existing collection before processing.
    
    Returns:
        JSON response with the status and file path.
    """
    try:
        # Validate file type
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

        # Save the uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Uploaded PDF: {file_path}")

        # Process the PDF in the background
        background_tasks.add_task(process_pdf_in_background, str(file_path), clear_existing)
        
        return {
            "status": "success",
            "message": "PDF uploaded and processing started in the background.",
            "file_path": str(file_path)
        }
    except Exception as e:
        logger.error(f"Error uploading PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")

@app.post("/query/")
async def query_rag(
    question: str,
    filters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Query the RAG pipeline with a question.
    
    Args:
        question: The question to query.
        filters: Optional filters for retrieval.
    
    Returns:
        JSON response with the answer, sources, and confidence.
    """
    try:
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        result = rag_pipeline.query(question, filters)
        logger.info(f"Processed query: {question}")
        return result
    except Exception as e:
        logger.error(f"Error processing query '{question}': {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/query-by-page/")
async def query_by_page(
    page_number: int
) -> Dict[str, Any]:
    """
    Retrieve all chunks from a specific page.
    
    Args:
        page_number: The page number to query.
    
    Returns:
        JSON response with the summary and chunks.
    """
    try:
        if page_number < 1:
            raise HTTPException(status_code=400, detail="Page number must be greater than 0.")

        result = rag_pipeline.query_by_page(page_number)
        logger.info(f"Retrieved chunks from page {page_number}")
        return result
    except Exception as e:
        logger.error(f"Error retrieving page {page_number}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving page: {str(e)}")

@app.post("/query-by-section/")
async def query_by_section(
    section_type: str
) -> Dict[str, Any]:
    """
    Retrieve all chunks of a specific section type.
    
    Args:
        section_type: The section type to query (e.g., 'abstract', 'introduction').
    
    Returns:
        JSON response with the summary and chunks.
    """
    try:
        if not section_type:
            raise HTTPException(status_code=400, detail="Section type cannot be empty.")

        result = rag_pipeline.query_by_section(section_type)
        logger.info(f"Retrieved chunks for section type: {section_type}")
        return result
    except Exception as e:
        logger.error(f"Error retrieving section type '{section_type}': {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving section: {str(e)}")

@app.get("/stats/")
async def get_stats() -> Dict[str, Any]:
    """
    Retrieve collection statistics.
    
    Returns:
        JSON response with collection statistics.
    """
    try:
        count = rag_pipeline.collection.count()
        stats = {
            "total_chunks": count,
            "collection_name": rag_pipeline.collection.name,
            "embedding_model": str(rag_pipeline.embedding_model),
            "ollama_model": rag_pipeline.ollama_model
        }
        logger.info("Retrieved collection statistics.")
        return stats
    except Exception as e:
        logger.error(f"Error retrieving stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving stats: {str(e)}")

@app.get("/health/")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint to verify the API is running.
    
    Returns:
        JSON response indicating the API status.
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)