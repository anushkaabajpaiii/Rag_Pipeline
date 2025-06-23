import os
import traceback  # Add this import
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import shutil
from rag_pipeline import AdvancedRAGPipeline

app = FastAPI(
    title="RAG Pipeline API",
    description="API for processing PDFs and querying content using an advanced RAG pipeline.",
    version="1.0.0"
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"Incoming request: {request.method} {request.url.path}")
    response = await call_next(request)
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_pipeline = AdvancedRAGPipeline(
    ollama_model="llama3.1:8b",
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    db_path="./chromadb",
    collection_name="advanced_rag"
)

class QueryRequest(BaseModel):
    question: str
    filters: Optional[dict] = None

@app.get("/")
async def root():
    return {"message": "Welcome to the RAG Pipeline API!"}

@app.get("/status")
async def get_status():
    try:
        collection_count = rag_pipeline.collection.count()
        return {
            "status": "healthy",
            "collection_chunks": collection_count,
            "message": "RAG Pipeline API is running."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API is unhealthy: {str(e)}")

@app.post("/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
        chunk_count = rag_pipeline.process_pdf(temp_file_path, clear_existing=True)
        os.remove(temp_file_path)
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Successfully processed PDF: {file.filename}",
                "chunk_count": chunk_count
            }
        )
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/query")
async def query_document(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = rag_pipeline.query(request.question, request.filters)
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
       
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")