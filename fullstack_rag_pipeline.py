import os
import json
import argparse
from typing import List, Dict, Any, Optional
import pymupdf  # PyMuPDF for PDF extraction
import chromadb
from chromadb.config import Settings
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class PDFProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def extract_text(self, pdf_path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        doc = pymupdf.open(pdf_path)
        content_blocks = []
        doc_title = doc.metadata.get('title', 'Unknown Document')
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    text_content = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text_content += span["text"] + " "
                    text_content = text_content.strip()
                    if text_content:
                        content_blocks.append({
                            "text": text_content,
                            "page": page_num + 1,
                            "doc_title": doc_title
                        })
        doc.close()
        print(f"Extracted {len(content_blocks)} content blocks from {pdf_path}")
        return content_blocks

class Chunker:
    def __init__(self, chunk_size: int = 300, min_chunk_size: int = 50):
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk_text(self, content_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        chunk_id = 1
        current_chunk = {"text": "", "pages": set(), "doc_title": ""}
        
        for block in content_blocks:
            text = block["text"]
            page = block["page"]
            doc_title = block["doc_title"]
            
            sentences = sent_tokenize(text)
            for sentence in sentences:
                if not current_chunk["text"]:
                    current_chunk["doc_title"] = doc_title
                
                current_chunk["text"] += sentence + " "
                current_chunk["pages"].add(page)
                
                word_count = len(current_chunk["text"].split())
                if word_count >= self.chunk_size:
                    if word_count >= self.min_chunk_size:
                        chunks.append({
                            "chunk_id": f"chunk_{chunk_id}",
                            "text": current_chunk["text"].strip(),
                            "pages": sorted(list(current_chunk["pages"])),
                            "doc_title": current_chunk["doc_title"]
                        })
                        chunk_id += 1
                    current_chunk = {"text": "", "pages": set(), "doc_title": ""}
        
        if len(current_chunk["text"].split()) >= self.min_chunk_size:
            chunks.append({
                "chunk_id": f"chunk_{chunk_id}",
                "text": current_chunk["text"].strip(),
                "pages": sorted(list(current_chunk["pages"])),
                "doc_title": current_chunk["doc_title"]
            })
        
        print(f"Created {len(chunks)} chunks")
        return chunks

class Retriever:
    def __init__(self, collection, embedding_model, top_k: int = 5):
        self.collection = collection
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.stop_words = set(stopwords.words('english'))

    def hybrid_retrieve(self, query: str, page_filter: Optional[int] = None) -> List[Dict[str, Any]]:
        print(f"Retrieving chunks for query: {query}")
        if page_filter:
            print(f"Filtering for page: {page_filter}")
        
        query_embedding = self.embedding_model.encode([query])
        filters = {"page": page_filter} if page_filter else None
        
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=self.top_k * 2,
            where=filters,
            include=['documents', 'metadatas', 'distances']
        )
        
        if not results['documents'][0]:
            print("No relevant chunks found.")
            return []
        
        query_words = set(word for word in query.lower().split() if word not in self.stop_words)
        scored_results = []
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            semantic_score = 1 - distance
            doc_words = set(word for word in doc.lower().split() if word not in self.stop_words)
            keyword_score = len(query_words.intersection(doc_words)) / max(len(query_words), 1)
            
            final_score = semantic_score * 0.7 + keyword_score * 0.3
            
            # Convert pages string back to list for output
            pages_str = metadata.get("pages", "")
            pages = [int(p) for p in pages_str.split(",") if p] if pages_str else []
            
            scored_results.append({
                "chunk_id": metadata.get("chunk_id", f"chunk_{i}"),
                "text": doc,
                "pages": pages,
                "doc_title": metadata.get("doc_title", "Unknown"),
                "score": final_score
            })
        
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        print(f"Retrieved {len(scored_results[:self.top_k])} chunks after ranking")
        return scored_results[:self.top_k]

class RAGPipeline:
    def __init__(self, db_path: str = "C:/Users/KIIT/Desktop/RAGCOMPANY/chromadb_data", collection_name: str = "rag_collection"):
        print("Initializing RAG Pipeline...")
        # Ensure the database path exists
        os.makedirs(db_path, exist_ok=True)
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2")
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.pdf_processor = PDFProcessor()
        self.chunker = Chunker()
        # Defer Retriever initialization until after collection is updated
        self.retriever = None
        self.ollama_model = "llama3.1:8b"
        print(f"ChromaDB path: {db_path}")
        print("RAG Pipeline initialized.")

    def process_pdf(self, pdf_path: str, clear_existing: bool = False) -> int:
        print(f"Processing PDF: {pdf_path}")
        if clear_existing:
            print("Clearing existing collection...")
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.create_collection(
                name=self.collection.name,
                metadata={"hnsw:space": "cosine"}
            )
            print("Collection cleared and recreated.")
        
        try:
            content_blocks = self.pdf_processor.extract_text(pdf_path)
        except Exception as e:
            print(f"Failed to extract text from PDF: {str(e)}")
            raise
        
        try:
            chunks = self.chunker.chunk_text(content_blocks)
        except Exception as e:
            print(f"Failed to chunk text: {str(e)}")
            raise
        
        if not chunks:
            print("No chunks created. Check PDF content.")
            return 0
        
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                "chunk_id": chunk["chunk_id"],
                "pages": ",".join(map(str, chunk["pages"])),  # Convert list to string
                "doc_title": chunk["doc_title"]
            } for chunk in chunks
        ]
        ids = [chunk["chunk_id"] for chunk in chunks]
        
        try:
            embeddings = self.embedding_model.encode(documents)
        except Exception as e:
            print(f"Failed to generate embeddings: {str(e)}")
            raise
        
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings.tolist()
            )
            print(f"Added {len(chunks)} chunks to ChromaDB.")
        except Exception as e:
            print(f"Failed to add chunks to ChromaDB: {str(e)}")
            raise
        
        # Initialize or update Retriever after collection is populated
        self.retriever = Retriever(self.collection, self.embedding_model)
        print("Retriever initialized with updated collection.")
        
        # Verify the collection exists
        try:
            collection_count = self.collection.count()
            print(f"Collection contains {collection_count} items after processing.")
        except Exception as e:
            print(f"Error verifying collection: {str(e)}")
            raise
        
        return len(chunks)

    def query(self, question: str, page_number: Optional[int] = None) -> Dict[str, Any]:
        if self.retriever is None:
            raise RuntimeError("Retriever not initialized. Ensure process_pdf is called first.")
        
        retrieved_chunks = self.retriever.hybrid_retrieve(question, page_filter=page_number)
        
        if not retrieved_chunks:
            return {
                "answer": "No relevant information found in the document.",
                "sources": [],
                "confidence": 0.0
            }
        
        context = "\n\n".join(
            f"[Source {i+1}] (Pages: {', '.join(map(str, chunk['pages']))})\n{chunk['text']}"
            for i, chunk in enumerate(retrieved_chunks)
        )
        
        system_prompt = """
You are an expert document analyst tasked with answering queries based on provided PDF content. Use the following context to provide a precise, concise, and accurate answer. Do not make up information. If the context is insufficient, state so clearly.
"""
        
        user_prompt = f"""
Context:
{context}

Question: {question}

Answer the question in a structured JSON format according to the schema provided.
"""
        
        # Define JSON schema for structured output
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "rag_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "sources": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "chunk_id": {"type": "string"},
                                    "doc_title": {"type": "string"},
                                    "pages": {
                                        "type": "array",
                                        "items": {"type": "integer"}
                                    },
                                    "score": {"type": "number"}
                                },
                                "required": ["chunk_id", "doc_title", "pages", "score"],
                                "additionalProperties": False
                            }
                        },
                        "confidence": {"type": "number"}
                    },
                    "required": ["answer", "sources", "confidence"],
                    "additionalProperties": False
                }
            }
        }
        
        sources = [
            {
                "chunk_id": chunk["chunk_id"],
                "doc_title": chunk["doc_title"],
                "pages": chunk["pages"],
                "score": round(chunk["score"], 3)
            }
            for chunk in retrieved_chunks
        ]
        
        avg_score = np.mean([chunk["score"] for chunk in retrieved_chunks])
        confidence = min(avg_score * 1.1, 1.0)
        
        # Fallback response if LLM fails
        fallback_response = {
            "answer": "No relevant information found in the document.",
            "sources": sources,
            "confidence": round(confidence, 3)
        }
        
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": 0.2,
                    "max_tokens": 500
                },
                format="json",
                response_format=response_format
            )
            result = json.loads(response["message"]["content"])
            
            # Ensure the response matches the expected schema
            if not all(key in result for key in ["answer", "sources", "confidence"]):
                raise ValueError("LLM response does not match expected schema")
            
            # Override sources and confidence with computed values
            result["sources"] = sources
            result["confidence"] = round(confidence, 3)
            
            return result
        
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return fallback_response

def main():
    parser = argparse.ArgumentParser(description="Full-Stack RAG Pipeline for PDF Querying")
    parser.add_argument("--pdf_path", type=str, required=True, help="Path to PDF file")
    parser.add_argument("--query", type=str, help="Query to process")
    parser.add_argument("--page", type=int, help="Specific page to query")
    parser.add_argument("--clear", action="store_true", help="Clear existing collection")
    
    args = parser.parse_args()
    
    pipeline = RAGPipeline()
    chunk_count = pipeline.process_pdf(args.pdf_path, clear_existing=args.clear)
    if chunk_count == 0:
        print("No chunks were created. Cannot proceed with query.")
        return
    
    if args.query:
        result = pipeline.query(args.query, page_number=args.page)
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()