import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

import pymupdf  # PyMuPDF for better PDF extraction
import chromadb
from chromadb.config import Settings
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
import re
from dataclasses import dataclass
from collections import defaultdict
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

@dataclass
class ChunkMetadata:
    chunk_id: str
    title: str
    page_numbers: List[int]
    importance: str
    section_type: str
    word_count: int
    sentence_count: int
    doc_title: str
    reading_time: float
    keywords: List[str]

class AdvancedPDFProcessor:
    """Enhanced PDF processor with better text extraction and structure detection"""
    
    def __init__(self):
        self.section_patterns = {
            'title': r'^[A-Z\s]{3,50}$',
            'heading': r'^\d+\.?\s+[A-Z].*$|^[A-Z][a-z\s]{2,50}$',
            'subheading': r'^\d+\.\d+\.?\s+.*$',
            'abstract': r'(?i)^abstract\s*$',
            'introduction': r'(?i)^introduction\s*$',
            'conclusion': r'(?i)^conclusion\s*$',
            'references': r'(?i)^references?\s*$',
            'bibliography': r'(?i)^bibliography\s*$'
        }
    
    def extract_pdf_content(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract structured content from PDF with enhanced metadata"""
        doc = pymupdf.open(pdf_path)
        content_blocks = []
        doc_title = self._extract_document_title(doc)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text blocks with formatting info
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    text_content = ""
                    font_sizes = []
                    
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text_content += span["text"]
                            font_sizes.append(span["size"])
                    
                    if text_content.strip():
                        avg_font_size = np.mean(font_sizes) if font_sizes else 12
                        section_type = self._classify_section_type(text_content.strip())
                        
                        content_blocks.append({
                            "text": text_content.strip(),
                            "page": page_num + 1,
                            "font_size": avg_font_size,
                            "section_type": section_type,
                            "doc_title": doc_title
                        })
        
        doc.close()
        return content_blocks
    
    def _extract_document_title(self, doc) -> str:
        """Extract document title from metadata or first page"""
        # Try metadata first
        metadata = doc.metadata
        if metadata.get('title'):
            return metadata['title']
        
        # Extract from first page
        if len(doc) > 0:
            first_page_text = doc[0].get_text()
            lines = first_page_text.split('\n')[:10]  # Check first 10 lines
            for line in lines:
                if len(line.strip()) > 10 and len(line.strip()) < 100:
                    return line.strip()
        
        return "Unknown Document"
    
    def _classify_section_type(self, text: str) -> str:
        """Classify text block by section type"""
        text_clean = text.strip()
        
        for section_type, pattern in self.section_patterns.items():
            if re.match(pattern, text_clean):
                return section_type
        
        return "body"

class SmartChunker:
    """Advanced chunking with semantic awareness and overlap"""
    
    def __init__(self, 
                 chunk_size: int = 400,
                 overlap_size: int = 50,
                 min_chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
        self.stop_words = set(stopwords.words('english'))
    
    def create_smart_chunks(self, content_blocks: List[Dict[str, Any]]) -> List[ChunkMetadata]:
        """Create semantically aware chunks with proper overlap"""
        chunks = []
        current_chunk_text = ""
        current_pages = set()
        current_section = "body"
        chunk_id = 1
        
        for i, block in enumerate(content_blocks):
            text = block["text"]
            page = block["page"]
            section_type = block.get("section_type", "body")
            doc_title = block.get("doc_title", "")
            
            # Handle section boundaries
            if section_type != "body" and current_chunk_text:
                # Finalize current chunk before starting new section
                if len(current_chunk_text.strip()) >= self.min_chunk_size:
                    chunk = self._create_chunk_metadata(
                        chunk_id, current_chunk_text, current_pages, 
                        current_section, doc_title
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                
                current_chunk_text = ""
                current_pages = set()
                current_section = section_type
            
            # Add text to current chunk
            if current_chunk_text:
                current_chunk_text += " " + text
            else:
                current_chunk_text = text
            current_pages.add(page)
            
            # Check if chunk is large enough to split
            if len(current_chunk_text.split()) >= self.chunk_size:
                # Split at sentence boundary
                sentences = sent_tokenize(current_chunk_text)
                chunk_sentences = []
                word_count = 0
                
                for sentence in sentences:
                    sentence_words = len(sentence.split())
                    if word_count + sentence_words <= self.chunk_size:
                        chunk_sentences.append(sentence)
                        word_count += sentence_words
                    else:
                        break
                
                if chunk_sentences:
                    chunk_text = " ".join(chunk_sentences)
                    chunk = self._create_chunk_metadata(
                        chunk_id, chunk_text, current_pages, 
                        current_section, doc_title
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                    
                    # Create overlap for next chunk
                    overlap_sentences = chunk_sentences[-2:] if len(chunk_sentences) > 1 else []
                    remaining_sentences = sentences[len(chunk_sentences):]
                    current_chunk_text = " ".join(overlap_sentences + remaining_sentences)
        
        # Handle remaining text
        if len(current_chunk_text.strip()) >= self.min_chunk_size:
            chunk = self._create_chunk_metadata(
                chunk_id, current_chunk_text, current_pages, 
                current_section, doc_title
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk_metadata(self, chunk_id: int, text: str, pages: set, 
                             section_type: str, doc_title: str) -> ChunkMetadata:
        """Create comprehensive metadata for chunk"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        
        # Extract keywords (removing stop words)
        keywords = [word for word in words if word.isalpha() and word not in self.stop_words]
        keyword_freq = defaultdict(int)
        for word in keywords:
            keyword_freq[word] += 1
        
        top_keywords = [word for word, _ in sorted(keyword_freq.items(), 
                                                 key=lambda x: x[1], reverse=True)[:10]]
        
        # Determine importance based on section type and content
        importance = self._calculate_importance(section_type, text)
        
        # Calculate reading time (average 200 words per minute)
        reading_time = len(words) / 200
        
        return ChunkMetadata(
            chunk_id=f"chunk_{chunk_id}",
            title=self._extract_chunk_title(text, section_type),
            page_numbers=sorted(list(pages)),
            importance=importance,
            section_type=section_type,
            word_count=len(words),
            sentence_count=len(sentences),
            doc_title=doc_title,
            reading_time=reading_time,
            keywords=top_keywords
        )
    
    def _extract_chunk_title(self, text: str, section_type: str) -> str:
        """Extract or generate appropriate title for chunk"""
        lines = text.split('\n')
        first_line = lines[0].strip()
        
        if section_type != "body" and len(first_line) < 100:
            return first_line
        
        # Generate title from first sentence
        sentences = sent_tokenize(text)
        if sentences:
            first_sentence = sentences[0]
            if len(first_sentence) <= 80:
                return first_sentence
            else:
                return first_sentence[:77] + "..."
        
        return text[:50] + "..." if len(text) > 50 else text
    
    def _calculate_importance(self, section_type: str, text: str) -> str:
        """Calculate importance score based on section type and content"""
        high_importance_sections = ['abstract', 'conclusion', 'title']
        medium_importance_sections = ['introduction', 'heading', 'subheading']
        
        if section_type in high_importance_sections:
            return "high"
        elif section_type in medium_importance_sections:
            return "medium"
        elif section_type == "references" or section_type == "bibliography":
            return "low"
        else:
            # Analyze content for importance indicators
            important_keywords = ['result', 'conclusion', 'finding', 'significant', 'important', 
                                'key', 'main', 'primary', 'crucial', 'essential']
            text_lower = text.lower()
            
            importance_score = sum(1 for keyword in important_keywords if keyword in text_lower)
            
            if importance_score >= 3:
                return "high"
            elif importance_score >= 1:
                return "medium"
            else:
                return "low"

class AdvancedRetriever:
    """Enhanced retrieval with hybrid search and re-ranking"""
    
    def __init__(self, collection, embedding_model, top_k: int = 10):
        self.collection = collection
        self.embedding_model = embedding_model
        self.top_k = top_k
    
    def retrieve_with_reranking(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Retrieve and re-rank results using multiple strategies"""
        
        # Step 1: Semantic search
        query_embedding = self.embedding_model.encode([query])
        
        semantic_results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=self.top_k * 2,  # Get more results for re-ranking
            where=filters,
            include=['documents', 'metadatas', 'distances']
        )
        
        if not semantic_results['documents'][0]:
            return []
        
        # Step 2: Keyword-based scoring
        query_words = set(query.lower().split())
        
        scored_results = []
        for i, (doc, metadata, distance) in enumerate(zip(
            semantic_results['documents'][0],
            semantic_results['metadatas'][0], 
            semantic_results['distances'][0]
        )):
            # Semantic similarity (convert distance to similarity)
            semantic_score = 1 - distance
            
            # Keyword overlap score
            doc_words = set(doc.lower().split())
            keyword_overlap = len(query_words.intersection(doc_words)) / len(query_words)
            
            # Importance boost
            importance_boost = {'high': 0.3, 'medium': 0.1, 'low': 0.0}.get(
                metadata.get('importance', 'low'), 0.0
            )
            
            # Section type boost
            section_boost = 0.2 if metadata.get('section_type') in ['abstract', 'conclusion', 'introduction'] else 0.0
            
            # Combined score
            final_score = (semantic_score * 0.6 + 
                          keyword_overlap * 0.2 + 
                          importance_boost + 
                          section_boost)
            
            scored_results.append({
                'document': doc,
                'metadata': metadata,
                'semantic_score': semantic_score,
                'keyword_score': keyword_overlap,
                'final_score': final_score,
                'chunk_id': metadata.get('chunk_id', f'chunk_{i}')
            })
        
        # Step 3: Re-rank and return top results
        scored_results.sort(key=lambda x: x['final_score'], reverse=True)
        return scored_results[:self.top_k // 2]  # Return fewer but higher quality results

class AdvancedRAGPipeline:
    """Complete RAG pipeline with local models and advanced features"""
    
    def __init__(self, 
                 ollama_model: str = "llama3.1:8b",
                 embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
                 db_path: str = "./chromadb",
                 collection_name: str = "advanced_rag"):
        
        print("🚀 Initializing Advanced RAG Pipeline...")
        
        # Initialize embedding model
        print(f"📥 Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB
        print(f"💾 Connecting to ChromaDB at: {db_path}")
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize components
        self.pdf_processor = AdvancedPDFProcessor()
        self.chunker = SmartChunker()
        self.retriever = AdvancedRetriever(self.collection, self.embedding_model)
        
        # Ollama model
        self.ollama_model = ollama_model
        print(f"🤖 Using Ollama model: {ollama_model}")
        
        # Test Ollama connection
        try:
            ollama.chat(model=self.ollama_model, messages=[{'role': 'user', 'content': 'test'}])
            print("✅ Ollama connection successful")
        except Exception as e:
            print(f"❌ Ollama connection failed: {e}")
            print("Please ensure Ollama is running and the model is installed")
    
    def process_pdf(self, pdf_path: str, clear_existing: bool = False) -> int:
        """Process PDF and add to vector database"""
        
        if clear_existing:
            print("🧹 Clearing existing collection...")
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.create_collection(
                name=self.collection.name,
                metadata={"hnsw:space": "cosine"}
            )
        
        print(f"📖 Processing PDF: {pdf_path}")
        
        # Extract content
        content_blocks = self.pdf_processor.extract_pdf_content(pdf_path)
        print(f"📄 Extracted {len(content_blocks)} content blocks")
        
        # Create chunks
        chunks = self.chunker.create_smart_chunks(content_blocks)
        print(f"🔄 Created {len(chunks)} semantic chunks")
        
        # Generate embeddings and store
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            # Get the actual text content (assuming first content block contains text)
            chunk_text = self._get_chunk_text(chunk, content_blocks)
            
            documents.append(chunk_text)
            metadatas.append({
                'chunk_id': chunk.chunk_id,
                'title': chunk.title,
                'page_numbers': ','.join(map(str, chunk.page_numbers)),
                'importance': chunk.importance,
                'section_type': chunk.section_type,
                'word_count': chunk.word_count,
                'sentence_count': chunk.sentence_count,
                'doc_title': chunk.doc_title,
                'reading_time': chunk.reading_time,
                'keywords': ','.join(chunk.keywords)
            })
            ids.append(chunk.chunk_id)
        
        # Add to ChromaDB
        print("💾 Adding chunks to ChromaDB...")
        embeddings = self.embedding_model.encode(documents)
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings.tolist()
        )
        
        print(f"✅ Successfully processed {len(chunks)} chunks")
        return len(chunks)
    
    def _get_chunk_text(self, chunk_metadata: ChunkMetadata, content_blocks: List[Dict]) -> str:
        """Extract actual text content for chunk based on pages and title"""
        # This is a simplified approach - in a real implementation, 
        # you'd maintain the mapping between chunks and their text content
        relevant_blocks = [
            block for block in content_blocks 
            if block['page'] in chunk_metadata.page_numbers
        ]
        
        # Find blocks that match the chunk title or are close to it
        chunk_text = ""
        for block in relevant_blocks:
            if chunk_metadata.title in block['text'] or block['text'] in chunk_metadata.title:
                chunk_text = block['text']
                break
        
        # If no exact match, use the first relevant block
        if not chunk_text and relevant_blocks:
            chunk_text = relevant_blocks[0]['text']
        
        return chunk_text if chunk_text else chunk_metadata.title
    
    def query(self, question: str, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Query the RAG system and generate response"""
        
        print(f"🔍 Processing query: {question}")
        
        # Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve_with_reranking(question, filters)
        
        if not retrieved_chunks:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": [],
                "confidence": 0.0
            }
        
        print(f"📚 Retrieved {len(retrieved_chunks)} relevant chunks")
        
        # Prepare context
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(retrieved_chunks):
            context_parts.append(f"[Source {i+1}] {chunk['document']}")
            sources.append({
                'chunk_id': chunk['chunk_id'],
                'title': chunk['metadata'].get('title', 'Unknown'),
                'pages': chunk['metadata'].get('page_numbers', ''),
                'importance': chunk['metadata'].get('importance', 'unknown'),
                'score': round(chunk['final_score'], 3)
            })
        
        context = "\n\n".join(context_parts)
        
        # Generate response using Ollama
        print("🤖 Generating response...")
        response = self._generate_response(question, context)
        
        # Calculate confidence based on retrieval scores
        avg_score = np.mean([chunk['final_score'] for chunk in retrieved_chunks])
        confidence = min(avg_score * 1.2, 1.0)  # Scale and cap at 1.0
        
        return {
            "answer": response,
            "sources": sources,
            "confidence": round(confidence, 3),
            "context_used": len(retrieved_chunks)
        }
    
    def _generate_response(self, question: str, context: str) -> str:
        """Generate response using Ollama"""
        
        prompt = f"""You are a helpful AI assistant that answers questions based on provided context. 

Context Information:
{context}

Question: {question}

Instructions:
- Answer the question based ONLY on the provided context
- Be comprehensive and detailed in your response
- If the context doesn't contain enough information, clearly state this
- Use specific examples and details from the context when available
- Structure your answer clearly with proper paragraphs
- Do not make up information not present in the context

Answer:"""

        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a knowledgeable assistant that provides accurate, detailed answers based on given context.'
                    },
                    {
                        'role': 'user', 
                        'content': prompt
                    }
                ],
                options={
                    'temperature': 0.3,
                    'top_p': 0.9,
                    'max_tokens': 1000
                }
            )
            return response['message']['content']
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while generating the response."
    
    def interactive_session(self):
        """Start interactive Q&A session"""
        print("\n" + "="*60)
        print("🎯 Advanced RAG System - Interactive Mode")
        print("="*60)
        print("Enter your questions below. Type 'exit' to quit.")
        print("You can also use commands:")
        print("  - 'stats' to see collection statistics")
        print("  - 'clear' to clear the screen")
        print("-"*60)
        
        while True:
            try:
                query = input("\n💭 Your question: ").strip()
                
                if query.lower() == 'exit':
                    print("👋 Goodbye!")
                    break
                elif query.lower() == 'stats':
                    self._show_stats()
                    continue
                elif query.lower() == 'clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue
                elif not query:
                    continue
                
                # Process query
                result = self.query(query)
                
                # Display results
                print(f"\n🤖 Answer (Confidence: {result['confidence']}):")
                print("-" * 50)
                print(result['answer'])
                
                if result['sources']:
                    print(f"\n📚 Sources ({len(result['sources'])} chunks used):")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"  {i}. {source['title']} (Pages: {source['pages']}, Score: {source['score']})")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _show_stats(self):
        """Show collection statistics"""
        try:
            count = self.collection.count()
            print(f"\n📊 Collection Statistics:")
            print(f"  - Total chunks: {count}")
            print(f"  - Collection name: {self.collection.name}")
            print(f"  - Embedding model: {self.embedding_model}")
            print(f"  - Ollama model: {self.ollama_model}")
        except Exception as e:
            print(f"Error getting stats: {e}")

def main():
    parser = argparse.ArgumentParser(description="Advanced RAG Pipeline with Local Models")
    parser.add_argument("--pdf_path", type=str, help="Path to PDF file to process")
    parser.add_argument("--db_path", type=str, default="./chromadb", help="ChromaDB path")
    parser.add_argument("--collection", type=str, default="advanced_rag", help="Collection name")
    parser.add_argument("--ollama_model", type=str, default="llama3.1:8b", help="Ollama model name")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-mpnet-base-v2", help="Embedding model")
    parser.add_argument("--clear", action="store_true", help="Clear existing collection")
    parser.add_argument("--query", type=str, help="Single query to process")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    rag = AdvancedRAGPipeline(
        ollama_model=args.ollama_model,
        embedding_model=args.embedding_model,
        db_path=args.db_path,
        collection_name=args.collection
    )
    
    # Process PDF if provided
    if args.pdf_path:
        if not os.path.exists(args.pdf_path):
            print(f"❌ PDF file not found: {args.pdf_path}")
            return
        
        chunk_count = rag.process_pdf(args.pdf_path, clear_existing=args.clear)
        print(f"✅ Processed PDF successfully: {chunk_count} chunks added")
    
    # Handle single query
    if args.query:
        result = rag.query(args.query)
        print(f"\nQuery: {args.query}")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']}")
        return
    
    # Start interactive session
    rag.interactive_session()

if __name__ == "__main__":
    main()