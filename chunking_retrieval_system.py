import re
import json
import uuid
from typing import List, Dict, Tuple, Optional
import chromadb
from chromadb.config import Settings
import nltk
from sentence_transformers import SentenceTransformer
import numpy as np
from sentence_transformers.util import cos_sim

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class AdvancedTextChunker:
    def __init__(self, 
                 chunk_size: int = 100,
                 chunk_overlap: int = 20,
                 min_chunk_size: int = 15):
        """
        Advanced text chunker that maintains semantic boundaries
        
        Args:
            chunk_size: Target size for each chunk (in tokens/words)
            chunk_overlap: Number of tokens to overlap between chunks
            min_chunk_size: Minimum size for a chunk to be valid
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\'\/]', '', text)
        text = re.sub(r'\s+([\.,!?;:])', r'\1', text)
        return text.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK"""
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def get_word_count(self, text: str) -> int:
        """Get approximate word count"""
        return len(text.split())
    
    def _is_reference_chunk(self, text: str, title: str) -> bool:
        """Detect if a chunk is from references section"""
        reference_patterns = [
            r'\[\d+\]\s*[A-Z]',  # [1] Author
            r'In\s+[A-Z]\.\s*[A-Z]',  # In C. Cortes, N. D. Lawrence
            r'[A-Z]\.\s*[A-Z]\.\s*[A-Z]',  # A. B. Smith
            r'et\s+al\.',  # et al.
            r'Proceedings\s+of',  # Proceedings of
            r'Journal\s+of',  # Journal of
            r'Conference\s+on',  # Conference on
            r'\d{4}\.\s*$',  # Year at end (e.g., 2015.)
            r'arXiv:\d+\.\d+',  # arXiv:1706.03762
            r'pages\s+\d+\s+\d+',  # pages 2440 2448
        ]
        
        if "reference" in title.lower() or "bibliography" in title.lower():
            return True
            
        reference_score = 0
        for pattern in reference_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                reference_score += 1
        
        return reference_score >= 2
    
    def create_semantic_chunks(self, text: str, title: str = "", 
                             page_numbers: List[int] = None) -> List[Dict]:
        """
        Create chunks that respect sentence boundaries and maintain context
        """
        cleaned_text = self.clean_text(text)
        sentences = self.split_into_sentences(cleaned_text)
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for i, sentence in enumerate(sentences):
            sentence_size = self.get_word_count(sentence)
            
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                if self.get_word_count(chunk_text) >= self.min_chunk_size:
                    chunks.append(self._create_chunk_metadata(
                        chunk_text, title, page_numbers, len(chunks)
                    ))
                
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_size = sum(self.get_word_count(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if self.get_word_count(chunk_text) >= self.min_chunk_size:
                chunks.append(self._create_chunk_metadata(
                    chunk_text, title, page_numbers, len(chunks)
                ))
        
        return chunks
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap based on overlap size"""
        if not sentences:
            return []
        
        overlap_text = ""
        overlap_sentences = []
        
        for sentence in reversed(sentences):
            if self.get_word_count(overlap_text + sentence) <= self.chunk_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_text = ' '.join(overlap_sentences)
            else:
                break
        
        return overlap_sentences
    
    def _create_chunk_metadata(self, text: str, title: str, 
                              page_numbers: List[int], chunk_index: int) -> Dict:
        """Create chunk with metadata"""
        is_reference = self._is_reference_chunk(text, title)
        section_title = "References" if is_reference else title
        
        return {
            "text": text,
            "metadata": {
                "chunk_id": f"chunk_{chunk_index}",
                "title": section_title,
                "original_title": title,
                "page_numbers": page_numbers or [],
                "word_count": self.get_word_count(text),
                "chunk_index": chunk_index,
                "has_complete_sentences": True,
                "is_reference": is_reference
            }
        }

class ChromaDBRetriever:
    def __init__(self, 
                 collection_name: str = "document_chunks",
                 persist_directory: str = "./chroma_db",
                 embedding_model: str = "multi-qa-mpnet-base-dot-v1"):
        """
        ChromaDB retriever for semantic search
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_model: Sentence transformer model for embeddings
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        
        self.embedding_model = SentenceTransformer(embedding_model)
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Document chunks for semantic search", "hnsw:space": "cosine"}
        )
    
    def _serialize_metadata(self, metadata: Dict) -> Dict:
        """Convert metadata values to ChromaDB-compatible types"""
        serialized = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                serialized[key] = ",".join(map(str, value))
            elif isinstance(value, (str, int, float, bool)) or value is None:
                serialized[key] = value
            else:
                serialized[key] = str(value)
        return serialized
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit length"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        normalized = embeddings / norms
        return normalized
    
    def add_chunks(self, chunks: List[Dict]) -> None:
        """Add chunks to ChromaDB with embeddings"""
        if not chunks:
            return
        
        seen_texts = set()
        unique_chunks = []
        duplicates_found = 0
        
        for chunk in chunks:
            text = re.sub(r'\s+', ' ', chunk["text"]).strip().lower()
            if text not in seen_texts:
                seen_texts.add(text)
                chunk_copy = chunk.copy()
                chunk_copy["metadata"] = self._serialize_metadata(chunk["metadata"])
                unique_chunks.append(chunk_copy)
            else:
                duplicates_found += 1
        
        if not unique_chunks:
            print("No unique chunks to add after deduplication.")
            return
        
        print(f"Removed {duplicates_found} duplicate chunks during deduplication.")
        print(f"Sample unique chunk texts (first 2): {[chunk['text'][:100] for chunk in unique_chunks[:2]]}")
        if unique_chunks:
            print(f"Sample serialized metadata (first chunk): {unique_chunks[0]['metadata']}")
        
        texts = [chunk["text"] for chunk in unique_chunks]
        metadatas = [chunk["metadata"] for chunk in unique_chunks]
        ids = [str(uuid.uuid4()) for _ in unique_chunks]
        
        embeddings = self.embedding_model.encode(texts)
        embeddings = self._normalize_embeddings(embeddings).tolist()
        
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Added {len(unique_chunks)} unique chunks to ChromaDB collection '{self.collection_name}'")
    
    def search(self, 
               query: str, 
               n_results: int = 5,
               filter_metadata: Dict = None) -> List[Dict]:
        """
        Search for relevant chunks
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Metadata filters (e.g., {"page_numbers": "1"})
        
        Returns:
            List of relevant chunks with scores
        """
        query_embedding = self.embedding_model.encode([query])
        query_embedding = self._normalize_embeddings(query_embedding).tolist()[0]
        
        exclude_references = {"title": {"$ne": "References"}}
        
        if filter_metadata:
            filter_metadata = self._serialize_metadata(filter_metadata)
            where_clause = {
                "$and": [
                    exclude_references,
                    filter_metadata
                ]
            }
        else:
            where_clause = exclude_references
        
        print(f"Applying where clause: {where_clause}")
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause
            )
        except Exception as e:
            print(f"Error in search with filter: {e}")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
        
        formatted_results = []
        for i in range(len(results['documents'][0])):
            metadata = results['metadatas'][0][i].copy()
            if 'page_numbers' in metadata and isinstance(metadata['page_numbers'], str):
                metadata['page_numbers'] = [int(x) for x in metadata['page_numbers'].split(',') if x]
            distance = results['distances'][0][i]
            similarity = max(0, min(1, 1 - distance))
            formatted_results.append({
                "text": results['documents'][0][i],
                "metadata": metadata,
                "score": similarity,
                "id": results['ids'][0][i]
            })
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model_name
        }
    
    def delete_collection(self) -> None:
        """Delete the entire collection"""
        self.client.delete_collection(name=self.collection_name)
        print(f"Deleted collection '{self.collection_name}'")

def process_document_chunks(raw_chunks: List[Dict], 
                          rechunk: bool = True) -> List[Dict]:
    """
    Process raw document chunks to create better semantic chunks
    
    Args:
        raw_chunks: List of raw chunks from document processing
        rechunk: Whether to rechunk the text for better boundaries
    
    Returns:
        List of processed chunks ready for embedding
    """
    chunker = AdvancedTextChunker(chunk_size=100, chunk_overlap=20, min_chunk_size=15)
    processed_chunks = []
    
    for raw_chunk in raw_chunks:
        text = raw_chunk.get("text", "")
        metadata = raw_chunk.get("metadata", {})
        
        title = metadata.get("title", "")
        page_numbers = metadata.get("page_numbers", [])
        
        if rechunk:
            semantic_chunks = chunker.create_semantic_chunks(
                text, title, page_numbers
            )
            processed_chunks.extend(semantic_chunks)
        else:
            cleaned_text = chunker.clean_text(text)
            if chunker.get_word_count(cleaned_text) >= chunker.min_chunk_size:
                is_reference = chunker._is_reference_chunk(cleaned_text, title)
                section_title = "References" if is_reference else title
                
                processed_chunks.append({
                    "text": cleaned_text,
                    "metadata": {
                        **metadata,
                        "title": section_title,
                        "original_title": title,
                        "word_count": chunker.get_word_count(cleaned_text),
                        "has_complete_sentences": True,
                        "is_reference": is_reference
                    }
                })
    
    return processed_chunks

def main():
    """Example usage of the chunking and retrieval system"""
    sample_text = """gated recurrent neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures. Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states as a function of the previous hidden state and the input for position This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples."""
    
    chunker = AdvancedTextChunker(chunk_size=100, chunk_overlap=20)
    
    chunks = chunker.create_semantic_chunks(
        sample_text, 
        title="Transformer Paper", 
        page_numbers=[2]
    )
    
    print("Created chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"Text: {chunk['text'][:100]}...")
        print(f"Word count: {chunk['metadata']['word_count']}")
    
    retriever = ChromaDBRetriever(
        collection_name="test_collection",
        persist_directory="./test_chroma_db",
        embedding_model="multi-qa-mpnet-base-dot-v1"
    )
    
    retriever.add_chunks(chunks)
    
    query = "neural networks sequence modeling"
    results = retriever.search(query, n_results=3)
    
    print(f"\nSearch results for '{query}':")
    for i, result in enumerate(results):
        print(f"\nResult {i+1} (Score: {result['score']:.3f}):")
        print(f"Text: {result['text'][:150]}...")
        print(f"Metadata: {result['metadata']}")
    
    stats = retriever.get_collection_stats()
    print(f"\nCollection stats: {stats}")

if __name__ == "__main__":
    main()