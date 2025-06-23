# requirements.txt
"""
chromadb>=0.4.0
sentence-transformers>=2.2.0
nltk>=3.8
numpy>=1.21.0
transformers>=4.21.0
torch>=1.12.0
"""

# usage_example.py - Fixed version addressing all issues

import json
from pathlib import Path
from typing import List, Dict
from chunking_retrieval_system import AdvancedTextChunker, ChromaDBRetriever, process_document_chunks
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

def fix_your_chunks():
    """Fix problematic chunks with better parameters"""
    chunks_file = "output/semantic_chunks.json"
    
    if not Path(chunks_file).exists():
        print(f"Error: {chunks_file} not found. Run enhanced_chunking.py first.")
        return []
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        your_chunks = json.load(f)
    
    print(f"Loaded {len(your_chunks)} original chunks from {chunks_file}")
    
    # Use smaller chunk size to capture key sentences
    print("Processing chunks with optimized parameters...")
    fixed_chunks = process_document_chunks(your_chunks, rechunk=True)
    
    print(f"\nOriginal chunks: {len(your_chunks)}")
    print(f"Fixed chunks: {len(fixed_chunks)}")
    
    # Save fixed chunks
    Path("output").mkdir(exist_ok=True)
    fixed_chunks_file = "output/fixed_semantic_chunks.json"
    with open(fixed_chunks_file, 'w', encoding='utf-8') as f:
        json.dump(fixed_chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(fixed_chunks)} fixed chunks to {fixed_chunks_file}")
    
    # Show sample fixed chunks
    print("\nSample fixed chunks:")
    for i, chunk in enumerate(fixed_chunks[:3]):
        print(f"\n--- Fixed Chunk {i+1} ---")
        print(f"Text: {chunk['text']}")
        print(f"Title: {chunk['metadata']['title']}")
        print(f"Word count: {chunk['metadata']['word_count']}")
        print(f"Is Reference: {chunk['metadata'].get('is_reference', False)}")
        print(f"Has complete sentences: {chunk['metadata']['has_complete_sentences']}")
    
    return fixed_chunks

def setup_complete_retrieval_system(fixed_chunks):
    """Setup retrieval system with correct embedding model"""
    print("Initializing ChromaDB retriever with multi-qa-mpnet-base-dot-v1...")
    
    # Ensure we use the correct embedding model
    retriever = ChromaDBRetriever(
        collection_name="research_papers",
        persist_directory="./paper_chunks_db",
        embedding_model="multi-qa-mpnet-base-dot-v1"  # High-quality model for better similarity
    )
    
    print(f"Adding {len(fixed_chunks)} chunks to database...")
    retriever.add_chunks(fixed_chunks)
    
    # Test queries with better expected results
    test_queries = [
        "neural networks sequence modeling",
        "machine translation methods", 
        "transformer attention mechanism"
    ]
    
    for query in test_queries:
        print(f"\n=== Query: '{query}' ===")
        results = retriever.search(query, n_results=3)
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i} (Similarity: {result['score']:.3f})")
            print(f"Source: {result['metadata']['title']}")
            print(f"Chunk ID: {result['metadata']['chunk_id']}")
            print(f"Is Reference: {result['metadata'].get('is_reference', False)}")
            print(f"Text: {result['text'][:150]}...")
            if len(result['text']) > 150:
                print("...")
    
    return retriever

def generate_answer(query: str, retrieved_chunks: List[Dict]) -> str:
    """Generate answer with proper error handling and debugging"""
    print(f"\n--- Starting Answer Generation for: '{query}' ---")
    
    if not retrieved_chunks:
        print("No chunks provided for generation.")
        return "No relevant information found to answer the query."
    
    # Check similarity scores
    max_score = max([chunk["score"] for chunk in retrieved_chunks])
    print(f"Max similarity score: {max_score:.3f}")
    
    if max_score < 0.3:
        print("Similarity scores too low for reliable generation.")
        return f"No highly relevant information found (max similarity: {max_score:.3f}). The query might not be well-covered in the document."
    
    # Filter out reference chunks
    content_chunks = [chunk for chunk in retrieved_chunks 
                     if not chunk['metadata'].get('is_reference', False)]
    
    if not content_chunks:
        print("All retrieved chunks are references, cannot generate content answer.")
        return "Retrieved chunks are primarily references. Try a different query."
    
    print(f"Using {len(content_chunks)} content chunks for generation.")
    
    try:
        print("Loading GPT-2 model...")
        generator = pipeline(
            "text-generation", 
            model="gpt2-medium",
            device=-1,  # Force CPU
            torch_dtype="auto"
        )
        print("GPT-2 model loaded successfully.")
        
        # Create context from relevant chunks
        context = " ".join([chunk["text"] for chunk in content_chunks[:2]])  # Use top 2 chunks
        context = context[:800]  # Limit context length
        
        print(f"Context length: {len(context)} characters")
        
        # Create focused prompt
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        
        print("Generating response...")
        generated = generator(
            prompt,
            max_new_tokens=60,
            num_return_sequences=1,
            truncation=True,
            pad_token_id=generator.tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            temperature=0.7,
            do_sample=True
        )[0]["generated_text"]
        
        # Extract just the answer part
        answer = generated[len(prompt):].strip()
        
        # Clean up the answer
        if answer:
            # Remove incomplete sentences
            sentences = answer.split('.')
            if len(sentences) > 1 and not sentences[-1].strip():
                answer = '.'.join(sentences[:-1]) + '.'
            
            # Limit to reasonable length
            if len(answer) > 200:
                answer = answer[:200] + "..."
        
        if not answer or len(answer) < 10:
            print("Generated answer too short or empty.")
            return "Could not generate a comprehensive answer. Try a more specific query."
        
        print(f"Generated answer: {answer}")
        return answer
        
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        return f"Error generating answer: {str(e)}"

def advanced_search_examples(retriever):
    """Advanced search with proper filtering"""
    print("\n=== Advanced Search Examples ===")
    
    # Test metadata filtering
    print("\n1. Filtering by page numbers...")
    try:
        results = retriever.search(
            "attention mechanism",
            n_results=3,
            filter_metadata={"page_numbers": [1]}
        )
        print(f"Results from page 1: {len(results)} found")
        for i, result in enumerate(results, 1):
            print(f"  Result {i} (Score: {result['score']:.3f}): {result['text'][:100]}...")
    except Exception as e:
        print(f"Error in filtered search: {e}")
    
    # Multi-query search
    print("\n2. Multi-query search...")
    queries = ["transformer", "attention", "neural network"]
    all_results = []
    
    for query in queries:
        results = retriever.search(query, n_results=2)
        all_results.extend(results)
    
    # Remove duplicates and sort by score
    unique_results = {}
    for result in all_results:
        if result['id'] not in unique_results:
            unique_results[result['id']] = result
        elif result['score'] > unique_results[result['id']]['score']:
            unique_results[result['id']] = result
    
    sorted_results = sorted(unique_results.values(), 
                          key=lambda x: x['score'], reverse=True)
    
    print(f"Multi-query results: {len(sorted_results)} unique chunks")
    for i, result in enumerate(sorted_results[:3], 1):
        print(f"  {i}. Score: {result['score']:.3f} - {result['text'][:80]}...")

def interactive_search(retriever):
    """Interactive search with generation"""
    print("\n=== Interactive Search with Generation ===")
    print("Enter queries to search and generate answers. Type 'exit' to quit.")
    
    while True:
        try:
            query = input("\nEnter your search query: ").strip()
            if query.lower() in ['exit', 'quit', 'q']:
                print("Exiting interactive search.")
                break
            
            if not query:
                print("Please enter a valid query.")
                continue
            
            print(f"\nProcessing query: '{query}'")
            
            # Search for relevant chunks
            results = retriever.search(query, n_results=5)
            
            print(f"\nRetrieved {len(results)} chunks:")
            for i, result in enumerate(results, 1):
                print(f"\nChunk {i} (Similarity: {result['score']:.3f})")
                print(f"  Source: {result['metadata']['title']}")
                print(f"  Text: {result['text'][:100]}...")
                print(f"  Is Reference: {result['metadata'].get('is_reference', False)}")
            
            # Generate answer
            print("\n--- Generating Answer ---")
            answer = generate_answer(query, results)
            print(f"\nGenerated Answer:\n{answer}")
            
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            break
        except Exception as e:
            print(f"Error in interactive search: {e}")

def main():
    """Main function with proper error handling"""
    try:
        print("🔧 Step 1: Fixing problematic chunks...")
        fixed_chunks = fix_your_chunks()
        
        if not fixed_chunks:
            print("❌ No chunks to process. Exiting.")
            return
        
        print(f"\n✅ Successfully processed {len(fixed_chunks)} chunks")
        
        print("\n" + "="*60)
        print("🔍 Step 2: Setting up retrieval system...")
        retriever = setup_complete_retrieval_system(fixed_chunks)
        
        print("\n" + "="*60)
        print("🧪 Step 3: Advanced search examples...")
        advanced_search_examples(retriever)
        
        print("\n" + "="*60)
        print("💬 Step 4: Interactive search with generation...")
        interactive_search(retriever)
        
        # Final stats
        stats = retriever.get_collection_stats()
        print(f"\n📊 Final database stats: {stats}")
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

def fix_your_chunks():
    """Example of how to fix your existing problematic chunks"""
    
    # Your original problematic chunks
    your_chunks = [
        {
            "text": "gated recurrent neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [38, 24, 15]. Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states as a function of the previous hidden state and the input for position This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples.",
            "metadata": {
                "chunk_id": "chunk_2",
                "title": "arXiv:1706.03762 v7 [cs.CL] 2 Aug 2023",
                "page_numbers": [2],
                "importance": "high",
                "has_heading": False,
                "word_count": 487
            }
        }
    ]
    
    # Process and fix the chunks
    print("Processing problematic chunks...")
    fixed_chunks = process_document_chunks(your_chunks, rechunk=True)
    
    print(f"\nOriginal chunks: 1")
    print(f"Fixed chunks: {len(fixed_chunks)}")
    
    for i, chunk in enumerate(fixed_chunks):
        print(f"\n--- Fixed Chunk {i+1} ---")
        print(f"Text: {chunk['text']}")
        print(f"Word count: {chunk['metadata']['word_count']}")
        print(f"Has complete sentences: {chunk['metadata']['has_complete_sentences']}")

def setup_complete_retrieval_system():
    """Complete example of setting up the retrieval system"""
    
    # 1. Initialize the chunker with your preferred settings
    chunker = AdvancedTextChunker(
        chunk_size=400,      # Target chunk size
        chunk_overlap=60,    # Overlap between chunks
        min_chunk_size=80    # Minimum chunk size
    )
    
    # 2. Initialize ChromaDB retriever
    retriever = ChromaDBRetriever(
        collection_name="research_papers",
        persist_directory="./paper_chunks_db",
        embedding_model="all-MiniLM-L6-v2"  # Fast and good quality
    )
    
    # 3. Process your documents
    sample_documents = [
        {
            "title": "Attention Is All You Need",
            "text": """The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.""",
            "page_numbers": [1]
        }
    ]
    
    all_chunks = []
    for doc in sample_documents:
        chunks = chunker.create_semantic_chunks(
            doc["text"], 
            doc["title"], 
            doc["page_numbers"]
        )
        all_chunks.extend(chunks)
    
    # 4. Add to ChromaDB
    retriever.add_chunks(all_chunks)
    
    # 5. Test retrieval
    queries = [
        "transformer attention mechanism",
        "neural machine translation BLEU score",
        "parallel training GPU"
    ]
    
    for query in queries:
        print(f"\n=== Query: '{query}' ===")
        results = retriever.search(query, n_results=3)
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i} (Similarity: {result['score']:.3f})")
            print(f"Source: {result['metadata']['title']}")
            print(f"Text: {result['text'][:200]}...")
    
    return retriever

def advanced_search_examples(retriever):
    """Examples of advanced search functionality"""
    
    # Search with metadata filters
    print("\n=== Advanced Search Examples ===")
    
    # 1. Search only in specific pages
    results = retriever.search(
        "attention mechanism",
        n_results=3,
        filter_metadata={"page_numbers": [1]}
    )
    print(f"\nResults from page 1 only: {len(results)} found")
    
    # 2. Multi-query search (you can implement this)
    queries = ["transformer", "attention", "parallel"]
    all_results = []
    for query in queries:
        results = retriever.search(query, n_results=2)
        all_results.extend(results)
    
    # Remove duplicates and sort by score
    unique_results = {}
    for result in all_results:
        if result['id'] not in unique_results:
            unique_results[result['id']] = result
        elif result['score'] > unique_results[result['id']]['score']:
            unique_results[result['id']] = result
    
    sorted_results = sorted(unique_results.values(), 
                          key=lambda x: x['score'], reverse=True)
    
    print(f"\nMulti-query results: {len(sorted_results)} unique chunks")
    for i, result in enumerate(sorted_results[:3], 1):
        print(f"{i}. Score: {result['score']:.3f} - {result['text'][:100]}...")

def save_and_load_chunks():
    """Example of saving chunks to file and loading them back"""
    
    chunker = AdvancedTextChunker()
    
    # Create some chunks
    text = "Your long document text here..."
    chunks = chunker.create_semantic_chunks(text, "Sample Document")
    
    # Save to JSON file
    output_file = "processed_chunks.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(chunks)} chunks to {output_file}")
    
    # Load from JSON file
    with open(output_file, 'r', encoding='utf-8') as f:
        loaded_chunks = json.load(f)
    
    print(f"Loaded {len(loaded_chunks)} chunks from {output_file}")
    
    return loaded_chunks

def main():
    """Run all examples"""
    print("1. Fixing problematic chunks...")
    fix_your_chunks()
    
    print("\n" + "="*50)
    print("2. Setting up complete retrieval system...")
    retriever = setup_complete_retrieval_system()
    
    print("\n" + "="*50)
    print("3. Advanced search examples...")
    advanced_search_examples(retriever)
    
    print("\n" + "="*50)
    print("4. Save/load chunks example...")
    save_and_load_chunks()
    
    # Get final stats
    stats = retriever.get_collection_stats()
    print(f"\nFinal database stats: {stats}")

if __name__ == "__main__":
    main()