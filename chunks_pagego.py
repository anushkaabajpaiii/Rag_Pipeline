import fitz  # PyMuPDF
import json
import re
import os

# Removed heading detection function as we're using simple chunking

def clean_text(text):
    """Clean up line text."""
    return re.sub(r'\s+', ' ', text).strip()

def extract_chunks_from_pdf(pdf_path, chunk_size=500):
    """Extract text chunks from PDF with fixed size chunking."""
    doc = fitz.open(pdf_path)
    chunks = []
    chunk_id = 1
    
    for page_num, page in enumerate(doc, 1):
        # Extract all text from the page
        page_text = page.get_text()
        page_text = clean_text(page_text)
        
        if not page_text or len(page_text) < 10:
            continue
            
        # Split page text into chunks of specified size
        words = page_text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > chunk_size and current_chunk:
                # Create chunk
                chunk_text = " ".join(current_chunk).strip()
                if chunk_text:
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "chunk_id": f"chunk_{chunk_id}",
                            "page_number": page_num
                        }
                    })
                    chunk_id += 1
                
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        # Add remaining chunk if any
        if current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "chunk_id": f"chunk_{chunk_id}",
                        "page_number": page_num
                    }
                })
                chunk_id += 1
    
    doc.close()
    return chunks

def save_chunks_to_json(chunks, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

# ✅ Run end-to-end
if __name__ == "__main__":
    pdf_path = "data/sample.pdf"  # change to your actual PDF path
    output_path = "output/structured_chunks.json"

    chunks = extract_chunks_from_pdf(pdf_path, chunk_size=500)  # 500 words per chunk
    
    # Filter out any empty chunks
    valid_chunks = [chunk for chunk in chunks if chunk.get("text", "").strip()]
    
    save_chunks_to_json(valid_chunks, output_path)

    print(f"✅ Extracted {len(valid_chunks)} valid chunks to {output_path}")
    
    # Debug: Show first chunk structure
    if valid_chunks:
        print("\n📋 Sample chunk structure:")
        print(json.dumps(valid_chunks[0], indent=2))
        print(f"\n📊 Average chunk length: {sum(len(chunk['text']) for chunk in valid_chunks) // len(valid_chunks)} characters")