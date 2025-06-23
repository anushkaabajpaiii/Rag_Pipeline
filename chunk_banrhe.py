import fitz  # PyMuPDF : open, read, analyze pdfs
import json  #save extracted chunks to JSON
import re    #for cleaning up text
import os    #to handle file paths and directory creation

def is_heading(text, font_size, flags):
    """Determine if a line is likely a heading based on font_size >= 16: large font → likely heading,flags & 2: means the text is bold,font_size >= 14 and text.isupper(): all caps also often indicates headings"""
    if len(text.strip()) < 3:
        return False
    if font_size >= 16 or (flags & 2):  # bold or large font
        return True
    if font_size >= 14 and text.strip().isupper():
        return True
    return False

def clean_text(text):
    """Clean up tabs, newlines,edges."""
    return re.sub(r'\s+', ' ', text).strip()

def extract_chunks_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []
    chapter_title = None
    section_title = None
    buffer = [] #accumulates paragraph text under the current heading
    chunk_id = 1

    for page_num, page in enumerate(doc, 1): #goes through every page
        blocks = page.get_text("dict")["blocks"] #extracts layout-aware blocks of text
        for b in blocks:
            if b["type"] != 0:
                continue
            for line in b["lines"]: #find the maximum font size in the line detects heading
                text_line = ""
                max_font_size = 0
                flags = 0
                for span in line["spans"]: #collects spans(individual styled text parts) to get the full line text
                    text_line += span["text"]
                    if span["size"] > max_font_size:
                        max_font_size = span["size"]
                        flags = span["flags"]

                text = clean_text(text_line)
                if not text:
                    continue
#chunk making algo
                if is_heading(text, max_font_size, flags):
                    if buffer:
                        chunk_text = " ".join(buffer).strip() #makes buffer as a chunk
                        if chunk_text and len(chunk_text) > 10:  # Only add non-empty chunks
                            chunks.append({
                                "text": chunk_text,  # ✅ Direct 'text' key as expected
                                "metadata": {
                                    "id": f"chunk_{chunk_id}",
                                    "page_number": page_num,
                                    "chapter_title": chapter_title or "Not Detected",
                                    "section_title": section_title or "Not Detected"
                                }
                            })
                            chunk_id += 1 #this chunk is been made now skip to other chunk id as a new chunk
                        buffer = []

                    # Update titles
                    if not chapter_title:
                        chapter_title = text
                    else:
                        section_title = text
                else:
                    buffer.append(text) #non heading lines : They’re accumulated in the buffer until the next heading or end of page.

        # End of page flush
        if buffer:
            chunk_text = " ".join(buffer).strip()
            if chunk_text and len(chunk_text) > 10:  # Only add non-empty chunks
                chunks.append({
                    "text": chunk_text,  # ✅ Direct 'text' key as expected
                    "metadata": {
                        "id": f"chunk_{chunk_id}",
                        "page_number": page_num,
                        "chapter_title": chapter_title or "Not Detected",
                        "section_title": section_title or "Not Detected"
                    }
                })
                chunk_id += 1
            buffer = []

    return chunks

def save_chunks_to_json(chunks, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

# ✅ Run end-to-end
if __name__ == "__main__":
    pdf_path = "data/sample.pdf"  # change to your actual PDF path
    output_path = "output/struct_chunks.json"

    chunks = extract_chunks_from_pdf(pdf_path)
    
    # Filter out any empty chunks
    valid_chunks = [chunk for chunk in chunks if chunk.get("text", "").strip()]
    
    save_chunks_to_json(valid_chunks, output_path)

    print(f"✅ Extracted {len(valid_chunks)} valid chunks to {output_path}")
    
    # Debug: Show first chunk structure
    if valid_chunks:
        print("\n📋 Sample chunk structure:")
        print(json.dumps(valid_chunks[0], indent=2))