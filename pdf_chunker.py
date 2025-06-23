import fitz  # PyMuPDF
import json
import re
import os
from typing import List, Dict
import statistics
import unicodedata
import nltk
from nltk.tokenize import sent_tokenize
import spacy

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except:
    pass

class SemanticPDFChunker:
    def __init__(self, chunk_size: int = 100, overlap_sentences: int = 2):
        """
        Initialize the Semantic PDF Chunker for general-purpose PDF processing.
        
        Args:
            chunk_size: Target number of words per chunk
            overlap_sentences: Number of sentences to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap_sentences = overlap_sentences

        # Initialize spacy for better sentence segmentation
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️ Warning: spacy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Noise patterns to remove
        self.noise_patterns = [
            r'<EOS>|<pad>|<unk>|<s>|</s>|<UNK>|<PAD>|<CLS>|<SEP>',
            r'\[UNK\]|\[PAD\]|\[CLS\]|\[SEP\]',
            r'^\s*\d+\s*$',  # Standalone numbers
            r'^[^\w\s]*$',   # Only special characters
            r'^\s*[•\-\*]\s*$',  # Standalone bullets
            r'^\s*\.{3,}\s*$',   # Just dots
            r'^\s*_{3,}\s*$',    # Just underscores
            r'^\s*\-{3,}\s*$',   # Just dashes
            r'(?i)^\s*(figure|fig|table|chart|image)\s*\d*\s*:?\s*$',
            r'^\s*\d+\s*of\s*\d+\s*$',  # Page indicators
            r'(?i)^\s*\d+\s*of\s*\d+\s*$',  # Page indicators
            r'^\s*\|\s*\|\s*$',  # Empty table cells
        ]
        
        # Important section patterns
        self.important_sections = [
            r'(?i)^(abstract|summary)(\s|$)',
            r'(?i)^(introduction|intro)(\s|$)', 
            r'(?i)^(conclusion|conclusions)(\s|$)',
            r'(?i)^(methodology|methods?)(\s|$)',
            r'(?i)^(results?|findings?)(\s|$)',
            r'(?i)^(discussion)(\s|$)',
            r'(?i)^(literature\s+review|related\s+work)(\s|$)',
            r'(?i)^(background)(\s|$)',
            r'(?i)^(executive\s+summary)(\s|$)',
        ]
        
        # Low priority sections
        self.low_priority_sections = [
            r'(?i)^(references?|bibliography)(\s|$)',
            r'(?i)^(appendix|appendices)(\s|$)',
            r'(?i)^(acknowledgments?|thanks)(\s|$)',
            r'(?i)^(table\s+of\s+contents|contents)(\s|$)',
            r'(?i)^(index)(\s|$)',
            r'(?i)^(glossary)(\s|$)',
        ]

    def is_noise_text(self, text: str) -> bool:
        """Enhanced noise detection."""
        if not text or len(text.strip()) < 3:
            return True
            
        text_clean = text.strip()
        
        for pattern in self.noise_patterns:
            if re.search(pattern, text_clean):
                return True
        
        if len(text_clean) > 0:
            alnum_chars = sum(1 for c in text_clean if c.isalnum())
            if alnum_chars / len(text_clean) < 0.3:
                return True
            
            unique_chars = len(set(text_clean.replace(' ', '')))
            if unique_chars <= 3 and len(text_clean) > 10:
                return True
        
        return False

    def clean_text_advanced(self, text: str) -> str:
        """Advanced text cleaning with OCR artifact removal."""
        if not text:
            return ""
        
        text = unicodedata.normalize('NFKD', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
        text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        ocr_fixes = {
            r'\bl\b(?=\s+[A-Z])': 'I',
            r'\brn\b': 'm',
            r'\bvv\b': 'w',
            r'(\w)\s+(\w)(?=\s)': r'\1\2',
        }
        
        for pattern, replacement in ocr_fixes.items():
            text = re.sub(pattern, replacement, text)
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+[^\w\s]{1,2}\s+', ' ', text)
        
        return text.strip()

    def segment_into_sentences(self, text: str) -> List[str]:
        """Segment text into sentences using multiple methods."""
        if not text.strip():
            return []
        
        sentences = []
        
        if self.nlp:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            try:
                sentences = sent_tokenize(text)
            except:
                sentences = re.split(r'[.!?]+\s+', text)
        
        valid_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 10 and not self.is_noise_text(sent):
                if not sent[-1] in '.!?':
                    sent += '.'
                valid_sentences.append(sent)
        
        return valid_sentences

    def detect_section_importance(self, text: str) -> str:
        """Classify section importance."""
        text_lower = text.lower().strip()
        
        for pattern in self.important_sections:
            if re.search(pattern, text_lower):
                return "high"
        
        for pattern in self.low_priority_sections:
            if re.search(pattern, text_lower):
                return "low"
                
        return "medium"

    def extract_structured_content(self, pdf_path: str) -> List[Dict]:
        """Extract content with structure preservation."""
        print(f"📖 Extracting content from {pdf_path}")
        doc = fitz.open(pdf_path)
        structured_content = []
        
        doc_title = doc.metadata.get('title', os.path.basename(pdf_path))
        
        for page_num, page in enumerate(doc, 1):
            blocks = page.get_text("dict")["blocks"]
            
            font_sizes = []
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["text"].strip():
                                font_sizes.append(span["size"])
            
            for block in blocks:
                if "lines" not in block:
                    continue
                
                block_text = ""
                max_font_size = 0
                
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        span_text = span["text"]
                        if not self.is_noise_text(span_text):
                            cleaned = self.clean_text_advanced(span_text)
                            if cleaned:
                                line_text += cleaned + " "
                                max_font_size = max(max_font_size, span["size"])
                    
                    if line_text.strip():
                        block_text += line_text.strip() + " "
                
                if block_text.strip():
                    is_heading = False
                    if font_sizes:
                        avg_font = statistics.median(font_sizes)
                        is_heading = max_font_size > avg_font + 1.5
                    
                    structured_content.append({
                        "text": block_text.strip(),
                        "page_number": page_num,
                        "font_size": max_font_size,
                        "is_heading": is_heading,
                        "importance": self.detect_section_importance(block_text),
                        "doc_title": doc_title
                    })
        
        doc.close()
        print(f"📄 Extracted {len(structured_content)} content blocks")
        return structured_content

    def create_semantic_chunks(self, structured_content: List[Dict]) -> List[Dict]:
        """Create semantically coherent chunks with sentence boundaries."""
        print("🔄 Creating semantic chunks...")
        
        chunks = []
        current_chunk_sentences = []
        current_chunk_info = {
            "page_numbers": set(),
            "importance": "medium",
            "has_heading": False,
            "section_title": "",
            "doc_title": ""
        }
        
        chunk_id = 1
        current_section_title = ""
        
        for item in structured_content:
            text = item["text"]
            
            if item["is_heading"]:
                current_section_title = text if len(text) < 100 else text[:97] + "..."
            
            sentences = self.segment_into_sentences(text)
            
            for sentence in sentences:
                current_text = " ".join(current_chunk_sentences + [sentence])
                word_count = len(current_text.split())
                
                if word_count > self.chunk_size and current_chunk_sentences:
                    chunk_text = " ".join(current_chunk_sentences)
                    
                    if chunk_text.strip():
                        chunks.append({
                            "text": chunk_text,
                            "metadata": {
                                "chunk_id": f"chunk_{chunk_id}",
                                "title": current_section_title or current_chunk_info["doc_title"],
                                "page_numbers": sorted(list(current_chunk_info["page_numbers"])),
                                "importance": current_chunk_info["importance"],
                                "has_heading": current_chunk_info["has_heading"],
                                "word_count": len(chunk_text.split()),
                                "sentence_count": len(current_chunk_sentences),
                                "doc_title": current_chunk_info["doc_title"]
                            }
                        })
                        chunk_id += 1
                    
                    overlap_start = max(0, len(current_chunk_sentences) - self.overlap_sentences)
                    current_chunk_sentences = current_chunk_sentences[overlap_start:] + [sentence]
                    
                    current_chunk_info = {
                        "page_numbers": {item["page_number"]},
                        "importance": item["importance"],
                        "has_heading": item["is_heading"],
                        "section_title": current_section_title,
                        "doc_title": item["doc_title"]
                    }
                else:
                    current_chunk_sentences.append(sentence)
                    current_chunk_info["page_numbers"].add(item["page_number"])
                    
                    if item["importance"] == "high":
                        current_chunk_info["importance"] = "high"
                    elif (item["importance"] == "medium" and 
                          current_chunk_info["importance"] == "low"):
                        current_chunk_info["importance"] = "medium"
                    
                    if item["is_heading"]:
                        current_chunk_info["has_heading"] = True
                        
                    current_chunk_info["doc_title"] = item["doc_title"]
        
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "chunk_id": f"chunk_{chunk_id}",
                        "title": current_section_title or current_chunk_info["doc_title"],
                        "page_numbers": sorted(list(current_chunk_info["page_numbers"])),
                        "importance": current_chunk_info["importance"],
                        "has_heading": current_chunk_info["has_heading"],
                        "word_count": len(chunk_text.split()),
                        "sentence_count": len(current_chunk_sentences),
                        "doc_title": current_chunk_info["doc_title"]
                    }
                })
        
        print(f"📦 Created {len(chunks)} semantic chunks")
        return chunks

    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Main processing pipeline."""
        print(f"🚀 Starting semantic PDF processing for {pdf_path}...")
        
        structured_content = self.extract_structured_content(pdf_path)
        chunks = self.create_semantic_chunks(structured_content)
        
        return chunks

def save_chunks_to_json(chunks: List[Dict], output_path: str):
    """Save chunks to JSON with proper serialization."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    json_chunks = []
    for chunk in chunks:
        json_chunk = chunk.copy()
        json_chunk["metadata"] = chunk["metadata"].copy()
        
        for key, value in json_chunk["metadata"].items():
            if isinstance(value, set):
                json_chunk["metadata"][key] = list(value)
        
        json_chunks.append(json_chunk)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_chunks, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    processor = SemanticPDFChunker(chunk_size=100, overlap_sentences=2)
    
    pdf_path = "data/sample.pdf"
    output_path = "output/semantic_chunks.json"
    
    try:
        chunks = processor.process_pdf(pdf_path)
        
        save_chunks_to_json(chunks, output_path)
        
        print(f"\n🎉 Successfully processed {pdf_path}")
        print(f"💾 Saved {len(chunks)} chunks to {output_path}")
        
        if chunks:
            high_imp = sum(1 for c in chunks if c["metadata"]["importance"] == "high")
            med_imp = sum(1 for c in chunks if c["metadata"]["importance"] == "medium")
            low_imp = sum(1 for c in chunks if c["metadata"]["importance"] == "low")
            
            avg_words = sum(c["metadata"]["word_count"] for c in chunks) // len(chunks)
            avg_sentences = sum(c["metadata"]["sentence_count"] for c in chunks) // len(chunks)
            
            print(f"\n📊 Chunk Statistics:")
            print(f"   🔴 High importance: {high_imp}")
            print(f"   🟡 Medium importance: {med_imp}")
            print(f"   🔵 Low importance: {low_imp}")
            print(f"   📏 Average words per chunk: {avg_words}")
            print(f"   📝 Average sentences per chunk: {avg_sentences}")
            
            print(f"\n📋 Sample chunk:")
            sample_chunk = next((c for c in chunks if c["metadata"]["importance"] == "high"), chunks[0])
            print(f"Title: {sample_chunk['metadata']['title']}")
            print(f"Importance: {sample_chunk['metadata']['importance']}")
            print(f"Text Preview: {sample_chunk['text'][:200]}...")
            
    except Exception as e:
        print(f"❌ Error processing PDF: {str(e)}")
        raise