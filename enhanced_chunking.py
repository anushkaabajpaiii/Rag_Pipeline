'''
import fitz  # PyMuPDF
import json
import re
import os
from typing import List, Dict, Tuple
import statistics
from collections import Counter
import unicodedata

chunk_size = 500        # Target words per chunk
overlap_size = 50       # Words to overlap between chunks
target_chunk_size = 500 # Unified and passed where needed

class AdvancedPDFProcessor:
    def __init__(self):
        # Common noise patterns found in PDFs
        self.noise_patterns = [
            r'<EOS>|<pad>|<unk>|<s>|</s>',  # Special tokens
            r'\[UNK\]|\[PAD\]|\[CLS\]|\[SEP\]',  # BERT-style tokens
            r'^\s*\d+\s*$',  # Standalone numbers (likely page numbers)
            r'^[^\w\s]*$',  # Lines with only special characters
            r'^\s*[•\-\*]\s*$',  # Standalone bullet points
            r'^\s*\.{3,}\s*$',  # Lines with just dots
            r'^\s*_{3,}\s*$',  # Lines with just underscores
            r'^\s*\-{3,}\s*$',  # Lines with just dashes
            r'(?i)^\s*(figure|table|chart|image)\s*\d*\s*$',  # Figure/table labels
            r'^\s*\d+\s*of\s*\d+\s*$',  # Page indicators
            r'^\s*\|\s*\|\s*$',  # Empty table cells
        ]
        
        # Academic section patterns (case-insensitive)
        self.important_sections = [
            r'(?i)^(abstract|summary)$',
            r'(?i)^(introduction|intro)$', 
            r'(?i)^(conclusion|conclusions)$',
            r'(?i)^(methodology|methods)$',
            r'(?i)^(results|findings)$',
            r'(?i)^(discussion)$',
            r'(?i)^(literature\s+review|related\s+work)$',
            r'(?i)^(background)$',
            r'(?i)^(executive\s+summary)$',
        ]
        
        # Patterns for less important content
        self.low_priority_patterns = [
            r'(?i)^(references|bibliography)$',
            r'(?i)^(appendix|appendices)$',
            r'(?i)^(acknowledgments?|thanks)$',
            r'(?i)^(table\s+of\s+contents|contents)$',
            r'(?i)^(index)$',
            r'(?i)^(glossary)$',
        ]

    def is_noise_line(self, text: str) -> bool:
        """Check if a line is likely noise/artifact."""
        if not text or len(text.strip()) < 3:
            return True
            
        # Check against noise patterns
        for pattern in self.noise_patterns:
            if re.search(pattern, text.strip()):
                return True
        
        # Check character composition
        clean_text = re.sub(r'\s', '', text)
        if len(clean_text) == 0:
            return True
            
        # Too many special characters
        special_char_ratio = sum(1 for c in clean_text if not c.isalnum()) / len(clean_text)
        if special_char_ratio > 0.7:
            return True
            
        # Repeated character patterns (OCR artifacts)
        if len(set(clean_text)) <= 3 and len(clean_text) > 5:
            return True
            
        return False

    def clean_text_advanced(self, text: str) -> str:
        """Advanced text cleaning for OCR artifacts and noise."""
        if not text:
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove common OCR artifacts
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Fix missing spaces
        text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)  # Space between numbers and letters
        text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)  # Space between letters and numbers
        
        # Remove standalone special characters
        text = re.sub(r'\s+[^\w\s]{1,2}\s+', ' ', text)
        
        # Fix common OCR errors
        ocr_fixes = {
            r'\bl\b': 'I',  # lowercase l to uppercase I
            r'\b0\b': 'O',  # zero to O in words
            r'rn': 'm',     # common OCR error
            r'\s+': ' ',    # multiple spaces
        }
        
        for pattern, replacement in ocr_fixes.items():
            text = re.sub(pattern, replacement, text)
        
        return text.strip()

    def detect_section_importance(self, text: str) -> str:
        """Classify text importance based on section headers."""
        text_lower = text.lower().strip()
        
        # Check for high-importance sections
        for pattern in self.important_sections:
            if re.match(pattern, text_lower):
                return "high"
        
        # Check for low-importance sections  
        for pattern in self.low_priority_patterns:
            if re.match(pattern, text_lower):
                return "low"
                
        return "medium"

    def extract_text_with_structure(self, pdf_path: str) -> List[Dict]:
        """Extract text with structural information and cleaning."""
        doc = fitz.open(pdf_path)
        structured_content = []
        
        for page_num, page in enumerate(doc, 1):
            # Get text blocks with formatting info
            blocks = page.get_text("dict")["blocks"]
            
            # Extract font information for heading detection
            font_sizes = []
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["text"].strip():
                                font_sizes.append(span["size"])
            
            # Determine heading thresholds
            if font_sizes:
                avg_font_size = statistics.median(font_sizes)
                heading_threshold = avg_font_size + 2
            else:
                heading_threshold = 12
            
            # Process blocks
            for block in blocks:
                if "lines" not in block:
                    continue
                    
                block_text = ""
                max_font_size = 0
                
                for line in block["lines"]:
                    line_text = ""
                    line_font_sizes = []
                    
                    for span in line["spans"]:
                        span_text = span["text"]
                        if not self.is_noise_line(span_text):
                            cleaned_span = self.clean_text_advanced(span_text)
                            if cleaned_span:
                                line_text += cleaned_span + " "
                                line_font_sizes.append(span["size"])
                    
                    if line_text.strip() and not self.is_noise_line(line_text):
                        block_text += line_text.strip() + " "
                        if line_font_sizes:
                            max_font_size = max(max_font_size, max(line_font_sizes))
                
                if block_text.strip():
                    is_heading = max_font_size >= heading_threshold
                    importance = self.detect_section_importance(block_text)
                    
                    structured_content.append({
                        "text": block_text.strip(),
                        "page_number": page_num,
                        "font_size": max_font_size,
                        "is_heading": is_heading,
                        "importance": importance,
                        "word_count": len(block_text.split())
                    })
        
        doc.close()
        return structured_content

    def smart_chunking(self, structured_content: List[Dict], 
                      target_chunk_size: int = 500, 
                      overlap_size: int = 50) -> List[Dict]:
        """Create semantically aware chunks with overlap."""
        chunks = []
        current_chunk = {
            "text": "",
            "metadata": {
                "chunk_id": "",
                "title": "",
                "page_numbers": set(),
                "importance": "medium",
                "has_heading": False,
                "word_count": 0
            }
        }
        
        current_section_title = ""
        chunk_id = 1
        
        for item in structured_content:
            text = item["text"]
            word_count = item["word_count"] 
            
            # Update section title if this is a heading
            if item["is_heading"] and item["importance"] != "low":
                current_section_title = text[:100]  # Truncate long headings
            
            # Check if adding this text would exceed chunk size
            if (current_chunk["metadata"]["word_count"] + word_count > target_chunk_size 
                and current_chunk["text"].strip()):
                
                # Finalize current chunk
                self._finalize_chunk(current_chunk, chunk_id, current_section_title)
                chunks.append(current_chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk["text"], overlap_size)
                current_chunk = {
                    "text": overlap_text,
                    "metadata": {
                        "chunk_id": "",
                        "title": current_section_title,
                        "page_numbers": set(),
                        "importance": "medium",
                        "has_heading": False,
                        "word_count": len(overlap_text.split()) if overlap_text else 0
                    }
                }
                chunk_id += 1
            
            # Add current text to chunk
            if current_chunk["text"]:
                current_chunk["text"] += " " + text
            else:
                current_chunk["text"] = text
                
            current_chunk["metadata"]["page_numbers"].add(item["page_number"])
            current_chunk["metadata"]["word_count"] += word_count
            
            # Update chunk importance (prioritize higher importance)
            if item["importance"] == "high":
                current_chunk["metadata"]["importance"] = "high"
            elif (item["importance"] == "medium" and 
                  current_chunk["metadata"]["importance"] == "low"):
                current_chunk["metadata"]["importance"] = "medium"
            
            if item["is_heading"]:
                current_chunk["metadata"]["has_heading"] = True
        
        # Add final chunk
        if current_chunk["text"].strip():
            self._finalize_chunk(current_chunk, chunk_id, current_section_title)
            chunks.append(current_chunk)
        
        return chunks

    def _finalize_chunk(self, chunk: Dict, chunk_id: int, section_title: str):
        """Finalize chunk metadata."""
        chunk["metadata"]["chunk_id"] = f"chunk_{chunk_id}"
        chunk["metadata"]["title"] = section_title
        chunk["metadata"]["page_numbers"] = sorted(list(chunk["metadata"]["page_numbers"]))
        
        # Final text cleaning
        chunk["text"] = self.clean_text_advanced(chunk["text"])

    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get last N words for overlap."""
        words = text.split()
        if len(words) <= overlap_size:
            return text
        return " ".join(words[-overlap_size:])

    def process_pdf(self, pdf_path: str, chunk_size: int = 500, 
                   overlap_size: int = 50) -> List[Dict]:
        """Main processing pipeline."""
        print(f"🔄 Processing {pdf_path}...")
        
        # Extract structured content
        structured_content = self.extract_text_with_structure(pdf_path)
        print(f"📄 Extracted {len(structured_content)} text blocks")
        
        # Create smart chunks
        chunks = self.smart_chunking(structured_content, chunk_size, overlap_size)
        print(f"📦 Created {len(chunks)} semantic chunks")
        
        # Filter and rank chunks
        filtered_chunks = self._filter_and_rank_chunks(chunks)
        print(f"✅ Filtered to {len(filtered_chunks)} high-quality chunks")
        
        return filtered_chunks

    def _filter_and_rank_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Filter out low-quality chunks and rank by importance."""
        filtered_chunks = []
        
        for chunk in chunks:
            # Skip very short chunks
            if len(chunk["text"].split()) < 10:
                continue
                
            # Skip chunks that are mostly noise
            words = chunk["text"].split()
            meaningful_words = sum(1 for word in words if len(word) > 2 and word.isalpha())
            if meaningful_words / len(words) < 0.6:
                continue
                
            filtered_chunks.append(chunk)
        
        # Sort by importance (high first, then medium, then low)
        importance_order = {"high": 0, "medium": 1, "low": 2}
        filtered_chunks.sort(key=lambda x: (
            importance_order.get(x["metadata"]["importance"], 1),
            -x["metadata"]["word_count"]  # Longer chunks within same importance
        ))
        
        return filtered_chunks

def save_chunks_to_json(chunks: List[Dict], output_path: str):
    """Save chunks to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert sets to lists for JSON serialization
    json_chunks = []
    for chunk in chunks:
        json_chunk = chunk.copy()
        if isinstance(json_chunk["metadata"]["page_numbers"], set):
            json_chunk["metadata"]["page_numbers"] = list(json_chunk["metadata"]["page_numbers"])
        json_chunks.append(json_chunk)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_chunks, f, indent=2, ensure_ascii=False)

# Main execution
if __name__ == "__main__":
    # Initialize processor
    processor = AdvancedPDFProcessor()
    
    # Configuration
    pdf_path = "data/sample.pdf"  # Change to your PDF path
    output_path = "output/enhanced_chunks.json"
    chunk_size = 500  # Target words per chunk
    overlap_size = 50  # Words to overlap between chunks
    
    try:
        # Process PDF
        chunks = processor.process_pdf(pdf_path, chunk_size, overlap_size)
        
        # Save results
        save_chunks_to_json(chunks, output_path)
        
        # Display results
        print(f"\n✅ Successfully processed {pdf_path}")
        print(f"📁 Saved {len(chunks)} chunks to {output_path}")
        
        if chunks:
            # Show statistics
            high_importance = sum(1 for c in chunks if c["metadata"]["importance"] == "high")
            medium_importance = sum(1 for c in chunks if c["metadata"]["importance"] == "medium")
            low_importance = sum(1 for c in chunks if c["metadata"]["importance"] == "low")
            
            print(f"\n📊 Chunk Statistics:")
            print(f"   🔴 High importance: {high_importance}")
            print(f"   🟡 Medium importance: {medium_importance}")
            print(f"   🔵 Low importance: {low_importance}")
            
            avg_length = sum(len(c["text"]) for c in chunks) // len(chunks)
            print(f"   📏 Average chunk length: {avg_length} characters")
            
            # Show sample chunk
            print(f"\n📋 Sample high-quality chunk:")
            sample_chunk = next((c for c in chunks if c["metadata"]["importance"] == "high"), chunks[0])
            print(json.dumps(sample_chunk, indent=2, default=str))
            
    except Exception as e:
        print(f"❌ Error processing PDF: {str(e)}")
        raise
'''

import fitz  # PyMuPDF
import json
import re
import os
from typing import List, Dict, Tuple
import statistics
from collections import Counter
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
    def __init__(self):
        # Initialize spacy for better sentence segmentation
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️  Warning: spacy model not found. Install with: python -m spacy download en_core_web_sm")
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
        
        # Check noise patterns
        for pattern in self.noise_patterns:
            if re.search(pattern, text_clean):
                return True
        
        # Character composition analysis
        if len(text_clean) > 0:
            # Too many special characters
            alnum_chars = sum(1 for c in text_clean if c.isalnum())
            if alnum_chars / len(text_clean) < 0.3:
                return True
            
            # Repeated character patterns (OCR artifacts)
            unique_chars = len(set(text_clean.replace(' ', '')))
            if unique_chars <= 3 and len(text_clean) > 10:
                return True
        
        return False

    def clean_text_advanced(self, text: str) -> str:
        """Advanced text cleaning with OCR artifact removal."""
        if not text:
            return ""
        
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Remove problematic characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Non-ASCII
        
        # Fix common OCR spacing issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Missing spaces
        text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)  # Number-letter
        text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)  # Letter-number
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Sentence boundaries
        
        # Fix common OCR character errors
        ocr_fixes = {
            r'\bl\b(?=\s+[A-Z])': 'I',  # lowercase l to I before capitals
            r'\brn\b': 'm',             # rn to m
            r'\bvv\b': 'w',             # vv to w
            r'(\w)\s+(\w)(?=\s)': r'\1\2',  # Fix broken words
        }
        
        for pattern, replacement in ocr_fixes.items():
            text = re.sub(pattern, replacement, text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def segment_into_sentences(self, text: str) -> List[str]:
        """Segment text into sentences using multiple methods for robustness."""
        if not text.strip():
            return []
        
        sentences = []
        
        # Use spacy if available (more accurate)
        if self.nlp:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Fallback to NLTK
            try:
                sentences = sent_tokenize(text)
            except:
                # Manual sentence splitting as last resort
                sentences = re.split(r'[.!?]+\s+', text)
        
        # Clean and validate sentences
        valid_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 10 and not self.is_noise_text(sent):
                # Ensure sentence ends properly
                if not sent[-1] in '.!?':
                    sent += '.'
                valid_sentences.append(sent)
        
        return valid_sentences

    def detect_section_importance(self, text: str) -> str:
        """Classify section importance."""
        text_lower = text.lower().strip()
        
        # High importance sections
        for pattern in self.important_sections:
            if re.search(pattern, text_lower):
                return "high"
        
        # Low importance sections
        for pattern in self.low_priority_sections:
            if re.search(pattern, text_lower):
                return "low"
                
        return "medium"

    def extract_structured_content(self, pdf_path: str) -> List[Dict]:
        """Extract content with structure preservation."""
        print(f"📖 Extracting content from {pdf_path}")
        doc = fitz.open(pdf_path)
        structured_content = []
        
        # Get document title
        doc_title = doc.metadata.get('title', os.path.basename(pdf_path))
        
        for page_num, page in enumerate(doc, 1):
            blocks = page.get_text("dict")["blocks"]
            
            # Collect font sizes for heading detection
            font_sizes = []
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["text"].strip():
                                font_sizes.append(span["size"])
            
            # Process each block
            for block in blocks:
                if "lines" not in block:
                    continue
                
                block_text = ""
                max_font_size = 0
                
                # Combine text from all spans in block
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
                    # Determine if heading
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

    def create_semantic_chunks(self, structured_content: List[Dict], 
                             target_chunk_size: int = 500,
                             overlap_sentences: int = 2) -> List[Dict]:
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
            
            # Update section title if this is a heading
            if item["is_heading"]:
                current_section_title = text if len(text) < 100 else text[:97] + "..."
            
            # Segment text into sentences
            sentences = self.segment_into_sentences(text)
            
            for sentence in sentences:
                # Check if adding this sentence would exceed target size
                current_text = " ".join(current_chunk_sentences + [sentence])
                word_count = len(current_text.split())
                
                if word_count > target_chunk_size and current_chunk_sentences:
                    # Finalize current chunk
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
                    
                    # Start new chunk with overlap
                    overlap_start = max(0, len(current_chunk_sentences) - overlap_sentences)
                    current_chunk_sentences = current_chunk_sentences[overlap_start:] + [sentence]
                    
                    # Reset chunk info but keep some context
                    current_chunk_info = {
                        "page_numbers": {item["page_number"]},
                        "importance": item["importance"],
                        "has_heading": item["is_heading"],
                        "section_title": current_section_title,
                        "doc_title": item["doc_title"]
                    }
                else:
                    # Add sentence to current chunk
                    current_chunk_sentences.append(sentence)
                    current_chunk_info["page_numbers"].add(item["page_number"])
                    
                    # Update importance (prioritize higher)
                    if item["importance"] == "high":
                        current_chunk_info["importance"] = "high"
                    elif (item["importance"] == "medium" and 
                          current_chunk_info["importance"] == "low"):
                        current_chunk_info["importance"] = "medium"
                    
                    if item["is_heading"]:
                        current_chunk_info["has_heading"] = True
                        
                    current_chunk_info["doc_title"] = item["doc_title"]
        
        # Add final chunk
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

    def filter_and_enhance_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Filter low-quality chunks and enhance metadata."""
        print("🔍 Filtering and enhancing chunks...")
        
        filtered_chunks = []
        
        for chunk in chunks:
            text = chunk["text"]
            words = text.split()
            
            # Skip very short chunks
            if len(words) < 15:
                continue
            
            # Skip chunks with too little meaningful content
            meaningful_words = sum(1 for word in words 
                                 if len(word) > 2 and word.isalpha())
            if meaningful_words / len(words) < 0.5:
                continue
            
            # Add quality metrics
            chunk["metadata"]["quality_score"] = self._calculate_quality_score(text)
            chunk["metadata"]["readability_score"] = self._calculate_readability_score(text)
            
            filtered_chunks.append(chunk)
        
        # Sort by importance and quality
        importance_order = {"high": 0, "medium": 1, "low": 2}
        filtered_chunks.sort(key=lambda x: (
            importance_order.get(x["metadata"]["importance"], 1),
            -x["metadata"]["quality_score"],
            -x["metadata"]["word_count"]
        ))
        
        print(f"✅ Filtered to {len(filtered_chunks)} high-quality chunks")
        return filtered_chunks

    def _calculate_quality_score(self, text: str) -> float:
        """Calculate a quality score for the chunk."""
        words = text.split()
        if not words:
            return 0.0
        
        score = 0.0
        
        # Length score (optimal around 200-600 words)
        word_count = len(words)
        if 200 <= word_count <= 600:
            score += 1.0
        elif 100 <= word_count < 200 or 600 < word_count <= 800:
            score += 0.7
        else:
            score += 0.3
        
        # Sentence structure score
        sentences = self.segment_into_sentences(text)
        if sentences:
            avg_sentence_length = word_count / len(sentences)
            if 10 <= avg_sentence_length <= 25:
                score += 1.0
            else:
                score += 0.5
        
        # Content diversity score
        unique_words = len(set(word.lower() for word in words if word.isalpha()))
        diversity = unique_words / len(words) if words else 0
        score += diversity
        
        return min(score, 3.0)  # Cap at 3.0

    def _calculate_readability_score(self, text: str) -> float:
        """Simple readability score based on sentence and word complexity."""
        sentences = self.segment_into_sentences(text)
        if not sentences:
            return 0.0
        
        words = text.split()
        if not words:
            return 0.0
        
        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simple readability score (lower is more readable)
        readability = (avg_sentence_length * 0.39) + (avg_word_length * 11.8) - 15.59
        
        # Normalize to 0-1 scale (higher is more readable)
        return max(0.0, min(1.0, (100 - readability) / 100))

    def process_pdf(self, pdf_path: str, chunk_size: int = 500, 
                   overlap_sentences: int = 2) -> List[Dict]:
        """Main processing pipeline."""
        print(f"🚀 Starting semantic PDF processing...")
        
        # Extract structured content
        structured_content = self.extract_structured_content(pdf_path)
        
        # Create semantic chunks
        chunks = self.create_semantic_chunks(
            structured_content, chunk_size, overlap_sentences
        )
        
        # Filter and enhance
        final_chunks = self.filter_and_enhance_chunks(chunks)
        
        return final_chunks

def save_chunks_to_json(chunks: List[Dict], output_path: str):
    """Save chunks to JSON with proper serialization."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert sets to lists for JSON serialization
    json_chunks = []
    for chunk in chunks:
        json_chunk = chunk.copy()
        json_chunk["metadata"] = chunk["metadata"].copy()
        
        # Convert any sets to lists
        for key, value in json_chunk["metadata"].items():
            if isinstance(value, set):
                json_chunk["metadata"][key] = list(value)
        
        json_chunks.append(json_chunk)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_chunks, f, indent=2, ensure_ascii=False)

# Main execution
if __name__ == "__main__":
    # Initialize processor
    processor = SemanticPDFChunker()
    
    # Configuration
    pdf_path = "data/sample.pdf"  # Change to your PDF path
    output_path = "output/semantic_chunks.json"
    chunk_size = 500  # Target words per chunk
    overlap_sentences = 2  # Sentences to overlap between chunks
    
    try:
        # Process PDF
        chunks = processor.process_pdf(pdf_path, chunk_size, overlap_sentences)
        
        # Save results
        save_chunks_to_json(chunks, output_path)
        
        # Display comprehensive results
        print(f"\n🎉 Successfully processed {pdf_path}")
        print(f"💾 Saved {len(chunks)} chunks to {output_path}")
        
        if chunks:
            # Statistics
            high_imp = sum(1 for c in chunks if c["metadata"]["importance"] == "high")
            med_imp = sum(1 for c in chunks if c["metadata"]["importance"] == "medium")
            low_imp = sum(1 for c in chunks if c["metadata"]["importance"] == "low")
            
            avg_words = sum(c["metadata"]["word_count"] for c in chunks) // len(chunks)
            avg_sentences = sum(c["metadata"]["sentence_count"] for c in chunks) // len(chunks)
            avg_quality = sum(c["metadata"]["quality_score"] for c in chunks) / len(chunks)
            
            print(f"\n📊 Chunk Statistics:")
            print(f"   🔴 High importance: {high_imp}")
            print(f"   🟡 Medium importance: {med_imp}")
            print(f"   🔵 Low importance: {low_imp}")
            print(f"   📏 Average words per chunk: {avg_words}")
            print(f"   📝 Average sentences per chunk: {avg_sentences}")
            print(f"   ⭐ Average quality score: {avg_quality:.2f}/3.0")
            
            # Show sample high-quality chunk
            print(f"\n📋 Sample high-quality chunk:")
            best_chunk = max(chunks, key=lambda x: x["metadata"]["quality_score"])
            print(f"Title: {best_chunk['metadata']['title']}")
            print(f"Importance: {best_chunk['metadata']['importance']}")
            print(f"Quality Score: {best_chunk['metadata']['quality_score']:.2f}")
            print(f"Text Preview: {best_chunk['text'][:200]}...")
            
    except Exception as e:
        print(f"❌ Error processing PDF: {str(e)}")
        raise