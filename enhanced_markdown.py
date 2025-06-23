import fitz  # PyMuPDF
import re
import sys
import statistics
from typing import List, Dict, Tuple
import unicodedata

class EnhancedMarkdownExtractor:
    def __init__(self):
        # Noise patterns to filter out
        self.noise_patterns = [
            r'<EOS>|<pad>|<unk>|<s>|</s>',
            r'\[UNK\]|\[PAD\]|\[CLS\]|\[SEP\]',
            r'^\s*\d+\s*$',  # Standalone page numbers
            r'^[^\w\s]*$',   # Only special characters
            r'^\s*[•\-\*]\s*$',  # Standalone bullets
            r'^\s*\.{3,}\s*$',   # Just dots
            r'^\s*_{3,}\s*$',    # Just underscores
            r'^\s*\-{3,}\s*$',   # Just dashes
        ]
        
        # Section patterns for better organization
        self.section_patterns = {
            'title': r'(?i)^(title|document\s+title)[\s:]*(.+)$',
            'abstract': r'(?i)^(abstract|summary)[\s:]*(.*)$',
            'introduction': r'(?i)^(introduction|intro)[\s:]*(.*)$',
            'methodology': r'(?i)^(methodology|methods?)[\s:]*(.*)$',
            'results': r'(?i)^(results?|findings?)[\s:]*(.*)$',
            'conclusion': r'(?i)^(conclusions?|summary)[\s:]*(.*)$',
            'references': r'(?i)^(references?|bibliography)[\s:]*(.*)$',
        }

    def is_noise_text(self, text: str) -> bool:
        """Check if text is likely noise or artifact."""
        if not text or len(text.strip()) < 3:
            return True
            
        # Check noise patterns
        for pattern in self.noise_patterns:
            if re.search(pattern, text.strip()):
                return True
        
        # Check character composition
        clean_text = re.sub(r'\s', '', text)
        if len(clean_text) == 0:
            return True
            
        # Too many special characters (likely OCR artifact)
        if len(clean_text) > 0:
            special_char_ratio = sum(1 for c in clean_text if not c.isalnum()) / len(clean_text)
            if special_char_ratio > 0.7:
                return True
        
        # Repeated character patterns
        if len(set(clean_text)) <= 2 and len(clean_text) > 4:
            return True
            
        return False

    def clean_text_advanced(self, text: str) -> str:
        """Advanced text cleaning."""
        if not text:
            return ""
        
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Remove non-ASCII characters that might be OCR artifacts
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        
        # Fix common OCR issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Missing spaces
        text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)  # Number-letter spacing
        text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)  # Letter-number spacing
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Remove standalone special characters
        text = re.sub(r'\s+[^\w\s]{1,2}\s+', ' ', text)
        
        # Common OCR fixes
        text = re.sub(r'\bl\b', 'I', text)  # lowercase l to I
        text = re.sub(r'rn\b', 'm', text)   # rn to m
        
        return text.strip()

    def detect_heading_level(self, font_size: float, font_sizes: List[float]) -> int:
        """Determine heading level based on font size."""
        if not font_sizes:
            return 0
            
        # Calculate percentiles for heading levels
        sorted_sizes = sorted(set(font_sizes), reverse=True)
        
        if len(sorted_sizes) == 1:
            return 1 if font_size >= sorted_sizes[0] else 0
        
        # Define thresholds
        if font_size >= sorted_sizes[0]:
            return 1  # H1 - largest
        elif len(sorted_sizes) > 1 and font_size >= sorted_sizes[1]:
            return 2  # H2 - second largest
        elif len(sorted_sizes) > 2 and font_size >= sorted_sizes[2]:
            return 3  # H3 - third largest
        else:
            # Check if significantly larger than average
            avg_size = statistics.mean(font_sizes)
            if font_size > avg_size + 2:
                return 3
            return 0  # Not a heading

    def extract_pdf_to_markdown(self, pdf_path: str, md_path: str):
        """Enhanced PDF to Markdown conversion."""
        doc = fitz.open(pdf_path)
        markdown_lines = []
        all_font_sizes = []
        
        # First pass: collect all font sizes for heading detection
        print("🔍 Analyzing document structure...")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["text"].strip() and not self.is_noise_text(span["text"]):
                            all_font_sizes.append(span["size"])
        
        # Get unique font sizes for heading detection
        unique_sizes = sorted(set(all_font_sizes), reverse=True)
        print(f"📊 Found {len(unique_sizes)} distinct font sizes")
        
        # Second pass: generate markdown
        print("📝 Converting to markdown...")
        current_section = None
        last_headings = {}  # Track last heading at each level
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")["blocks"]
            
            # Add page marker (commented out by default)
            # markdown_lines.append(f"\n<!-- Page {page_num + 1} -->\n")
            
            for block in blocks:
                if "lines" not in block:
                    continue
                
                block_text = ""
                max_font_size = 0
                block_spans = []
                
                # Collect all spans in block
                for line in block["lines"]:
                    for span in line["spans"]:
                        if not self.is_noise_text(span["text"]):
                            cleaned_text = self.clean_text_advanced(span["text"])
                            if cleaned_text:
                                block_spans.append({
                                    "text": cleaned_text,
                                    "size": span["size"],
                                    "flags": span["flags"]
                                })
                                max_font_size = max(max_font_size, span["size"])
                
                if not block_spans:
                    continue
                
                # Combine text from spans
                block_text = " ".join(span["text"] for span in block_spans).strip()
                
                if not block_text:
                    continue
                
                # Determine if this is a heading
                heading_level = self.detect_heading_level(max_font_size, all_font_sizes)
                
                # Check for special sections
                section_type = self._identify_section_type(block_text)
                
                if heading_level > 0:
                    # This is a heading
                    heading_prefix = "#" * heading_level
                    
                    # Avoid duplicate headings
                    if block_text != last_headings.get(heading_level):
                        markdown_lines.append(f"\n{heading_prefix} {block_text}\n")
                        last_headings[heading_level] = block_text
                        current_section = section_type or block_text.lower()
                        
                        # Clear lower-level headings
                        for level in range(heading_level + 1, 7):
                            last_headings.pop(level, None)
                
                else:
                    # Regular paragraph text
                    if len(block_text.split()) > 2:  # Skip very short fragments
                        # Add special formatting for certain sections
                        if section_type == "abstract":
                            markdown_lines.append(f"\n**Abstract:** {block_text}\n")
                        elif section_type == "keywords":
                            markdown_lines.append(f"\n**Keywords:** {block_text}\n")
                        else:
                            # Check if it's a list item
                            if re.match(r'^\s*[\-\•\*]\s+', block_text):
                                markdown_lines.append(f"{block_text}\n")
                            elif re.match(r'^\s*\d+[\.\)]\s+', block_text):
                                markdown_lines.append(f"{block_text}\n")
                            else:
                                markdown_lines.append(f"{block_text}\n\n")
        
        # Post-process markdown
        processed_lines = self._post_process_markdown(markdown_lines)
        
        # Write to file
        with open(md_path, "w", encoding="utf-8") as f:
            f.writelines(processed_lines)
        
        # Generate statistics
        total_lines = len([line for line in processed_lines if line.strip()])
        headings = len([line for line in processed_lines if line.strip().startswith('#')])
        
        print(f"✅ Conversion complete!")
        print(f"📄 Output: {md_path}")
        print(f"📊 Stats: {total_lines} lines, {headings} headings")
        
        doc.close()

    def _identify_section_type(self, text: str) -> str:
        """Identify special section types."""
        text_lower = text.lower().strip()
        
        # Check for keywords section
        if re.match(r'(?i)^(keywords?|key\s+words?)[\s:]*', text_lower):
            return "keywords"
        
        # Check for abstract
        if re.match(r'(?i)^(abstract|summary)[\s:]*', text_lower):
            return "abstract"
        
        # Check other sections
        for section_name, pattern in self.section_patterns.items():
            if re.match(pattern, text_lower):
                return section_name
        
        return None

    def _post_process_markdown(self, lines: List[str]) -> List[str]:
        """Clean up the generated markdown."""
        processed = []
        prev_line = ""
        
        for line in lines:
            line = line.rstrip()
            
            # Skip empty lines after headings
            if prev_line.strip().startswith('#') and not line.strip():
                processed.append(line + '\n')
                prev_line = line
                continue
            
            # Avoid multiple consecutive empty lines
            if not line.strip() and not prev_line.strip():
                continue
            
            # Fix spacing around headings
            if line.strip().startswith('#'):
                if prev_line.strip() and not prev_line.strip().startswith('#'):
                    processed.append('\n')
                processed.append(line + '\n')
            else:
                processed.append(line + '\n')
            
            prev_line = line
        
        return processed

def main():
    """Main function for command line usage."""
    if len(sys.argv) != 3:
        print("Usage: python enhanced_pdf_to_markdown.py input.pdf output.md")
        print("\nExample:")
        print("  python enhanced_pdf_to_markdown.py research_paper.pdf paper.md")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    md_file = sys.argv[2]
    
    # Check if input file exists
    import os
    if not os.path.exists(pdf_file):
        print(f"❌ Error: File '{pdf_file}' not found")
        sys.exit(1)
    
    try:
        extractor = EnhancedMarkdownExtractor()
        extractor.extract_pdf_to_markdown(pdf_file, md_file)
    except Exception as e:
        print(f"❌ Error processing PDF: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()