"""
Multimodal RAG System for Maritime AKV PDF
Step 1: IMPROVED PDF Text Extraction with Enhanced OCR

Requirements:
pip install pymupdf pillow pytesseract pdf2image opencv-python numpy easyocr

Note: Install Tesseract OR use EasyOCR (no external dependencies):
- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
- Linux: sudo apt-get install tesseract-ocr
- Mac: brew install tesseract

OR use EasyOCR (recommended - no external installation needed)
"""

import os
import fitz  # PyMuPDF
from PIL import Image
import cv2
import numpy as np
import json
from pathlib import Path
import time
from typing import List, Dict, Tuple

# Try to import OCR libraries
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("EasyOCR not available. Install with: pip install easyocr")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Pytesseract not available. Install with: pip install pytesseract")

class PDFTextExtractor:
    """Extract text from PDFs with enhanced OCR capabilities"""
    
    def __init__(self, pdf_path: str, output_dir: str = "extracted_data", ocr_method: str = "auto"):
        """
        Args:
            pdf_path: Path to PDF file
            output_dir: Output directory for extracted content
            ocr_method: 'auto', 'easyocr', 'tesseract'
        """
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine OCR method
        if ocr_method == "auto":
            if EASYOCR_AVAILABLE:
                self.ocr_method = "easyocr"
                self.reader = easyocr.Reader(['en'], gpu=False)
                print("✓ Using EasyOCR for text extraction")
            elif TESSERACT_AVAILABLE:
                self.ocr_method = "tesseract"
                print("✓ Using Tesseract for text extraction")
            else:
                raise RuntimeError("No OCR library available. Install easyocr or pytesseract")
        else:
            self.ocr_method = ocr_method
            if ocr_method == "easyocr":
                if not EASYOCR_AVAILABLE:
                    raise RuntimeError("EasyOCR not available")
                self.reader = easyocr.Reader(['en'], gpu=False)
        
        # Create subdirectories
        self.text_dir = self.output_dir / "text"
        self.images_dir = self.output_dir / "images"
        self.metadata_dir = self.output_dir / "metadata"
        self.processed_images_dir = self.output_dir / "processed_images"
        
        for dir_path in [self.text_dir, self.images_dir, self.metadata_dir, self.processed_images_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.extraction_metadata = {
            "pdf_path": pdf_path,
            "ocr_method": self.ocr_method,
            "total_pages": 0,
            "pages_with_text": 0,
            "pages_with_ocr": 0,
            "total_images_extracted": 0,
            "extraction_time": 0
        }
    
    def preprocess_image_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image to improve OCR accuracy
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply thresholding to get better contrast
        # Try adaptive thresholding first
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        return denoised
    
    def extract_text_from_page(self, page) -> Tuple[str, bool]:
        """
        Extract text from a PDF page using PyMuPDF
        Returns: (text, is_scanned_flag)
        """
        text = page.get_text()
        
        # Check if page is likely scanned (very little or no text)
        is_scanned = len(text.strip()) < 50
        
        return text, is_scanned
    
    def ocr_with_easyocr(self, image_path: str) -> str:
        """
        Perform OCR using EasyOCR
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            
            # Preprocess
            processed = self.preprocess_image_for_ocr(img)
            
            # Save processed image for debugging
            processed_filename = Path(image_path).stem + "_processed.png"
            cv2.imwrite(
                str(self.processed_images_dir / processed_filename), 
                processed
            )
            
            # Perform OCR on both original and processed
            results_original = self.reader.readtext(image_path)
            results_processed = self.reader.readtext(processed)
            
            # Combine results (use the one with more text)
            text_original = ' '.join([result[1] for result in results_original])
            text_processed = ' '.join([result[1] for result in results_processed])
            
            # Return the longer result
            if len(text_processed) > len(text_original):
                return text_processed
            else:
                return text_original
        
        except Exception as e:
            print(f"EasyOCR failed: {str(e)}")
            return ""
    
    def ocr_with_tesseract(self, image_path: str) -> str:
        """
        Perform OCR using Tesseract
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            
            # Preprocess
            processed = self.preprocess_image_for_ocr(img)
            
            # Save processed image
            processed_filename = Path(image_path).stem + "_processed.png"
            cv2.imwrite(
                str(self.processed_images_dir / processed_filename), 
                processed
            )
            
            # Try both original and processed
            pil_img_original = Image.open(image_path)
            pil_img_processed = Image.fromarray(processed)
            
            text_original = pytesseract.image_to_string(pil_img_original)
            text_processed = pytesseract.image_to_string(pil_img_processed)
            
            # Return the longer result
            if len(text_processed) > len(text_original):
                return text_processed
            else:
                return text_original
        
        except Exception as e:
            print(f"Tesseract OCR failed: {str(e)}")
            return ""
    
    def ocr_page_from_pdf(self, page_num: int, dpi: int = 300) -> str:
        """
        Convert PDF page to image and perform OCR
        """
        from pdf2image import convert_from_path
        
        try:
            # Convert PDF page to image
            images = convert_from_path(
                self.pdf_path,
                first_page=page_num + 1,
                last_page=page_num + 1,
                dpi=dpi
            )
            
            if not images:
                return ""
            
            # Save temporary image
            temp_image_path = self.processed_images_dir / f"temp_page_{page_num}.png"
            images[0].save(temp_image_path)
            
            # Perform OCR
            if self.ocr_method == "easyocr":
                text = self.ocr_with_easyocr(str(temp_image_path))
            else:
                text = self.ocr_with_tesseract(str(temp_image_path))
            
            return text
        
        except Exception as e:
            print(f"OCR failed for page {page_num}: {str(e)}")
            return ""
    
    def extract_images_from_page(self, page, page_num: int) -> List[Dict]:
        """
        Extract images from a PDF page
        """
        images_metadata = []
        image_list = page.get_images(full=True)
        
        for img_index, img_info in enumerate(image_list):
            try:
                xref = img_info[0]
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Save image
                image_filename = f"page_{page_num:03d}_img_{img_index:03d}.{image_ext}"
                image_path = self.images_dir / image_filename
                
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                # Get image dimensions
                img = Image.open(image_path)
                width, height = img.size
                
                # Extract text from this image using OCR
                if self.ocr_method == "easyocr":
                    image_text = self.ocr_with_easyocr(str(image_path))
                else:
                    image_text = self.ocr_with_tesseract(str(image_path))
                
                images_metadata.append({
                    "image_id": f"img_{page_num:03d}_{img_index:03d}",
                    "page_num": page_num,
                    "path": str(image_path),
                    "filename": image_filename,
                    "width": width,
                    "height": height,
                    "format": image_ext,
                    "extracted_text": image_text,
                    "text_length": len(image_text)
                })
                
            except Exception as e:
                print(f"Failed to extract image {img_index} from page {page_num}: {str(e)}")
        
        return images_metadata
    
    def process_pdf(self) -> Dict:
        """
        Main processing function: extract text and images from PDF
        """
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Processing PDF: {self.pdf_path}")
        print(f"OCR Method: {self.ocr_method}")
        print(f"{'='*60}\n")
        
        doc = fitz.open(self.pdf_path)
        
        self.extraction_metadata["total_pages"] = len(doc)
        
        all_text_content = []
        all_images_metadata = []
        
        for page_num in range(len(doc)):
            print(f"\nProcessing page {page_num + 1}/{len(doc)}...")
            
            page = doc[page_num]
            page_data = {
                "page_num": page_num,
                "text": "",
                "ocr_used": False,
                "images": [],
                "word_count": 0
            }
            
            # First, try to extract native text
            text, is_scanned = self.extract_text_from_page(page)
            
            # If page appears scanned or has minimal text, use OCR
            if is_scanned:
                print(f"  📄 Page appears to be scanned/image-based. Running OCR...")
                
                # Try OCR from PDF page
                ocr_text = self.ocr_page_from_pdf(page_num)
                
                if ocr_text and len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
                    page_data["ocr_used"] = True
                    self.extraction_metadata["pages_with_ocr"] += 1
                    print(f"  ✓ OCR extracted {len(ocr_text)} characters")
            else:
                self.extraction_metadata["pages_with_text"] += 1
                print(f"  ✓ Native text extraction: {len(text)} characters")
            
            # Extract images and their text
            images_metadata = self.extract_images_from_page(page, page_num)
            
            # If main text is empty but images have text, use image text
            if len(text.strip()) < 50 and images_metadata:
                combined_image_text = "\n\n".join([
                    img["extracted_text"] for img in images_metadata 
                    if img.get("extracted_text")
                ])
                if combined_image_text and len(combined_image_text) > len(text):
                    text = combined_image_text
                    print(f"  ✓ Used text from images: {len(text)} characters")
            
            page_data["text"] = text
            page_data["word_count"] = len(text.split())
            page_data["images"] = images_metadata
            
            # Save page text
            text_filename = f"page_{page_num:03d}.txt"
            with open(self.text_dir / text_filename, "w", encoding="utf-8") as f:
                f.write(text)
            
            all_text_content.append(page_data)
            all_images_metadata.extend(images_metadata)
            
            print(f"  📊 Final stats: {len(text)} chars, {len(text.split())} words, {len(images_metadata)} images")
        
        doc.close()
        
        self.extraction_metadata["total_images_extracted"] = len(all_images_metadata)
        self.extraction_metadata["extraction_time"] = time.time() - start_time
        self.extraction_metadata["total_words"] = sum(p["word_count"] for p in all_text_content)
        
        # Save metadata
        metadata_file = self.metadata_dir / "extraction_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": self.extraction_metadata,
                "pages": all_text_content,
                "images": all_images_metadata
            }, f, indent=2, ensure_ascii=False)
        
        # Save summary
        self.print_summary()
        
        return {
            "text_content": all_text_content,
            "images_metadata": all_images_metadata,
            "extraction_metadata": self.extraction_metadata
        }
    
    def print_summary(self):
        """Print extraction summary"""
        print("\n" + "="*60)
        print("PDF EXTRACTION SUMMARY")
        print("="*60)
        print(f"OCR Method: {self.extraction_metadata['ocr_method']}")
        print(f"Total Pages: {self.extraction_metadata['total_pages']}")
        print(f"Pages with native text: {self.extraction_metadata['pages_with_text']}")
        print(f"Pages requiring OCR: {self.extraction_metadata['pages_with_ocr']}")
        print(f"Total words extracted: {self.extraction_metadata.get('total_words', 0)}")
        print(f"Total images extracted: {self.extraction_metadata['total_images_extracted']}")
        print(f"Extraction time: {self.extraction_metadata['extraction_time']:.2f} seconds")
        print(f"\n📁 Output directory: {self.output_dir.absolute()}")
        print("="*60)


def main():
    """
    Example usage
    """
    # IMPORTANT: Replace with your actual PDF path
    pdf_path = "MAKV-2047.pdf"  # Update this path
    
    if not os.path.exists(pdf_path):
        print(f"❌ ERROR: PDF file not found at {pdf_path}")
        print("Please update the pdf_path variable with the correct path to your Maritime AKV PDF")
        return
    
    # Initialize extractor with EasyOCR (recommended) or Tesseract
    extractor = PDFTextExtractor(
        pdf_path=pdf_path,
        output_dir="maritime_akv_extracted",
        ocr_method="auto"  # Will use easyocr if available, else tesseract
    )
    
    # Process PDF
    results = extractor.process_pdf()
    
    print("\n✅ Step 1 Complete: PDF Text Extraction with Enhanced OCR")
    print("\n📂 Check extracted content:")
    print("   • Text files: maritime_akv_extracted/text/")
    print("   • Images: maritime_akv_extracted/images/")
    print("   • Processed images: maritime_akv_extracted/processed_images/")
    print("   • Metadata: maritime_akv_extracted/metadata/")
    print("\n➡️  Ready for Step 2: Text Chunking and Embedding")


if __name__ == "__main__":
    main()