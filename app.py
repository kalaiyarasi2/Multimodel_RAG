import os
import time
import io
import base64
from groq import Groq as GroqClient
import warnings
from typing import List, Dict, Any
from PIL import Image
from pathlib import Path
import dotenv
import concurrent.futures
import psutil
import GPUtil
import pickle

import fitz  # PyMuPDF
import pdf2image
import cv2
import numpy as np

# OCR imports (try easyocr first, fallback to tesseract)
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    print("✓ EasyOCR available for OCR")
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    print("✓ Tesseract available for OCR")
except ImportError:
    TESSERACT_AVAILABLE = False

import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    CLIPProcessor,
    CLIPModel
)
from sentence_transformers import CrossEncoder
import faiss

# LlamaIndex
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
from llama_index.core.retrievers import VectorIndexRetriever


dotenv.load_dotenv()

warnings.filterwarnings("ignore")

# ==============================
# POPPLER PATH CONFIGURATION
# ==============================


POPPLER_PATH = r"D:\poppler\poppler-25.12.0\Library\bin"

# ==============================
# Utility: Timing Decorator
# ==============================
def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"[⏱️] {func.__name__} took {duration:.2f}s")
        return result, duration
    return wrapper

# ==============================
# Resource Manager: Monitor and throttle to 80% usage
# ==============================
class ResourceManager:
    def __init__(self, max_cpu_percent=80.0, max_gpu_percent=80.0):
        self.max_cpu_percent = max_cpu_percent
        self.max_gpu_percent = max_gpu_percent
        self.cpu_count = psutil.cpu_count(logical=True)
        self.max_workers = max(1, int(self.cpu_count * 0.8))  # 80% of logical cores

    def get_cpu_usage(self):
        return psutil.cpu_percent(interval=1)

    def get_gpu_usage(self):
        try:
            gpus = GPUtil.getGPUs()
            return max([gpu.load * 100 for gpu in gpus]) if gpus else 0
        except:
            return 0

    def should_throttle(self):
        cpu_usage = self.get_cpu_usage()
        gpu_usage = self.get_gpu_usage()
        return cpu_usage > self.max_cpu_percent or gpu_usage > self.max_gpu_percent

    def wait_if_needed(self):
        if self.should_throttle():
            cpu_usage = self.get_cpu_usage()
            gpu_usage = self.get_gpu_usage()
            print(f"⚠️  Resource usage high (CPU: {cpu_usage:.1f}%, GPU: {gpu_usage:.1f}%), throttling...")
            time.sleep(2.0)

    def get_thread_pool(self):
        return concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)

# ==============================
# Multimodal RAG System
# ==============================
class MultimodalRAGSystem:
    def __init__(self, groq_api_key: str, output_dir: str = "./rag_output"):
        self.output_dir = output_dir
        self.processed_data_path = f"{output_dir}/processed_data.pkl"
        os.makedirs(f"{output_dir}/images", exist_ok=True)

        # Resource manager for 80% utilization
        self.resource_mgr = ResourceManager(max_cpu_percent=80.0, max_gpu_percent=80.0)

        print("🔄 Initializing models with GPU acceleration...")
        # Force GPU priority
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # Use first GPU
            torch.set_default_device('cuda')
            self.device = torch.device('cuda')
            print("✓ GPU priority enabled - using GTX 1650")
        else:
            self.device = torch.device('cpu')
            print("⚠️ GPU not available, using CPU")

        print(f"Using device: {self.device}")

        # LlamaIndex Settings
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.llm = Groq(model="openai/gpt-oss-20b", api_key=groq_api_key)  # Valid Groq model
        Settings.node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=50)  # ✅ Token-aware via tiktoken

        # CLIP - GPU prioritized
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(self.device)

        # Groq Client for Vision (LLaVA) - using the passed API key
        try:
            self.groq_client = GroqClient(api_key=groq_api_key)
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize Groq client: {e}")
            print("   Image captioning will use OCR fallback only.")
            self.groq_client = None

        # Cross-Encoder for reranking
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

        # Storage - check for pre-processed data
        self.image_metadata: List[Dict] = []
        self.documents: List[Document] = []
        self.text_index = None
        self.clip_image_index = None
        self.clip_embeddings = None

        # Try to load pre-processed data
        self.processed_data_loaded = self.load_processed_data()

        print("✅ Initialization complete with GPU acceleration.")
        if self.processed_data_loaded:
            print("📂 Pre-processed data loaded - ready for queries!")
        else:
            print("📝 No pre-processed data found - will process PDF first")

    def load_processed_data(self) -> bool:
        """Load pre-processed data from disk"""
        try:
            # Check if all required files exist
            faiss_index_path = f"{self.output_dir}/clip_faiss.index"
            embeddings_path = f"{self.output_dir}/clip_embeddings.npy"
        
            if not (os.path.exists(self.processed_data_path) and 
                    os.path.exists(faiss_index_path) and 
                    os.path.exists(embeddings_path)):
                print("📂 No complete pre-processed data found")
                return False
        
            print("📄 Loading pre-processed data...")
        
            # Load pickled data
            with open(self.processed_data_path, 'rb') as f:
                data = pickle.load(f)
        
            self.image_metadata = data['image_metadata']
            self.documents = data['documents']
        
            # Load FAISS index
            self.clip_image_index = faiss.read_index(faiss_index_path)
            print(f"  ✅ Loaded FAISS index: {self.clip_image_index.ntotal} vectors")
        
            # Load embeddings
            self.clip_embeddings = np.load(embeddings_path)
            print(f"  ✅ Loaded embeddings: {self.clip_embeddings.shape}")
        
            # Verify counts match
            if len(self.image_metadata) != self.clip_image_index.ntotal:
                print(f"⚠️ Warning: Metadata count ({len(self.image_metadata)}) != "
                      f"FAISS index count ({self.clip_image_index.ntotal})")
        
            # Recreate text index from persisted storage - use docstore.json path
            storage_context_path = f"{self.output_dir}/docstore.json"
            default_storage_path = f"{self.output_dir}/default__storage__.json"
            
            if os.path.exists(storage_context_path) or os.path.exists(default_storage_path):
                from llama_index.core import StorageContext, load_index_from_storage
                storage_context = StorageContext.from_defaults(persist_dir=self.output_dir)
                self.text_index = load_index_from_storage(storage_context)
                print("  ✅ Loaded text index from storage")
            else:
                print("  ℹ️ Text index not found, will rebuild from documents")
                self.text_index = None
        
            print(f"✅ Pre-processed data loaded successfully:")
            print(f"   - {len(self.image_metadata)} images")
            print(f"   - {len(self.documents)} documents")
            print(f"   - {self.clip_embeddings.shape[0]} embeddings")
        
            return True
        
        except Exception as e:
            print(f"⚠️ Failed to load pre-processed data: {e}")
            import traceback
            traceback.print_exc()
            return False    

    def save_processed_data(self):
        """Save processed data to disk for persistence"""
        try:
            print("💾 Saving processed data...")
        
            # Save FAISS index separately using faiss.write_index
            if self.clip_image_index is not None:
                faiss_index_path = f"{self.output_dir}/clip_faiss.index"
                faiss.write_index(self.clip_image_index, faiss_index_path)
                print(f"  ✅ Saved FAISS index to {faiss_index_path}")
        
            # Save embeddings separately as numpy array
            embeddings_path = f"{self.output_dir}/clip_embeddings.npy"
            if self.clip_embeddings is not None:
                np.save(embeddings_path, self.clip_embeddings)
                print(f"  ✅ Saved embeddings to {embeddings_path}")
        
            # Pickle the rest (without FAISS index)
            data = {
                'image_metadata': self.image_metadata,
                'documents': self.documents,
            }
        
            with open(self.processed_data_path, 'wb') as f:
                pickle.dump(data, f)
        
            print("✅ Processed data saved successfully")
        except Exception as e:
            print(f"⚠️ Failed to save processed data: {e}")
            import traceback
            traceback.print_exc()

    # ------------------------------
    # 1. PDF Text Extraction (Load from files or PyMuPDF + OCR)
    # ------------------------------
    def load_pre_extracted_text(self) -> List[Dict]:
        """Load text from pre-extracted files (from text_extraction.py)"""
        pages_text = []
        text_dir = Path("maritime_akv_extracted_unstructured/text")
        if text_dir.exists():
            for txt_file in sorted(text_dir.glob("*.txt")):
                page_num = int(txt_file.stem.split('_')[1]) + 1
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                if text:
                    pages_text.append({'page_num': page_num, 'text': text})
        return pages_text

    def extract_page_text(self, pdf_path: str, page_num: int) -> Dict:
        """Extract text from a single page (for parallel processing)"""
        doc = fitz.open(pdf_path)
        page = doc[page_num]

        # Try native text extraction first
        text = page.get_text().strip()
        extraction_method = "native"

        # If text is minimal (<50 chars), likely scanned - use OCR
        if len(text) < 50:
            ocr_text = self.ocr_page_from_pdf(pdf_path, page_num)
            if ocr_text and len(ocr_text.strip()) > len(text):
                text = ocr_text.strip()
                extraction_method = "ocr"

        doc.close()

        return {
            'page_num': page_num + 1,
            'text': text,
            'extraction_method': extraction_method,
            'char_count': len(text)
        }

    @timed
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        # First, try to load pre-extracted text
        pages_text = self.load_pre_extracted_text()
        if pages_text:
            print(f"📄 Loaded pre-extracted text from {len(pages_text)} pages")
            return pages_text

        # Fallback to extraction with parallel processing
        print("📄 No pre-extracted text found. Extracting with PyMuPDF + OCR fallback (parallel)...")
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()

        # Use thread pool for parallel page processing
        with self.resource_mgr.get_thread_pool() as executor:
            futures = [executor.submit(self.extract_page_text, pdf_path, page_num)
                      for page_num in range(total_pages)]
            pages_text = [f.result() for f in futures]

        # Sort by page number and log results
        pages_text.sort(key=lambda x: x['page_num'])
        for page_data in pages_text:
            page_num = page_data['page_num']
            char_count = page_data['char_count']
            method = page_data['extraction_method']
            print(f"  Page {page_num}: {char_count} chars ({method})")

        print(f"✅ Extracted text from {len(pages_text)} pages using parallel processing")
        return pages_text

    # ------------------------------
    # 2. Image Extraction (pdf2image + OpenCV)
    # ------------------------------
    @timed
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict]:
        print("🖼️ Extracting images with pdf2image + OpenCV...")
        # Convert PDF to list of PIL images (one per page)
        page_images = pdf2image.convert_from_path(pdf_path, dpi=200, poppler_path=POPPLER_PATH )
        extracted_images = []

        for page_num, pil_img in enumerate(page_images):
            # Convert to OpenCV
            open_cv_image = np.array(pil_img)
            img_cv = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            img_count = 0
            
            # Calculate adaptive thresholds based on page size
            page_area = open_cv_image.shape[0] * open_cv_image.shape[1]
            min_contour_area = page_area * 0.001  # 0.1% of page
            min_width = open_cv_image.shape[1] * 0.05  # 5% of page width
            min_height = open_cv_image.shape[0] * 0.05  # 5% of page height
            
            for contour in contours:
                if cv2.contourArea(contour) > min_contour_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    cropped = open_cv_image[y:y+h, x:x+w]
                    if w > min_width or h > min_height:  # OR instead of AND
                        img_id = f"img_{page_num+1:03d}_{img_count:02d}"
                        img_path = f"{self.output_dir}/images/{img_id}.png"
                        Image.fromarray(cropped).save(img_path)
                        extracted_images.append({
                            'image_id': img_id,
                            'page_num': page_num + 1,
                            'path': img_path,
                            'bbox': (x, y, w, h)
                        })
                        img_count += 1
        return extracted_images

    # ------------------------------
    # OCR Helper Methods
    # ------------------------------
    def preprocess_image_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR accuracy"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply simple thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresh

    def ocr_with_easyocr(self, image_path: str) -> str:
        """Perform OCR using EasyOCR - CPU only to save GPU memory"""
        if not EASYOCR_AVAILABLE:
            return ""

        try:
            # Initialize reader if not exists - ALWAYS use CPU for OCR
            if not hasattr(self, 'easyocr_reader'):
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False)  # CPU only!
                print("✓ EasyOCR initialized on CPU (GPU reserved for CLIP/BLIP)")

            # Perform OCR directly on image
            results = self.easyocr_reader.readtext(image_path)

            # Combine results
            text = ' '.join([result[1] for result in results])

            return text

        except Exception as e:
            print(f"EasyOCR failed: {str(e)}")
            return ""

    def ocr_with_tesseract(self, image_path: str) -> str:
        """Perform OCR using Tesseract"""
        if not TESSERACT_AVAILABLE:
            return ""

        try:
            # Use PIL image directly
            pil_img = Image.open(image_path)

            # OCR
            text = pytesseract.image_to_string(pil_img)

            return text

        except Exception as e:
            print(f"Tesseract OCR failed: {str(e)}")
            return ""

    def ocr_page_from_pdf(self, pdf_path: str, page_num: int, dpi: int = 300) -> str:
        """Convert PDF page to image and perform OCR"""
        try:
            # Convert PDF page to image
            images = pdf2image.convert_from_path(
                pdf_path,
                first_page=page_num + 1,
                last_page=page_num + 1,
                dpi=dpi,
                poppler_path=POPPLER_PATH
            )

            if not images:
                return ""

            # Save temporary image
            temp_path = f"{self.output_dir}/temp_page_{page_num}.png"
            images[0].save(temp_path)

            # Perform OCR
            if EASYOCR_AVAILABLE:
                text = self.ocr_with_easyocr(temp_path)
            elif TESSERACT_AVAILABLE:
                text = self.ocr_with_tesseract(temp_path)
            else:
                print("No OCR library available")
                text = ""

            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

            return text

        except Exception as e:
            print(f"OCR failed for page {page_num}: {str(e)}")
            return ""

    # ------------------------------
    # 3. Visual Processing (GPU-accelerated with batching)
    # ------------------------------
    def generate_image_caption(self, image_path: str) -> str:
        """Generate caption using Groq LLaVA model"""
        if self.groq_client is None:
            print("⚠️  Groq client not available, using OCR fallback")
            return ""
        
        try:
            # Encode image to base64
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Call Groq API
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in detail. If there is text, transcribe it."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_string}",
                                },
                            },
                        ],
                    }
                ],
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
            )
            
            caption = chat_completion.choices[0].message.content
            return caption

        except Exception as e:
            print(f"⚠️ Groq caption generation failed: {e}. Falling back to OCR.")
            # Fallback to OCR if Groq fails
            return self.ocr_with_easyocr(image_path)


    def _is_text_heavy_image(self, img_array: np.ndarray) -> bool:
        """Check if image is primarily text vs visual content"""
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        return edge_density > 0.15  # High edge density = likely text

    def embed_image_clip(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.clip_model.get_image_features(**inputs).cpu().numpy()
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        torch.cuda.empty_cache()  # Clear GPU memory
        return emb[0]

    def process_images_batch(self, image_paths: List[str], batch_size: int = 4) -> Dict[str, List]:
        """Process images in batches for better GPU utilization"""
        captions = []
        embeddings = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]

            # Throttle if needed
            self.resource_mgr.wait_if_needed()

            # Process batch with thread pool
            with self.resource_mgr.get_thread_pool() as executor:
                caption_futures = [executor.submit(self.generate_image_caption, path) for path in batch_paths]
                embed_futures = [executor.submit(self.embed_image_clip, path) for path in batch_paths]

                batch_captions = [f.result() for f in caption_futures]
                batch_embeddings = [f.result() for f in embed_futures]

            captions.extend(batch_captions)
            embeddings.extend(batch_embeddings)

            # Clear cache after batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {'captions': captions, 'embeddings': embeddings}

    # ------------------------------
    # 4. Main Processing Pipeline
    # ------------------------------
    def process_pdf(self, pdf_path: str):
        print("\n" + "="*60)
        print("🚀 STARTING PDF PROCESSING (FULL REQUIREMENTS COMPLIANT)")
        print("="*60)

        # 1. Extract text
        pages_text, _ = self.extract_text_from_pdf(pdf_path)
        print("\n📄 Extracted Text:")
        for pt in pages_text:
            print(f"Page {pt['page_num']}: {pt['text']}")

        # 2. Extract images
        images_data, _ = self.extract_images_from_pdf(pdf_path)
        self.image_metadata = images_data
        print("\n🖼️ Extracted Images:")
        for img in images_data:
            print(f"Image {img['image_id']}: Page {img['page_num']}, Path: {img['path']}, Bbox: {img['bbox']}")

        # 3. Process images: caption + CLIP embed (batched with threading)
        print("🧠 Generating image captions and CLIP embeddings with batch processing...")
        image_paths = [img['path'] for img in images_data]
        batch_results = self.process_images_batch(image_paths, batch_size=4)

        clip_embeddings = []
        for i, (caption, embedding) in enumerate(zip(batch_results['captions'], batch_results['embeddings'])):
            images_data[i]['caption'] = caption
            clip_embeddings.append(embedding)
            print(f"  → {images_data[i]['image_id']}: {caption}")

        # Build CLIP FAISS index
        if clip_embeddings:
            self.clip_embeddings = np.array(clip_embeddings).astype('float32')
            self.clip_image_index = faiss.IndexFlatIP(512)
            self.clip_image_index.add(self.clip_embeddings)

        # 4. Build LlamaIndex documents (text + image captions)
        print("🧱 Building LlamaIndex documents...")
        self.documents = []

        # Add text pages
        for pt in pages_text:
            self.documents.append(Document(
                text=pt['text'],
                metadata={'page_num': pt['page_num'], 'type': 'text'}
            ))

        # Add image captions as standalone documents
        for img in images_data:
            self.documents.append(Document(
                text=img['caption'],
                metadata={
                    'page_num': img['page_num'],
                    'image_id': img['image_id'],
                    'image_path': img['path'],
                    'type': 'image_caption'
                }
            ))

        # Print chunking
        parsed_nodes = Settings.node_parser.get_nodes_from_documents(self.documents)
        print(f"\n📄 Text Chunks: {len(parsed_nodes)} total")
        for i, node in enumerate(parsed_nodes[:3]):  # Show first 3 chunks
            print(f"Chunk {i+1}: {node.text[:150]}...")
        if len(parsed_nodes) > 3:
            print(f"... and {len(parsed_nodes) - 3} more chunks")

        # 5. Build FAISS text index via LlamaIndex
        print("💾 Building FAISS text index...")
        faiss_index = faiss.IndexFlatIP(384)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        if not self.documents:
            print("⚠️  No documents to index. Check text and image extraction.")
            return

        self.text_index = VectorStoreIndex.from_documents(
            self.documents,
            storage_context=storage_context,
            show_progress=True
        )
        # Persist
        storage_context.persist(persist_dir=self.output_dir)

        # Save processed data for persistence
        self.save_processed_data()

        print("✅ PDF processing complete.")

    # ------------------------------
    # 5. Hybrid Retrieval + Reranking
    # ------------------------------
    def retrieve_hybrid(self, query: str, top_k: int = 10, rerank_top: int = 15):
        """
        Improved retrieval with better image-text fusion
        """
        # 1. TEXT RETRIEVAL (unchanged)
        text_retriever = VectorIndexRetriever(index=self.text_index, similarity_top_k=rerank_top)
        text_nodes = text_retriever.retrieve(query)
        
        text_items = []
        for node in text_nodes:
            text_items.append({
                'text': node.text,
                'metadata': {
                    **node.metadata,
                    'type': 'text',
                    'retrieval_score': float(node.score) if hasattr(node, 'score') else 0.0
                }
            })
        
        # 2. IMPROVED IMAGE RETRIEVAL
        image_items = []
        if self.clip_image_index is not None:
            # Method A: CLIP visual search
            inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True)
            with torch.no_grad():
                text_emb = self.clip_model.get_text_features(**inputs).cpu().numpy()
            text_emb = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)
            
            scores, idxs = self.clip_image_index.search(text_emb.astype('float32'), rerank_top)
            
            for score, idx in zip(scores[0], idxs[0]):
                if idx < len(self.image_metadata):
                    img = self.image_metadata[idx]
                    image_items.append({
                        'text': img['caption'],  # Use caption as text
                        'metadata': {
                            'page_num': img['page_num'],
                            'image_id': img['image_id'],
                            'type': 'image_caption',
                            'retrieval_score': float(score)
                        }
                    })
        
        # 3. COMBINE ALL CANDIDATES
        all_items = text_items + image_items
        
        if not all_items:
            print("⚠️ No items retrieved")
            return []
        
        # 4. RERANK with Cross-Encoder
        pairs = [(query, item['text']) for item in all_items]
        cross_scores = self.cross_encoder.predict(pairs)
        
        # 5. COMBINED SCORING (normalized)
        all_scores = [item['metadata']['retrieval_score'] for item in all_items]
        min_score, max_score = min(all_scores), max(all_scores)
        score_range = max_score - min_score if max_score != min_score else 1.0
        
        # Detect visual intent in query
        visual_keywords = ['image', 'picture', 'visual', 'diagram', 'show', 'cover', 'describe', 'photo', 'illustration', 'figure', 'chart', 'graph']
        has_visual_intent = any(kw in query.lower() for kw in visual_keywords)
        
        for i, item in enumerate(all_items):
            # Normalize retrieval score to 0-1
            norm_retrieval = (item['metadata']['retrieval_score'] - min_score) / score_range
            # Cross-encoder score (already roughly 0-1)
            cross_score = float(cross_scores[i])
            
            # Combined: 50% retrieval + 50% cross-encoder
            item['metadata']['combined_score'] = 0.5 * norm_retrieval + 0.5 * cross_score
            item['metadata']['rerank_score'] = cross_score
            
            # Boost image scores for visual queries
            if has_visual_intent and item['metadata'].get('type') == 'image_caption':
                item['metadata']['combined_score'] *= 1.2
                print(f"   🎨 Boosted image score: {item['metadata']['image_id']} -> {item['metadata']['combined_score']:.3f}")
        
        # 6. SORT by combined score
        all_items = sorted(all_items, key=lambda x: x['metadata']['combined_score'], reverse=True)
        
        # 7. MODALITY BALANCING - Ensure at least 2 images in results
        final_results = []
        text_added = 0
        images_added = 0
        
        for item in all_items:
            if len(final_results) >= top_k:
                break
            
            item_type = item['metadata'].get('type', 'text')
            
            # Prioritize adding missing modality
            if item_type == 'image_caption' and images_added < 2:
                final_results.append(item)
                images_added += 1
            elif item_type == 'text' and text_added < 8:
                final_results.append(item)
                text_added += 1
        
        # Fill remaining slots if needed
        for item in all_items:
            if len(final_results) >= top_k:
                break
            if item not in final_results:
                final_results.append(item)
        
        # 8. DEBUG OUTPUT
        print(f"\n📋 Retrieved {len(final_results)} items:")
        text_count = sum(1 for r in final_results if r['metadata'].get('type') == 'text')
        img_count = sum(1 for r in final_results if r['metadata'].get('type') == 'image_caption')
        print(f"   Text: {text_count} | Images: {img_count}")
        
        for i, item in enumerate(final_results[:5]):
            meta = item['metadata']
            score = meta.get('combined_score', 0.0)
            item_type = meta.get('type', 'text')[:4]
            page = meta.get('page_num', 'N/A')
            preview = item['text'][:50] + "..."
            print(f"   [{i+1}] {item_type} | Score: {score:.3f} | Page {page} | {preview}")
        
        return final_results

    def detect_query_type(self, query: str) -> str:
        """Detect if query needs images, text, or both"""
        query_lower = query.lower()
        
        # Visual keywords
        visual_keywords = ['image', 'picture', 'visual', 'diagram', 'chart', 'graph', 
                          'cover', 'figure', 'photo', 'illustration', 'show', 'describe']
        
        # Check for visual focus
        has_visual = any(kw in query_lower for kw in visual_keywords)
        
        if has_visual:
            return 'multimodal'  # Need both text and images
        else:
            return 'text'  # Text only is sufficient

    def retrieve_adaptive(self, query: str, top_k: int = 5):
        """Adaptive retrieval based on query type"""
        query_type = self.detect_query_type(query)
        
        if query_type == 'text':
            # Text-heavy retrieval: 90% text, 10% images
            print(f"🔍 Query type: TEXT-FOCUSED")
        else:  # multimodal
            # Balanced retrieval: 50% text, 50% images
            print(f"🔍 Query type: MULTIMODAL")
        
        # Call improved hybrid retrieval
        return self.retrieve_hybrid(query, top_k=top_k)

    # ------------------------------
    # 6. Query Engine
    # ------------------------------
    def query(self, query: str, top_k: int = 5):
        """Query with performance metrics tracking"""
        print(f"\n❓ Query: {query}")
        
        # Track retrieval time
        retrieval_start = time.time()
        context_items = self.retrieve_hybrid(query, top_k=top_k)
        retrieval_time = time.time() - retrieval_start
        
        if not context_items:
            return "Information not available in the document.", 0.0

        # Build context string with citations and truncation
        context_parts = []
        total_chars = 0
        MAX_CONTEXT_CHARS = 2000  # Leave room for prompt and answer
        
        for item in context_items:
            meta = item['metadata']
            page = meta.get('page_num', 'N/A')
            img_id = meta.get('image_id', None)
            img_path = meta.get('image_path', '')
            text = item['text']
            
            # Include image path in reference for better display
            if img_id:
                context_parts.append(f"[Page {page}, Image {img_id}] {text}")
            else:
                context_parts.append(f"[Page {page}] {text}")
            
            total_chars += len(text)
            if total_chars >= MAX_CONTEXT_CHARS:
                break

        context_str = "\n\n".join(context_parts)

        # Generate response
        prompt = f"""Based on the following retrieved context from the Maritime AKV document (including text and image descriptions), answer the user's query factually. Cite the relevant page number or image reference (e.g., [Page X], [Image ID Y]).

CONTEXT:
{context_str}

QUERY: {query}

ANSWER:"""
        
        # Track LLM generation time
        llm_start = time.time()
        try:    
            response = Settings.llm.complete(prompt)
            answer = str(response).strip()
            llm_time = time.time() - llm_start
            
            if not answer or len(answer) < 20:
                return "Unable to generate a complete answer. Please try rephrasing your question.", 0.0
            
            # Print performance metrics
            print(f"\n⏱️ PERFORMANCE METRICS:")
            print(f"   Retrieval: {retrieval_time:.3f}s")
            print(f"   LLM Generation: {llm_time:.3f}s")
            print(f"   Total: {retrieval_time + llm_time:.3f}s")
            
            return answer, 1.0
        except Exception as e:
            print(f"LLM Error : {e}")
            return "Error generating response:{str(e)}", 0.0

# ==============================
# Interactive Query System
# ==============================
def interactive_query_loop(rag_system):
    """Interactive query loop for multiple questions"""
    print("\n" + "="*60)
    print("🎯 INTERACTIVE QUERY MODE")
    print("="*60)
    print("Ask any questions about the Maritime AKV document!")
    print("Type 'quit', 'exit', or 'q' to end the session.")
    print("="*60)

    query_count = 0
    total_query_time = 0

    while True:
        try:
            # Get user query
            query = input(f"\n[Query #{query_count + 1}] ❓ ").strip()

            # Check for exit commands
            if query.lower() in ['quit', 'exit', 'q']:
                break

            if not query:
                print("⚠️  Please enter a question (or 'quit' to exit)")
                continue

            # Show resource usage before query
            cpu_usage = rag_system.resource_mgr.get_cpu_usage()
            gpu_usage = rag_system.resource_mgr.get_gpu_usage()
            print(f"📊 Resources - CPU: {cpu_usage:.1f}%, GPU: {gpu_usage:.1f}%")

            # Process query
            answer, query_time = rag_system.query(query)
            total_query_time += query_time
            query_count += 1

            # Display answer
            print(f"\n✅ Answer ({query_time:.3f}s):")
            print("-" * 40)
            print(answer)
            print("-" * 40)

        except KeyboardInterrupt:
            print("\n\n👋 Session ended by user.")
            break
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            continue

    # Show session summary
    if query_count > 0:
        avg_time = total_query_time / query_count
        print("📈 Session Summary:")     
        print(f" Total queries: {query_count}")    
        print("\n🎉 Thank you for using the Maritime AKV RAG System!")

# ==============================
# Main Execution
# ==============================
if __name__ == "__main__":
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')  # Use env var or fallback

    print("\n🚀 Starting Optimized Multimodal RAG System")
    print("Features: GPU acceleration, parallel processing, 80% resource limits")

    # Initialize system
    rag = MultimodalRAGSystem(groq_api_key=GROQ_API_KEY, output_dir="./maritime_rag")

    # Process PDF if not already processed
    if not rag.processed_data_loaded:
        print("\n📝 No pre-processed data found. Processing PDF once...")
        pdf_path = "MAKV-2047.pdf"

        if not os.path.exists(pdf_path):
            print(f"❌ PDF file not found: {pdf_path}")
            exit(1)

        # Time the processing
        start_time = time.time()
        rag.process_pdf(pdf_path)
        processing_time = time.time() - start_time

        print("💾 Processed data saved for future sessions!")
    else:
        print("\n✅ Using pre-processed data - skipping PDF processing!")

    # Enter interactive query loop
    interactive_query_loop(rag)





