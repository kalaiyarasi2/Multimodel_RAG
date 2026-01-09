# Maritime AKV Multimodal RAG System

A production-grade Retrieval-Augmented Generation (RAG) system designed for processing maritime documentation with advanced OCR, visual understanding, and intelligent retrieval capabilities.

## 🎯 Overview

This system extracts, processes, and enables intelligent querying of maritime PDF documents (specifically Maritime AKV PDFs) containing both textual and visual content. It leverages state-of-the-art multimodal AI models to understand text, images, diagrams, and scanned pages.

## ✨ Key Features

### 🔍 Advanced Text Extraction
- **Hybrid extraction pipeline**: Native text + OCR fallback
- **Dual OCR support**: EasyOCR (GPU-accelerated) and Tesseract
- **Intelligent preprocessing**: Adaptive thresholding, denoising, and contrast enhancement
- **Parallel processing**: Multi-threaded extraction for faster processing

### 🖼️ Visual Understanding
- **Automatic image extraction**: Contour-based detection with adaptive thresholds
- **Visual captioning**: BLIP model for semantic image descriptions
- **OCR for embedded text**: Extracts text from diagrams, charts, and tables
- **Smart image classification**: Distinguishes text-heavy vs. visual content

### 🧠 Intelligent Retrieval
- **Hybrid search**: Combines text embeddings (BGE) and visual embeddings (CLIP)
- **Cross-encoder reranking**: MS-MARCO model for relevance refinement
- **Query-adaptive retrieval**: Automatically adjusts text/image ratios based on query intent
- **Modality balancing**: Ensures diverse results (text + images) for comprehensive answers

### ⚡ Performance Optimizations
- **GPU acceleration**: Prioritizes GPU for CLIP/BLIP, CPU for OCR
- **Resource management**: 80% utilization caps with intelligent throttling
- **Batch processing**: Efficient GPU memory usage
- **Data persistence**: Pre-processed data saved to disk (skip reprocessing)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PDF Input (MAKV-2047.pdf)                │
└───────────────────────────┬─────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
    ┌───────▼────────┐            ┌────────▼─────────┐
    │ Text Extraction│            │ Image Extraction │
    │ • PyMuPDF      │            │ • pdf2image      │
    │ • EasyOCR/Tess │            │ • OpenCV         │
    └───────┬────────┘            └────────┬─────────┘
            │                               │
    ┌───────▼────────┐            ┌────────▼─────────┐
    │ Text Chunking  │            │ Visual Processing│
    │ • 500 tokens   │            │ • BLIP Captions  │
    │ • 50 overlap   │            │ • CLIP Embeddings│
    └───────┬────────┘            └────────┬─────────┘
            │                               │
            └───────────────┬───────────────┘
                            │
                ┌───────────▼────────────┐
                │   LlamaIndex Storage   │
                │ • Text: FAISS (BGE)    │
                │ • Images: FAISS (CLIP) │
                └───────────┬────────────┘
                            │
                ┌───────────▼────────────┐
                │   Hybrid Retrieval     │
                │ • Vector similarity    │
                │ • Cross-encoder rerank │
                │ • Modality balancing   │
                └───────────┬────────────┘
                            │
                ┌───────────▼────────────┐
                │   LLM Response (Groq)  │
                │ • Contextual answers   │
                │ • Page citations       │
                └────────────────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional but recommended)
- Poppler utilities (for PDF to image conversion)

### Step 1: Install Poppler
**Windows:**
```bash
# Download from: https://github.com/oschwartz10612/poppler-windows/releases
# Extract to: D:\poppler\poppler-25.12.0\Library\bin
```

**Linux:**
```bash
sudo apt-get install poppler-utils
```

**macOS:**
```bash
brew install poppler
```

### Step 2: Install Python Dependencies
```bash
pip install pymupdf pillow pytesseract pdf2image opencv-python numpy easyocr
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers sentence-transformers faiss-cpu llama-index
pip install llama-index-embeddings-huggingface llama-index-llms-groq
pip install llama-index-vector-stores-faiss psutil gputil python-dotenv
```

### Step 3: Configure Environment
Create a `.env` file:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

Get your Groq API key from: https://console.groq.com/

## 🚀 Usage

### Option 1: Full Pipeline (Recommended for First Run)

```python
# Run the main application
python app.py
```

**First Run Workflow:**
1. System checks for pre-processed data
2. If not found, automatically processes PDF:
   - Extracts text from all pages
   - Extracts and processes images
   - Generates captions and embeddings
   - Builds search indices
   - Saves everything to `./maritime_rag/`
3. Enters interactive query mode

**Subsequent Runs:**
- Loads pre-processed data instantly (no reprocessing needed)
- Directly enters query mode

### Option 2: Separate Extraction (Advanced)

```python
# Step 1: Extract text and images
python text_extraction.py

# This creates:
# - maritime_akv_extracted/text/         (page text files)
# - maritime_akv_extracted/images/       (extracted images)
# - maritime_akv_extracted/metadata/     (extraction metadata)

# Step 2: Run main RAG system
python app.py
# (Will use pre-extracted files if available)
```

### Interactive Query Examples

```
[Query #1] ➜ What is this document about?
✅ Answer (2.345s):
This is the Maritime AKV-2047 document, which covers vessel specifications,
safety procedures, and operational guidelines for maritime operations...
[Page 1, Page 3]

[Query #2] ➜ Show me images from the cover page
✅ Answer (1.892s):
The cover page contains the vessel identification diagram showing...
[Page 1, Image img_001_00]

[Query #3] ➜ What are the safety protocols mentioned?
✅ Answer (2.156s):
The document outlines the following safety protocols:
1. Emergency evacuation procedures [Page 15]
2. Fire suppression systems [Page 18, Image img_018_02]
...
```

## 🔧 Configuration

### File Paths
Update paths in `app.py`:
```python
# Poppler path (Windows)
POPPLER_PATH = r"D:\poppler\poppler-25.12.0\Library\bin"

# PDF input file
pdf_path = "MAKV-2047.pdf"

# Output directory
output_dir = "./maritime_rag"
```

### Resource Limits
Adjust in `ResourceManager`:
```python
ResourceManager(
    max_cpu_percent=80.0,  # Max CPU usage
    max_gpu_percent=80.0   # Max GPU usage
)
```

### Retrieval Parameters
Modify in `query()` method:
```python
top_k = 5          # Number of results to retrieve
rerank_top = 5     # Items to rerank with cross-encoder
```

### Chunking Strategy
Update in `Settings`:
```python
Settings.node_parser = SentenceSplitter(
    chunk_size=500,     # Tokens per chunk
    chunk_overlap=50    # Overlap between chunks
)
```

## 📊 Output Structure

```
maritime_rag/
├── images/                          # Extracted images
│   ├── img_001_00.png
│   ├── img_002_00.png
│   └── ...
├── processed_images/                # OCR-processed images
├── text/                           # Extracted text files
│   ├── page_000.txt
│   ├── page_001.txt
│   └── ...
├── metadata/                       # Extraction metadata
│   └── extraction_metadata.json
├── processed_data.pkl              # Cached processed data
├── clip_faiss.index               # CLIP image index
├── clip_embeddings.npy            # CLIP embeddings
├── docstore.json                  # LlamaIndex document store
└── default__storage__.json        # LlamaIndex storage
```

## 🧪 Technical Details

### Models Used
- **Text Embeddings**: BAAI/bge-small-en-v1.5 (384-dim)
- **Visual Embeddings**: CLIP ViT-B/32 (512-dim)
- **Image Captioning**: BLIP base
- **Reranking**: MS-MARCO MiniLM-L-12-v2
- **LLM**: Groq (Mixtral/Llama via API)

### OCR Pipeline
1. **Native extraction** (PyMuPDF)
2. **Scanned page detection** (word/line count heuristics)
3. **Preprocessing** (grayscale → adaptive threshold → denoising)
4. **OCR execution** (EasyOCR or Tesseract)
5. **Result validation** (choose longer output)

### Retrieval Strategy
1. **Text retrieval**: BGE embeddings → FAISS search
2. **Image retrieval**: CLIP text-image similarity
3. **Candidate pooling**: Combine top results
4. **Cross-encoder reranking**: Semantic relevance scoring
5. **Modality balancing**: Ensure text/image diversity
6. **Query adaptation**: Boost images for visual queries

## 🎯 Performance Benchmarks

**System Specs**: GTX 1650, 16GB RAM, i5 CPU

| Operation | Time | Notes |
|-----------|------|-------|
| PDF Processing (one-time) | ~45s | 25-page document |
| Text Extraction | ~8s | Parallel processing |
| Image Processing | ~25s | BLIP + CLIP batch |
| Index Building | ~5s | FAISS + persistence |
| Query (cold) | ~2-3s | Includes retrieval + LLM |
| Query (warm) | ~1-2s | Cached embeddings |

## 🐛 Troubleshooting

### Common Issues

**1. "No OCR library available"**
```bash
# Install EasyOCR (recommended)
pip install easyocr

# OR Tesseract
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
# Linux: sudo apt-get install tesseract-ocr
```

**2. "Poppler not found"**
```python
# Update POPPLER_PATH in app.py
POPPLER_PATH = r"C:\path\to\poppler\bin"
```

**3. "CUDA out of memory"**
```python
# Reduce batch size in process_images_batch()
batch_results = self.process_images_batch(image_paths, batch_size=2)
```

**4. "Unable to generate answer"**
```bash
# Check Groq API key
echo $GROQ_API_KEY

# Verify PDF exists
ls MAKV-2047.pdf
```

**5. Images not appearing in results**
- Check `maritime_rag/images/` for extracted images
- Verify CLIP index built: `clip_faiss.index` exists
- Enable debug: Look for "Boosted image score" messages

## 🔒 Privacy & Security

- **Local processing**: All OCR and embeddings run locally
- **API usage**: Only LLM generation uses external API (Groq)
- **No data sharing**: PDF content never leaves your system
- **Cached data**: Stored locally in `maritime_rag/`

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Support for more document types (DOCX, images)
- Multi-language OCR support
- Fine-tuned domain-specific models
- Web interface (Gradio/Streamlit)

## 📄 License

MIT License - See LICENSE file for details

##  Acknowledgments

- **LlamaIndex**: Orchestration framework
- **Hugging Face**: Pre-trained models
- **FAISS**: Efficient similarity search
- **Groq**: Fast LLM inference
- **OpenAI**: CLIP and BLIP models



---

**Built with ❤️ for maritime document intelligence**
