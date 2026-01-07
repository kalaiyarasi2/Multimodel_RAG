import os
import fitz  # PyMuPDF
import numpy as np
from pdf2image import convert_from_path
import cv2
from PIL import Image
import torch
from transformers import (
    AutoTokenizer, 
    AutoModel,
    CLIPProcessor, 
    CLIPModel,
    BlipProcessor, 
    BlipForConditionalGeneration
)
import faiss
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    Settings
)

from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine


class MultimodalRAGSystem:
    """
    Multimodal RAG system using LlamaIndex orchestration.
    Processes PDFs with text and images using FAISS, CLIP, BLIP, and Groq LLM.  
    """
    
    def __init__(self, groq_api_key: str, output_dir: str = "./rag_output"):
        """
        Initialize the RAG system with LlamaIndex orchestration.
        
        Args:
            groq_api_key: API key for Groq LLM
            output_dir: Directory to store extracted images and indices
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        
        print("Initializing models and LlamaIndex components...")
        
        # Configure LlamaIndex Settings
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )
        Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
        Settings.chunk_size = 500
        Settings.chunk_overlap = 50
        
        # Text embedding model (for direct access)
        self.text_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
        self.text_model = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5')
        self.text_model.eval()
        
        # CLIP for image embeddings
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()
        
        # BLIP for image captioning
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model.eval()
        
        # Storage
        self.image_metadata = []
        self.text_nodes = []
        self.image_nodes = []
        
        # LlamaIndex components
        self.text_index = None
        self.image_index = None
        self.query_engine = None
        
        # FAISS indices for CLIP
        self.clip_image_index = None
        
        print("✓ Models and LlamaIndex initialized successfully")
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text content from PDF pages."""
        print(f"Extracting text from {pdf_path}...")
        doc = fitz.open(pdf_path)
        pages_text = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            if text.strip():
                pages_text.append({
                    'page_num': page_num + 1,
                    'text': text,
                    'char_count': len(text)
                })
        
        doc.close()
        print(f"✓ Extracted text from {len(pages_text)} pages")
        return pages_text
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract images from PDF pages using pdf2image and OpenCV."""
        print(f"Extracting images from {pdf_path}...")
        images = convert_from_path(pdf_path, dpi=200, poppler_path=r"D:\poppler\poppler-25.12.0\Library\bin")
        extracted_images = []
        
        for page_num, page_image in enumerate(images):
            page_array = np.array(page_image)
            page_cv = cv2.cvtColor(page_array, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(page_cv, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            img_count = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area > 5000:
                    x, y, w, h = cv2.boundingRect(contour)
                    cropped = page_array[y:y+h, x:x+w]
                    
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
        
        print(f"✓ Extracted {len(extracted_images)} images")
        return extracted_images
    
    def generate_image_caption(self, image_path: str) -> str:
        """Generate caption for an image using BLIP."""
        image = Image.open(image_path).convert('RGB')
        inputs = self.blip_processor(image, return_tensors="pt")
        
        with torch.no_grad():
            out = self.blip_model.generate(**inputs, max_length=50)
        
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    def embed_image_clip(self, image_path: str) -> np.ndarray:
        """Generate CLIP embedding for an image."""
        image = Image.open(image_path).convert('RGB')
        inputs = self.clip_processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            embedding = image_features.numpy()
        
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        return embedding[0]
    
    def process_pdf(self, pdf_path: str):
        """
        Complete pipeline using LlamaIndex orchestration:
        Extract text/images, generate embeddings, build indices.
        """
        print("\n" + "="*50)
        print("STARTING PDF PROCESSING WITH LLAMAINDEX")
        print("="*50)
        
        # Step 1: Extract text
        pages_text = self.extract_text_from_pdf(pdf_path)
        
        # Step 2: Extract images
        images_data = self.extract_images_from_pdf(pdf_path)
        self.image_metadata = images_data
        
        # Step 3: Process images (caption + CLIP embed)
        print("\nGenerating image captions and embeddings...")
        clip_embeddings = []
        
        for img_data in images_data:
            caption = self.generate_image_caption(img_data['path'])
            img_data['caption'] = caption
            
            clip_embedding = self.embed_image_clip(img_data['path'])
            clip_embeddings.append(clip_embedding)
            
            print(f"  {img_data['image_id']} (Page {img_data['page_num']}): {caption}")
        
        # Build CLIP FAISS index for visual similarity search
        if clip_embeddings:
            clip_embeddings = np.array(clip_embeddings).astype('float32')
            self.clip_image_index = faiss.IndexFlatIP(512)
            self.clip_image_index.add(clip_embeddings)
            print(f"✓ Built CLIP image index with {len(clip_embeddings)} images")
        
        # Step 4: Create LlamaIndex Document nodes for text
        print("\nCreating LlamaIndex text nodes...")
        documents = []
        
        for page_data in pages_text:
            # Find images on same page
            page_images = [img for img in images_data if img['page_num'] == page_data['page_num']]
            image_refs = ", ".join([img['image_id'] for img in page_images]) if page_images else "None"
            
            doc = Document(
                text=page_data['text'],
                metadata={
                    'page_num': page_data['page_num'],
                    'images_on_page': image_refs,
                    'source': 'maritime_akv_text'
                }
            )
            documents.append(doc)
        
        # Step 5: Create LlamaIndex nodes for image captions
        print("Creating LlamaIndex image caption nodes...")
        for img_data in images_data:
            caption_text = f"[IMAGE {img_data['image_id']}]: {img_data['caption']}"
            
            doc = Document(
                text=caption_text,
                metadata={
                    'page_num': img_data['page_num'],
                    'image_id': img_data['image_id'],
                    'image_path': img_data['path'],
                    'source': 'maritime_akv_image'
                }
            )
            documents.append(doc)
        
        # Step 6: Build FAISS vector store and index with LlamaIndex
        print(f"\nBuilding LlamaIndex vector store with {len(documents)} documents...")
        
        # Create FAISS vector store
        faiss_index = faiss.IndexFlatIP(384)  # BGE-small-en dimension
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index
        self.text_index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        
        print("✓ Built LlamaIndex vector store and index")
        
        # Step 7: Create query engine
        retriever = VectorIndexRetriever(
            index=self.text_index,
            similarity_top_k=5
        )
        
        self.query_engine = RetrieverQueryEngine(retriever=retriever)
        
        print("\n" + "="*50)
        print("PDF PROCESSING COMPLETE")
        print("="*50)
    
    def retrieve_images_clip(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve relevant images using CLIP text-to-image matching.
        """
        if self.clip_image_index is None or len(self.image_metadata) == 0:
            return []
        
        inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True)
        
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            query_embedding = text_features.numpy()
        
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        distances, indices = self.clip_image_index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(self.image_metadata):
                result = self.image_metadata[idx].copy()
                result['score'] = float(score)
                results.append(result)
        
        return results
    
    def query(self, query: str, include_images: bool = True) -> str:
        """
        Complete query pipeline using LlamaIndex orchestration.
        
        Args:
            query: User query
            include_images: Whether to include image retrieval
            
        Returns:
            Generated response with citations
        """
        print(f"\nQuery: {query}")
        print("-" * 50)
        
        # Retrieve using LlamaIndex query engine
        response = self.query_engine.query(query)
        
        # Get source nodes for citation
        source_info = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                page_num = node.metadata.get('page_num', 'N/A')
                source_type = node.metadata.get('source', 'unknown')
                score = node.score if hasattr(node, 'score') else 0.0
                
                if source_type == 'maritime_akv_image':
                    img_id = node.metadata.get('image_id', 'N/A')
                    source_info.append(f"[Page {page_num}, Image {img_id}, Score: {score:.3f}]")
                else:
                    source_info.append(f"[Page {page_num}, Score: {score:.3f}]")
        
        # Additional visual retrieval if requested
        visual_context = ""
        if include_images:
            image_results = self.retrieve_images_clip(query, top_k=2)
            if image_results:
                visual_context = "\n\nRelevant Images Found:\n"
                for img in image_results:
                    visual_context += f"- {img['image_id']} (Page {img['page_num']}): {img['caption']} [Visual Score: {img['score']:.3f}]\n"
        
        # Format final response
        final_response = str(response)
        
        if source_info:
            final_response += f"\n\nSources: {', '.join(source_info)}"
        
        if visual_context:
            final_response += visual_context
        
        return final_response
    
    def custom_query_with_context(self, query: str, text_k: int = 5, image_k: int = 3) -> str:
        """
        Advanced query with custom retrieval and Groq generation.
        Combines LlamaIndex text retrieval with CLIP image retrieval.
        """
        print(f"\nCustom Query: {query}")
        print("-" * 50)
        
        # Retrieve text using LlamaIndex
        retriever = VectorIndexRetriever(
            index=self.text_index,
            similarity_top_k=text_k
        )
        text_nodes = retriever.retrieve(query)
        
        # Retrieve images using CLIP
        image_results = self.retrieve_images_clip(query, image_k)
        
        # Build context
        context_parts = []
        
        for node in text_nodes:
            page_num = node.metadata.get('page_num', 'N/A')
            text_content = node.text
            score = node.score if hasattr(node, 'score') else 0.0
            context_parts.append(f"[Page {page_num}, Score: {score:.3f}] {text_content}")
        
        for img in image_results:
            context_parts.append(
                f"[Page {img['page_num']}, Image {img['image_id']}, Visual Score: {img['score']:.3f}] "
                f"Image shows: {img['caption']}"
            )
        
        context_str = "\n\n".join(context_parts)
        
        if not context_str.strip():
            return "Information not available in the document."
        
        # Generate response using Groq via LlamaIndex
        prompt = f"""Based on the following retrieved context from the Maritime AKV document (including text and image descriptions), answer the user's query factually. Cite page numbers and image IDs when relevant.

CONTEXT:
{context_str}

QUERY: {query}

ANSWER (be concise and cite sources):"""
        
        response = Settings.llm.complete(prompt)
        
        return str(response)


# Example usage
if __name__ == "__main__":
    # Initialize system
    GROQ_API_KEY = XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    
    rag_system = MultimodalRAGSystem(
        groq_api_key=GROQ_API_KEY,
        output_dir="./maritime_rag"
    )
    
    # Process PDF
    pdf_path = "MAKV-2047-1-3.pdf"
    rag_system.process_pdf(pdf_path)
    
    # Example queries using LlamaIndex orchestration
    print("\n" + "="*50)
    print("QUERYING WITH LLAMAINDEX")
    print("="*50)
    
    queries = input("Enter your queries (separated by ';'): ").split(';')
    
    for query in queries:
        # Method 1: Standard LlamaIndex query
        response = rag_system.query(query)
        print(f"\n{response}\n")
        print("="*50)
        
        # Method 2: Custom query with hybrid retrieval
        # response = rag_system.custom_query_with_context(query)
        # print(f"\n{response}\n")
        # print("="*50)