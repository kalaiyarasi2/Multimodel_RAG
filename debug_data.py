import pickle
import os
from llama_index.core import Document

def check_page_content(page_num):
    pkl_path = "maritime_rag/processed_data.pkl"
    if not os.path.exists(pkl_path):
        print("❌ PROCESSED DATA NOT FOUND")
        return

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    documents = data['documents']
    print(f"📚 Total Documents: {len(documents)}")
    
    found = False
    print(f"\n🔍 Searching for Page {page_num} content...")
    
    for doc in documents:
        # Access metadata from the LlamaIndex Document object
        # Note: Depending on version, it might be doc.metadata or doc.extra_info
        meta = getattr(doc, 'metadata', {})
        if str(meta.get('page_num')) == str(page_num):
            found = True
            print("-" * 40)
            print(f"Type: {meta.get('type', 'text')}")
            print(f"Content: {doc.text[:500]}...") # Print first 500 chars
            print("-" * 40)

    if not found:
        print(f"❌ No content found for Page {page_num}")

if __name__ == "__main__":
    check_page_content(12)
