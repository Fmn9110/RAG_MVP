import sys
import os

# Ensure core modules can be imported
sys.path.append(os.getcwd())
try:
    from kb_desktop.core.storage import DBManager
    from kb_desktop.core.ingest import Ingestor
    from kb_desktop.core.chunker import Chunker
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'kb_desktop'))
    from core.storage import DBManager
    from core.ingest import Ingestor
    from core.chunker import Chunker

def test_day3():
    print("Testing Day 3 Logic (Chunking)...")
    
    # 1. Initialize DB
    db_path = "kb_desktop/data/kb.sqlite"
    # Clean previous run chunks just in case? No, persistence is key. 
    # But for test repeatability, maybe using a test file name is better.
    db = DBManager(db_path=db_path)
    
    test_file_name = f"test_day3_{os.getpid()}.txt"
    with open(test_file_name, "w", encoding="utf-8") as f:
        # Generate some long text
        text = "这是一个段落。这是第二句。" * 20 + "\n\n" + "第二段开始。" + "中文切分测试。" * 10
        f.write(text)
        
    try:
        # 2. Ingest
        content = Ingestor.load_file(test_file_name)
        
        # 3. Chunk
        chunks = Chunker.split_text(content, max_len=100) # Small max_len for testing
        print(f"Original Length: {len(content)}")
        print(f"Generated Chunks: {len(chunks)}")
        
        if len(chunks) < 2:
            print("FAILURE: Text should have been split into multiple chunks.")
            sys.exit(1)
            
        print(f"First Chunk: {chunks[0][:20]}...")
        
        # 4. Store
        doc_id = db.add_document(test_file_name, os.path.abspath(test_file_name), content)
        if doc_id:
            db.add_chunks(doc_id, chunks)
            print(f"Stored Doc ID: {doc_id} with {len(chunks)} chunks.")
        
            # 5. Retrieve
            stored_chunks = db.get_document_chunks(doc_id)
            print(f"Retrieved Chunks Count: {len(stored_chunks)}")
            
            if len(stored_chunks) == len(chunks):
                print("SUCCESS: Chunk retrieval matches generation.")
            else:
                print("FAILURE: Chunk count mismatch.")
                sys.exit(1)
        else:
            print("Document already existed, skipped storage test for this run.")
            
    finally:
        if os.path.exists(test_file_name):
            os.remove(test_file_name)

if __name__ == "__main__":
    test_day3()
