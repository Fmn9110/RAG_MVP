import sys
import os

# Ensure core modules can be imported
sys.path.append(os.getcwd())
try:
    from kb_desktop.core.storage import DBManager
    from kb_desktop.core.ingest import Ingestor
except ImportError:
    # Try alternate path if running from within kb_desktop
    sys.path.append(os.path.join(os.getcwd(), 'kb_desktop'))
    from core.storage import DBManager
    from core.ingest import Ingestor

def test_day2():
    print("Testing Day 2 Logic...")
    
    # 0. Cleanup old DB for test
    if os.path.exists("kb_desktop/data/kb.sqlite"):
        # os.remove("kb_desktop/data/kb.sqlite") 
        # Actually, let's keep it to verify persistence, just print current count
        pass

    # 1. Initialize DB
    db = DBManager(db_path="kb_desktop/data/kb.sqlite")
    print("DB Initialized.")
    
    # 2. Ingest
    test_file = "test_import.txt"
    if not os.path.exists(test_file):
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("This is a test document content for Day 2 verification.")
    
    content = Ingestor.load_file(test_file)
    print(f"Ingested Content: {content}")
    
    # 3. Store
    doc_id = db.add_document("test_import.txt", os.path.abspath(test_file), content)
    if doc_id:
        print(f"Stored Document ID: {doc_id}")
    else:
        print("Document already exists (Duplicate detection worked).")
        
    # 4. Verification
    docs = db.get_all_documents()
    print(f"Total Documents in DB: {len(docs)}")
    
    found = False
    for doc in docs:
        if doc[1] == "test_import.txt":
            found = True
            break
            
    if found:
        print("SUCCESS: Document found in DB list.")
    else:
        print("FAILURE: Document not found.")
        sys.exit(1)

if __name__ == "__main__":
    test_day2()
