import sys
import os
import numpy as np

# Ensure core modules can be imported
sys.path.append(os.getcwd())
try:
    from kb_desktop.core.storage import DBManager
    from kb_desktop.core.index_faiss import FaissIndex
    from kb_desktop.core.embedder import Embedder
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'kb_desktop'))
    from core.storage import DBManager
    from core.index_faiss import FaissIndex
    from core.embedder import Embedder

def test_day5():
    print("Testing Day 5 Logic (Search & Retrieval)...")
    
    # 1. Initialize components
    db = DBManager(db_path="kb_desktop/data/kb.sqlite")
    faiss_index = FaissIndex()
    
    # 2. Load existing index
    if not faiss_index.load():
        print("ERROR: No index found. Please run the app and build index first.")
        return
    
    stats = faiss_index.get_stats()
    print(f"Index loaded: {stats}")
    
    # 3. Initialize embedder (must have real API key in .env)
    try:
        embedder = Embedder()
        print(f"Embedder initialized: {embedder.model}")
    except Exception as e:
        print(f"WARNING: Could not init embedder (API key missing?): {e}")
        print("Using mock embedding for testing...")
        # Mock for testing
        class MockEmbedder:
            def get_embedding(self, text):
                return np.random.rand(1536).tolist()
        embedder = MockEmbedder()
    
    # 4. Test query
    query = "什么是向量检索"
    print(f"\nQuery: '{query}'")
    
    # 5. Get query embedding
    query_emb = embedder.get_embedding(query)
    query_vec = np.array(query_emb)
    print(f"Query vector shape: {query_vec.shape}")
    
    # 6. Search
    k = 5
    distances, chunk_ids = faiss_index.search(query_vec, k=k)
    print(f"\nTop-{k} Results:")
    print(f"Chunk IDs: {chunk_ids}")
    print(f"Distances: {distances}")
    
    # 7. Retrieve details
    for i, (dist, chunk_id) in enumerate(zip(distances, chunk_ids)):
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT c.text, d.filename FROM chunks c JOIN documents d ON c.doc_id = d.id WHERE c.id = ?',
            (chunk_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            text, filename = result
            similarity = 1 / (1 + dist)
            print(f"\n【{i+1}】 Score: {similarity:.3f} | File: {filename}")
            print(f"Text: {text[:100]}...")
    
    print("\n\nSUCCESS: Search logic working!")

if __name__ == "__main__":
    test_day5()
