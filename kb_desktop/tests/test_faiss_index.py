import sys
import os
import numpy as np

# Ensure core modules can be imported
sys.path.append(os.getcwd())
try:
    from kb_desktop.core.storage import DBManager
    from kb_desktop.core.index_faiss import FaissIndex
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'kb_desktop'))
    from core.storage import DBManager
    from core.index_faiss import FaissIndex

# Mock Embedder for testing (to avoid requiring API key)
class MockEmbedder:
    def __init__(self):
        self.dimension = 128  # Smaller dimension for testing
    
    def get_embeddings(self, texts):
        """Generate random embeddings for testing."""
        return [np.random.rand(self.dimension).tolist() for _ in texts]
    
    def get_dimension(self):
        return self.dimension

def test_day4():
    print("Testing Day 4 Logic (FAISS Indexing)...")
    
    # 1. Initialize
    db = DBManager(db_path="kb_desktop/data/kb.sqlite")
    faiss_index = FaissIndex(
        index_path="kb_desktop/data/test_faiss.index",
        meta_path="kb_desktop/data/test_meta.json"
    )
    embedder = MockEmbedder()
    
    # 2. Get chunks from DB
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, text FROM chunks ORDER BY id ASC LIMIT 10')  # Test with first 10
    chunks = cursor.fetchall()
    conn.close()
    
    if not chunks:
        print("WARNING: No chunks in DB. Run test_chunking.py first or import documents.")
        print("Creating mock data for testing...")
        # Create some test data
        chunks = [(i, f"Test chunk {i}") for i in range(5)]
    
    chunk_ids = [row[0] for row in chunks]
    chunk_texts = [row[1] for row in chunks]
    
    print(f"Processing {len(chunk_texts)} chunks...")
    
    # 3. Generate embeddings
    embeddings = embedder.get_embeddings(chunk_texts)
    vectors = np.array(embeddings)
    
    print(f"Generated vectors shape: {vectors.shape}")
    
    # 4. Build index
    dimension = embedder.get_dimension()
    faiss_index.build_index(vectors, chunk_ids, dimension)
    
    # 5. Save
    faiss_index.save()
    
    # 6. Load and verify
    faiss_index2 = FaissIndex(
        index_path="kb_desktop/data/test_faiss.index",
        meta_path="kb_desktop/data/test_meta.json"
    )
    loaded = faiss_index2.load()
    
    if not loaded:
        print("FAILURE: Could not load saved index.")
        sys.exit(1)
    
    stats = faiss_index2.get_stats()
    print(f"Loaded index stats: {stats}")
    
    # 7. Test search
    query_vector = np.random.rand(dimension)
    distances, result_ids = faiss_index2.search(query_vector, k=min(3, len(chunk_ids)))
    
    print(f"Search results: {len(result_ids)} chunks")
    print(f"Top chunk IDs: {result_ids}")
    print(f"Distances: {distances}")
    
    if len(result_ids) > 0:
        print("SUCCESS: Index built, saved, loaded, and searched successfully!")
    else:
        print("FAILURE: Search returned no results.")
        sys.exit(1)
    
    # Cleanup test files
    if os.path.exists("kb_desktop/data/test_faiss.index"):
        os.remove("kb_desktop/data/test_faiss.index")
    if os.path.exists("kb_desktop/data/test_meta.json"):
        os.remove("kb_desktop/data/test_meta.json")

if __name__ == "__main__":
    test_day4()
