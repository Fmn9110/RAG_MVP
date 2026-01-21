import sys
import os

sys.path.append(os.getcwd())
try:
    from kb_desktop.core.rag import RAGGenerator
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'kb_desktop'))
    from core.rag import RAGGenerator

def test_day6():
    print("Testing Day 6 Logic (RAG Generation)...")
    
    # Mock context chunks
    context_chunks = [
        {
            'text': '本系统采用了向量数据库技术，将所有文本片段转换为高维向量。',
            'filename': 'test_doc.txt',
            'chunk_id': 1,
            'similarity': 0.85
        },
        {
            'text': '系统使用FAISS作为向量索引引擎，支持快速相似度搜索。',
            'filename': 'test_doc.txt',
            'chunk_id': 2,
            'similarity': 0.78
        }
    ]
    
    query = "什么是向量检索？"
    
    print(f"\nQuery: {query}")
    print(f"Context chunks: {len(context_chunks)}")
    
    # Set dummy API key for testing
    os.environ["OPENAI_API_KEY"] = "sk-test-key"
    
    try:
        rag = RAGGenerator()
        print("RAG Generator initialized")
        
        # Mock LLM chat method to avoid real API calls
        def mock_chat(*args, **kwargs):
            yield "这是一个测试回答，基于向量检索技术。[1]"
            
        rag.llm.chat = mock_chat
        
        # Generate answer
        answer, citations = rag.generate_answer(query, context_chunks)
        
        print("\n" + "="*60)
        print("ANSWER:")
        print(answer)
        
        print("\n" + "="*60)
        print("CITATIONS:")
        for i, cite in enumerate(citations):
            print(f"[{i+1}] {cite['filename']}")
            print(f"    Excerpt: {cite['excerpt']}")
        
        print("\n" + "="*60)
        print("SUCCESS: RAG generation working!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_day6()
