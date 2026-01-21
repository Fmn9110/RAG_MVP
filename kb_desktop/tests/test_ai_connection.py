import sys
import os
import time

# Ensure core modules can be imported
sys.path.append(os.getcwd())
try:
    from kb_desktop.core.llm import LLMClient
    from kb_desktop.core.embedder import Embedder
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'kb_desktop'))
    from core.llm import LLMClient
    from core.embedder import Embedder

def test_integration():
    print("Testing OpenRouter Integration...")
    
    # 1. Test Embedder (OpenAI v1 compatibility)
    print("\n--- Testing Embedder ---")
    try:
        # Should pick up from .env automatically via load_dotenv in embedder module
        embedder = Embedder() 
        print(f"Embedder initialized with model: {embedder.model}")
        print(f"Base URL: {embedder.base_url}")
        
        # Test embedding
        text = "Test embedding generation."
        print(f"Embedding text: '{text}'")
        emb = embedder.get_embedding(text)
        print(f"Success! Embedding dimension: {len(emb)}")
        
    except Exception as e:
        print(f"Embedder failed: {e}")
        # Don't exit yet, try LLM

    # 2. Test LLM Client
    print("\n--- Testing LLM Client ---")
    try:
        llm = LLMClient()
        print(f"LLM Client initialized with model: {llm.model}")
        
        messages = [
            {"role": "user", "content": "Say 'Hello' in Chinese."}
        ]
        
        print("Sending chat request...")
        full_response = ""
        for chunk in llm.chat(messages, stream=True):
            full_response += chunk
            print(chunk, end="", flush=True)
            
        print("\n\nLLM Chat Success!")
        
    except Exception as e:
        print(f"\nLLM Client failed: {e}")

if __name__ == "__main__":
    test_integration()
