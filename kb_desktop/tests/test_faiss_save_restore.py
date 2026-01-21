
import os
import shutil
import sys
import numpy as np
import faiss

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kb_desktop.core.index_faiss import FaissIndex

def test_index_save_with_missing_dir():
    print("Testing FaissIndex save with missing directory and CHINESE PATH...")
    
    # 获取当前脚本所在目录的绝对路径（包含“中文知识库助手”）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构造一个包含中文的测试子目录
    test_dir = os.path.join(current_dir, "kb_desktop", "临时测试数据")
    
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        
    index_path = os.path.join(test_dir, "测试索引.index")
    meta_path = os.path.join(test_dir, "测试元数据.json")
    
    try:
        # Initialize FaissIndex
        # Note: We need to override the default paths which we hardcoded in __init__
        # But wait, the __init__ logic forces kb_desktop/data if None.
        # So we pass our test paths explicitly.
        
        fi = FaissIndex(index_path=index_path, meta_path=meta_path)
        
        # Create a dummy index
        d = 1536
        index = faiss.IndexFlatL2(d)
        fi.index = index
        fi.dimension = d
        
        # Add some dummy data
        vectors = np.random.random((5, d)).astype('float32')
        fi.index.add(vectors)
        fi.chunk_ids = [f"chunk_{i}" for i in range(5)]
        
        print(f"Attempting to save index to {index_path}...")
        # This should trigger the os.makedirs logic we added
        fi.save()
        
        if os.path.exists(index_path) and os.path.exists(meta_path):
            print("✅ Success: Index and meta files created successfully in new directory!")
        else:
            print("❌ Failure: Files were not created.")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
    finally:
        # Cleanup
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print("Cleanup: Removed test directory.")

if __name__ == "__main__":
    test_index_save_with_missing_dir()
