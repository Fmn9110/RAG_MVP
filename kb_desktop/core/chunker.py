import re

class Chunker:
    @staticmethod
    def split_text(text, max_len=300, overlap=50):
        """
        Splits text into chunks optimized for Chinese.
        Strategy:
        1. Split by coarse separators (\n\n, \n)
        2. If chunk > max_len, split by sentence endings (。！?...)
        3. Merge small chunks to reach max_len (with overlap)
        """
        if not text:
            return []
            
        # 1. First cleanup
        text = text.replace("\r\n", "\n")
        
        # 2. 首先按段落分割（强语义断开）
        # 如果需要，使用先行断言保留分隔符，但简单分割对MVP更简单
        paragraphs = text.split("\n")
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # 如果添加这个段落超过 max_len，我们需要处理 current_chunk
            if len(current_chunk) + len(para) + 1 <= max_len:
                current_chunk += ("\n" + para) if current_chunk else para
            else:
                # current_chunk 已经足够满，保存它
                if current_chunk:
                    chunks.append(current_chunk)
                    
                # 现在处理新的段落。
                # 如果段落本身很大 (> max_len)，我们需要按句子分割它
                if len(para) > max_len:
                    sub_chunks = Chunker._split_long_sentence(para, max_len, overlap)
                    # 如果我们有 current_chunk，第一个 sub_chunk 可能需要重叠
                    # 对于 MVP，只需附加所有 sub_chunks
                    chunks.extend(sub_chunks)
                    current_chunk = "" # 重置
                else:
                    # 用这个段落开始新的文本块，可能从前一个重用重叠
                    # （段落的简单滑动窗口很棘手，让我们坚持简单积累）
                    # 要正确地做重叠：从前一个文本块拿最后 N 个字符？
                    # MVP 简化：段落之间没有重叠，仅在长段落内部
                    current_chunk = para
        
        # 添加最后一部分
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    @staticmethod
    def _split_long_sentence(text, max_len, overlap):
        """
        按句子分隔符分割长字符串。
        """
        # 用于按中文句子结束符分割的正则表达式
        # split 返回: [part1, sep1, part2, sep2, ...]
        pattern = r'([。！？.!?])' 
        parts = re.split(pattern, text)
        
        sentences = []
        # 重新组装带标点的句子
        for i in range(0, len(parts), 2):
            sent = parts[i]
            if i + 1 < len(parts):
                sent += parts[i+1] # 把标点加回来
            if sent.strip():
                sentences.append(sent)
        
        chunks = []
        current_chunk = ""
        
        for sent in sentences:
            if len(current_chunk) + len(sent) <= max_len:
                current_chunk += sent
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # 重叠逻辑：用前一个文本块的结尾开始新文本块
                # 对于 MVP 的简化：只是重新开始还是带一点上下文？
                # 让我们从 current_chunk 的结尾拿 'overlap' 个字符（如果可能）
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
                current_chunk = overlap_text + sent
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
