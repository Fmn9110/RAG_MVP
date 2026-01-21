from typing import List, Dict, Tuple, Optional
from core.llm import LLMClient
import re

# 置信度阈值
LOW_CONFIDENCE_THRESHOLD = 0.3  # 如果 top-1 相似度 < 这个值，触发备用回复
MIN_TOPK_VARIANCE = 0.05  # 如果所有 TopK 分数太相似，可能是噪声

class RAGGenerator:
    """
    RAG 生成器，组装上下文并生成带有强制引用的回答。
    """
    
    def __init__(self):
        self.llm = LLMClient()
    
    def check_confidence(self, context_chunks: List[Dict]) -> Tuple[bool, str]:
        """
        检查检索结果是否具有足够的置信度。
        
        Args:
            context_chunks: 带有 'similarity' 分数的检索文本块
            
        Returns:
            (is_confident: bool, reason: str)
        """
        if not context_chunks:
            return False, "未找到相关文档"
        
        # 检查 top-1 相似度
        top1_score = context_chunks[0].get('similarity', 0)
        if top1_score < LOW_CONFIDENCE_THRESHOLD:
            return False, f"最佳匹配相似度过低 ({top1_score:.3f} < {LOW_CONFIDENCE_THRESHOLD})"
        
        # 检查 TopK 之间的方差（它们是否都差不多？）
        if len(context_chunks) >= 3:
            scores = [chunk.get('similarity', 0) for chunk in context_chunks]
            variance = max(scores) - min(scores)
            if variance < MIN_TOPK_VARIANCE:
                return False, f"所有结果分数过低且相近 (方差: {variance:.3f})"
        
        return True, "置信度足够"

    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        生成带有强制引用的回答。
        
        Args:
            query: 用户的问题
            context_chunks: 字典列表，包含键: 'text', 'filename', 'chunk_id', 'similarity'
            
        Returns:
            (answer_text, citations) 的元组
            citations 是字典列表: {'filename': str, 'chunk_id': int, 'excerpt': str}
        """
        # 1. 用上下文构建提示
        prompt = self._build_prompt(query, context_chunks)
        
        # 2. 准备消息
        messages = [
            {"role": "system", "content": "你是一个知识库助手。请基于提供的上下文回答问题，并在回答末尾列出引用来源。"},
            {"role": "user", "content": prompt}
        ]
        
        # 3. 调用 LLM（流式传输）
        full_response = ""
        for chunk in self.llm.chat(messages, stream=True):
            full_response += chunk
        
        # 4. 解析引用（简单方法：从响应中提取）
        # 对于 MVP，如果 LLM 没有提供，我们将手动附加引用
        citations = self._extract_or_force_citations(full_response, context_chunks)
        
        return full_response, citations
    
    def _build_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        """用上下文构建 RAG 提示。"""
        context_text = ""
        
        for i, chunk in enumerate(context_chunks):
            context_text += f"\n【文档 {i+1}】来源: {chunk['filename']}\n"
            context_text += f"{chunk['text']}\n"
            context_text += "-" * 60 + "\n"
        
        prompt = f"""基于以下文档片段回答问题。请务必在回答末尾列出引用的文档编号。

已知文档:
{context_text}

问题: {query}

要求:
1. 仅基于上述文档内容回答
2. 如果文档中没有相关信息，请明确说明
3. 在回答末尾列出引用来源，格式: 【引用】文档1, 文档2...
"""
        return prompt
    
    def _extract_or_force_citations(self, response: str, context_chunks: List[Dict]) -> List[Dict]:
        """
        从响应中提取引用，或如果缺少则强制添加。
        现在包括引用验证以防止幻觉。
        """
        citations = []
        
        # 尝试从响应中提取引用的文档编号
        # 查找类似“文档1”、“文档 2”、“[1]”等模式
        cited_pattern = re.findall(r'(?:文档|[\[\(])(\d+)(?:[\]\)])?', response)
        cited_indices = set()
        
        for match in cited_pattern:
            try:
                idx = int(match) - 1  # Convert to 0-based
                if 0 <= idx < len(context_chunks):
                    cited_indices.add(idx)
            except ValueError:
                continue
        
        # 如果 LLM 引用了特定文本块，验证并使用它们
        if cited_indices:
            for idx in sorted(cited_indices):
                chunk = context_chunks[idx]
                citations.append({
                    'filename': chunk['filename'],
                    'chunk_id': chunk.get('chunk_id', idx),
                    'excerpt': chunk['text'][:100] + "..."
                })
        else:
            # 回退：将所有提供的文本块作为潜在引用包含进来
            # （LLM 应该引用它们，但没有 - 标记这个）
            for i, chunk in enumerate(context_chunks):
                citations.append({
                    'filename': chunk['filename'],
                    'chunk_id': chunk.get('chunk_id', i),
                    'excerpt': chunk['text'][:100] + "...",
                    'verified': False  # 标记为未验证
                })
        
        return citations
    
    def verify_citations(self, response: str, context_chunks: List[Dict]) -> Tuple[bool, str]:
        """
        验证响应中的所有引用是否对应于检索的文本块。
        
        Returns:
            (is_valid: bool, issue: str)
        """
        # 从响应中提取引用的索引
        cited_pattern = re.findall(r'(?:文档|[\[\(])(\d+)(?:[\]\)])?', response)
        
        for match in cited_pattern:
            try:
                idx = int(match) - 1
                if idx < 0 or idx >= len(context_chunks):
                    return False, f"引用文档{match}超出范围 (有效范围: 1-{len(context_chunks)})"
            except ValueError:
                continue
        
        return True, "引用已验证"


    def generate_fallback_response(self, query: str, context_chunks: List[Dict], reason: str) -> Tuple[str, List[Dict]]:
        """
        当置信度太低时生成有用的备用回复。
        
        Args:
            query: 用户的问题
            context_chunks: 检索的文本块（即使置信度低）
            reason: 为什么置信度低
            
        Returns:
            (fallback_message, empty_citations) 的元组
        """
        # 从查询中提取关键词用于建议
        keywords = self._extract_keywords(query)
        
        # 构建备用消息
        fallback_msg = f"""⚠️ **知识库缺乏足够依据**

抱歉，当前知识库中没有找到足够相关的信息来回答您的问题。

**原因**: {reason}

**建议的追问方向**:
"""
        
        # 根据关键词生成后续问题
        if keywords:
            fallback_msg += f"\n1. 关于 '{keywords[0]}' 的具体定义或背景是什么？"
            if len(keywords) > 1:
                fallback_msg += f"\n2. '{keywords[1]}' 在哪些场景下适用？"
            fallback_msg += f"\n3. 能否提供更具体的场景或案例？"
        else:
            fallback_msg += "\n1. 能否提供更具体的关键词或背景信息？"
            fallback_msg += "\n2. 您想了解的是哪方面的内容？"
            fallback_msg += "\n3. 是否可以换个方式描述您的问题？"
        
        # 显示可用文档作为推荐
        if context_chunks:
            fallback_msg += "\n\n**可能相关的文档** (相似度较低，仅供参考):\n"
            for i, chunk in enumerate(context_chunks[:3]):  # Top 3
                fallback_msg += f"\n- {chunk['filename']} (相似度: {chunk.get('similarity', 0):.3f})"
        
        fallback_msg += "\n\n💡 **提示**: 您可以尝试导入更多相关文档，或使用不同的关键词重新提问。"
        
        return fallback_msg, []
    
    def _extract_keywords(self, query: str) -> List[str]:
        """从查询中提取简单关键词（移除常见词）。"""
        # 简单方法：分割并过滤
        common_words = {'的', '了', '是', '在', '有', '和', '就', '不', '人', '都', '我', '要', 
                       '他', '这', '着', '你', '会', '地', '个', '她', '到', '说', '们', '为',
                       '什么', '怎么', '如何', '能否', '可以', '吗', '呢', '？', '?'}
        
        words = re.findall(r'[\u4e00-\u9fa5]+', query)  # 提取中文词
        keywords = [w for w in words if w not in common_words and len(w) > 1]
        
        return keywords[:3]  # 返回前3个关键词

