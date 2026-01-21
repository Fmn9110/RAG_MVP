import sqlite3
import os
import hashlib
from datetime import datetime
from typing import List, Tuple

class DBManager:
    def __init__(self, db_path=None):
        # 如果没有指定路径，使用 kb_desktop/data/kb.sqlite
        if db_path is None:
            # 获取 storage.py 的目录（kb_desktop/core）
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 往上一级到 kb_desktop 目录
            kb_desktop_dir = os.path.dirname(current_dir)
            db_path = os.path.join(kb_desktop_dir, "data", "kb.sqlite")
        
        # 确保 data 目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_db(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 文档表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_hash TEXT UNIQUE,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                content TEXT,
                chunk_count INTEGER DEFAULT 0,
                last_indexed TIMESTAMP
            )
        ''')
        
        # 文本块表（为第3天准备，但现在创建也可以）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER,
                chunk_index INTEGER,
                text TEXT,
                meta_info TEXT, -- 额外信息的JSON字符串
                FOREIGN KEY(doc_id) REFERENCES documents(id)
            )
        ''')
        
        # 迁移：向已有数据库添加新列
        try:
            cursor.execute("PRAGMA table_info(documents)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'chunk_count' not in columns:
                cursor.execute('ALTER TABLE documents ADD COLUMN chunk_count INTEGER DEFAULT 0')
                print("✓ Migration: Added chunk_count column")
            
            if 'last_indexed' not in columns:
                cursor.execute('ALTER TABLE documents ADD COLUMN last_indexed TIMESTAMP')
                print("✓ Migration: Added last_indexed column")
        except sqlite3.OperationalError as e:
            print(f"Migration warning: {e}")
        
        conn.commit()
        conn.close()

    def add_document(self, filename, file_path, content):
        """
        如果成功返回文档ID，如果重复（按哈希值）则返回None。
        """
        # 计算哈希值以防止重复（使用SHA256以确保安全）
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO documents (filename, file_path, file_hash, content)
                VALUES (?, ?, ?, ?)
            ''', (filename, file_path, content_hash, content))
            doc_id = cursor.lastrowid
            conn.commit()
            return doc_id
        except sqlite3.IntegrityError:
            # 文件重复
            conn.rollback()
            return None
        finally:
            conn.close()

    def add_chunks(self, doc_id, chunks):
        """
        为文档批量插入文本块。
        chunks: 字符串列表
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        data = []
        for i, text in enumerate(chunks):
            # (doc_id, chunk_index, text, meta_info)
            data.append((doc_id, i, text, "{}"))
            
        cursor.executemany('''
            INSERT INTO chunks (doc_id, chunk_index, text, meta_info)
            VALUES (?, ?, ?, ?)
        ''', data)
        
        # 更新文档表中的文本块数量
        cursor.execute('''
            UPDATE documents SET chunk_count = ? WHERE id = ?
        ''', (len(chunks), doc_id))
        
        conn.commit()
        conn.close()

    def get_document_chunks(self, doc_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT chunk_index, text FROM chunks WHERE doc_id = ? ORDER BY chunk_index ASC', (doc_id,))
        rows = cursor.fetchall()
        conn.close()
        return rows

    def get_all_documents(self):
        """返回 (id, filename, upload_time, chunk_count, last_indexed)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, filename, upload_time, chunk_count, last_indexed FROM documents ORDER BY id DESC')
        rows = cursor.fetchall()
        conn.close()
        return rows

    def get_document_content(self, doc_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT content FROM documents WHERE id = ?', (doc_id,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    
    def mark_as_indexed(self, doc_id):
        """将文档标记为已索引，使用当前时间戳。"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE documents SET last_indexed = CURRENT_TIMESTAMP WHERE id = ?
        ''', (doc_id,))
        conn.commit()
        conn.close()
    
    def keyword_search(self, query: str, k: int = 10) -> List[Tuple[int, str, str, float]]:
        """
        对文本块执行简单的基于关键词的搜索。
        返回: (chunk_id, text, filename, score) 的列表
        
        为简单起见使用 SQL LIKE。生产环境请考虑使用 FTS5。
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 将查询分解为关键词（简单方法）
        keywords = query.split()
        
        # 构建查询条件
        conditions = []
        params = []
        for keyword in keywords:
            conditions.append("c.text LIKE ?")
            params.append(f"%{keyword}%")
        
        if not conditions:
            conn.close()
            return []
        
        where_clause = " OR ".join(conditions)
        
        cursor.execute(f'''
            SELECT c.id, c.text, d.filename
            FROM chunks c
            JOIN documents d ON c.doc_id = d.id
            WHERE {where_clause}
            LIMIT ?
        ''', params + [k * 2])  # 获取更多候选结果用于评分
        
        results = cursor.fetchall()
        conn.close()
        
        # 根据关键词匹配数量评分结果
        scored_results = []
        for chunk_id, text, filename in results:
            score = 0
            text_lower = text.lower()
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    score += 1
            scored_results.append((chunk_id, text, filename, score))
        
        # 按分数降序排序
        scored_results.sort(key=lambda x: x[3], reverse=True)
        
        return scored_results[:k]

