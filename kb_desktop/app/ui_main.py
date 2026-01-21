import sys
import os
import re
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QTextEdit, QListWidget, QTabWidget, 
    QFileDialog, QSplitter, QFrame, QStatusBar, QProgressBar, QMessageBox, QMenu
)
from PySide6.QtCore import Qt

# 导入核心模块
from core.storage import DBManager
from core.ingest import Ingestor
from core.chunker import Chunker
from core.embedder import Embedder
from core.index_faiss import FaissIndex
from core.rag import RAGGenerator
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("中文知识库助手 (RAG MVP)")
        self.resize(1200, 800)
        
        # 初始化核心组件
        self.db = DBManager()
        self.faiss_index = FaissIndex()
        self.embedder = None  # 需要时初始化（需要API密钥）
        
        # 尝试加载已有索引
        index_loaded = self.faiss_index.load()
        
        # 中央组件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局（横向）
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 可调整大小的分隔器
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        main_layout.addWidget(splitter)
        
        # --- 左侧面板：知识库管理 ---
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(15, 15, 15, 15)
        left_layout.setSpacing(10)
        
        lb_kb_title = QLabel("知识库管理")
        lb_kb_title.setObjectName("header")
        left_layout.addWidget(lb_kb_title)
        
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_file_selected) # 连接选择事件
        # 启用右键上下文菜单
        self.file_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self.on_file_list_context_menu)
        left_layout.addWidget(self.file_list)
        
        self.btn_import = QPushButton("导入文档 (txt/md/docx)")
        self.btn_import.clicked.connect(self.on_import_clicked)
        left_layout.addWidget(self.btn_import)
        
        self.reindex_btn = QPushButton("重建索引")
        self.reindex_btn.clicked.connect(self.on_build_index)  # 启用第4天功能
        left_layout.addWidget(self.reindex_btn)

        self.lb_doc_count = QLabel("已索引文档: 0")
        self.lb_doc_count.setStyleSheet("color: #909399; font-size: 12px;")
        left_layout.addWidget(self.lb_doc_count)
        
        splitter.addWidget(left_panel)
        
        # --- 中间面板：问题输入 ---
        middle_panel = QFrame()
        middle_layout = QVBoxLayout(middle_panel)
        middle_layout.setContentsMargins(15, 15, 15, 15)
        middle_layout.setSpacing(10)
        
        lb_input_title = QLabel("提问区")
        lb_input_title.setObjectName("header")
        middle_layout.addWidget(lb_input_title)
        
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("请输入您的问题...")
        middle_layout.addWidget(self.input_text)
        
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        self.btn_ask = QPushButton("提问")
        self.btn_ask.setObjectName("btn_primary")
        self.btn_ask.setMinimumHeight(40)
        self.btn_ask.clicked.connect(self.on_ask_question)  # 第5天：连接搜索
        
        self.btn_clear = QPushButton("清空")
        self.btn_clear.clicked.connect(self.input_text.clear)
        self.btn_clear.setMinimumHeight(40)
        
        btn_layout.addWidget(self.btn_ask)
        btn_layout.addWidget(self.btn_clear)
        middle_layout.addLayout(btn_layout)
        
        splitter.addWidget(middle_panel)
        
        # --- 右侧面板：输出区域 ---
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(15, 15, 15, 15)
        right_layout.setSpacing(10)
        
        lb_output_title = QLabel("输出结果")
        lb_output_title.setObjectName("header")
        right_layout.addWidget(lb_output_title)
        
        self.tabs = QTabWidget()
        
        # 选项卡1: 回答
        self.tab_answer = QWidget()
        tab1_layout = QVBoxLayout(self.tab_answer)
        tab1_layout.setContentsMargins(10, 10, 10, 10)
        self.text_answer = QTextEdit()
        self.text_answer.setReadOnly(True)
        tab1_layout.addWidget(self.text_answer)
        self.tabs.addTab(self.tab_answer, "回答")
        
        # 选项卡2: 源文本块
        self.tab_chunks = QWidget()
        tab2_layout = QVBoxLayout(self.tab_chunks)
        tab2_layout.setContentsMargins(10, 10, 10, 10)
        self.list_chunks = QListWidget()
        tab2_layout.addWidget(self.list_chunks)
        self.tabs.addTab(self.tab_chunks, "命中片段 (Top-K)")
        

        
        right_layout.addWidget(self.tabs)
        
        splitter.addWidget(right_panel)
        
        # 设置初始分隔器大小
        splitter.setSizes([250, 400, 550])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 1)

        # 状态栏
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("就绪")
        
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        self.progress.setFixedWidth(150)
        self.progress.setStyleSheet("QProgressBar { border: 1px solid #dcdfe6; border-radius: 4px; text-align: center; } QProgressBar::chunk { background-color: #409eff; }")
        self.status.addPermanentWidget(self.progress)
        
        # 初始数据加载
        self.refresh_doc_list()

    def refresh_doc_list(self):
        """从数据库加载文档并显示在列表中。"""
        self.file_list.clear()
        docs = self.db.get_all_documents()
        for doc in docs:
            # doc: (id, filename, upload_time, chunk_count, last_indexed)
            doc_id, filename, upload_time, chunk_count, last_indexed = doc
            
            # 格式化显示文本 - 不显示ID，只显示文件名
            indexed_status = "✓" if last_indexed else "⏳"
            chunk_info = f" ({chunk_count or 0} 个文本块)" if chunk_count else ""
            item = f"{indexed_status} {filename}{chunk_info}"
            self.file_list.addItem(item)
            # 将doc_id存储为item的用户数据，以便后续使用
            list_item = self.file_list.item(self.file_list.count() - 1)
            list_item.setData(Qt.UserRole, doc_id)
        
        self.lb_doc_count.setText(f"已索引文档: {len(docs)}")

    def on_file_selected(self, item):
        """点击文件时显示文本块。"""
        try:
            # 从 item 的用户数据中获取 doc_id
            doc_id = item.data(Qt.UserRole)
            if doc_id is None:
                raise ValueError("无法获取文档ID")
            
            chunks = self.db.get_document_chunks(doc_id)
            
            self.list_chunks.clear()
            for chunk_idx, text in chunks:
                self.list_chunks.addItem(f"--- 片段 {chunk_idx} ---\n{text}\n")
                
            # 切换到文本块选项卡
            self.tabs.setCurrentIndex(1)
            self.status.showMessage(f"已加载文档 {doc_id} 的 {len(chunks)} 个文本块")
            
        except Exception as e:
            print(f"加载文本块错误: {e}")

    def on_import_clicked(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择文档", "", "Documents (*.txt *.md *.docx)"
        )
        if not file_paths:
            return
            
        success_count = 0
        duplicate_count = 0
        fail_count = 0
        
        self.status.showMessage("正在导入文档...")
        self.progress.setVisible(True)
        self.progress.setValue(0)
        
        total = len(file_paths)
        
        for i, path in enumerate(file_paths):
            filename = os.path.basename(path)
            try:
                # 1. 提取
                content = Ingestor.load_file(path)
                
                # 2. 存储到数据库
                doc_id = self.db.add_document(filename, path, content)
                
                if doc_id:
                    # 3. 分块（第3天功能）
                    chunks = Chunker.split_text(content)
                    self.db.add_chunks(doc_id, chunks)
                    success_count += 1
                else:
                    duplicate_count += 1
            
            except Exception as e:
                print(f"导入 {filename} 时出错: {e}")
                fail_count += 1
            
            self.progress.setValue(int((i + 1) / total * 100))
            QApplication.processEvents() # 保持UI响应
        
        self.progress.setVisible(False)
        self.refresh_doc_list()
        
        msg = f"已导入: {success_count}\n重复: {duplicate_count}\n失败: {fail_count}"
        QMessageBox.information(self, "导入结果", msg)
        self.status.showMessage("就绪")

    def on_build_index(self):
        """从数据库中的所有文本块构建 FAISS 索引。"""
        self.status.showMessage("正在构建索引...")
        self.progress.setVisible(True)
        self.progress.setValue(0)
        
        try:
            # 1. 初始化嵌入器（如果还没有）
            if self.embedder is None:
                try:
                    self.embedder = Embedder()
                except ValueError as e:
                    QMessageBox.critical(
                        self, 
                        "需要 API 密钥", 
                        "请设置 OPENAI_API_KEY 环境变量。\n\n" +
                        "例如:\n" +
                        "set OPENAI_API_KEY=sk-xxxx (Windows)\n" +
                        "export OPENAI_API_KEY=sk-xxxx (Linux/Mac)"
                    )
                    self.progress.setVisible(False)
                    self.status.showMessage("就绪")
                    return
            
            # 2. 从数据库获取所有文本块
            conn = self.db.get_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT id, text FROM chunks ORDER BY id ASC')
            all_chunks = cursor.fetchall()
            conn.close()
            
            if not all_chunks:
                QMessageBox.warning(self, "无数据", "未找到文本块。请先导入文档。")
                self.progress.setVisible(False)
                self.status.showMessage("就绪")
                return
            
            chunk_ids = [row[0] for row in all_chunks]
            chunk_texts = [row[1] for row in all_chunks]
            
            total = len(chunk_texts)
            self.status.showMessage(f"正在嵌入 {total} 个文本块...")
            
            # 3. 获取嵌入（批量处理以提高效率）
            batch_size = 100  # 批量处理以提高效率
            all_embeddings = []
            
            for i in range(0, total, batch_size):
                batch = chunk_texts[i:i+batch_size]
                embeddings = self.embedder.get_embeddings(batch)
                all_embeddings.extend(embeddings)
                
                progress = int((i + len(batch)) / total * 90)  # 保留最后10%用于构建索引
                self.progress.setValue(progress)
                QApplication.processEvents()
            
            # 转换为numpy数组
            vectors = np.array(all_embeddings)
            
            self.status.showMessage("正在构建 FAISS 索引...")
            self.progress.setValue(95)
            
            # 4. 构建 FAISS 索引
            dimension = self.embedder.get_dimension()
            self.faiss_index.build_index(vectors, chunk_ids, dimension)
            
            # 5. 将所有文档标记为已索引
            self.status.showMessage("正在更新文档状态...")
            doc_ids = set()
            for chunk_id in chunk_ids:
                conn = self.db.get_connection()
                cursor = conn.cursor()
                cursor.execute('SELECT doc_id FROM chunks WHERE id = ?', (chunk_id,))
                result = cursor.fetchone()
                conn.close()
                if result:
                    doc_ids.add(result[0])
            
            for doc_id in doc_ids:
                self.db.mark_as_indexed(doc_id)
            
            # 6. 保存索引
            self.faiss_index.save()
            
            self.progress.setValue(100)
            self.status.showMessage(f"索引构建成功：{total} 个向量，{len(doc_ids)} 个文档")
            
            # 刷新文档列表以显示索引状态
            self.refresh_doc_list()
            
            QMessageBox.information(
                self, 
                "成功", 
                f"索引构建成功！\n\n向量数: {total}\n维度: {dimension}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"索引构建失败：\n{str(e)}")
            print(f"索引构建错误: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.progress.setVisible(False)
            self.status.showMessage("就绪")

    def on_ask_question(self):
        """处理用户问题：嵌入、搜索、显示 Top-K 文本块。"""
        query = self.input_text.toPlainText().strip()
        
        if not query:
            QMessageBox.warning(self, "问题为空", "请输入一个问题。")
            return
        
        self.status.showMessage("正在搜索...")
        self.progress.setVisible(True)
        self.progress.setValue(0)
        
        try:
            # 1. 检查索引是否加载
            stats = self.faiss_index.get_stats()
            if not stats.get("loaded"):
                QMessageBox.warning(
                    self, 
                    "无索引", 
                    "请先构建索引（点击“重建索引”按钮）。"
                )
                self.progress.setVisible(False)
                self.status.showMessage("就绪")
                return
            
            # 2. 如果还没有初始化嵌入器
            if self.embedder is None:
                try:
                    self.embedder = Embedder()
                except ValueError as e:
                    QMessageBox.critical(
                        self, 
                        "需要 API 密钥", 
                        "请在 .env 文件中设置 OPENAI_API_KEY。"
                    )
                    self.progress.setVisible(False)
                    self.status.showMessage("就绪")
                    return
            
            self.progress.setValue(30)
            QApplication.processEvents()
            
            # 3. 获取查询嵌入
            self.status.showMessage("正在嵌入查询...")
            query_embedding = self.embedder.get_embedding(query)
            query_vector = np.array(query_embedding)
            
            self.progress.setValue(60)
            QApplication.processEvents()
            
            # 4. 搜索 FAISS
            self.status.showMessage("正在搜索索引...")
            k = 5  # Top-5 结果
            distances, chunk_ids = self.faiss_index.search(query_vector, k=k)
            
            self.progress.setValue(80)
            
            # 5. 执行关键词搜索（P1：混合搜索）
            self.status.showMessage("正在执行关键词搜索...")
            keyword_results = self.db.keyword_search(query, k=k)
            
            # 6. 合并向量和关键词结果（混合搜索）
            combined_chunks = {}  # chunk_id -> 数据
            self.list_chunks.clear()
            
            # 添加向量搜索结果
            for dist, chunk_id in zip(distances, chunk_ids):
                vector_score = 1 / (1 + dist)
                conn = self.db.get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT c.text, d.filename FROM chunks c JOIN documents d ON c.doc_id = d.id WHERE c.id = ?',
                    (chunk_id,)
                )
                result = cursor.fetchone()
                conn.close()
                
                if result:
                    text, filename = result
                    combined_chunks[chunk_id] = {
                        'text': text,
                        'filename': filename,
                        'chunk_id': chunk_id,
                        'vector_score': vector_score,
                        'keyword_score': 0
                    }
            
            # 添加关键词搜索结果
            if keyword_results:
                max_kw = max(r[3] for r in keyword_results)
                for chunk_id, text, filename, score in keyword_results:
                    norm_kw = score / max_kw if max_kw > 0 else 0
                    if chunk_id in combined_chunks:
                        combined_chunks[chunk_id]['keyword_score'] = norm_kw
                    else:
                        combined_chunks[chunk_id] = {
                            'text': text,
                            'filename': filename,
                            'chunk_id': chunk_id,
                            'vector_score': 0,
                            'keyword_score': norm_kw
                        }
            
            # 计算综合分数
            for data in combined_chunks.values():
                data['combined_score'] = data['vector_score'] * 0.6 + data['keyword_score'] * 0.4
                data['similarity'] = data['combined_score']
            
            # 排序并显示
            sorted_chunks = sorted(combined_chunks.values(), key=lambda x: x['combined_score'], reverse=True)[:k]
            context_chunks = []
            
            for i, chunk_data in enumerate(sorted_chunks):
                self.list_chunks.addItem(
                    f"【{i+1}】 综合: {chunk_data['combined_score']:.3f} " +
                    f"(向量: {chunk_data['vector_score']:.3f}, 关键词: {chunk_data['keyword_score']:.3f})\n" +
                    f"来源: {chunk_data['filename']}\n{chunk_data['text']}\n{'='*60}"
                )
                context_chunks.append(chunk_data)
            
            self.progress.setValue(85)
            QApplication.processEvents()
            
            # 6. 生成回答前检查置信度（P0：防止幻觉）
            rag = RAGGenerator()
            is_confident, confidence_reason = rag.check_confidence(context_chunks)
            
            if not is_confident:
                # 使用备用回复而不是LLM
                self.status.showMessage(f"置信度低: {confidence_reason}")
                answer, citations = rag.generate_fallback_response(query, context_chunks, confidence_reason)
                
                # 在Tab 1显示备用回答
                self.text_answer.clear()
                self.text_answer.append(answer)
                
                # 切换到回答选项卡
                self.tabs.setCurrentIndex(0)
                
                self.progress.setValue(100)
                self.status.showMessage(f"返回备用回复（置信度低）")
                
                return  # 跳过LLM生成
            
            # 7. 生成 RAG 回答（第6天功能）- 仅在置信度足够时
            self.status.showMessage("正在生成回答...")
            try:
                answer, citations = rag.generate_answer(query, context_chunks)
                
                # 验证引用（P0：引用验证）
                is_valid, citation_issue = rag.verify_citations(answer, context_chunks)
                if not is_valid:
                    # 显示关于无效引用的警告
                    self.text_answer.clear()
                    self.text_answer.append("⚠️ **引用验证警告**\n")
                    self.text_answer.append(f"生成的回答存在引用问题: {citation_issue}\n")
                    self.text_answer.append("="*60 + "\n\n")
                    self.text_answer.append(answer)
                else:
                    # 显示正常回答
                    self.text_answer.clear()
                    self.text_answer.append(answer)
                
                self.text_answer.append("\n" + "="*60)
                self.text_answer.append("\n【引用来源】")
                for i, cite in enumerate(citations):
                    self.text_answer.append(
                        f"\n[{i+1}] {cite['filename']}\n摘录: {cite['excerpt']}"
                    )
                
                # 切换到回答选项卡
                self.tabs.setCurrentIndex(0)
                
            except Exception as e:
                # 如果 RAG 失败，仍显示文本块（回退到第5天行为）
                print(f"RAG 生成失败: {e}")
                import traceback
                traceback.print_exc()
                self.tabs.setCurrentIndex(1)  # 显示文本块而不是答案
                QMessageBox.warning(
                    self, 
                    "生成错误", 
                    f"无法生成回答。显示搜索结果。\n\n错误: {str(e)}"
                )
            
            self.progress.setValue(100)
            self.status.showMessage(f"找到 {len(chunk_ids)} 个相关文本块")
            
        except Exception as e:
            QMessageBox.critical(self, "搜索错误", f"搜索失败：\n{str(e)}")
            print(f"搜索错误: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.progress.setVisible(False)
            self.status.showMessage("就绪")

    def on_file_list_context_menu(self, position):
        """显示文件列表的上下文菜单（右键单击）。"""
        item = self.file_list.itemAt(position)
        if not item:
            return
        
        # 创建上下文菜单
        menu = QMenu(self)
        delete_action = menu.addAction("删除文档")
        
        # 显示菜单并获取操作
        action = menu.exec_(self.file_list.mapToGlobal(position))
        
        if action == delete_action:
            self.on_delete_document(item)
    
    def on_delete_document(self, item):
        """删除选中的文档及其文本块。"""
        try:
            # 从 item 的用户数据中获取 doc_id
            doc_id = item.data(Qt.UserRole)
            if doc_id is None:
                raise ValueError("无法获取文档ID")
            
            # 确认删除
            reply = QMessageBox.question(
                self,
                "确认删除",
                f"确定要删除文档 ID {doc_id} 吗？\n这将删除文档及其所有片段。",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                return
            
            # 从数据库删除
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            # 先删除文本块（外键约束）
            cursor.execute('DELETE FROM chunks WHERE doc_id = ?', (doc_id,))
            deleted_chunks = cursor.rowcount
            
            # 删除文档
            cursor.execute('DELETE FROM documents WHERE id = ?', (doc_id,))
            
            conn.commit()
            conn.close()
            
            # 刷新UI
            self.refresh_doc_list()
            
            QMessageBox.information(
                self,
                "删除成功",
                f"已删除文档 (ID: {doc_id}) 及其 {deleted_chunks} 个片段。\n\n请重新建立索引以更新向量库。"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "删除失败", f"删除出错:\n{str(e)}")
            print(f"Delete error: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # 尝试从 assets 加载 QSS，如果是独立运行
    # 假设从 kb_desktop 根目录或 app 文件夹运行
    # 这是为了独立测试的尽力而为
    possible_paths = [
        "assets/styles.qss",
        "../assets/styles.qss",
        "kb_desktop/assets/styles.qss"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                app.setStyleSheet(f.read())
            break
            
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
