# 中文知识库助手 (RAG Desktop)

本项目是一个功能完整的本地桌面知识库助手，基于 RAG (Retrieval-Augmented Generation) 技术架构。它允许用户构建私人文档库，并通过智能对话的方式获取基于文档的精准回答。

ui界面经过现代化美化，提供舒适的视觉体验。

## 📷 项目预览
*(在此处添加运行截图)*

## ✨ 核心特性

- **📂 多格式文档导入**: 支持 `.txt`, `.md`, `.docx` 格式文档的批量导入。
- **🧠 智能分块与索引**: 自动将文档切分为适合检索的语义片段，并使用 FAISS 建立高效向量索引。
- **💬 RAG 智能问答**: 结合语义检索与 LLM 生成能力，提供基于本地文档的准确回答。
- **📝 精确引用**: 每个回答都会清晰标注参考的文档来源及具体片段，杜绝幻觉。
- **🎨 现代化 UI**: 基于 PySide6 构建的“Light Modern”风格界面，简洁美观。

## 🛡️ 质量控制策略

### 1. 防幻觉机制
- **置信度阈值**: Top-1 相似度 < 0.6 时触发兜底
- **分数离散度检查**: TopK 结果分差过小时判定为"全盲猜"
- **拒答 + 追问**: 低置信度时不调用 LLM，改为返回追问建议

### 2. 引用强约束
- **引用解析**: 自动提取回答中的文档编号（如"文档1"、"[2]"）
- **范围校验**: 验证引用编号是否在有效范围内
- **警告标注**: 无效引用时在回答顶部显示警告

### 3. 数据完整性
- **SHA256 哈希去重**: 防止重复导入相同内容
- **增量索引**: 新增文档时无需重建整个索引
- **状态追踪**: 显示每个文档的 chunk 数量和最后索引时间

### 4. 混合检索稳定性
- **语义 + 关键词**: 向量检索(60%) + 关键词匹配(40%) 融合
- **覆盖盲点**: 对专有名词、编号等精确匹配类场景更稳健

### 5. 可评估性
- **eval.jsonl**: 标准化评测数据格式
- **自动化指标**: Hit@K 命中率、引用准确率、平均耗时
- **报告导出**: JSON/CSV 格式结果输出

## 🛠️ 技术栈

- **Frontend**: PySide6 (Qt for Python) + QSS Stylesheets
- **Backend Logic**: Python 3.8+
- **Database**: SQLite (元数据存储)
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Embedding**: OpenAI API 兼容接口 (如 DeepSeek, Moonshot 等)
- **LLM**: OpenAI API 兼容接口

## 🚀 快速开始

### 1. 环境准备
确保已安装 Python 3.8 或更高版本。

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置 API Key
本项目支持所有兼容 OpenAI 格式的 API。请在项目根目录创建 `.env` 文件或设置环境变量。

**Windows PowerShell:**
```powershell
$env:OPENAI_API_KEY = "sk-your-api-key-here"
# 可选: 自定义 Base URL (例如使用 DeepSeek)
# $env:OPENAI_BASE_URL = "https://api.deepseek.com/v1"
```

### 4. 运行应用
```bash
python app/main.py
```

## 📖 使用指南

### 第一步：构建知识库
1. 在左侧面板点击 **“导入文档”**。
2. 选择你需要索引的本地文档（支持多选）。
3. 导入完成后，点击 **“重建索引”** 按钮。系统将计算向量并存储，这可能需要一点时间。

### 第二步：智能提问
1. 在中间的 **“提问区”** 输入你的问题。
2. 点击 **“提问”** 按钮。
3. 系统将自动检索相关片段，并生成回答。

### 第三步：查看结果
- **回答 Tab**: 显示 AI 生成的最终答案及引用列表。
- **命中片段 Tab**: 显示 RAG 检索到的 Top-K 原始文本片段及其相似度得分。

## 📂 项目结构

```
kb_desktop/
├── app/
│   ├── main.py          # 应用程序入口
│   └── ui_main.py       # 主界面逻辑与布局
├── assets/
│   └── styles.qss       # UI 样式表
├── core/
│   ├── ingest.py        # 文档加载器
│   ├── chunker.py       # 文本分块器
│   ├── embedder.py      # 向量化处理
│   ├── index_faiss.py   # FAISS 索引管理
│   ├── rag.py           # RAG 生成逻辑
│   ├── llm.py           # LLM 客户端封装
│   └── storage.py       # SQLite 数据库管理
├── data/
│   ├── kb.sqlite        # 数据库文件
│   └── faiss.index      # 向量索引文件
└── requirements.txt     # 项目依赖
```

## ⚠️ 注意事项

- 请确保 API Key 有足够的余额。
- 首次运行会自动创建 `data` 目录和数据库。
- 建议单次索引文档量不要过大，以免内存溢出（FAISS 索引目前运行在内存中）。

## License

MIT License
