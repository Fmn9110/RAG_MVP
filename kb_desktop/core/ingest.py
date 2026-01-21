import os
import docx

class Ingestor:
    @staticmethod
    def load_file(file_path):
        """
        读取文件并返回其文本内容。
        支持: .txt, .md, .docx
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext == '.docx':
                return Ingestor._read_docx(file_path)
            elif ext in ['.txt', '.md']:
                return Ingestor._read_text(file_path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
        except Exception as e:
            raise Exception(f"Failed to read {file_path}: {str(e)}")

    @staticmethod
    def _read_text(path):
        # 明确支持中文编码
        encodings = ['utf-8', 'gb18030', 'gbk', 'big5', 'latin-1']
        
        for enc in encodings:
            try:
                with open(path, 'r', encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        # 如果全部失败，尝试 ignore（最后手段）
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    @staticmethod
    def _read_docx(path):
        doc = docx.Document(path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
