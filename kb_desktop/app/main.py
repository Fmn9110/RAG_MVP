import sys
import os

# 将项目根目录添加到 sys.path 以允许从 core 导入
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from PySide6.QtWidgets import QApplication
from app.ui_main import MainWindow

def main():
    app = QApplication(sys.argv)
    
    # 设置全局样式
    app.setStyle("Fusion")
    
    # 加载 QSS 样式表
    # project_root 是 .../kb_desktop
    qss_file = os.path.join(project_root, "assets", "styles.qss")
    if os.path.exists(qss_file):
        with open(qss_file, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())
    else:
        print(f"警告：未找到样式表于 {qss_file}")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
