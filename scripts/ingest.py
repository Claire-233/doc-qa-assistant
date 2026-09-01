import sys
import os

# 获取项目根目录（scripts 的父目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 打印调试信息
print(f"当前脚本目录: {current_dir}")
print(f"项目根目录: {project_root}")

from store import ingest_data

if __name__ == "__main__":
    ingest_data()