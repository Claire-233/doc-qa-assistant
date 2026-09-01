import os
from pathlib import Path

def health_check(text: str) -> bool:
    """检查文本是否为空或包含乱码"""
    if not text or len(text.strip()) == 0:
        print("❌ 空文本")
        return False
    # 简单乱码检测：如果出现 � 字符，视为乱码
    if '\ufffd' in text:
        print("❌ 检测到乱码字符")
        return False
    return True

def load_txt_files(folder_path: str = "data/raw"):
    """加载指定目录下的所有 .txt 文件，自动处理编码"""
    docs_dir = Path(folder_path)
    if not docs_dir.exists():
        print(f"❌ 目录 {folder_path} 不存在，请先创建并放入 .txt 文件")
        return []

    valid_docs = []
    for file_path in docs_dir.glob("*.txt"):
        # 尝试多种编码读取
        content = None
        for enc in ["utf-8", "gbk", "gb2312", "utf-16"]:
            try:
                content = file_path.read_text(encoding=enc)
                break  # 成功读取则跳出循环
            except (UnicodeDecodeError, LookupError):
                continue
        if content is None:
            print(f"❌ 无法识别编码: {file_path.name}")
            continue

        # 健康检查
        if health_check(content):
            valid_docs.append({"name": file_path.name, "content": content})
            print(f"✅ 加载成功: {file_path.name} ({len(content)} 字符)")
        else:
            print(f"⚠️ 健康检查未通过: {file_path.name}")

    print(f"\n🎉 共加载 {len(valid_docs)} 个有效文档")
    return valid_docs

if __name__ == "__main__":
    load_txt_files()