from loaders import load_txt_files

def hard_split(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """硬切分文本为多个 chunk"""
    if len(text) <= chunk_size:
        return [text]
        
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap) # 滑动窗口重叠
        
    return chunks

if __name__ == "__main__":
    # 1. 复用 loaders 加载文档
    docs = load_txt_files()
    
    if not docs:
        print("⚠️ 没有加载到有效文档，请检查 data/raw 目录")
    else:
        total_chunks = 0
        # 2. 对每个文档进行硬切分
        for doc in docs:
            chunks = hard_split(doc["content"])
            total_chunks += len(chunks)
            # 下面这行是修复重点，确保完整复制
            print(f"📄 {doc['name']} 切分为 {len(chunks)} 个 chunks")
            
        # 3. 打印总 chunk 数量
        print(f"🎉 切分完成！总共生成 {total_chunks} 个 chunks")