import re
from typing import List

def recursive_split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    separators: List[str] = None
) -> List[str]:
    """
    递归字符切分器
    
    参数:
        text: 待切分的原始文本
        chunk_size: 每个文本块的最大字符数（默认500）
        chunk_overlap: 相邻文本块之间的重叠字符数（默认80）
        separators: 分隔符优先级列表（从高到低）
    
    返回:
        切分后的文本块列表
    """
    if separators is None:
        separators = ["\n\n", "\n", "。", "；", "，", " ", ""]
    
    # 最终返回的文本块
    final_chunks = []
    
    def _split(text: str, separators: List[str], chunk_size: int) -> List[str]:
        """内部递归切分函数"""
        if len(text) <= chunk_size:
            return [text]
        
        # 尝试当前最高优先级的分隔符
        separator = separators[0]
        
        if separator == "":
            # 最后手段：按字符切分
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        # 按当前分隔符分割
        segments = text.split(separator)
        
        # 如果分隔符不在文本中，降级到下一个分隔符
        if len(segments) == 1:
            return _split(text, separators[1:], chunk_size)
        
        # 合并片段直到接近 chunk_size
        current_chunk = ""
        chunks = []
        for seg in segments:
            # 加上分隔符（如果是换行符则不加，保持原样）
            if separator in ["\n\n", "\n"]:
                candidate = current_chunk + seg + separator
            else:
                candidate = current_chunk + seg + separator
            
            if len(candidate) <= chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # 如果当前片段本身就超长，递归处理
                if len(seg) > chunk_size:
                    sub_chunks = _split(seg, separators[1:], chunk_size)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = seg + separator
        
        # 处理最后一个片段
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    # 第一次切分
    raw_chunks = _split(text, separators, chunk_size)
    
    # 应用重叠（overlap）
    if chunk_overlap > 0 and len(raw_chunks) > 1:
        final_chunks = [raw_chunks[0]]
        for i in range(1, len(raw_chunks)):
            prev_chunk = raw_chunks[i-1]
            curr_chunk = raw_chunks[i]
            
            # 从前一个块的尾部取 overlap 字符作为当前块的前缀
            overlap_text = prev_chunk[-chunk_overlap:] if len(prev_chunk) >= chunk_overlap else prev_chunk
            overlapped_chunk = overlap_text + curr_chunk
            final_chunks.append(overlapped_chunk)
    else:
        final_chunks = raw_chunks
    
    # 过滤空文本块
    final_chunks = [chunk for chunk in final_chunks if chunk.strip()]
    
    return final_chunks


# 测试代码（如果直接运行本文件）
if __name__ == "__main__":
    test_text = """苏轼字子瞻，眉州眉山人。生十年，父洵游学四方，母程氏亲授以书。
    嘉祐二年，试礼部。主司欧阳修得轼《刑赏忠厚论》，惊喜。
    苏轼与弟辙同科进士及第，父子三人名动京师，世称"三苏"。
    苏轼一生仕途坎坷，屡遭贬谪，然其文学成就极高，为唐宋八大家之一。"""
    
    chunks = recursive_split_text(test_text, chunk_size=150, chunk_overlap=30)
    print(f"切分后共 {len(chunks)} 个文本块：")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- 块 {i} (长度 {len(chunk)}) ---")
        print(chunk)