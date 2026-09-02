try:
    from splitter import recursive_split_text
except ImportError:
    import sys
    sys.path.append(str(SCRIPT_DIR / "scripts"))
    from splitter import recursive_split_text
#!/usr/bin/env python3
"""
pipeline.py - 文档入库、检索与大模型问答的完整流水线
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 如果你的环境比较旧，可能是：from langchain.text_splitter import RecursiveCharacterTextSplitter

# ------------------------------------------------------------
# 1. 路径与环境变量
# ------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # 项目根目录
DATA_DIR = SCRIPT_DIR / "data" / "raw"                # 原始文档目录
CHROMA_DIR = SCRIPT_DIR / "chroma_db"                 # ChromaDB 持久化目录

# 加载 .env 文件
load_dotenv(SCRIPT_DIR / ".env")

# ------------------------------------------------------------
# 2. 初始化客户端
# ------------------------------------------------------------
# 硅基流动客户端（用于 embedding 和 LLM）
client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1/"
)

# 嵌入函数（与入库时一致）
embedding_func = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    model_name="BAAI/bge-m3",
    api_base="https://api.siliconflow.cn/v1/"
)

# 初始化 ChromaDB
chroma_client = chromadb.PersistentClient(str(CHROMA_DIR))

# 创建或获取集合
collection = chroma_client.get_or_create_collection(
    name="doc_qa_collection",
    embedding_function=embedding_func
)

# ------------------------------------------------------------
# 3. 文档入库函数
# ------------------------------------------------------------
def ingest_documents(file_path, chunk_size=1024, chunk_overlap=80): # 确保默认参数已改
    # 1. 读取文件内容并赋值给 text
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 2. 实例化切分器
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    # 3. 切分文本
    chunks = splitter.split_text(text) 

# ✅ 新增：清洗文本中的转义字符和多余空白
    chunks = [chunk.replace('\\n', '\n').replace('\\u3000', '').strip() for chunk in chunks]

    if not chunks:
        print("⚠️ 未提取到任何文本块")
        return

   # 将 file_path 转换为 Path 对象
    p_file = Path(file_path)
    ids = [f"{p_file.stem}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": p_file.name} for _ in chunks]

    try:
        collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )
        print(f"✅ 已入库 {len(chunks)} 个文本块（chunk_size={chunk_size}）")
    except Exception as e:
        print(f"❌ 入库失败: {e}")

# ------------------------------------------------------------
# 4. 检索函数
# ------------------------------------------------------------
def retrieve_context(query: str, top_k: int = 5) -> List[str]:
    """
    根据问题检索最相关的 top_k 个文本块。
    返回文档内容列表。
    """
    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        return results['documents'][0] if results['documents'] else []
    except Exception as e:
        print(f"❌ 检索失败: {e}")
        return []

# ------------------------------------------------------------
# 5. 问答函数（RAG 核心）
# ------------------------------------------------------------
def answer_question(query: str, chat_history: Optional[List] = None) -> str:
    """
    检索上下文 + 调用大模型生成回答。
    chat_history 参数预留，当前未使用。
    """
    # 1. 检索
    documents = retrieve_context(query, top_k=5)
    if not documents:
        return "😕 未在知识库中找到相关信息，请换个问题试试。"

    # 2. 构建带引用的上下文
    context_parts = []
    for idx, doc in enumerate(documents):
        context_parts.append(f"[{idx+1}] {doc}")
    context = "\n\n".join(context_parts)

    # 3. 构造 Prompt
    system_prompt = (
    "你是一个基于本地知识库的问答助手。请严格根据提供的参考材料回答用户问题。"
    "回答时请标注引用来源编号，格式如 [1]、[2]。"
    "【严格约束】如果参考材料不足以回答或不包含相关信息，你必须直接回复：'根据知识库文档，未找到相关信息。'，"
    "绝对禁止使用自身知识进行补充、猜测或提供建议！"
)
    user_prompt = f"""参考材料：
{context}

用户问题：{query}

请根据参考材料回答："""

    # 4. 调用 LLM
    try:
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",   # 硅基流动上的模型 ID
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=2048
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ LLM 调用失败: {str(e)}"

# ------------------------------------------------------------
# 6. 主入口（测试用）
# ------------------------------------------------------------
if __name__ == "__main__":
    test_file = DATA_DIR / "东坡诗话.txt"
    if test_file.exists():
        ingest_documents(str(test_file), chunk_size=1024, chunk_overlap=80)
    else:
        print(f"⚠️ 测试文件不存在: {test_file}，请先放置文档。")

    # 测试检索
    context = retrieve_context("苏轼是谁？")
    print("检索到的上下文:", context)

    # 测试问答
    answer = answer_question("苏轼有哪些著名的诗句？")
    print("问答结果:", answer)