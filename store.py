#!/usr/bin/env python3
"""
store.py - 文档加载、递归切分、向量化并存入 ChromaDB
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# ------------------------------------------------------------
# 1. 动态计算项目根目录（store.py 位于项目根目录）
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# 将 scripts 目录加入 sys.path，便于导入 splitter.py（如果 splitter.py 在 scripts/ 下）
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
# 如果 splitter.py 在根目录，也可以直接加根目录
sys.path.insert(0, str(PROJECT_ROOT))

# 加载 .env 文件（从项目根目录）
load_dotenv(PROJECT_ROOT / ".env")

# ------------------------------------------------------------
# 2. 导入递归切分函数
# ------------------------------------------------------------
try:
    from splitter import recursive_split_text
except ImportError:
    # 如果 splitter.py 在 scripts/ 下，从那里导入
    sys.path.insert(0, str(SCRIPTS_DIR))
    from splitter import recursive_split_text

# ------------------------------------------------------------
# 3. 常量配置
# ------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "raw"          # 原始文档目录
CHROMA_DIR = PROJECT_ROOT / "chroma_db"           # ChromaDB 持久化目录
EMBEDDING_MODEL = "BAAI/bge-m3"                   # 硅基流动 embedding 模型
CHUNK_SIZE = 512                                  # 递归切分块大小（可调）
CHUNK_OVERLAP = 80                                # 重叠字符数（可调）

# ------------------------------------------------------------
# 4. 初始化 OpenAI 客户端（用于 embedding）
# ------------------------------------------------------------
client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),     # 从 .env 读取
    base_url="https://api.siliconflow.cn/v1/"
)

# ------------------------------------------------------------
# 5. 加载文本文件
# ------------------------------------------------------------
def load_txt_files() -> List[Dict[str, Any]]:
    """加载 DATA_DIR 下所有 .txt 文件"""
    docs = []
    if not DATA_DIR.exists():
        print(f"❌ 数据目录不存在: {DATA_DIR}")
        return docs

    for file_path in sorted(DATA_DIR.glob("*.txt")):
        try:
            content = file_path.read_text(encoding="utf-8")
            docs.append({
                "source": file_path.name,
                "content": content
            })
            print(f"✅ 加载文档: {file_path.name} ({len(content)} 字符)")
        except Exception as e:
            print(f"⚠️ 读取失败 {file_path.name}: {e}")
    return docs

# ------------------------------------------------------------
# 6. Embedding 函数（带重试）
# ------------------------------------------------------------
def get_embedding_with_retry(text: str, max_retries: int = 3) -> List[float]:
    """调用硅基流动 Embedding API，失败时指数退避重试"""
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            return resp.data[0].embedding
        except Exception as e:
            print(f"⚠️ Embedding 失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)   # 指数退避
            else:
                raise   # 最后一次失败则抛出异常
    return []   # 不会执行到这里

# ------------------------------------------------------------
# 7. 初始化 ChromaDB 集合
# ------------------------------------------------------------
def init_chroma() -> chromadb.Collection:
    """创建或获取持久化 ChromaDB 集合"""
    chroma_client = chromadb.PersistentClient(str(CHROMA_DIR))
    # 嵌入函数用于后续 query 时的自动向量化（需与入库时使用的模型一致）
    embedding_func = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        model_name=EMBEDDING_MODEL,
        api_base="https://api.siliconflow.cn/v1/"
    )
    collection = chroma_client.get_or_create_collection(
        name="doc_qa_collection",
        embedding_function=embedding_func
    )
    return collection

# ------------------------------------------------------------
# 8. 主入库流程
# ------------------------------------------------------------
def ingest_data():
    """加载文档 → 递归切分 → 向量化 → 存入 ChromaDB"""
    docs = load_txt_files()
    if not docs:
        print("⚠️ 没有加载到文档，退出")
        return

    # 切分所有文档
    all_chunks = []
    all_sources = []
    for doc in docs:
        chunks = recursive_split_text(
            doc["content"],
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        all_chunks.extend(chunks)
        all_sources.extend([doc["source"]] * len(chunks))

    print(f"📦 共切分为 {len(all_chunks)} 个文本块")

    # 初始化 ChromaDB 集合
    collection = init_chroma()

    # 分批入库（每批 5 条）
    batch_size = 5
    total_batches = (len(all_chunks) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(all_chunks))
        batch_chunks = all_chunks[start:end]
        batch_ids = [f"chunk_{start + j}" for j in range(end - start)]
        batch_metas = [{"source": src} for src in all_sources[start:end]]

        # 获取向量
        batch_embeddings = []
        skip_batch = False
        for chunk in batch_chunks:
            try:
                emb = get_embedding_with_retry(chunk)
                batch_embeddings.append(emb)
            except Exception as e:
                print(f"❌ 文本块向量化失败，跳过该批次: {e}")
                skip_batch = True
                break

        if skip_batch or not batch_embeddings:
            continue

        # 写入 ChromaDB
        try:
            collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_chunks,
                metadatas=batch_metas
            )
            print(f"✅ 批次 {batch_idx+1}/{total_batches} 入库成功 ({len(batch_chunks)} 条)")
        except Exception as e:
            print(f"❌ 批次 {batch_idx+1} 入库失败: {e}")

    print("🎉 数据入库完成！")

# ------------------------------------------------------------
# 9. 入口
# ------------------------------------------------------------
if __name__ == "__main__":
    ingest_data()