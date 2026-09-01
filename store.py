import os
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import chromadb

from loaders import load_txt_files
from splitter import hard_split

# 加载环境变量
load_dotenv()

# 初始化 OpenAI 兼容客户端
client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
    timeout=30
)

EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
CHROMA_DB_PATH = "./storage/chroma"

# 1. 带指数退避重试机制的 Embedding 调用
@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5)
)
def get_embedding_with_retry(text: str) -> list:
    """调用 Embedding API，遇到限流或网络问题自动重试"""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

# 2. 初始化 ChromaDB 持久化客户端
def init_chroma():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    # 每次重新入库前清空旧集合，确保数量准确
    try:
        client.delete_collection(name="doc_qa")
    except Exception:
        pass
    collection = client.get_or_create_collection(name="doc_qa")
    return collection

def ingest_data():
    """加载、切分、向量化并存入 ChromaDB"""
    docs = load_txt_files()
    if not docs:
        print("⚠️ 没有加载到文档，退出")
        return
    
    all_chunks = []
    for doc in docs:
        chunks = hard_split(doc["content"])
        all_chunks.extend(chunks)
    print(f"📦 共切分为 {len(all_chunks)} 个文本块")
    
    collection = init_chroma()
    
    # 批量处理
    batch_size = 5
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i+batch_size]
        batch_ids = [f"chunk_{i+j}" for j in range(len(batch_chunks))]
        batch_metas = [{"source": "东坡诗话.txt"} for _ in range(len(batch_chunks))]
        
        # 获取向量（自带重试机制）
        batch_embeddings = [get_embedding_with_retry(chunk) for chunk in batch_chunks]
        
        # 存入 ChromaDB
        collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_chunks,
            metadatas=batch_metas
        )
        print(f"  已入库: {min(i+batch_size, len(all_chunks))}/{len(all_chunks)}")
        
    # 打印库内总条数
    total_count = collection.count()
    print(f"🎉 入库完成！库内共 {total_count} 条")

if __name__ == "__main__":
    ingest_data()