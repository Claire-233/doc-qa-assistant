import os
from openai import OpenAI
from dotenv import load_dotenv
from loaders import load_txt_files
from splitter import hard_split

# 加载环境变量
load_dotenv()

# 初始化硅基流动的 OpenAI 兼容客户端
client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1"
)

def get_embedding(text: str, model: str = "BAAI/bge-large-zh-v1.5") -> list:
    """调用硅基流动 Embedding API 获取向量"""
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding

if __name__ == "__main__":
    # 1. 加载文档
    docs = load_txt_files()
    if not docs:
        print("⚠️ 没有加载到文档，退出")
        exit()
    
    # 2. 切分文档（复用 splitter）
    all_chunks = []
    for doc in docs:
        chunks = hard_split(doc["content"])
        all_chunks.extend(chunks)
    print(f"\n📦 共 {len(all_chunks)} 个文本块需要向量化")
    
    # 3. 对第一个文本块进行向量化（演示）
    sample_chunk = all_chunks[0]
    print(f"\n🔬 对第一个文本块进行向量化...")
    print(f"文本预览: {sample_chunk[:80]}...")
    
    vector = get_embedding(sample_chunk)
    print(f"✅ 向量维度: {len(vector)}")  # 期望输出 1024
    print(f"向量前5个元素: {vector[:5]}")
    
    # 4. 可选：对所有文本块进行向量化（耗时较长，先注释掉）
    # print("\n⏳ 正在批量向量化所有文本块...")
    # all_vectors = [get_embedding(chunk) for chunk in all_chunks]
    # print(f"✅ 共生成 {len(all_vectors)} 个向量")