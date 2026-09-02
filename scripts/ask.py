import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import os
from openai import OpenAI  # 新增导入

# 获取项目根目录
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 加载环境变量
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

# 初始化 ChromaDB
client = chromadb.PersistentClient(path=os.path.join(SCRIPT_DIR, "chroma_db"))
embedding_func = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    model_name="BAAI/bge-m3",
    api_base="https://api.siliconflow.cn/v1/"
)
collection = client.get_collection(
    name="doc_qa_collection",
    embedding_function=embedding_func
)

# 初始化硅基流动大模型客户端
llm_client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1/"
)

def ask_question(question: str, top_k: int = 3):
    # 1. 检索相关上下文
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )
    contexts = results['documents'][0]
    
    # 2. 清洗上下文并编号
    cleaned_contexts = []
    for i, ctx in enumerate(contexts, 1):
        clean_ctx = ctx.replace('\u3000', ' ').strip()
        cleaned_contexts.append(f"[{i}] {clean_ctx}")
    
    context_text = "\n".join(cleaned_contexts)
    
    # 3. 构造 Prompt（强调引用格式）
    prompt = f"""你是一个文档问答助手。以下是从文档中检索到的相关上下文，每个上下文前面有编号 [1]、[2] 等。

{context_text}

请根据以上上下文回答问题。**要求在回答中引用对应的上下文编号**，例如“根据资料[1]，苏轼……”。回答结束后，请另起一行列出“参考资料：”，并逐一写出每个编号对应的原文片段（简洁摘录即可）。

问题：{question}"""

    print("\n🤖 正在思考生成回答...")
    response = llm_client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",  # 保持你当前的模型
        messages=[
            {"role": "system", "content": "你是一个严谨的文档问答助手，回答时必须引用上下文编号，并在末尾列出参考资料。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1024  # 适当增加 token 上限，以便输出引用列表
    )
    
    # 4. 打印大模型生成的回答
    answer = response.choices[0].message.content
    print("\n💡 回答：")
    print(answer)

if __name__ == "__main__":
    print("🤖 文档问答助手已启动（输入 'exit' 退出）")
    while True:
        user_input = input("请输入你的问题：").strip()
        if user_input.lower() == "exit":
            print("再见！")
            break
        if not user_input:
            continue
            
        ask_question(user_input)