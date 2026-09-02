#!/usr/bin/env python3
"""
app.py - Gradio Web 界面，基于 ChromaDB + 硅基流动 DeepSeek-V3 的 RAG 问答
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import gradio as gr

# ------------------------------------------------------------
# 1. 动态计算项目根目录（app.py 也放在项目根目录）
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# 添加 scripts 目录到 sys.path（以便导入 splitter.py 等）
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# 加载 .env 文件
load_dotenv(PROJECT_ROOT / ".env")

# ------------------------------------------------------------
# 2. 导入递归切分函数（用于查询时可能的上下文截断，这里保留备用）
# ------------------------------------------------------------
try:
    from splitter import recursive_split_text
except ImportError:
    # 如果 splitter.py 在根目录
    pass

# ------------------------------------------------------------
# 3. 常量配置
# ------------------------------------------------------------
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "doc_qa_collection"
EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = "deepseek-ai/DeepSeek-V3"  # 硅基流动上的 DeepSeek-V3 模型 ID
MAX_RETRIEVAL_CHUNKS = 5               # 检索返回的最大文本块数量

# ------------------------------------------------------------
# 4. 初始化客户端和 ChromaDB 集合
# ------------------------------------------------------------
# 硅基流动客户端（用于 embedding 和 LLM）
client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1/"
)

# 初始化 ChromaDB
chroma_client = chromadb.PersistentClient(str(CHROMA_DIR))

# 嵌入函数（用于查询时的自动向量化，必须与入库时一致）
embedding_func = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    model_name=EMBEDDING_MODEL,
    api_base="https://api.siliconflow.cn/v1/"
)

# 获取集合
collection = chroma_client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func
)

# ------------------------------------------------------------
# 5. 核心问答函数
# ------------------------------------------------------------
def answer_question(query: str, history: List[Tuple[str, str]]) -> str:
    """
    接收用户问题，执行 RAG 检索 + LLM 生成，返回回答字符串。
    history 参数由 Gradio ChatInterface 自动维护，此处未使用，但保留签名。
    """
    if not query.strip():
        return "请输入您的问题。"

    # 1. 检索相关文本块
    try:
        results = collection.query(
            query_texts=[query],
            n_results=MAX_RETRIEVAL_CHUNKS
        )
    except Exception as e:
        return f"❌ 检索失败: {str(e)}"

    documents = results["documents"][0] if results["documents"] else []
    sources = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []

    if not documents:
        return "😕 未在知识库中找到相关信息，请换个问题试试。"

    # 2. 构建上下文（带引用编号）
    context_parts = []
    for idx, (doc, meta, dist) in enumerate(zip(documents, sources, distances)):
        source_name = meta.get("source", "未知来源")
        context_parts.append(f"[{idx+1}] (来源: {source_name}, 距离: {dist:.4f})\n{doc}")
    context = "\n\n".join(context_parts)

    # 3. 构造 Prompt
    system_prompt = (
        "你是一个基于本地知识库的问答助手。请根据以下提供的参考材料，用中文回答用户的问题。"
        "回答时请标注引用的来源编号，格式如 [1]、[2] 等。如果参考材料不足以回答问题，"
        "请如实说明。不要编造信息。"
    )
    user_prompt = f"""参考材料：
{context}

用户问题：{query}

请根据参考材料回答："""

    # 4. 调用 LLM
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=2048
        )
        answer = response.choices[0].message.content
    except Exception as e:
        return f"❌ LLM 调用失败: {str(e)}"

    # 5. 附加检索到的原文（可选，方便调试）
    final_output = answer + "\n\n---\n**📎 参考原文**\n" + context
    return final_output

# ------------------------------------------------------------
# 6. Gradio 界面
# 假设前面的组件定义(如 chatbot, msg)都在 create_gradio_app 内，且缩进正确
def create_gradio_app():
    # 1. 先定义界面组件（按顺序）
    with gr.Blocks(title="本地知识库问答系统") as demo:
        gr.Markdown("## 本地知识库问答系统")
        
        # 先定义 chatbot
        chatbot = gr.Chatbot(label="对话历史", height=500)
        
        # 再定义输入框
        msg = gr.Textbox(label="输入您的问题", placeholder="请输入...")
        
        # 再定义按钮
        submit_btn = gr.Button("发送")
        
        # 2. 清空按钮放在被清空的组件定义之后
        clear = gr.ClearButton([msg, chatbot])
        
        # 3. 绑定点击事件（写在组件定义之后）
        submit_btn.click(
            fn=respond, 
            inputs=[msg, chatbot], 
            outputs=[msg, chatbot]
        )
        
    return demo

# 将 respond 函数放在外部（与 create_gradio_app 平级）
def respond(message, chat_history):
    answer = answer_question(message, chat_history)
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": answer})
    return "", chat_history

# 启动应用
if __name__ == "__main__":
    demo = create_gradio_app()
    print("🚀 启动 Gradio 应用，请在浏览器中打开显示的 URL")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft()
    )