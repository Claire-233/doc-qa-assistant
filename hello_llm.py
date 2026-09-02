from dotenv import load_dotenv
import os
from openai import OpenAI

# 加载 .env 文件
load_dotenv()

# 初始化 DeepSeek 客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

print("🤖 问答助手已启动！（DeepSeek 模型，连续问三个问题）\n")

for i in range(1, 4):
    user_input = input(f"问题{i}: ")
    
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": "你是一个有帮助的助手"},
            {"role": "user", "content": user_input}
        ],
        stream=False
    )
    
    print(f"AI: {response.choices[0].message.content}\n")

print("👋 三个问题已问完，程序结束。")