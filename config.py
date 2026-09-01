from dotenv import load_dotenv
import os

def check_env():
    # 加载 .env 文件
    load_dotenv()
    
    # 校验 DeepSeek 和 硅基流动 的 Key（根据你的实际需求调整）
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    siliconflow_key = os.getenv("SILICONFLOW_API_KEY")
    
    if not deepseek_key:
        raise ValueError("环境变量 DEEPSEEK_API_KEY 未配置！请检查 .env 文件")
    
    if not siliconflow_key:
        raise ValueError("环境变量 SILICONFLOW_API_KEY 未配置！请检查 .env 文件")
        
    print("✅ 环境变量加载成功！")

if __name__ == "__main__":
    check_env()