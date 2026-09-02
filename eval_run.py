import sys
import time
from pathlib import Path

# 将项目根目录（当前文件所在目录）加入 Python 搜索路径
sys.path.append(str(Path(__file__).resolve().parent))

# 从 pipeline 导入函数（请务必把 answer_question 替换成你 pipeline.py 里真实的函数名！）
from pipeline import answer_question 

# ... 下面的 20 条测试问题和 run_evaluation() 代码保持不变 ...

# 20 条测试问题（假设本地知识库主要为《东坡诗话》相关内容）
TEST_QUESTIONS = [
    # --- 文档内问题 (15条) ---
    "苏轼有哪些著名的诗句？",
    "什么是乌台诗案？",
    "苏轼在狱中写了什么诗？",
    "苏轼的《酸枣》诗句是什么？",
    "苏轼对儿子苏迈的《林麓》诗有什么评价？",
    "苏轼关于诗画艺术有什么见解？",
    "苏轼评价吴道子画作时提出了什么艺术理念？",
    "根到九泉无觅处是谁的诗句？",
    "苏轼和宋神宗的关系如何？",
    "苏轼晚年的作品有哪些？",
    "水光潋滟晴方好是哪里的景色？", # 文档备注中提到但未详述
    "苏轼和王安石之间有什么故事？",
    "苏轼被贬黄州时作了哪些诗？",
    "什么是东坡肉？", # 关联苏轼生活常识，可能在扩展文档中
    "苏轼的弟弟是谁？",

    # --- 文档外问题 (5条，用于测试拒答/反幻觉能力) ---
    "火星上有几个人？",
    "李白的《静夜思》全文是什么？",
    "今天成都的天气怎么样？",
    "Python 怎么读取 Excel 文件？",
    "爱因斯坦发明了什么？"
]

def run_evaluation():
    print("🚀 开始运行问答系统评估...\n")
    
    results = []
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"[{i}/20] 提问: {question}")
        
        # 模拟空的聊天历史进行测试
        chat_history = []
        try:
            answer = answer_question(question, chat_history)
        except Exception as e:
            answer = f"❌ 调用出错: {e}"
            
        print(f"回答: {answer}\n")
        
        # 判断是否为文档外问题 (索引 15 及以后)
        is_out_of_doc = i > 15
        
        # 简单的拒答检测逻辑（可根据实际 prompt 约束调整关键词）
        reject_keywords = ["未找到相关信息", "未提及", "无法回答", "没有相关信息", "文档中"]
        is_rejected = any(kw in answer for kw in reject_keywords)
        
        results.append({
            "id": i,
            "question": question,
            "answer": answer,
            "is_out_of_doc": is_out_of_doc,
            "is_rejected": is_rejected
        })
        
        time.sleep(0.5) # 避免请求过快

    # 打印评估报告
    print("\n" + "="*50)
    print("📊 评估报告汇总")
    print("="*50)
    
    success_doc = sum(1 for r in results if not r["is_out_of_doc"] and not r["is_rejected"])
    reject_success = sum(1 for r in results if r["is_out_of_doc"] and r["is_rejected"])
    
    print(f"文档内问题正确回答数: {success_doc}/15")
    print(f"文档外问题正确拒答数: {reject_success}/5")
    print("\n详细结果已生成，请检查上方日志。")

if __name__ == "__main__":
    run_evaluation()