"""问答提示模板，包含拒答约束和引用编号要求"""

SYSTEM_PROMPT = """你是一个专业的文档问答助手。请根据给定的上下文信息回答问题。

## 回答规则：
1. **基于上下文**：只能使用提供的上下文信息回答问题，不要依赖自己的知识。
2. **引用来源**：在回答中引用相关段落时，在句末标注对应的引用编号，格式为 [1]、[2] 等。
3. **拒答处理**：如果问题与提供的上下文无关，或者上下文中没有足够的信息来回答问题，请明确回答："抱歉，根据提供的资料无法回答该问题。"
4. **语言**：请使用中文回答问题。
5. **简洁准确**：回答应简洁明了，直击要点，不要添加无关信息。"""

def build_prompt(query: str, context_chunks: list[str]) -> str:
    """
    构建完整的提示词
    
    Args:
        query: 用户的问题
        context_chunks: 相关的文本块列表
        
    Returns:
        完整的提示词字符串
    """
    # 构建上下文部分，给每个文本块编号
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        context_parts.append(f"[{i}] {chunk}")
    
    context_str = "\n\n".join(context_parts)
    
    # 组装完整的提示词
    prompt = f"""{SYSTEM_PROMPT}

## 上下文信息：
{context_str}

## 用户问题：
{query}

## 回答："""
    
    return prompt


def build_refuse_prompt(query: str) -> str:
    """
    构建拒答提示词（当没有找到相关上下文时使用）
    
    Args:
        query: 用户的问题
        
    Returns:
        拒答提示词字符串
    """
    prompt = f"""{SYSTEM_PROMPT}

## 上下文信息：
（未找到与问题相关的参考资料）

## 用户问题：
{query}

## 回答："""
    
    return prompt


# 简单的测试
if __name__ == "__main__":
    # 测试正常问答
    test_context = [
        "苏轼是北宋著名文学家，号东坡居士。",
        "《东坡诗话》记载了苏轼对诗歌创作的独到见解。"
    ]
    test_query = "苏轼是谁？"
    
    full_prompt = build_prompt(test_query, test_context)
    print("=== 正常问答提示词 ===")
    print(full_prompt)
    print("\n" + "="*50)
    
    # 测试拒答
    refuse_prompt = build_refuse_prompt("火星上有几个人？")
    print("\n=== 拒答提示词 ===")
    print(refuse_prompt)