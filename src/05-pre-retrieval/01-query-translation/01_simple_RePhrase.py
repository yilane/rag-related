"""
简单查询重写Demo - 直接使用LLM重写查询
展示查询重写的基本原理和效果
"""

from openai import OpenAI
import os
from dotenv import load_dotenv

# 加载.env文件中的环境变量，包括API密钥等敏感信息
load_dotenv()

# 初始化OpenAI客户端，指定DeepSeek URL
client = OpenAI(
    base_url="https://api.deepseek.com", 
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

print("=" * 60)
print("🔍 简单查询重写 - 基础版本")
print("=" * 60)

def rewrite_query(question: str) -> str:
    """使用大模型重写查询"""
    
    prompt = f"""
请将以下用户查询重写为更适合知识库检索的标准化表达：

原始查询：{question}

重写要求：
1. 使用更准确的技术术语和标准表达
2. 明确查询意图，避免歧义
3. 去除口语化表达和冗余词汇
4. 扩展缩写，补充完整信息
5. 保持原查询的核心意图不变

请直接给出重写后的查询（不要加任何前缀或说明）。
    """

    # 使用DeepSeek模型重写查询
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

# 测试多个查询案例
test_cases = [
    "那个机器学习的东西怎么调参数啊？",
    "血糖高了咋办？", 
    "DM有啥症状？",
    "小米手机拍照好不好？",
    "这个病严重吗？",
]

print(f"\n🚀 开始测试 {len(test_cases)} 个查询案例\n")

for i, query in enumerate(test_cases, 1):
    print(f"{'='*50}")
    print(f"🧪 测试案例 {i}")
    print(f"{'='*50}")
    
    print(f"📝 原始查询: 「{query}」")
    
    print("🔄 正在重写查询...")
    rewritten = rewrite_query(query)
    
    print(f"✨ 重写查询: 「{rewritten}」")
    
    if i < len(test_cases):
        print(f"\n⏸️  按回车键继续下一个测试案例...")
        input()

print(f"\n{'='*60}")
