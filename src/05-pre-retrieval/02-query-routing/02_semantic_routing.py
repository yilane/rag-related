"""
语义路由 (Semantic Routing) 示例

本脚本演示如何基于查询内容的语义相似性，将用户问题路由到最匹配的专门化提示模板。
与逻辑路由不同，语义路由使用向量相似度计算来确定最佳匹配。

核心思想：
1. 为不同领域预定义专门化的提示模板
2. 将所有提示模板进行向量化嵌入
3. 计算用户查询与各模板的余弦相似度
4. 选择相似度最高的模板进行问答

主要优势：
- 自动化的领域识别，无需手工规则
- 基于语义理解，比关键词匹配更准确
- 可扩展性强，易于添加新的专业领域
- 对查询表达方式的变化更具鲁棒性
"""

import os
import numpy as np
from dotenv import load_dotenv
from langchain.utils.math import cosine_similarity
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings

# 加载环境变量
load_dotenv()

# 初始化嵌入模型 - 使用中文优化的模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh",  # 使用中文优化的嵌入模型
    model_kwargs={'device': 'cpu'},   # 使用CPU计算
    encode_kwargs={'normalize_embeddings': True}  # 启用向量归一化
)

# 定义不同领域的专门化提示模板
physics_template = """你是一位非常优秀的物理学教授。\
你擅长用简洁易懂的方式回答物理学问题。\
当你不知道某个问题的答案时，你会诚实地承认不知道。

请回答以下物理学问题：
{query}"""

math_template = """你是一位非常优秀的数学家。你擅长回答数学问题。\
你之所以如此出色，是因为你能够将复杂问题分解为组成部分，\
逐一解决各个部分，然后将它们整合起来回答更广泛的问题。

请回答以下数学问题：
{query}"""

programming_template = """你是一位经验丰富的程序员和软件工程师。\
你精通多种编程语言和开发技术，能够提供清晰的代码示例和解释。\
你擅长调试代码问题，解释编程概念，并提供最佳实践建议。

请回答以下编程问题：
{query}"""

general_template = """你是一位知识渊博的助手，能够回答各种通用问题。\
你会尽力提供准确、有用的信息，并承认自己不确定的地方。\
请用清晰、结构化的方式组织你的回答。

请回答以下问题：
{query}"""

# 收集所有提示模板
prompt_templates = [
    ("物理学", physics_template),
    ("数学", math_template), 
    ("编程", programming_template),
    ("通用", general_template)
]

print("🔄 正在计算提示模板的向量嵌入...")
# 计算所有提示模板的向量嵌入
template_texts = [template for _, template in prompt_templates]
prompt_embeddings = embeddings.embed_documents(template_texts)
print("✅ 提示模板向量化完成")

# 初始化DeepSeek大语言模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,  # 适中的创造性
    max_tokens=2048,  # 允许较长的回答
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

def prompt_router(user_input: str):
    """
    基于语义相似度的提示路由器
    
    Args:
        user_input: 用户输入的查询
        
    Returns:
        tuple: (选中的领域名称, 格式化的提示)
    """
    print(f"\n🔍 正在分析查询: {user_input[:50]}{'...' if len(user_input) > 50 else ''}")
    
    # 1. 将用户查询进行向量化
    query_embedding = embeddings.embed_query(user_input)
    
    # 2. 计算查询与所有提示模板的余弦相似度
    similarity_scores = cosine_similarity([query_embedding], prompt_embeddings)[0]
    
    # 3. 找到相似度最高的模板
    best_match_idx = np.argmax(similarity_scores)
    best_score = similarity_scores[best_match_idx]
    
    # 4. 获取最匹配的领域和模板
    domain_name, most_similar_template = prompt_templates[best_match_idx]
    
    # 5. 显示路由信息
    print(f"📊 相似度分数:")
    for i, (domain, _) in enumerate(prompt_templates):
        score = similarity_scores[i]
        marker = "👉 " if i == best_match_idx else "   "
        print(f"{marker}{domain}: {score:.4f}")
    
    print(f"\n🎯 选择领域: {domain_name} (相似度: {best_score:.4f})")
    
    # 6. 创建提示模板对象并格式化
    template = PromptTemplate.from_template(most_similar_template)
    formatted_prompt = template.invoke({'query': user_input})
    
    return domain_name, formatted_prompt

def semantic_routing_qa(user_query: str) -> str:
    """
    完整的语义路由问答流程
    
    Args:
        user_query: 用户查询
        
    Returns:
        str: 大模型的回答
    """
    # 1. 路由到最适合的提示模板
    domain, refined_prompt = prompt_router(user_query)
    
    # 2. 使用DeepSeek生成回答
    print(f"\n🤖 正在使用{domain}专家模式生成回答...")
    response = llm.invoke(refined_prompt)
    
    return response.content

def main():
    """主函数：演示语义路由系统"""
    print("🎯 语义路由系统 - DeepSeek版本")
    print("=" * 60)
    print("💡 本系统会根据问题的语义内容自动选择最合适的专家回答")
    
    # 测试用例
    test_queries = [
        "什么是黑洞？它是如何形成的？",
        "如何求解二次方程？请详细解释步骤。",
        "Python中的装饰器是什么？如何使用？",
        "请介绍一下人工智能的发展历史。"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*20} 测试用例 {i} {'='*20}")
        print(f"📝 用户问题: {query}")
        
        try:
            # 执行语义路由问答
            answer = semantic_routing_qa(query)
            print(f"\n💬 系统回答:\n{answer}")
            
        except Exception as e:
            print(f"❌ 处理过程中出现错误: {e}")
        
        print("\n" + "-" * 60)

def interactive_mode():
    """交互模式：允许用户输入自定义问题"""
    print("\n🎮 进入交互模式 (输入 'quit' 或 'exit' 退出)")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n❓ 请输入您的问题: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q', '退出']:
                print("👋 感谢使用语义路由系统！")
                break
            
            if not user_input:
                print("⚠️  请输入有效问题")
                continue
            
            # 执行语义路由问答
            answer = semantic_routing_qa(user_input)
            print(f"\n💬 系统回答:\n{answer}")
            
        except KeyboardInterrupt:
            print("\n👋 感谢使用语义路由系统！")
            break
        except Exception as e:
            print(f"❌ 处理过程中出现错误: {e}")

if __name__ == "__main__":
    # 运行预设测试用例
    main()

