"""
逻辑路由 (Logical Routing) 示例

本脚本演示如何根据用户查询的编程语言类型，将问题路由到相应的数据源。
这是一个基于内容分析的智能路由系统，可以自动识别查询内容并选择最合适的处理链。

核心思想：
1. 分析用户查询内容，识别编程语言类型
2. 根据识别结果，路由到相应的专门处理链
3. 每个处理链针对特定编程语言优化，提供更精准的答案
"""

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

# 加载环境变量
load_dotenv()

# 初始化DeepSeek大语言模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.1,  # 设置较低的温度，确保路由结果的稳定性
    max_tokens=50,    # 路由只需要简短回答，限制token数量
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 设置路由提示模板
system_prompt = """你是一个专业的编程查询路由专家。

根据用户问题中涉及的编程语言，将其路由到相应的数据源：
- 如果问题涉及Python代码或Python相关概念，返回 "python_docs"
- 如果问题涉及JavaScript代码或JavaScript相关概念，返回 "js_docs" 
- 如果问题涉及Go/Golang代码或Go相关概念，返回 "golang_docs"
- 如果无法明确判断或涉及多种语言，返回 "general_docs"

请只返回数据源名称，不要包含其他内容。"""

# 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}"),
])

def route_query(question: str) -> str:
    """
    根据问题内容进行路由分析
    
    Args:
        question: 用户提出的问题
        
    Returns:
        str: 路由到的数据源名称
    """
    # 构建结构化提示
    formatted_prompt = prompt.invoke({'question': question})
    
    # 调用大模型进行路由判断
    result = llm.invoke(formatted_prompt)
    
    return result.content.strip().lower()

def python_docs_chain(question: str) -> str:
    """Python文档处理链"""
    return f"🐍 Python文档链处理: {question}\n这里会连接到Python专门的文档检索和回答系统"

def js_docs_chain(question: str) -> str:
    """JavaScript文档处理链"""
    return f"📜 JavaScript文档链处理: {question}\n这里会连接到JavaScript专门的文档检索和回答系统"

def golang_docs_chain(question: str) -> str:
    """Go语言文档处理链"""
    return f"🚀 Go语言文档链处理: {question}\n这里会连接到Go语言专门的文档检索和回答系统"

def general_docs_chain(question: str) -> str:
    """通用文档处理链"""
    return f"📚 通用文档链处理: {question}\n这里会连接到通用的文档检索和回答系统"

def choose_route(route_result: str, question: str) -> str:
    """
    根据路由结果选择相应的处理链
    
    Args:
        route_result: 路由分析的结果
        question: 原始问题
        
    Returns:
        str: 处理链的输出结果
    """
    # 清理路由结果，移除可能的引号和空格
    route = route_result.strip().lower().replace('"', '').replace("'", "")
    
    if "python_docs" in route:
        return python_docs_chain(question)
    elif "js_docs" in route:
        return js_docs_chain(question)
    elif "golang_docs" in route:
        return golang_docs_chain(question)
    else:
        return general_docs_chain(question)

def main():
    """主函数：演示逻辑路由的完整流程"""
    print("🎯 逻辑路由系统 - DeepSeek版本")
    print("=" * 50)
    
    # 测试用例：包含不同编程语言的查询
    test_questions = [
        # Python相关问题
        """为什么下面的代码不工作：
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
prompt.invoke("french")""",
        
        # JavaScript相关问题
        "如何在JavaScript中使用async/await处理异步操作？",
        
        # Go语言相关问题
        "Go语言中的goroutine和channel是如何工作的？",
        
        # 通用问题
        "什么是微服务架构的优缺点？"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📋 测试用例 {i}:")
        print(f"问题: {question[:50]}{'...' if len(question) > 50 else ''}")
        
        # 第一步：路由分析
        print("\n🔍 第一步：路由分析")
        route_result = route_query(question)
        print(f"路由结果: {route_result}")
        
        # 第二步：选择处理链
        print("\n⚡ 第二步：选择处理链")
        final_result = choose_route(route_result, question)
        print(f"处理结果: {final_result}")
        
        print("-" * 50)

if __name__ == "__main__":
    main()
