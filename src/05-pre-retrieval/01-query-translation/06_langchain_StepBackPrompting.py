"""
使用回溯提示(Step Back Prompting)进行查询翻译
将具体查询转换为更通用的问题，结合两种检索结果提供更全面的答案
"""

import logging
import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings

# 加载.env文件中的环境变量，包括API密钥等敏感信息
load_dotenv()

# 设置日志记录
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

print("\n📚 正在加载文档数据...")
# 使用项目根目录的数据文件
loader = TextLoader("data/txt/糖尿病.txt", encoding="utf-8")
data = loader.load()
print("✅ 文档加载完成")

# 文本分块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
all_splits = text_splitter.split_documents(data)

# 向量存储
print("\n🔤 正在构建向量存储...")
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embed_model)
print("✅ 向量存储构建完成")

# 设置LLM
llm = ChatDeepSeek(
    model="deepseek-chat", temperature=0, api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 创建基础检索器
print("\n🛠️ 正在设置检索器...")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("✅ 检索器设置完成")

def step_back_prompting(question: str, retriever, llm):
    """回溯提示：将具体查询转换为通用查询，结合两种检索结果"""
    
    print("🔄 步骤1: 生成回溯查询...")
    
    # 设计中文的少样本示例
    examples = [
        {
            "input": "胰岛素注射的最佳时间是什么时候？",
            "output": "胰岛素的使用方法和注意事项有哪些？",
        },
        {
            "input": "糖尿病患者可以吃哪些水果？",
            "output": "糖尿病患者的饮食原则是什么？",
        },
        {
            "input": "糖尿病会引起哪些眼部并发症？",
            "output": "糖尿病的并发症有哪些类型？",
        },
    ]
    
    # 创建示例提示模板
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    
    # 创建回溯提示模板
    step_back_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是一个专业的医疗知识专家。你的任务是将具体的医疗问题转换为更通用的、更容易回答的问题。这种转换能帮助我们获得更全面的背景知识。以下是一些示例：""",
            ),
            # 少样本示例
            few_shot_prompt,
            # 新问题
            ("user", "{question}"),
        ]
    )
    
    # 生成回溯查询
    step_back_response = llm.invoke(step_back_prompt.format(question=question))
    step_back_question = step_back_response.content.strip()
    
    print(f"   原始查询: 「{question}」")
    print(f"   回溯查询: 「{step_back_question}」")
    
    print("\n🔍 步骤2: 执行双重检索...")
    
    # 使用原始查询检索
    print("   • 使用原始查询检索...")
    normal_docs = retriever.invoke(question)
    print(f"     检索到 {len(normal_docs)} 个具体相关文档")
    
    # 使用回溯查询检索
    print("   • 使用回溯查询检索...")
    step_back_docs = retriever.invoke(step_back_question)
    print(f"     检索到 {len(step_back_docs)} 个通用背景文档")
    
    print("\n📊 步骤3: 整合上下文信息...")
    
    # 准备上下文内容
    normal_context = "\n".join([doc.page_content for doc in normal_docs])
    step_back_context = "\n".join([doc.page_content for doc in step_back_docs])
    
    print(f"   具体上下文长度: {len(normal_context)} 字符")
    print(f"   通用上下文长度: {len(step_back_context)} 字符")
    
    # 创建最终回答提示模板
    response_prompt_template = """你是一个专业的医疗知识专家。我将向你提出一个问题，你的回答应该全面准确，并充分利用以下两类上下文信息：

## 具体相关信息:
{normal_context}

## 通用背景信息:
{step_back_context}

请基于上述信息回答以下问题。如果上下文信息相关，请充分利用；如果不相关，请忽略。

原始问题: {question}

专业回答:"""

    response_prompt = ChatPromptTemplate.from_template(response_prompt_template)
    
    print("\n💡 步骤4: 生成综合回答...")
    
    # 生成最终回答
    final_prompt = response_prompt.format(
        normal_context=normal_context,
        step_back_context=step_back_context,
        question=question
    )
    
    final_response = llm.invoke(final_prompt)
    
    return {
        "original_question": question,
        "step_back_question": step_back_question,
        "normal_docs": normal_docs,
        "step_back_docs": step_back_docs,
        "final_answer": final_response.content
    }

# 设计测试查询案例
test_question = "糖尿病会引起哪些眼部并发症？"

print(f"\n🔎 测试查询: 「{test_question}」")
print("🚀 开始回溯提示检索...")

# 执行回溯提示
result = step_back_prompting(test_question, retriever, llm)

print(f"\n📄 检索到的具体相关文档:")
for i, doc in enumerate(result["normal_docs"], 1):
    content = doc.page_content.replace('\n', ' ').strip()
    preview = content[:120] + "..." if len(content) > 120 else content
    print(f"[{i}] {preview}")

print(f"\n📄 检索到的通用背景文档:")
for i, doc in enumerate(result["step_back_docs"], 1):
    content = doc.page_content.replace('\n', ' ').strip()
    preview = content[:120] + "..." if len(content) > 120 else content
    print(f"[{i}] {preview}")

print(f"\n🎯 综合回答:")
print("=" * 60)
print(result["final_answer"])
print("=" * 60)
print(f"\n✅ 回溯提示检索完成！")
