"""
假设文档嵌入（HyDE - Hypothetical Document Embedding）
先由大模型生成假设答案，再用假设答案的嵌入去检索真实文档，提高检索精准度
"""

import logging
import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
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

def hyde_retrieval(question: str, retriever, llm):
    """
    假设文档嵌入（HyDE）检索
    先生成假设答案，再用假设答案检索真实文档
    """
    
    print("🎯 步骤1: 生成假设文档...")
    
    # HyDE 假设文档生成提示模板
    hyde_template = """请基于以下问题生成一个详细、专业的假设答案。这个假设答案将用于文档检索，因此需要包含可能的关键词和相关概念。

问题: {question}

请生成一个假设的专业回答（不要说"假设"或"可能"等不确定词汇，直接给出肯定的回答）:"""

    hyde_prompt = ChatPromptTemplate.from_template(hyde_template)
    
    # 生成假设文档
    hypothetical_response = llm.invoke(hyde_prompt.format(question=question))
    hypothetical_document = hypothetical_response.content.strip()
    
    print(f"   原始问题: 「{question}」")
    print(f"   假设答案长度: {len(hypothetical_document)} 字符")
    print(f"   假设答案预览: 「{hypothetical_document[:100]}...」")
    
    print("\n🔍 步骤2: 使用假设文档检索相关文档...")
    
    # 使用假设文档进行检索
    retrieved_docs = retriever.invoke(hypothetical_document)
    print(f"   检索到 {len(retrieved_docs)} 个相关文档")
    
    print("\n📋 步骤3: 基于检索文档生成最终答案...")
    
    # 准备检索到的上下文
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    print(f"   上下文总长度: {len(context)} 字符")
    
    # 最终答案生成提示模板
    final_template = """你是一个专业的医疗知识专家。请基于以下检索到的相关文档信息，准确回答用户的问题。

相关文档信息:
{context}

用户问题: {question}

请基于上述文档信息提供准确、专业的回答。如果文档信息不足以完全回答问题，请明确说明：

专业回答:"""

    final_prompt = ChatPromptTemplate.from_template(final_template)
    
    # 生成最终回答
    final_response = llm.invoke(final_prompt.format(
        context=context,
        question=question
    ))
    
    return {
        "original_question": question,
        "hypothetical_document": hypothetical_document,
        "retrieved_docs": retrieved_docs,
        "final_answer": final_response.content
    }

def compare_retrieval_methods(question: str, retriever, llm):
    """
    比较传统检索与HyDE检索的效果
    """
    print(f"\n🆚 检索方法对比分析")
    print("=" * 60)
    
    print("\n🔍 方法1: 传统直接检索")
    print("-" * 30)
    
    # 传统检索
    traditional_docs = retriever.invoke(question)
    print(f"检索到 {len(traditional_docs)} 个文档")
    
    for i, doc in enumerate(traditional_docs, 1):
        content = doc.page_content.replace('\n', ' ').strip()
        preview = content[:80] + "..." if len(content) > 80 else content
        print(f"[{i}] {preview}")
    
    print("\n🎯 方法2: HyDE假设文档检索")
    print("-" * 30)
    
    # HyDE检索
    hyde_result = hyde_retrieval(question, retriever, llm)
    
    for i, doc in enumerate(hyde_result["retrieved_docs"], 1):
        content = doc.page_content.replace('\n', ' ').strip()
        preview = content[:80] + "..." if len(content) > 80 else content
        print(f"[{i}] {preview}")
    
    return {
        "traditional_docs": traditional_docs,
        "hyde_result": hyde_result
    }

# 设计测试查询案例
test_questions = [
    "糖尿病患者应该如何控制血糖？",
    "胰岛素的作用机制是什么？",
    "糖尿病会引起哪些并发症？"
]

print(f"\n🚀 开始HyDE检索测试")
print("=" * 60)

for i, question in enumerate(test_questions, 1):
    print(f"\n🧪 测试案例 {i}/{len(test_questions)}")
    print(f"🔎 测试问题: 「{question}」")
    print("=" * 50)
    
    # 执行HyDE检索
    result = hyde_retrieval(question, retriever, llm)
    
    print(f"\n📄 检索到的相关文档:")
    for j, doc in enumerate(result["retrieved_docs"], 1):
        content = doc.page_content.replace('\n', ' ').strip()
        preview = content[:100] + "..." if len(content) > 100 else content
        print(f"[{j}] {preview}")
    
    print(f"\n🎯 HyDE最终答案:")
    print("-" * 40)
    print(result["final_answer"])
    print("-" * 40)
    
    # 如果不是最后一个案例，等待用户输入
    if i < len(test_questions):
        print(f"\n⏸️  按回车键继续下一个测试案例...")
        input()

print(f"\n📊 详细对比测试")
print("=" * 60)

# 选择一个案例进行详细对比
comparison_question = "糖尿病患者应该如何控制血糖？"
print(f"🔎 对比问题: 「{comparison_question}」")

comparison_result = compare_retrieval_methods(comparison_question, retriever, llm)

print(f"\n🎯 HyDE方法的最终答案:")
print("=" * 50)
print(comparison_result["hyde_result"]["final_answer"])
print("=" * 50)
print(f"\n✅ HyDE检索测试完成！")
