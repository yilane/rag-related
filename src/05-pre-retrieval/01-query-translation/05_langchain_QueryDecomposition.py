"""
使用查询分解技术处理复杂查询
将复杂查询拆分成多个子问题，提供更全面的检索结果
"""

import logging
import os
from dotenv import load_dotenv
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
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("✅ 检索器设置完成")

def decompose_query(complex_query, llm):
    """将复杂查询分解为多个子问题"""
    decompose_prompt = f"""
请将以下复杂查询分解为3-4个相互独立的子问题，每个子问题都应该是可以单独回答的简单问题。

复杂查询: {complex_query}

分解要求:
1. 每个子问题都应该相互独立
2. 子问题组合起来应该能完整回答原始查询
3. 每个子问题应该简洁明确
4. 每个子问题一行，不要编号

分解后的子问题:
    """
    
    response = llm.invoke(decompose_prompt)
    sub_queries = [line.strip() for line in response.content.strip().split('\n') if line.strip()]
    sub_queries = [q.lstrip('0123456789. ').rstrip('？?') + '？' for q in sub_queries if q.strip()]
    
    return sub_queries

def query_decomposition_retrieve(complex_query, retriever, llm):
    """查询分解检索：复杂查询分解 + 子问题检索 + 结果整合"""
    
    print("🔄 步骤1: 分解复杂查询...")
    sub_queries = decompose_query(complex_query, llm)
    print(f"   将复杂查询分解为 {len(sub_queries)} 个子问题:")
    for i, sub_q in enumerate(sub_queries, 1):
        print(f"   [{i}] {sub_q}")
    
    print("\n🔍 步骤2: 对各子问题进行检索...")
    all_docs = []
    all_docs_content = set()  # 用于去重
    
    for i, sub_query in enumerate(sub_queries, 1):
        print(f"\n   子问题{i}: {sub_query}")
        docs = retriever.invoke(sub_query)
        print(f"   检索到 {len(docs)} 个相关文档")
        
        # 去重添加文档
        new_docs = 0
        for doc in docs:
            if doc.page_content not in all_docs_content:
                all_docs.append(doc)
                all_docs_content.add(doc.page_content)
                new_docs += 1
        print(f"   新增 {new_docs} 个独特文档")
    
    print(f"\n📊 步骤3: 整合检索结果...")
    print(f"   总共收集到 {len(all_docs)} 个相关文档")
    print(f"   涵盖了 {len(sub_queries)} 个不同维度的信息")
    
    return all_docs[:5]  # 返回前5个最相关的文档

# 设计一个复杂查询案例，体现分解的价值
complex_query = "糖尿病患者应该如何控制血糖水平，预防哪些并发症，以及日常饮食需要注意什么？"

print(f"\n🔎 复杂查询: 「{complex_query}」")
print("🚀 开始查询分解检索...")

# 使用查询分解进行检索
docs = query_decomposition_retrieve(complex_query, base_retriever, llm)

print(f"\n📄 查询分解检索结果:")
if isinstance(docs, list):
    for i, doc in enumerate(docs, 1):
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        preview = content.replace('\n', ' ').strip()
        if len(preview) > 150:
            preview = preview[:150] + "..."
        print(f"[{i}] {preview}")
else:
    print("❌ 未找到相关文档")

print(f"\n✅ 查询分解检索完成！从 {len(docs) if isinstance(docs, list) else 0} 个文档中获得全面信息")
