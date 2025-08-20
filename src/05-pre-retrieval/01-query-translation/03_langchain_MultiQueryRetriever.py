"""
使用LangChain的MultiQueryRetriever进行多查询生成
展示将单个查询扩展为多个相关查询的效果
"""

import logging
import os
from dotenv import load_dotenv
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings

# 加载.env文件中的环境变量，包括API密钥等敏感信息
load_dotenv()

# 设置日志记录，查看多查询生成过程
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
multi_query_logger = logging.getLogger("langchain.retrievers.multi_query")
multi_query_logger.setLevel(logging.DEBUG)

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
    model="deepseek-chat", temperature=0.1, api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 创建多查询检索器
print("\n🛠️ 正在设置检索器...")
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}), llm=llm
)
print("✅ 检索器设置完成")

query = "糖尿病有什么症状？"

print(f"\n🔎 测试查询: 「{query}」")
print("🔄 正在生成多个查询并检索...")

# 使用MultiQueryRetriever进行多查询生成和检索
docs = multi_query_retriever.invoke(query)

print(f"\n📄 检索到的文档片段:")
if isinstance(docs, list):
    for i, doc in enumerate(docs, 1):
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        preview = content.replace('\n', ' ').strip()
        if len(preview) > 150:
            preview = preview[:150] + "..."
        print(f"[{i}] {preview}")
else:
    print("❌ 未找到相关文档")

print(f"\n✅ 多查询检索完成！检索到 {len(docs) if isinstance(docs, list) else 0} 个相关文档")