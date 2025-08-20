"""
使用LangChain的RePhraseQueryRetriever进行查询重写对比
展示查询重写对检索效果的改进作用
"""

import logging
import os
from dotenv import load_dotenv
from langchain.retrievers import RePhraseQueryRetriever
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings

# 加载.env文件中的环境变量，包括API密钥等敏感信息
load_dotenv()

# 设置日志记录，可以看到查询重写的具体过程
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("langchain.retrievers.re_phraser")
logger.setLevel(logging.INFO)

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

# 创建重写检索器
print("\n🛠️ 正在设置检索器...")
rewrite_retriever = RePhraseQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}), llm=llm
)
print("✅ 检索器设置完成")

query = "血糖高了怎么办？"

print(f"\n🔎 测试查询: 「{query}」")
print("🔄 正在进行查询重写和检索...")

# 使用RePhraseQueryRetriever进行查询重写和检索
docs = rewrite_retriever.invoke(query)

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

print(f"\n✅ 查询重写检索完成！检索到 {len(docs) if isinstance(docs, list) else 0} 个相关文档")
