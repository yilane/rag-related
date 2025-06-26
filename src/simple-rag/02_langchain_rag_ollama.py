# 1. 加载文档
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(web_paths=("https://zh.wikipedia.org/wiki/深度求索",))
docs = loader.load()

# 2. 文档分块
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 每个文本块的最大字符数
    chunk_overlap=200,  # 相邻文本块之间的重叠字符数
)
all_splits = text_splitter.split_documents(docs)

# 3. 设置嵌入模型
from langchain_huggingface import HuggingFaceEmbeddings

# 初始化中文嵌入模型
# 使用BAAI/bge-small-zh模型，这是一个专门针对中文优化的嵌入模型
# model_kwargs指定使用CPU进行计算，适合没有GPU的环境
# encode_kwargs中的normalize_embeddings=True确保嵌入向量被归一化，提高检索精度
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh",  # 中文嵌入模型，体积小但效果良好
    model_kwargs={"device": "cpu"},   # 使用CPU设备进行计算
    encode_kwargs={"normalize_embeddings": True},  # 启用向量归一化
)

# 4. 创建向量存储
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)
# 将分割后的文档添加到向量存储中
# 这一步会将所有文档块转换为向量并存储在内存中，用于后续的相似性搜索
vector_store.add_documents(all_splits)

# 5. 构建用户查询
question = "DeepSeek核心技术是什么？"

# 6. 在向量存储中搜索相关文档，并准备上下文内容
retrieved_docs = vector_store.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

# 7. 构建提示模板
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    """
                基于以下上下文，回答问题。如果上下文中没有相关信息，
                请说"我无法从提供的上下文中找到相关信息"。
                上下文: {context}
                问题: {question}
                回答:"""
)

# 8. 使用ollama本地大语言模型生成答案
from langchain_ollama import ChatOllama # pip install langchain-ollama
llm = ChatOllama(
    model="qwen3:8b",  # 可以根据需要更换其他模型，如 llama2, mistral 等
    request_timeout=300.0  # 增加超时时间
)
answer = llm.invoke(prompt.format(question=question, context=docs_content))

# 格式化输出答案
print("=" * 80)
print(f"📝 问题: {question}")
print("-" * 80)
print("💡 答案:")
print(answer.content if hasattr(answer, "content") else str(answer))
print("-" * 80)
print("📚 参考文档数量:", len(retrieved_docs))
print("🔍 检索到的相关片段:")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"\n片段 {i}:")
    print(
        f"  内容: {doc.page_content[:200]}{'...' if len(doc.page_content) > 200 else ''}"
    )
    if hasattr(doc, "metadata") and doc.metadata:
        print(f"  来源: {doc.metadata}")
print("=" * 80)
