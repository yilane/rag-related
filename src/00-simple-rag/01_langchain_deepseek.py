"""
LangChain RAG智能问答系统 - DeepSeek版本
使用传统的LangChain链式调用实现检索增强生成(RAG)
包含文档加载、向量化、检索、生成等完整流程
"""
# 第一步：索引阶段

# 1. 加载文档
# 导入必要的模块和环境变量配置
import os
from dotenv import load_dotenv

# 加载.env文件中的环境变量，包括API密钥等敏感信息
load_dotenv()

# 导入网页文档加载器，用于从网页爬取内容
from langchain_community.document_loaders import WebBaseLoader

# 创建网页加载器实例，指定要爬取的URL
loader = WebBaseLoader(web_paths=("https://zh.wikipedia.org/wiki/深度求索",))  # 深度求索的维基百科页面
docs = loader.load()  # 执行加载操作，返回Document对象列表

# 2. 文本分块
# 导入递归字符文本分割器，用于将长文本切分成小块
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 创建文本分割器实例
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 每个文本块的最大字符数（平衡内容完整性和处理效率）
    chunk_overlap=200,  # 相邻文本块之间的重叠字符数（确保信息不丢失）
)
# 将加载的文本分割成多个小块，便于向量化和检索
all_splits = text_splitter.split_documents(docs)

# 3. 信息嵌入
# 导入HuggingFace嵌入模型接口
from langchain_huggingface import HuggingFaceEmbeddings

# 初始化中文嵌入模型
# 使用BAAI/bge-small-zh模型，这是一个专门针对中文优化的嵌入模型
# model_kwargs指定使用CPU进行计算，适合没有GPU的环境
# encode_kwargs中的normalize_embeddings=True确保嵌入向量被归一化，提高检索精度
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh",  # 中文嵌入模型，体积小但效果良好
    model_kwargs={"device": "cpu"},   # 使用CPU设备进行计算（可改为"cuda"使用GPU）
    encode_kwargs={"normalize_embeddings": True},  # 启用向量归一化，提升相似度计算准确性
)

# 4. 向量存储
# 导入内存向量存储，用于存储和检索文档向量
from langchain_core.vectorstores import InMemoryVectorStore

# 创建向量存储实例，传入嵌入模型
vector_store = InMemoryVectorStore(embeddings)
# 将分割后的文档添加到向量存储中
# 这一步会将所有文档块转换为向量并存储在内存中，用于后续的相似性搜索
vector_store.add_documents(all_splits)

# 第二步：检索阶段

# 5. 构建用户查询
# 定义要询问的问题（这里可以改为动态输入）
question = "DeepSeek有哪些核心技术？"

# 6. 在向量存储中搜索相关文档，并准备上下文内容
# 使用余弦相似度搜索最相关的文档块
retrieved_docs = vector_store.similarity_search(question, k=3)  # k=3表示检索前3个最相关的文档块
# 将检索到的文档内容拼接成一个字符串，作为大模型的上下文
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

# 7. 构建提示模板
# 导入聊天提示模板，用于格式化用户问题和检索到的上下文
from langchain_core.prompts import ChatPromptTemplate

# 创建提示模板，定义大模型的角色和任务
prompt = ChatPromptTemplate.from_template(
    """基于以下上下文，请详细回答问题。如果上下文中没有相关信息，
请说"我无法从提供的上下文中找到相关信息"。

上下文: {context}

问题: {question}

回答:"""
)

# 第三步：生成阶段

# 8. 使用大语言模型生成答案
# 导入DeepSeek聊天模型
from langchain_deepseek import ChatDeepSeek  # 需要安装: pip install langchain-deepseek

# 初始化DeepSeek聊天模型
llm = ChatDeepSeek(
    model="deepseek-chat",  # DeepSeek API 支持的聊天模型名称
    temperature=0.7,  # 控制输出的随机性（0-1之间，越高越随机，越低越确定）
    max_tokens=2048,  # 最大输出token数量，控制回答长度
    api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量加载API密钥
)

# 使用提示模板格式化问题和上下文，然后调用大模型生成答案
formatted_prompt = prompt.format(question=question, context=docs_content)
answer = llm.invoke(formatted_prompt)

# 格式化输出答案
print("=" * 80)
print("🤖 LangChain RAG 智能问答系统 (DeepSeek版本)")
print("=" * 80)
print(f"📝 问题: {question}")
print("-" * 80)
print("💡 答案:")
# 安全地提取答案内容，兼容不同的返回格式
print(answer.content if hasattr(answer, "content") else str(answer))
print("-" * 80)
print("📚 参考文档数量:", len(retrieved_docs))
print("🔍 检索到的相关片段:")

# 逐个显示检索到的文档片段，便于用户了解答案来源
for i, doc in enumerate(retrieved_docs, 1):
    print(f"\n片段 {i}:")
    # 截取前200个字符显示，避免输出过长
    content_preview = doc.page_content[:200]
    if len(doc.page_content) > 200:
        content_preview += "..."
    print(f"  内容: {content_preview}")
    
    # 如果文档有元数据（如来源URL），则显示
    if hasattr(doc, "metadata") and doc.metadata:
        print(f"  来源: {doc.metadata}")

print("=" * 80)
