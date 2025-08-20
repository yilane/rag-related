"""
使用RAG-Fusion进行增强型多查询检索
在多查询基础上增加智能排序，提供最优化的检索结果
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

# 创建RAG-Fusion检索器（基于MultiQueryRetriever的增强版）
print("\n🛠️ 正在设置检索器...")

class RAGFusionRetriever(MultiQueryRetriever):
    """RAG-Fusion检索器：继承MultiQueryRetriever并添加RRF排序"""
    
    def reciprocal_rank_fusion(self, results: list, k=60):
        """RRF算法：对多个检索结果进行融合排序"""
        fused_scores = {}
        
        # 遍历每个查询的检索结果
        for docs in results:
            # 遍历文档及其排名位置
            for rank, doc in enumerate(docs):
                # 将文档转换为字符串作为键
                doc_str = doc.page_content
                # 如果文档不在融合分数字典中，添加初始分数0
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = {"score": 0, "doc": doc}
                # 使用RRF公式更新分数: 1 / (rank + k)
                fused_scores[doc_str]["score"] += 1 / (rank + k)
        
        # 按融合分数降序排序
        sorted_results = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_results]
    
    def _get_relevant_documents(self, query: str, **kwargs):
        """重写检索方法，增加RRF融合排序"""
        print("🔄 步骤1: MultiQueryRetriever生成多个查询...")
        
        # 使用父类的完整检索逻辑先获取所有查询的结果
        # 但我们需要分步骤来展示过程，所以直接调用父类方法获取结果
        from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
        run_manager = CallbackManagerForRetrieverRun.get_noop_manager()
        
        # 生成多个查询
        queries = self.generate_queries(query, run_manager)
        print(f"   生成了 {len(queries)} 个查询变体:")
        for i, q in enumerate(queries, 1):
            print(f"   [{i}] {q}")
        
        print("\n🔍 步骤2: 执行多查询检索...")
        # 对每个查询分别检索
        all_results = []
        for i, q in enumerate(queries, 1):
            docs = self.retriever.get_relevant_documents(q)
            print(f"   查询{i}检索到 {len(docs)} 个文档")
            all_results.append(docs)
        
        print("\n📊 步骤3: RRF融合排序...")
        # 使用RRF算法融合排序
        if all_results:
            fused_docs = self.reciprocal_rank_fusion(all_results, k=60)
            print(f"   RRF融合完成，返回前3个最相关文档")
            return fused_docs[:3]
        
        return []

# 创建RAG-Fusion检索器
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
rag_fusion_retriever = RAGFusionRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)
print("✅ RAG-Fusion检索器设置完成")

query = "糖尿病的并发症有哪些？"

print(f"\n🔎 测试查询: 「{query}」")
print("🚀 开始RAG-Fusion检索...")

# 使用RAG-Fusion检索器进行检索
docs = rag_fusion_retriever.invoke(query)

print(f"\n📄 RAG-Fusion检索结果:")
if isinstance(docs, list):
    for i, doc in enumerate(docs, 1):
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        preview = content.replace('\n', ' ').strip()
        if len(preview) > 150:
            preview = preview[:150] + "..."
        print(f"[{i}] {preview}")
else:
    print("❌ 未找到相关文档")

print(f"\n✅ RAG-Fusion检索完成！使用RRF算法返回 {len(docs) if isinstance(docs, list) else 0} 个最相关文档")

