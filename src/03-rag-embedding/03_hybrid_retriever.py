"""
混合检索器
结合BM25稀疏检索和密集嵌入检索，实现混合检索
"""

# 导入依赖
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from typing import List, Tuple, Dict

# 导入已完成的检索器
import importlib.util

# 动态导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))

# 导入BM25检索器
bm25_spec = importlib.util.spec_from_file_location(
    "bm25_retriever", os.path.join(current_dir, "01_bm25_retriever.py")
)
bm25_module = importlib.util.module_from_spec(bm25_spec)
bm25_spec.loader.exec_module(bm25_module)
BM25Retriever = bm25_module.BM25Retriever

# 导入密集嵌入检索器
dense_spec = importlib.util.spec_from_file_location(
    "dense_retriever", os.path.join(current_dir, "02_dense_embedding_retriever.py")
)
dense_module = importlib.util.module_from_spec(dense_spec)
dense_spec.loader.exec_module(dense_module)
DenseEmbeddingRetriever = dense_module.DenseEmbeddingRetriever


class HybridRetriever:
    def __init__(self, documents: List[str], model_name: str = "BAAI/bge-m3"):
        print("正在初始化混合检索器...")
        self.bm25_retriever = BM25Retriever(documents)
        self.dense_retriever = DenseEmbeddingRetriever(documents, model_name)
        self.documents = documents
        print("混合检索器初始化完成。")

    def search(self, query: str, top_k: int = 3, k_rrf: int = 60) -> List[Tuple[str, float]]:
        # 1. 从每个检索器获取结果
        # 为了RRF能更好地工作，我们从每个检索器获取更多的结果
        bm25_results = self.bm25_retriever.search(query, top_k=len(self.documents))
        dense_results = self.dense_retriever.search(query, top_k=len(self.documents))

        # 2. RRF融合
        rrf_scores = {}
        
        # 处理BM25结果
        for rank, (doc, _) in enumerate(bm25_results):
            if doc not in rrf_scores:
                rrf_scores[doc] = 0
            rrf_scores[doc] += 1 / (k_rrf + rank + 1) # rank从0开始

        # 处理密集搜索结果
        for rank, (doc, _) in enumerate(dense_results):
            if doc not in rrf_scores:
                rrf_scores[doc] = 0
            rrf_scores[doc] += 1 / (k_rrf + rank + 1)

        # 3. 排序并返回top-k
        sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        
        return sorted_docs[:top_k]

# --- 主函数 ---
def main():
    documents = [
        "自动驾驶技术是未来汽车行业的重要发展方向，涉及人工智能、机器学习等多个领域。",
        "人工智能在医疗诊断中的应用越来越广泛，可以帮助医生更准确地诊断疾病。",
        "机器学习算法在金融风控中发挥着重要作用，能够识别潜在的风险。",
        "深度学习是机器学习的一个重要分支，在图像识别、自然语言处理等领域表现出色。",
        "自动驾驶汽车需要强大的计算能力和先进的传感器技术来感知周围环境。",
        "量子计算有望在未来解决一些传统计算机难以处理的复杂问题。",
    ]

    hybrid_retriever = HybridRetriever(documents)

    query = "无人驾驶汽车怎么样？"
    results = hybrid_retriever.search(query, top_k=3)

    print(f"\n查询: {query}")
    print("混合搜索 (RRF) 结果:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. [RRF分数: {score:.6f}] {doc}")

if __name__ == "__main__":
    main()

