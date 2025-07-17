"""
密集嵌入检索器
使用BGE-M3模型对文档进行密集嵌入编码，并计算查询与文档的余弦相似度
"""

# pip install FlagEmbedding numpy
from FlagEmbedding import BGEM3FlagModel
import numpy as np
from typing import List, Tuple


class DenseEmbeddingRetriever:

    def __init__(self, documents: List[str], model_name: str = "BAAI/bge-m3"):
        self.documents = documents
        self.model = BGEM3FlagModel(model_name, use_fp16=True)  # 使用半精度以加速
        print("正在对文档进行嵌入编码...")
        # 对所有文档进行编码，并存储为numpy数组
        self.doc_embeddings = self.model.encode(documents, return_dense=True)[
            "dense_vecs"
        ]
        print("文档嵌入完成。")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        # 对查询进行编码
        query_embedding = self.model.encode([query], return_dense=True)["dense_vecs"]
        # 计算余弦相似度
        similarities = np.dot(query_embedding, self.doc_embeddings.T).flatten()

        # 获取top-k结果的索引
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(self.documents[i], float(similarities[i])) for i in top_indices]


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

    retriever = DenseEmbeddingRetriever(documents)

    # 使用和BM25相同的查询
    query = "无人驾驶汽车怎么样？"
    results = retriever.search(query, top_k=3)

    print(f"\n查询: {query}")
    print("密集嵌入搜索结果:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. [分数: {score:.4f}] {doc}")


if __name__ == "__main__":
    main()
