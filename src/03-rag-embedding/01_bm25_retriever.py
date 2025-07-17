"""
BM25检索器
使用BM25算法对文档进行检索，并返回top-k个结果
"""

from rank_bm25 import BM25Okapi  # pip install rank-bm25
import jieba  # pip install jieba
from typing import List, Tuple


class BM25Retriever:

    def __init__(self, documents: List[str]):
        """
        初始化BM25检索器
        :param documents: 文档列表
        """
        self.documents = documents
        # 对每个文档进行分词
        self.tokenized_corpus = [list(jieba.cut(doc)) for doc in documents]
        # 构建BM25模型
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        搜索相关文档
        :param query: 查询字符串
        :param top_k: 返回top-k个结果
        :return: (文档, 分数)的列表
        """
        # 对查询进行分词
        tokenized_query = list(jieba.cut(query))
        # 计算每个文档的BM25分数
        scores = self.bm25.get_scores(tokenized_query)
        # 获取top-k结果
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 只返回有得分的结果
                results.append((self.documents[idx], scores[idx]))

        return results


def main():
    # 示例文档库
    documents = [
        "自动驾驶技术是未来汽车行业的重要发展方向，涉及人工智能、机器学习等多个领域。",
        "人工智能在医疗诊断中的应用越来越广泛，可以帮助医生更准确地诊断疾病。",
        "机器学习算法在金融风控中发挥着重要作用，能够识别潜在的风险。",
        "深度学习是机器学习的一个重要分支，在图像识别、自然语言处理等领域表现出色。",
        "自动驾驶汽车需要强大的计算能力和先进的传感器技术来感知周围环境。",
        "量子计算有望在未来解决一些传统计算机难以处理的复杂问题。",
    ]

    # 创建BM25检索器
    retriever = BM25Retriever(documents)

    # 进行搜索
    query = "无人驾驶汽车怎么样？"
    results = retriever.search(query, top_k=3)

    print(f"查询: {query}")
    print("BM25搜索结果:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. [分数: {score:.4f}] {doc}")


if __name__ == "__main__":
    main()
