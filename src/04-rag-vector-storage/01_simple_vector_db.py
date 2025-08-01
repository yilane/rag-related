# 需要安装的依赖包
# pip install faiss-cpu sentence-transformers numpy
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class SimpleVectorDB:

    def __init__(self):
        """
        初始化向量数据库
        """
        # 这个数组用来保存原始文本，检索时要返回给用户看
        # 我刚开始忘了这步，结果只能返回向量，用户根本看不懂 
        self.texts = []
        
        # 加载预训练模型，这里用的是中文优化的BGE模型
        # 选这个是因为对中文支持比较好，你也可以试试其他的
        self.model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        
        # 获取模型的向量维度
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # 创建Faiss索引 - 使用L2距离（欧几里得距离）
        self.index = faiss.IndexFlatL2(self.dimension)

    def add_documents(self, documents):
        """
        步骤1: 批量将文本转换为向量，并添加到向量数据库
        """
        print("=== 开始向量化 ===")
        vectors = []

        for i, doc in enumerate(documents):
            print(f"正在处理第 {i+1} 个文档（总共{len(documents)}个）:")
            # 这一步就是把文字"翻译"成数字的过程
            # 第一次看到一句话变成512个小数点，还挺震撼的
            vector = self.model.encode([doc])
            print(f"原文: '{doc}'")
            print(f"向量维度: {vector.shape} - 一句话变成了{vector.shape[1]}个数字！")
            print(f"向量前10个值: {vector[0][:10]} - 看起来很随机，但其实包含了语义信息")
            print(f"向量数据类型: {vector.dtype}")
            print("-" * 50)
            vectors.append(vector[0])
            self.texts.append(doc)

        # 转换为numpy数组（Faiss要求的格式）
        vectors_array = np.array(vectors, dtype=np.float32)
        print(f"所有向量的形状: {vectors_array.shape}")
        print(f"向量矩阵大小: {vectors_array.nbytes} 字节")
		
        """
        步骤2 索引构建，添加到Faiss索引中
        """
        self.index.add(vectors_array)
        print(f"索引中的向量数量: {self.index.ntotal}")
        return vectors_array

    def search(self, query, k=3):
        """
        步骤3: 向量检索
        """
        print(f"\n=== 向量检索过程 ===")
        print(f"查询: '{query}'")

        # 将查询文本转换为向量
        query_vector = self.model.encode([query])[0]
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        
        # 在索引中搜索最相似的k个向量
        distances, indices = self.index.search(query_vector, k)

        print(f"搜索结果:")
        results = []
        for i in range(k):
            idx = indices[0][i]
            distance = distances[0][i]
            similarity = 1 / (1 + distance)  # 转换距离为相似度分数
            results.append(
                {
                    "text": self.texts[idx],
                    "distance": float(distance),
                    "similarity": float(similarity),
                    "index": int(idx),
                }
            )
            print(
                f"  {i+1}. 相似度: {similarity:.4f} | 距离: {distance:.4f} | 文本: '{self.texts[idx]}'"
            )

        return results

    def show_storage_details(self):
        """
        展示存储细节
        """
        print(f"\n=== 向量存储详情 ===")
        print(f"索引类型: {type(self.index).__name__}")
        print(f"向量维度: {self.dimension}")
        print(f"存储的向量数量: {self.index.ntotal}")
        print(
            f"索引大小估算: {self.index.ntotal * self.dimension * 4} 字节"
        )  # float32 = 4字节

        # 展示向量在内存中的存储形式
        if self.index.ntotal > 0:
            # 重构第一个向量来展示存储形式
            first_vector = self.index.reconstruct(0)
            print(f"第一个向量的存储形式:")
            print(f"  类型: {type(first_vector)}")
            print(f"  形状: {first_vector.shape}")
            print(f"  前5个值: {first_vector[:5]}")
            print(f"  后5个值: {first_vector[-5:]}")


def main():
    # 示例文档集合
    documents = [
        "苹果是一种营养丰富的水果",
        "Python是一种编程语言",
        "机器学习是人工智能的重要分支",
        "香蕉含有丰富的钾元素",
        "深度学习使用神经网络进行训练",
        "橙子富含维生素C",
        "自然语言处理是AI的应用领域",
    ]

    # 创建向量数据库
    vector_db = SimpleVectorDB()
    # 存储文档
    vectors = vector_db.add_documents(documents)
    # 显示存储详情
    vector_db.show_storage_details()
    # 执行检索
    queries = ["什么水果比较健康？", "编程相关的内容", "人工智能技术"]
    for query in queries:
        vector_db.search(query, k=3)

if __name__ == "__main__":
    main()