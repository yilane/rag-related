"""
FLAT索引演示 - 精确暴力搜索
==============================

FLAT索引是最简单直接的向量索引方法，每次搜索都会对所有向量进行暴力比较。
虽然搜索速度相对较慢，但能保证100%的召回率，绝不会漏掉正确答案。

特点：
- 100%精确召回率
- 搜索速度相对较慢，适合小规模数据
- 无需训练，即插即用
- 内存占用相对较小
"""

from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
import numpy as np
import random

def main():
    # 准备数据
    dimension = 128
    num_vectors = 1000

    # 连接Milvus Lite
    client = MilvusClient("./milvus_flat_demo.db")
    collection_name = "flat_index_demo"

    # 定义集合结构
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=100),
    ]
    schema = CollectionSchema(fields, description="FLAT索引演示")

    # 创建集合
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    client.create_collection(collection_name=collection_name, schema=schema)

    # 创建FLAT索引 (FLAT = 精确暴力搜索)
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="FLAT",           # 使用FLAT索引类型
        metric_type="L2",            # L2距离（欧几里得距离）
        params={}                    # FLAT索引无需额外参数
    )
    client.create_index(collection_name=collection_name, index_params=index_params)
    print("✅ FLAT索引创建完成")

    # 准备测试数据
    data_to_insert = []
    for i in range(num_vectors):
        vector = np.random.random(dimension).astype(np.float32)
        data_to_insert.append({
            "vector": vector.tolist(),
            "text": f"文档{i}"
        })

    # 批量插入数据
    client.insert(collection_name=collection_name, data=data_to_insert)
    print(f"✅ 插入 {num_vectors} 条数据")

    # 搜索测试
    query_vector = np.random.random(dimension).astype(np.float32)
    search_results = client.search(
        collection_name=collection_name,
        data=[query_vector.tolist()],
        limit=5,                     # 返回前5个最相似的结果
        output_fields=["text"]       # 返回文本字段
    )

    print("\nFLAT索引搜索结果:")
    for i, result in enumerate(search_results[0]):
        print(f"排名{i+1}: ID={result['id']}, 距离={result['distance']:.4f}, 文本={result['entity']['text']}")

    print(f"\nFLAT索引特点:")
    print("- 100%精确召回率，绝不漏掉正确答案")
    print("- 搜索速度相对较慢，适合小规模数据")
    print("- 无需训练，即插即用")
    print("- 内存占用相对较小")

    client.close()

if __name__ == "__main__":
    main()