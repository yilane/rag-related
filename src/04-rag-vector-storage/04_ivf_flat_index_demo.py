"""
IVF_FLAT索引演示 - 倒排文件索引 + 精确搜索
==========================================

IVF_FLAT索引先将所有向量分成nlist个"堆"（聚类），搜索时只在最相近的几个堆中进行暴力搜索。
这是性能和召回率之间的一个极佳平衡点，适合大多数生产环境使用。

特点：
- 性能和召回率的最佳平衡点
- 通过聚类大幅提高搜索速度 
- nprobe参数可调节精度与速度的权衡
- 适合大多数生产环境使用
"""

from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
import numpy as np

def main():
    # 准备数据
    dimension = 128
    num_vectors = 10000

    # 连接Milvus Lite
    client = MilvusClient("./milvus_ivf_flat_demo.db")
    collection_name = "ivf_flat_demo"

    # 定义集合结构
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50),
    ]
    schema = CollectionSchema(fields, description="IVF_FLAT索引演示")

    # 创建集合
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    client.create_collection(collection_name=collection_name, schema=schema)

    # 创建IVF_FLAT索引
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="IVF_FLAT",      # 使用IVF_FLAT索引
        metric_type="L2",           # L2距离度量
        params={
            "nlist": 128,           # 聚类数量，通常设为sqrt(数据量)
        }
    )
    client.create_index(collection_name=collection_name, index_params=index_params)
    print("✅ IVF_FLAT索引创建完成")

    # 准备测试数据
    categories = ["科技", "健康", "娱乐", "体育", "财经"]
    data_to_insert = []

    print("📝 准备插入数据...")
    for i in range(num_vectors):
        vector = np.random.random(dimension).astype(np.float32)
        category = categories[i % len(categories)]
        data_to_insert.append({
            "vector": vector.tolist(),
            "category": category
        })

    # 批量插入数据
    client.insert(collection_name=collection_name, data=data_to_insert)
    print(f"✅ 插入 {num_vectors} 条数据")

    # 设置搜索参数
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 16}    # 搜索时检查的聚类数量，越大召回率越高但速度越慢
    }

    # 搜索测试
    query_vector = np.random.random(dimension).astype(np.float32)
    search_results = client.search(
        collection_name=collection_name,
        data=[query_vector.tolist()],
        limit=5,
        search_params=search_params,
        output_fields=["category"]
    )

    print(f"\nIVF_FLAT索引搜索结果 (nprobe=16):")
    for i, result in enumerate(search_results[0]):
        print(f"排名{i+1}: ID={result['id']}, 距离={result['distance']:.4f}, 类别={result['entity']['category']}")

    # 测试不同nprobe值的影响
    print(f"\n=== 不同nprobe参数对比 ===")
    nprobe_values = [4, 8, 16, 32]

    for nprobe in nprobe_values:
        search_params["params"]["nprobe"] = nprobe
        results = client.search(
            collection_name=collection_name,
            data=[query_vector.tolist()],
            limit=3,
            search_params=search_params,
            output_fields=["category"]
        )
        
        avg_distance = sum(r['distance'] for r in results[0]) / len(results[0])
        print(f"nprobe={nprobe:2d}: 平均距离={avg_distance:.4f} (nprobe越大，召回率越高)")

    # 条件搜索演示
    print(f"\n=== 结合条件过滤的搜索 ===")
    search_results_filtered = client.search(
        collection_name=collection_name,
        data=[query_vector.tolist()],
        filter="category == '科技'",    # 只搜索科技类别的文档
        limit=3,
        search_params=search_params,
        output_fields=["category"]
    )

    print("科技类别文档的搜索结果:")
    for i, result in enumerate(search_results_filtered[0]):
        print(f"排名{i+1}: ID={result['id']}, 距离={result['distance']:.4f}, 类别={result['entity']['category']}")

    print(f"\nIVF_FLAT索引特点:")
    print("- 性能和召回率的最佳平衡点")
    print("- 通过聚类大幅提高搜索速度") 
    print("- nprobe参数可调节精度与速度的权衡")
    print("- 适合大多数生产环境使用")

    client.close()

if __name__ == "__main__":
    main()