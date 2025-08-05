"""
Milvus向量数据库示例 - 医疗病例检索系统
===========================================

本示例演示如何使用Milvus向量数据库构建医疗病例检索系统：
1. 创建医疗病例数据
2. 使用BGE嵌入模型进行向量化
3. 在Milvus中存储和检索向量数据
4. 支持语义搜索和条件查询
"""

# 准备示例数据集
import pandas as pd

# 定义医疗病例示例数据
data_records = [
    {
        "case_id": "MD001",
        "patient_name": "张三",
        "disease_name": "心肌梗死",
        "department": "心脏内科",
        "severity": "危重",
        "aliases": "心脏病发作, 心梗",
        "case_description": "患者因急性胸痛入院，心电图显示ST段抬高，诊断为急性心肌梗死。",
    },
    {
        "case_id": "MD002",
        "patient_name": "李四",
        "disease_name": "普通感冒",
        "department": "呼吸内科",
        "severity": "轻微",
        "aliases": "上呼吸道感染, 伤风",
        "case_description": "患者出现流涕、鼻塞、轻微咳嗽等症状，无发热，诊断为普通感冒。",
    },
]
df = pd.DataFrame(data_records)

# ==================== Milvus数据库连接 ====================
# 导入Milvus相关模块
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema

# 设置数据库路径和连接客户端
# Milvus支持本地文件存储，这里使用本地数据库文件
db_path = "./milvus_demo.db"
client = MilvusClient(db_path)

# 定义集合名称（相当于关系数据库中的表名）
collection_name = "medical_cases"
# ==================== 嵌入模型初始化 ====================
# 导入Milvus内置的SentenceTransformer嵌入函数
# 这个函数封装了sentence-transformers库，简化了向量化过程
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction

embedding_function = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-zh-v1.5")
# 获取模型的向量维度，用于定义Milvus集合的向量字段
# 通过编码示例文本来确定向量维度
sample_embedding = embedding_function(["示例文本"])[0]
vector_dim = len(sample_embedding)

# ==================== 集合模式定义 ====================
# 定义Milvus集合的字段结构，类似于关系数据库的表结构
fields = [
    # 主键字段：自动递增的整数ID
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    # 向量字段：存储文本的向量表示，维度由嵌入模型决定
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
    # 标量字段：存储医疗病例的各个属性
    FieldSchema(name="case_id", dtype=DataType.VARCHAR, max_length=50),      # 病例ID
    FieldSchema(name="patient_name", dtype=DataType.VARCHAR, max_length=100), # 患者姓名
    FieldSchema(name="disease_name", dtype=DataType.VARCHAR, max_length=100), # 疾病名称
    FieldSchema(name="department", dtype=DataType.VARCHAR, max_length=100),   # 科室
    FieldSchema(name="severity", dtype=DataType.VARCHAR, max_length=20),      # 病情严重程度
    FieldSchema(name="aliases", dtype=DataType.VARCHAR, max_length=200),      # 疾病别名
    FieldSchema(name="case_description", dtype=DataType.VARCHAR, max_length=500), # 病例描述
]
schema = CollectionSchema(fields, description="medical_cases", enable_dynamic_field=True)

if not client.has_collection(collection_name):
    client.create_collection(collection_name=collection_name, schema=schema)
    print(f"✅ 创建新集合: {collection_name}")
else:
    # 如果集合已存在，先清空数据
    client.delete(collection_name=collection_name, filter="id >= 0")
    print("🗑️  清空现有数据，准备重新插入...")

# ==================== 向量索引创建 ====================
# 为向量字段创建索引，以支持高效的相似性搜索
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector",           # 索引字段名
    index_type="AUTOINDEX",        # 自动选择最优索引类型
    metric_type="L2",              # 使用L2距离度量（欧几里得距离）
    params={"nlist": 1024},        # 聚类数量，影响索引构建速度和搜索精度
)
client.create_index(collection_name=collection_name, index_params=index_params)
print("✅ 向量索引创建完成")

# ==================== 数据向量化和插入 ====================
from tqdm import tqdm

# 确保集合为空
if client.has_collection(collection_name):
    client.delete(collection_name=collection_name, filter="id >= 0")
    print(f"🗑️  清空集合 '{collection_name}' 中的现有数据")

print(f"📝 准备插入 {len(df)} 条数据...")

# 逐条处理数据，进行向量化并插入
for start_idx in tqdm(range(0, len(df)), desc="插入数据"):
    row = df.iloc[start_idx]
    
    # ==================== 文本向量化准备 ====================
    # 将医疗病例的各个字段组合成结构化文本，用于向量化
    # 这种组合方式有助于模型理解病例的完整语义
    doc_parts = [str(row["case_id"])]  # 病例ID作为基础
    
    # 按重要性顺序添加各个字段，构建语义丰富的文本
    if row["patient_name"]:
        doc_parts.append(f"(患者姓名：{row['patient_name']})")
    if row["disease_name"]:
        doc_parts.append(f"疾病名称：{row['disease_name']}")
    if row["department"]:
        doc_parts.append(f"科室：{row['department']}")
    if row["severity"]:
        doc_parts.append(f"病情：{row['severity']}")
    if row["aliases"]:
        doc_parts.append(f"别名：{row['aliases']}")
    if row["case_description"]:
        doc_parts.append(f"病例描述：{row['case_description']}")
    
    # 将所有字段用分号连接，形成完整的病例描述文本
    doc_text = "；".join(doc_parts)
    
    # ==================== 向量生成和插入 ====================
    # 使用BGE模型将文本转换为向量表示
    embedding = embedding_function([doc_text])[0]
    
    # 准备插入数据，包含向量和所有标量字段
    data_to_insert = [
        {
            "vector": embedding,                    # 向量字段
            "case_id": str(row["case_id"]),        # 病例ID
            "patient_name": str(row["patient_name"]), # 患者姓名
            "disease_name": str(row["disease_name"]), # 疾病名称
            "department": str(row["department"]),   # 科室
            "severity": str(row["severity"]),       # 病情严重程度
            "aliases": str(row["aliases"]),         # 疾病别名
            "case_description": str(row["case_description"]), # 病例描述
        }
    ]
    
    # 将数据插入到Milvus集合中
    client.insert(collection_name=collection_name, data=data_to_insert)

# ==================== 向量相似性检索测试 ====================
# 演示如何使用语义搜索找到相关的医疗病例
search_query = "感冒患者"  # 查询文本：寻找与感冒相关的病例

# 将查询文本转换为向量表示
search_embedding = embedding_function([search_query])[0]

# 在Milvus中执行向量相似性搜索
# 使用L2距离找到与查询向量最相似的病例
search_result = client.search(
    collection_name=collection_name,  # 搜索的集合名称
    data=[search_embedding.tolist()], # 查询向量（需要转换为列表）
    limit=3,                         # 返回前3个最相似的结果
    output_fields=[                  # 指定返回的字段
        "case_id",
        "patient_name", 
        "disease_name",
        "department",
        "severity",
        "aliases",
        "case_description",
    ],
)
print(f"\n{'='*50}")
print(f"🔍 向量检索结果: '{search_query}'")
print(f"{'='*50}")

if search_result and search_result[0]:
    results = search_result[0]
    print(f"找到 {len(results)} 个相关病例:")
    print()

    for i, result in enumerate(results, 1):
        print(f"【第{i}名】相似度: {result.get('distance', 0):.4f}")
        print(f"  病例ID: {result.get('case_id', 'N/A')}")
        print(f"  患者姓名: {result.get('patient_name', 'N/A')}")
        print(f"  疾病名称: {result.get('disease_name', 'N/A')}")
        print(f"  科室: {result.get('department', 'N/A')}")
        print(f"  病情: {result.get('severity', 'N/A')}")
        print(f"  别名: {result.get('aliases', 'N/A')}")
        print(f"  病例描述: {result.get('case_description', 'N/A')}")
        print("-" * 40)
else:
    print("❌ 未找到相关病例")

# ==================== 条件查询测试 ====================
# 演示如何使用标量字段进行精确的条件查询
# 这种查询不涉及向量相似性，而是基于字段值的精确匹配
query_result = client.query(
    collection_name=collection_name,  # 查询的集合名称
    filter="severity == '危重'",     # 过滤条件：查找病情为危重的病例
    output_fields=[                  # 指定返回的字段
        "case_id",
        "patient_name",
        "disease_name", 
        "department",
        "severity",
        "aliases",
        "case_description",
    ],
)
print(f"\n{'='*50}")
print(f"📋 条件查询结果: 'severity == 危重'")
print(f"{'='*50}")

if query_result:
    print(f"找到 {len(query_result)} 个危重病例:")
    print()

    for i, result in enumerate(query_result, 1):
        print(f"【病例 {i}】") 
        print(f"  病例ID: {result.get('case_id', 'N/A')}")
        print(f"  患者姓名: {result.get('patient_name', 'N/A')}")
        print(f"  疾病名称: {result.get('disease_name', 'N/A')}")
        print(f"  科室: {result.get('department', 'N/A')}")
        print(f"  病情: {result.get('severity', 'N/A')}")
        print(f"  别名: {result.get('aliases', 'N/A')}")
        print(f"  病例描述: {result.get('case_description', 'N/A')}")
        print("-" * 40)
else:
    print("❌ 未找到危重病例")

# ==================== 系统统计信息 ====================
# 显示系统的基本统计信息，帮助了解数据规模和配置
print(f"\n{'='*50}")
print("📊 数据统计信息")
print(f"{'='*50}")
print(f"集合名称: {collection_name}")      # 当前使用的集合名称
print(f"向量维度: {vector_dim}")          # BGE模型的向量维度
print(f"示例数据条数: {len(df)}")         # 插入的医疗病例数量
print(f"{'='*50}")
