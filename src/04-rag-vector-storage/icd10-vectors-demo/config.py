"""
ICD-10 RAG检索Demo项目配置文件
"""

# 数据库配置
DATABASE_CONFIG = {
    'milvus_host': "localhost",
    'milvus_port': 19530,
    'collection_name': "icd10_diseases"
}

# 向量化配置
EMBEDDING_CONFIG = {
    'model_name': "BAAI/bge-large-zh-v1.5",
    'dimension': 1024,
    'batch_size': 32
}

# NER配置
NER_CONFIG = {
    'model_name': "lixin12345/chinese-medical-ner",
    'confidence_threshold': 0.7
}

# Gradio界面配置
GRADIO_CONFIG = {
    'server_port': 7860,
    'server_name': "0.0.0.0",
    'share': False
}

# 数据处理配置
PROCESSING_CONFIG = {
    'chunk_size': 1000,
    'log_level': "INFO"
}

# Milvus Collection Schema配置
COLLECTION_SCHEMA = {
    "collection_name": "icd10_diseases",
    "description": "ICD-10医疗编码向量数据库",
    "fields": [
        {
            "name": "id",
            "type": "INT64",
            "is_primary": True,
            "auto_id": True,
            "description": "主键ID"
        },
        {
            "name": "disease_code", 
            "type": "VARCHAR",
            "max_length": 50,
            "description": "疾病编码"
        },
        {
            "name": "disease_name",
            "type": "VARCHAR", 
            "max_length": 500,
            "description": "疾病名称"
        },
        {
            "name": "description_text",
            "type": "VARCHAR",
            "max_length": 2000,
            "description": "完整描述文本"
        },
        {
            "name": "embedding_vector",
            "type": "FLOAT_VECTOR",
            "dim": 1024,
            "description": "文本向量"
        },
        {
            "name": "chapter_name",
            "type": "VARCHAR",
            "max_length": 100,
            "description": "章名称"
        },
        {
            "name": "section_name", 
            "type": "VARCHAR",
            "max_length": 200,
            "description": "节名称"
        }
    ]
}

# 索引配置
INDEX_CONFIG = {
    "index_type": "IVF_FLAT",
    "metric_type": "IP",  # 内积相似度
    "params": {
        "nlist": 1024  # 聚类中心数量
    }
}

# 检索配置
SEARCH_CONFIG = {
    'default_top_k': 10,
    'score_threshold': 0.7,
    'nprobe': 16,
    'ner_confidence': 0.7
}