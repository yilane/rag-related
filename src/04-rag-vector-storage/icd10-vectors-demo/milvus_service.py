"""
Milvus向量数据库服务模块
"""

import logging
from typing import List, Dict, Any, Optional
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from config import DATABASE_CONFIG, COLLECTION_SCHEMA, INDEX_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MilvusService:
    """Milvus向量数据库服务类"""
    
    def __init__(self):
        self.host = DATABASE_CONFIG['milvus_host']
        self.port = DATABASE_CONFIG['milvus_port']
        self.collection_name = DATABASE_CONFIG['collection_name']
        self.collection = None
        self._connect()
    
    def _connect(self):
        """连接到Milvus数据库"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            logger.info(f"成功连接到Milvus数据库: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            raise
    
    def create_collection(self) -> bool:
        """创建向量集合"""
        try:
            if utility.has_collection(self.collection_name):
                logger.info(f"集合 {self.collection_name} 已存在")
                self.collection = Collection(self.collection_name)
                return True
            
            # 构建字段Schema
            fields = []
            for field_config in COLLECTION_SCHEMA['fields']:
                if field_config['type'] == 'INT64':
                    field = FieldSchema(
                        name=field_config['name'],
                        dtype=DataType.INT64,
                        is_primary=field_config.get('is_primary', False),
                        auto_id=field_config.get('auto_id', False),
                        description=field_config.get('description', '')
                    )
                elif field_config['type'] == 'VARCHAR':
                    field = FieldSchema(
                        name=field_config['name'],
                        dtype=DataType.VARCHAR,
                        max_length=field_config['max_length'],
                        description=field_config.get('description', '')
                    )
                elif field_config['type'] == 'FLOAT_VECTOR':
                    field = FieldSchema(
                        name=field_config['name'],
                        dtype=DataType.FLOAT_VECTOR,
                        dim=field_config['dim'],
                        description=field_config.get('description', '')
                    )
                fields.append(field)
            
            schema = CollectionSchema(
                fields=fields,
                description=COLLECTION_SCHEMA['description']
            )
            
            self.collection = Collection(
                name=self.collection_name,
                schema=schema
            )
            
            logger.info(f"成功创建集合: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False
    
    def create_index(self) -> bool:
        """为向量字段创建索引"""
        try:
            if not self.collection:
                logger.error("集合未初始化")
                return False
            
            if self.collection.has_index():
                logger.info("索引已存在")
                return True
            
            index_params = {
                "metric_type": INDEX_CONFIG["metric_type"],
                "index_type": INDEX_CONFIG["index_type"],
                "params": INDEX_CONFIG["params"]
            }
            
            self.collection.create_index(
                field_name="embedding_vector",
                index_params=index_params
            )
            
            logger.info(f"成功创建索引: {INDEX_CONFIG['index_type']}")
            return True
            
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            return False
    
    def insert_data(self, entities: List[Dict[str, Any]]) -> bool:
        """批量插入数据"""
        try:
            if not self.collection:
                logger.error("集合未初始化")
                return False
            
            if not entities:
                logger.warning("没有数据需要插入")
                return True
            
            # 准备插入数据 - 按列组织数据
            disease_codes = [entity['disease_code'] for entity in entities]
            disease_names = [entity['disease_name'] for entity in entities]
            description_texts = [entity['description_text'] for entity in entities]
            embedding_vectors = [entity['embedding_vector'] for entity in entities]
            chapter_names = [entity['chapter_name'] for entity in entities]
            section_names = [entity['section_name'] for entity in entities]
            
            data = [
                disease_codes,
                disease_names,
                description_texts,
                embedding_vectors,
                chapter_names,
                section_names
            ]
            
            self.collection.insert(data)
            self.collection.flush()
            
            logger.info(f"成功插入 {len(entities)} 条数据")
            return True
            
        except Exception as e:
            logger.error(f"插入数据失败: {e}")
            return False
    
    def load_collection(self) -> bool:
        """加载集合到内存"""
        try:
            if not self.collection:
                logger.error("集合未初始化")
                return False
            
            self.collection.load()
            logger.info(f"成功加载集合到内存: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"加载集合失败: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            if not self.collection:
                return {"error": "集合未初始化"}
            
            stats = {
                "collection_name": self.collection_name,
                "num_entities": self.collection.num_entities,
                "is_empty": self.collection.is_empty
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取集合统计信息失败: {e}")
            return {"error": str(e)}
    
    def search_vectors(self, vectors: List[List[float]], top_k: int = 10, 
                      search_params: Dict = None, output_fields: List[str] = None) -> List:
        """向量相似度搜索"""
        try:
            if not self.collection:
                logger.error("集合未初始化")
                return []
            
            # 默认搜索参数
            if search_params is None:
                search_params = {
                    "metric_type": "IP",
                    "params": {"nprobe": 16}
                }
            
            # 默认输出字段
            if output_fields is None:
                output_fields = ["disease_code", "disease_name", "description_text", "chapter_name", "section_name"]
            
            # 执行搜索
            results = self.collection.search(
                data=vectors,
                anns_field="embedding_vector",
                param=search_params,
                limit=top_k,
                output_fields=output_fields
            )
            
            logger.info(f"向量搜索完成，返回 {len(results)} 组结果")
            return results
            
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []
    
    def query_by_code(self, disease_code: str) -> Optional[Dict[str, Any]]:
        """根据疾病编码精确查询"""
        try:
            if not self.collection:
                logger.error("集合未初始化")
                return None
            
            # 构建查询表达式
            expr = f'disease_code == "{disease_code}"'
            
            # 执行查询
            results = self.collection.query(
                expr=expr,
                output_fields=["disease_code", "disease_name", "description_text", "chapter_name", "section_name"]
            )
            
            if results and len(results) > 0:
                return results[0]
            else:
                logger.warning(f"未找到编码为 {disease_code} 的疾病信息")
                return None
                
        except Exception as e:
            logger.error(f"根据编码查询失败: {e}")
            return None
    
    def query_by_name(self, disease_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """根据疾病名称模糊查询"""
        try:
            if not self.collection:
                logger.error("集合未初始化")
                return []
            
            # 构建查询表达式（模糊匹配）
            expr = f'disease_name like "%{disease_name}%"'
            
            # 执行查询
            results = self.collection.query(
                expr=expr,
                output_fields=["disease_code", "disease_name", "description_text", "chapter_name", "section_name"],
                limit=limit
            )
            
            logger.info(f"根据名称 '{disease_name}' 查询到 {len(results)} 条结果")
            return results
            
        except Exception as e:
            logger.error(f"根据名称查询失败: {e}")
            return []
    
    def cleanup_resources(self) -> bool:
        """清理Milvus资源"""
        try:
            if self.collection:
                logger.info("正在释放Milvus集合资源...")
                # 释放集合内存
                self.collection.release()
                logger.info("Milvus集合资源已释放")
                self.collection = None
            
            # 断开连接
            connections.disconnect("default")
            logger.info("Milvus连接已断开")
            return True
            
        except Exception as e:
            logger.error(f"清理Milvus资源失败: {e}")
            return False
    
    def __del__(self):
        """析构函数：确保资源被清理"""
        try:
            self.cleanup_resources()
        except:
            pass


def test_milvus_connection():
    """测试Milvus连接和基本功能"""
    try:
        milvus_service = MilvusService()
        
        success = milvus_service.create_collection()
        if success:
            print("✅ 成功创建集合")
        else:
            print("❌ 创建集合失败")
            return
        
        success = milvus_service.create_index()
        if success:
            print("✅ 成功创建索引")
        else:
            print("❌ 创建索引失败")
            return
        
        stats = milvus_service.get_collection_stats()
        print(f"📊 集合统计信息: {stats}")
        
        print("🎉 Milvus服务测试通过！")
        
    except Exception as e:
        print(f"❌ Milvus服务测试失败: {e}")


if __name__ == "__main__":
    test_milvus_connection()