"""
ICD-10数据库构建模块
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging

from config import EMBEDDING_CONFIG, PROCESSING_CONFIG
from milvus_service import MilvusService

logging.basicConfig(level=getattr(logging, PROCESSING_CONFIG['log_level']))
logger = logging.getLogger(__name__)


class ICD10Vectorizer:
    """ICD-10文本向量化器"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or EMBEDDING_CONFIG['model_name']
        self.dimension = EMBEDDING_CONFIG['dimension']
        self.batch_size = EMBEDDING_CONFIG['batch_size']
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载向量化模型"""
        try:
            logger.info(f"正在加载向量化模型: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"模型加载完成，向量维度: {self.dimension}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """文本向量化"""
        try:
            if not texts:
                return np.array([])
            
            logger.info(f"开始向量化处理，文本数量: {len(texts)}")
            
            vectors = self.model.encode(
                texts, 
                batch_size=self.batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            
            logger.info(f"向量化完成，输出形状: {vectors.shape}")
            return vectors
            
        except Exception as e:
            logger.error(f"向量化处理失败: {e}")
            raise
    
    def cleanup_resources(self) -> bool:
        """清理向量化模型资源"""
        try:
            logger.info("正在清理向量化模型资源...")
            
            # 清理SentenceTransformer模型
            if hasattr(self, 'model') and self.model is not None:
                # 清理模型组件
                if hasattr(self.model, '_modules'):
                    del self.model._modules
                if hasattr(self.model, 'tokenizer'):
                    del self.model.tokenizer
                del self.model
                self.model = None
            
            # 尝试清理CUDA缓存
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("向量化模型CUDA缓存已清理")
            except ImportError:
                pass
            
            logger.info("向量化模型资源清理完成")
            return True
            
        except Exception as e:
            logger.error(f"清理向量化资源失败: {e}")
            return False
    
    def __del__(self):
        """析构函数：确保资源被清理"""
        try:
            self.cleanup_resources()
        except:
            pass


class DatabaseBuilder:
    """数据库构建器"""
    
    def __init__(self):
        self.vectorizer = ICD10Vectorizer()
        self.milvus_service = MilvusService()
        self.chunk_size = PROCESSING_CONFIG['chunk_size']
    
    def load_csv_data(self, csv_path: str) -> pd.DataFrame:
        """加载CSV数据"""
        try:
            logger.info(f"正在加载CSV文件: {csv_path}")
            
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
            
            df = pd.read_csv(csv_path, encoding='utf-8')
            df = df.dropna(subset=['疾病编码', '疾病名称'])
            df = df.drop_duplicates(subset=['疾病编码'])
            
            logger.info(f"CSV加载完成，原始数据: {len(df)}条记录")
            
            logger.info("数据样本:")
            for col in df.columns:
                logger.info(f"  {col}: {df[col].iloc[0] if not df.empty else 'N/A'}")
            
            return df
            
        except Exception as e:
            logger.error(f"加载CSV文件失败: {e}")
            raise
    
    def generate_descriptions(self, df: pd.DataFrame) -> List[str]:
        """生成描述文本"""
        try:
            logger.info("开始生成描述文本...")
            
            descriptions = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc="生成描述"):
                parts = []
                
                if pd.notna(row.get('章名称')):
                    parts.append(f"分类: {row['章名称']}")
                
                if pd.notna(row.get('节名称')):
                    parts.append(f"类别: {row['节名称']}")
                
                if pd.notna(row.get('三位名称')):
                    parts.append(f"疾病组: {row['三位名称']}")
                
                if pd.notna(row.get('四位名称')) and row.get('四位名称') != row.get('疾病名称'):
                    parts.append(f"亚型: {row['四位名称']}")
                
                parts.append(f"疾病: {row['疾病名称']}")
                parts.append(f"编码: {row['疾病编码']}")
                
                description = " | ".join(parts)
                descriptions.append(description)
            
            logger.info(f"描述文本生成完成，共{len(descriptions)}条")
            
            if descriptions:
                logger.info(f"描述示例: {descriptions[0]}")
            
            return descriptions
            
        except Exception as e:
            logger.error(f"生成描述文本失败: {e}")
            raise
    
    def prepare_entities(self, df: pd.DataFrame, descriptions: List[str], vectors: np.ndarray) -> List[Dict[str, Any]]:
        """准备插入实体数据"""
        try:
            logger.info("开始准备入库数据...")
            
            entities = []
            for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="准备数据")):
                entity = {
                    'disease_code': str(row['疾病编码']),
                    'disease_name': str(row['疾病名称']),
                    'description_text': descriptions[i],
                    'embedding_vector': vectors[i].tolist(),
                    'chapter_name': str(row.get('章名称', '')),
                    'section_name': str(row.get('节名称', ''))
                }
                entities.append(entity)
            
            logger.info(f"数据准备完成，共{len(entities)}条记录")
            return entities
            
        except Exception as e:
            logger.error(f"准备入库数据失败: {e}")
            raise
    
    def build_database(self, csv_path: str) -> Dict[str, Any]:
        """完整的数据库构建流程"""
        try:
            logger.info("🚀 开始构建ICD-10向量数据库...")
            
            df = self.load_csv_data(csv_path)
            descriptions = self.generate_descriptions(df)
            vectors = self.vectorizer.encode(descriptions)
            entities = self.prepare_entities(df, descriptions, vectors)
            
            logger.info("📊 创建Milvus集合...")
            if not self.milvus_service.create_collection():
                raise Exception("创建Milvus集合失败")
            
            if not self.milvus_service.create_index():
                raise Exception("创建Milvus索引失败")
            
            logger.info("💾 开始插入数据到Milvus...")
            
            total_inserted = 0
            for i in range(0, len(entities), self.chunk_size):
                chunk = entities[i:i + self.chunk_size]
                if self.milvus_service.insert_data(chunk):
                    total_inserted += len(chunk)
                    logger.info(f"已插入: {total_inserted}/{len(entities)}")
                else:
                    logger.error(f"插入第{i//self.chunk_size + 1}批数据失败")
            
            logger.info("🔄 加载集合到内存...")
            if not self.milvus_service.load_collection():
                logger.warning("加载集合到内存失败，但数据已插入成功")
            
            stats = self.milvus_service.get_collection_stats()
            
            result = {
                'success': True,
                'total_records': len(entities),
                'inserted_records': total_inserted,
                'vector_dimension': self.vectorizer.dimension,
                'collection_stats': stats,
                'csv_path': csv_path
            }
            
            logger.info("🎉 数据库构建完成！")
            logger.info(f"📈 统计信息: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 数据库构建失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'csv_path': csv_path
            }


def main():
    """主函数：构建ICD-10向量数据库"""
    try:
        csv_path = "data/ICD-10医保1.0版.csv"
        
        if not os.path.exists(csv_path):
            print(f"❌ CSV文件不存在: {csv_path}")
            print("请确保数据文件在正确的位置")
            return
        
        builder = DatabaseBuilder()
        result = builder.build_database(csv_path)
        
        if result['success']:
            print("\n🎉 数据库构建成功!")
            print(f"📊 总记录数: {result['total_records']}")
            print(f"💾 已插入记录: {result['inserted_records']}")
            print(f"🔢 向量维度: {result['vector_dimension']}")
            print(f"📈 集合统计: {result['collection_stats']['num_entities']}条记录")
            
            print("\n✅ 数据库构建完成，可以开始使用检索功能!")
            
        else:
            print(f"\n❌ 数据库构建失败: {result['error']}")
            
    except KeyboardInterrupt:
        print("\n⚠️  用户中断操作")
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")


if __name__ == "__main__":
    main()