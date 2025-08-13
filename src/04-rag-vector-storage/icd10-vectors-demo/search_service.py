# -*- coding: utf-8 -*-
"""
向量检索服务模块
集成NER实体识别和向量相似度检索，实现智能ICD-10编码匹配
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from build_database import ICD10Vectorizer
from milvus_service import MilvusService
from medical_ner_service import MedicalNERService
from config import SEARCH_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchService:
    """智能检索服务类"""
    
    def __init__(self):
        # 初始化各个服务组件
        self.vectorizer = ICD10Vectorizer()
        self.milvus_service = MilvusService()
        self.ner_service = MedicalNERService()
        
        # 检索配置参数
        self.default_top_k = SEARCH_CONFIG['default_top_k']
        self.score_threshold = SEARCH_CONFIG['score_threshold']
        self.nprobe = SEARCH_CONFIG['nprobe']
        
        # 确保Milvus集合已加载
        self._ensure_collection_loaded()
    
    def _ensure_collection_loaded(self):
        """确保Milvus集合已加载到内存"""
        try:
            # 先创建集合
            if not self.milvus_service.create_collection():
                logger.error("无法创建或连接到Milvus集合")
                return
            
            # 加载集合到内存
            if not self.milvus_service.load_collection():
                logger.warning("Milvus集合未成功加载，检索功能可能受影响")
            else:
                logger.info("Milvus集合已成功加载")
        except Exception as e:
            logger.error(f"加载Milvus集合失败: {e}")
    
    def search_icd_codes(self, 
                        query_text: str, 
                        top_k: int = None, 
                        score_threshold: float = None,
                        use_ner: bool = True) -> Dict[str, Any]:
        """
        搜索匹配的ICD-10编码
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            score_threshold: 相似度阈值
            use_ner: 是否使用NER实体识别
            
        Returns:
            包含搜索结果的字典
        """
        try:
            if not query_text or not query_text.strip():
                return self._empty_result("查询文本不能为空")
            
            # 设置默认参数
            top_k = top_k or self.default_top_k
            score_threshold = score_threshold or self.score_threshold
            
            logger.info(f"开始检索ICD-10编码，查询: '{query_text}'")
            
            # 第一步：NER实体识别（可选）
            entities = []
            if use_ner:
                entities = self.ner_service.extract_entities(query_text)
                logger.info(f"NER识别到 {len(entities)} 个医学实体")
            
            # 第二步：构建检索查询
            search_queries = self._build_search_queries(query_text, entities, use_ner)
            
            # 第三步：执行向量检索
            all_results = []
            for query_info in search_queries:
                results = self._execute_vector_search(
                    query_info['text'], 
                    query_info['weight'],
                    top_k, 
                    score_threshold
                )
                all_results.extend(results)
            
            # 第四步：结果合并和排序
            final_results = self._merge_and_rank_results(all_results, top_k)
            
            # 第五步：格式化输出结果
            return self._format_search_results(
                query_text, 
                entities, 
                final_results, 
                top_k, 
                score_threshold
            )
            
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return self._empty_result(f"检索过程出现错误: {str(e)}")
    
    def _build_search_queries(self, original_text: str, entities: List[Dict], use_ner: bool) -> List[Dict[str, Any]]:
        """构建检索查询"""
        queries = []
        
        # 主查询：原始文本
        queries.append({
            'text': original_text,
            'weight': 1.0,
            'type': 'original'
        })
        
        if use_ner and entities:
            # 实体查询：基于识别的医学实体
            entity_texts = []
            
            # 优先处理疾病名称
            disease_entities = [e for e in entities if 'disease' in e['label'].lower() or 'symptom' in e['label'].lower()]
            if disease_entities:
                entity_text = ' '.join([e['text'] for e in disease_entities])
                entity_texts.append(entity_text)
            
            # 添加症状相关实体
            symptom_entities = [e for e in entities if 'symptom' in e['label'].lower()]
            if symptom_entities:
                symptom_text = ' '.join([e['text'] for e in symptom_entities])
                entity_texts.append(symptom_text)
            
            # 为每个实体文本创建查询
            for i, entity_text in enumerate(entity_texts):
                if entity_text.strip():
                    queries.append({
                        'text': entity_text,
                        'weight': 0.8 - i * 0.1,  # 递减权重
                        'type': f'entity_{i}'
                    })
        
        logger.info(f"构建了 {len(queries)} 个检索查询")
        return queries
    
    def _execute_vector_search(self, query_text: str, weight: float, top_k: int, score_threshold: float) -> List[Dict]:
        """执行单个向量检索"""
        try:
            # 向量化查询文本
            query_vector = self.vectorizer.encode([query_text])
            if query_vector.size == 0:
                return []
            
            # 执行Milvus检索
            search_params = {
                "metric_type": "IP",  # 内积相似度
                "params": {"nprobe": self.nprobe}
            }
            
            results = self.milvus_service.search_vectors(
                vectors=query_vector.tolist(),
                top_k=top_k * 2,  # 获取更多结果用于合并
                search_params=search_params,
                output_fields=["disease_code", "disease_name", "description_text", "chapter_name", "section_name"]
            )
            
            # 处理检索结果
            processed_results = []
            if results and len(results) > 0:
                for hit in results[0]:  # results[0]对应第一个查询向量的结果
                    score = float(hit.score)
                    if score >= score_threshold:
                        result = {
                            'id': hit.id,
                            'score': score * weight,  # 应用权重
                            'weighted_score': score * weight,
                            'original_score': score,
                            'query_weight': weight,
                            'disease_code': hit.entity.get('disease_code', ''),
                            'disease_name': hit.entity.get('disease_name', ''),
                            'description_text': hit.entity.get('description_text', ''),
                            'chapter_name': hit.entity.get('chapter_name', ''),
                            'section_name': hit.entity.get('section_name', '')
                        }
                        processed_results.append(result)
            
            logger.info(f"查询 '{query_text}' 返回 {len(processed_results)} 个结果")
            return processed_results
            
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []
    
    def _merge_and_rank_results(self, all_results: List[Dict], top_k: int) -> List[Dict]:
        """合并和排序检索结果"""
        if not all_results:
            return []
        
        # 按疾病编码去重，保留最高分数
        unique_results = {}
        for result in all_results:
            code = result['disease_code']
            if code not in unique_results or result['weighted_score'] > unique_results[code]['weighted_score']:
                unique_results[code] = result
        
        # 按加权分数排序
        sorted_results = sorted(unique_results.values(), key=lambda x: x['weighted_score'], reverse=True)
        
        # 返回Top-K结果
        return sorted_results[:top_k]
    
    def _format_search_results(self, query_text: str, entities: List[Dict], results: List[Dict], 
                             top_k: int, score_threshold: float) -> Dict[str, Any]:
        """格式化搜索结果"""
        
        # 计算实体统计
        entity_stats = {}
        if entities:
            for entity in entities:
                label = entity['label']
                entity_stats[label] = entity_stats.get(label, 0) + 1
        
        # 格式化结果列表
        formatted_results = []
        for i, result in enumerate(results):
            formatted_result = {
                'rank': i + 1,
                'disease_code': result['disease_code'],
                'disease_name': result['disease_name'],
                'similarity_score': round(result['original_score'], 4),
                'weighted_score': round(result['weighted_score'], 4),
                'chapter_name': result['chapter_name'],
                'section_name': result['section_name'],
                'description': result['description_text']
            }
            formatted_results.append(formatted_result)
        
        # 获取最佳匹配详情
        best_match = formatted_results[0] if formatted_results else None
        
        return {
            'success': True,
            'query_text': query_text,
            'search_params': {
                'top_k': top_k,
                'score_threshold': score_threshold,
                'total_found': len(formatted_results)
            },
            'ner_analysis': {
                'entities_found': len(entities),
                'entity_stats': entity_stats,
                'entities': entities[:5]  # 只返回前5个实体
            },
            'results': formatted_results,
            'best_match': best_match,
            'summary': {
                'has_results': len(formatted_results) > 0,
                'best_score': best_match['similarity_score'] if best_match else 0,
                'avg_score': round(sum(r['similarity_score'] for r in formatted_results) / len(formatted_results), 4) if formatted_results else 0
            }
        }
    
    def _empty_result(self, message: str) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'success': False,
            'error': message,
            'query_text': '',
            'search_params': {},
            'ner_analysis': {},
            'results': [],
            'best_match': None,
            'summary': {
                'has_results': False,
                'best_score': 0,
                'avg_score': 0
            }
        }
    
    def get_disease_detail(self, disease_code: str) -> Dict[str, Any]:
        """获取特定疾病编码的详细信息"""
        try:
            # 通过编码精确查询
            result = self.milvus_service.query_by_code(disease_code)
            
            if result:
                return {
                    'success': True,
                    'disease_code': result.get('disease_code', ''),
                    'disease_name': result.get('disease_name', ''),
                    'description': result.get('description_text', ''),
                    'chapter_name': result.get('chapter_name', ''),
                    'section_name': result.get('section_name', ''),
                    'hierarchy': self._build_hierarchy_info(result)
                }
            else:
                return {
                    'success': False,
                    'error': f'未找到编码为 {disease_code} 的疾病信息'
                }
                
        except Exception as e:
            logger.error(f"获取疾病详情失败: {e}")
            return {
                'success': False,
                'error': f'查询失败: {str(e)}'
            }
    
    def _build_hierarchy_info(self, disease_info: Dict) -> Dict[str, str]:
        """构建疾病分类层次信息"""
        return {
            'chapter': disease_info.get('chapter_name', ''),
            'section': disease_info.get('section_name', ''),
            'disease_code': disease_info.get('disease_code', ''),
            'disease_name': disease_info.get('disease_name', '')
        }
    
    def batch_search(self, queries: List[str], top_k: int = None) -> List[Dict[str, Any]]:
        """批量检索多个查询"""
        results = []
        top_k = top_k or self.default_top_k
        
        for query in queries:
            result = self.search_icd_codes(query, top_k=top_k)
            results.append(result)
        
        return results
    
    def cleanup_resources(self) -> bool:
        """清理搜索服务资源"""
        try:
            logger.info("正在清理搜索服务资源...")
            
            # 清理向量化器
            if hasattr(self, 'vectorizer') and self.vectorizer is not None:
                self.vectorizer.cleanup_resources()
                self.vectorizer = None
            
            # 清理Milvus服务
            if hasattr(self, 'milvus_service') and self.milvus_service is not None:
                self.milvus_service.cleanup_resources()
                self.milvus_service = None
            
            # 清理NER服务
            if hasattr(self, 'ner_service') and self.ner_service is not None:
                self.ner_service.cleanup_resources()
                self.ner_service = None
            
            logger.info("搜索服务资源清理完成")
            return True
            
        except Exception as e:
            logger.error(f"清理搜索服务资源失败: {e}")
            return False
    
    def __del__(self):
        """析构函数：确保资源被清理"""
        try:
            self.cleanup_resources()
        except:
            pass


def main():
    """测试检索服务"""
    try:
        # 创建检索服务实例
        search_service = SearchService()
        
        # 测试查询
        test_queries = [
            "急性心肌梗死",
            "患者主诉胸痛3天，伴有呼吸困难和心悸",
            "高血压病",
            "糖尿病并发症"
        ]
        
        print("🔍 ICD-10智能检索测试")
        print("=" * 60)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 测试查询 {i}: {query}")
            print("-" * 40)
            
            # 执行检索
            result = search_service.search_icd_codes(query, top_k=5)
            
            if result['success']:
                print(f"✅ 检索成功!")
                print(f"📊 NER识别: {result['ner_analysis']['entities_found']} 个实体")
                print(f"🎯 找到结果: {result['search_params']['total_found']} 条")
                
                if result['best_match']:
                    best = result['best_match']
                    print(f"🏆 最佳匹配: {best['disease_code']} - {best['disease_name']}")
                    print(f"    相似度: {best['similarity_score']}")
                    print(f"    分类: {best['chapter_name']}")
                
                print("\n📋 Top-3 结果:")
                for j, res in enumerate(result['results'][:3], 1):
                    print(f"  {j}. {res['disease_code']} - {res['disease_name']}")
                    print(f"     相似度: {res['similarity_score']:.4f}")
                
            else:
                print(f"❌ 检索失败: {result.get('error', '未知错误')}")
            
            print("=" * 60)
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")


if __name__ == "__main__":
    main()