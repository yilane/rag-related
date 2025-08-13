# -*- coding: utf-8 -*-
"""
医学命名实体识别(NER)服务模块
基于中文医学预训练模型进行实体识别和分类
"""

import re
import logging
from typing import List, Dict, Any
import torch

from config import NER_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalNERService:
    """医学命名实体识别服务类"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or NER_CONFIG['model_name']
        self.confidence_threshold = NER_CONFIG['confidence_threshold']
        self.ner_pipeline = None
        self._load_model()
    
    def _load_model(self):
        """加载NER模型"""
        try:
            logger.info(f"正在加载医学NER模型: {self.model_name}")
            
            # 尝试加载预训练模型
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            
            self.ner_pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("医学NER模型加载完成")
            
        except Exception as e:
            logger.error(f"加载NER模型失败: {e}")
            logger.info("使用备用NER方案...")
            self._load_backup_ner()
    
    def _load_backup_ner(self):
        """备用NER方案：基于规则的实体识别"""
        logger.info("使用基于规则的备用NER识别器")
        self.ner_pipeline = None
        
        # 定义医学实体关键词词典
        self.medical_keywords = {
            'DISEASE': [
                '心肌梗死', '高血压', '糖尿病', '肺炎', '哮喘', '肝炎', 
                '胃炎', '肾炎', '关节炎', '肿瘤', '癌症', '白血病',
                '脑梗', '中风', '骨折', '感冒', '发烧', '咳嗽',
                '头痛', '腹痛', '胸痛', '心悸', '眩晕', '失眠'
            ],
            'SYMPTOM': [
                '疼痛', '发热', '咳嗽', '气喘', '乏力', '恶心', '呕吐',
                '腹泻', '便秘', '头晕', '头痛', '胸闷', '心悸', '失眠',
                '食欲不振', '体重下降', '出血', '肿胀', '瘙痒', '皮疹'
            ],
            'BODY_PART': [
                '心脏', '肺', '肝脏', '肾脏', '大脑', '胃', '肠道',
                '脊椎', '关节', '血管', '神经', '皮肤', '眼睛', '耳朵',
                '鼻子', '喉咙', '手', '脚', '胸部', '腹部', '背部'
            ],
            'EXAMINATION': [
                'CT', 'MRI', 'X光', 'B超', '心电图', '血常规', '尿常规',
                '肝功能', '肾功能', '血糖', '血压', '体温', '脉搏'
            ]
        }
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """从文本中提取医学实体"""
        try:
            if not text or not text.strip():
                return []
            
            # 预处理文本
            cleaned_text = self._preprocess_text(text)
            
            if self.ner_pipeline is not None:
                # 使用预训练模型进行NER
                return self._extract_with_model(cleaned_text)
            else:
                # 使用备用规则方法
                return self._extract_with_rules(cleaned_text)
                
        except Exception as e:
            logger.error(f"实体提取失败: {e}")
            return []
    
    def _preprocess_text(self, text: str) -> str:
        """文本预处理"""
        # 去除多余空格和特殊字符
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。；：！？]', '', text)
        return text.strip()
    
    def _extract_with_model(self, text: str) -> List[Dict[str, Any]]:
        """使用预训练模型提取实体"""
        try:
            # 调用NER pipeline
            ner_results = self.ner_pipeline(text)
            
            entities = []
            for result in ner_results:
                if result['score'] >= self.confidence_threshold:
                    entity = {
                        'text': result['word'],
                        'label': result['entity_group'],
                        'confidence': round(result['score'], 3),
                        'start': result['start'],
                        'end': result['end']
                    }
                    entities.append(entity)
            
            # 去重和合并相邻实体
            entities = self._merge_adjacent_entities(entities)
            
            return entities
            
        except Exception as e:
            logger.error(f"模型提取实体失败: {e}")
            return self._extract_with_rules(text)
    
    def _extract_with_rules(self, text: str) -> List[Dict[str, Any]]:
        """使用规则方法提取实体"""
        entities = []
        
        for label, keywords in self.medical_keywords.items():
            for keyword in keywords:
                # 查找关键词在文本中的位置
                for match in re.finditer(keyword, text):
                    entity = {
                        'text': keyword,
                        'label': label,
                        'confidence': 0.8,  # 规则方法给定默认置信度
                        'start': match.start(),
                        'end': match.end()
                    }
                    entities.append(entity)
        
        # 按位置排序并去重
        entities = sorted(entities, key=lambda x: x['start'])
        entities = self._remove_overlapping_entities(entities)
        
        return entities
    
    def _merge_adjacent_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并相邻的同类型实体"""
        if not entities:
            return entities
        
        merged = []
        current = entities[0]
        
        for next_entity in entities[1:]:
            # 如果是相邻的同类型实体，则合并
            if (current['label'] == next_entity['label'] and 
                next_entity['start'] <= current['end'] + 2):
                current['text'] += next_entity['text']
                current['end'] = next_entity['end']
                current['confidence'] = max(current['confidence'], next_entity['confidence'])
            else:
                merged.append(current)
                current = next_entity
        
        merged.append(current)
        return merged
    
    def _remove_overlapping_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """移除重叠的实体，保留置信度更高的"""
        if not entities:
            return entities
        
        # 按置信度降序排序
        entities = sorted(entities, key=lambda x: x['confidence'], reverse=True)
        
        filtered = []
        for entity in entities:
            # 检查是否与已选择的实体重叠
            overlapped = False
            for selected in filtered:
                if (entity['start'] < selected['end'] and 
                    entity['end'] > selected['start']):
                    overlapped = True
                    break
            
            if not overlapped:
                filtered.append(entity)
        
        # 按位置重新排序
        return sorted(filtered, key=lambda x: x['start'])
    
    def analyze_entities(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析提取的实体统计信息"""
        if not entities:
            return {
                'total_entities': 0,
                'entity_types': {},
                'confidence_stats': {},
                'entities_by_type': {}
            }
        
        # 统计各类型实体数量
        entity_types = {}
        entities_by_type = {}
        confidences = [e['confidence'] for e in entities]
        
        for entity in entities:
            label = entity['label']
            entity_types[label] = entity_types.get(label, 0) + 1
            
            if label not in entities_by_type:
                entities_by_type[label] = []
            entities_by_type[label].append(entity)
        
        return {
            'total_entities': len(entities),
            'entity_types': entity_types,
            'confidence_stats': {
                'mean': round(sum(confidences) / len(confidences), 3),
                'max': round(max(confidences), 3),
                'min': round(min(confidences), 3)
            },
            'entities_by_type': entities_by_type
        }
    
    def highlight_entities(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """在原文中高亮显示识别的实体"""
        if not entities:
            return text
        
        # 按位置倒序排序，避免插入标记时位置偏移
        entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        highlighted_text = text
        color_map = {
            'DISEASE': '#FF6B6B',
            'SYMPTOM': '#4ECDC4', 
            'BODY_PART': '#45B7D1',
            'EXAMINATION': '#96CEB4',
            'DRUG': '#FFEAA7',
            'TREATMENT': '#DDA0DD'
        }
        
        for entity in entities:
            color = color_map.get(entity['label'], '#95A5A6')
            start, end = entity['start'], entity['end']
            original_text = highlighted_text[start:end]
            
            confidence_str = entity['confidence']
            highlighted = f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; color: white; font-weight: bold;" title="{entity["label"]} (置信度: {confidence_str})">{original_text}</span>'
            
            highlighted_text = highlighted_text[:start] + highlighted + highlighted_text[end:]
        
        return highlighted_text
    
    def cleanup_resources(self) -> bool:
        """清理NER模型资源"""
        try:
            logger.info("正在清理NER模型资源...")
            
            # 清理模型引用
            if hasattr(self, 'ner_pipeline') and self.ner_pipeline is not None:
                # 如果是Transformers pipeline，尝试清理
                if hasattr(self.ner_pipeline, 'model'):
                    del self.ner_pipeline.model
                if hasattr(self.ner_pipeline, 'tokenizer'):
                    del self.ner_pipeline.tokenizer
                del self.ner_pipeline
                self.ner_pipeline = None
            
            # 清理关键词词典（如果使用备用方案）
            if hasattr(self, 'medical_keywords'):
                self.medical_keywords = None
            
            # 尝试清理CUDA缓存
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("CUDA缓存已清理")
            except ImportError:
                pass
            
            logger.info("NER模型资源清理完成")
            return True
            
        except Exception as e:
            logger.error(f"清理NER资源失败: {e}")
            return False
    
    def __del__(self):
        """析构函数：确保资源被清理"""
        try:
            self.cleanup_resources()
        except:
            pass


def main():
    """测试医学NER服务"""
    try:
        # 创建NER服务实例
        ner_service = MedicalNERService()
        
        # 测试文本
        test_texts = [
            "患者主诉胸痛3天，伴有呼吸困难和心悸，既往有高血压病史",
            "诊断为急性心肌梗死，建议立即进行心电图检查",
            "患者出现发热、咳嗽症状，体温38.5度，建议胸部CT检查"
        ]
        
        print("🔍 医学NER实体识别测试")
        print("=" * 50)
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n📝 测试文本 {i}: {text}")
            
            # 提取实体
            entities = ner_service.extract_entities(text)
            
            if entities:
                print(f"✅ 识别到 {len(entities)} 个实体:")
                for entity in entities:
                    print(f"  - {entity['text']} ({entity['label']}, 置信度: {entity['confidence']})")
                
                # 分析统计
                stats = ner_service.analyze_entities(entities)
                print(f"📊 实体统计: {stats['entity_types']}")
                
            else:
                print("❌ 未识别到任何实体")
            
            print("-" * 30)
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")


if __name__ == "__main__":
    main()