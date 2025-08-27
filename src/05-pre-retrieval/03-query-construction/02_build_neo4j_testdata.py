#!/usr/bin/env python3
"""
Neo4j测试数据初始化脚本
创建医疗知识图谱的示例数据
"""

from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# Neo4j连接配置
uri = "bolt://localhost:7687"
username = "neo4j"  
password = "password123"  # 使用Docker容器设置的密码

# 初始化Neo4j驱动
driver = GraphDatabase.driver(uri, auth=(username, password))

def clear_database():
    """清空数据库"""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        print("✓ 数据库已清空")

def create_constraints():
    """创建约束和索引"""
    with driver.session() as session:
        # 为疾病节点创建唯一约束
        try:
            session.run("CREATE CONSTRAINT disease_name_unique IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE")
            print("✓ 疾病名称唯一约束已创建")
        except Exception as e:
            print(f"约束创建跳过（可能已存在）: {e}")
        
        # 为症状节点创建唯一约束
        try:
            session.run("CREATE CONSTRAINT symptom_name_unique IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE")
            print("✓ 症状名称唯一约束已创建")
        except Exception as e:
            print(f"约束创建跳过（可能已存在）: {e}")
            
        # 为药物节点创建唯一约束
        try:
            session.run("CREATE CONSTRAINT drug_name_unique IF NOT EXISTS FOR (drug:Drug) REQUIRE drug.name IS UNIQUE")
            print("✓ 药物名称唯一约束已创建")
        except Exception as e:
            print(f"约束创建跳过（可能已存在）: {e}")

def create_test_data():
    """创建医疗知识图谱测试数据"""
    with driver.session() as session:
        # 创建疾病节点
        diseases_data = [
            {"name": "糖尿病", "code": "E10-E14", "type": "内分泌疾病", "description": "一组以高血糖为特征的代谢性疾病"},
            {"name": "高血压", "code": "I10-I15", "type": "心血管疾病", "description": "以体循环动脉血压增高为主要特征"},
            {"name": "感冒", "code": "J00-J06", "type": "呼吸系统疾病", "description": "急性上呼吸道感染"},
            {"name": "肺炎", "code": "J12-J18", "type": "呼吸系统疾病", "description": "肺部感染性疾病"},
            {"name": "胃炎", "code": "K29", "type": "消化系统疾病", "description": "胃黏膜炎症"}
        ]
        
        for disease in diseases_data:
            session.run(
                "MERGE (d:Disease {name: $name, code: $code, type: $type, description: $description})",
                **disease
            )
        print("✓ 疾病节点已创建")
        
        # 创建症状节点
        symptoms_data = [
            {"name": "发热", "severity": "轻度到重度", "description": "体温升高超过正常范围"},
            {"name": "咳嗽", "severity": "轻度到重度", "description": "呼吸道刺激引起的反射性动作"},
            {"name": "头痛", "severity": "轻度到重度", "description": "头部疼痛感觉"},
            {"name": "乏力", "severity": "轻度到中度", "description": "全身无力、疲劳感"},
            {"name": "胸闷", "severity": "中度到重度", "description": "胸部闷胀感"},
            {"name": "腹痛", "severity": "轻度到重度", "description": "腹部疼痛"},
            {"name": "恶心", "severity": "轻度到中度", "description": "想要呕吐的感觉"},
            {"name": "多尿", "severity": "轻度到中度", "description": "尿量增多"},
            {"name": "多饮", "severity": "轻度到中度", "description": "饮水量增加"},
            {"name": "体重下降", "severity": "轻度到重度", "description": "体重明显减轻"}
        ]
        
        for symptom in symptoms_data:
            session.run(
                "MERGE (s:Symptom {name: $name, severity: $severity, description: $description})",
                **symptom
            )
        print("✓ 症状节点已创建")
        
        # 创建药物节点
        drugs_data = [
            {"name": "阿莫西林", "type": "抗生素", "indication": "细菌感染", "dosage": "250mg-500mg"},
            {"name": "布洛芬", "type": "解热镇痛药", "indication": "发热、疼痛", "dosage": "200mg-400mg"},
            {"name": "二甲双胍", "type": "降糖药", "indication": "2型糖尿病", "dosage": "500mg-850mg"},
            {"name": "硝苯地平", "type": "降压药", "indication": "高血压", "dosage": "10mg-20mg"},
            {"name": "奥美拉唑", "type": "质子泵抑制剂", "indication": "胃炎、胃溃疡", "dosage": "20mg-40mg"},
            {"name": "止咳糖浆", "type": "镇咳药", "indication": "咳嗽", "dosage": "10ml-15ml"}
        ]
        
        for drug in drugs_data:
            session.run(
                "MERGE (drug:Drug {name: $name, type: $type, indication: $indication, dosage: $dosage})",
                **drug
            )
        print("✓ 药物节点已创建")
        
        # 创建疾病-症状关系
        disease_symptom_relations = [
            ("糖尿病", "多尿", "常见症状"),
            ("糖尿病", "多饮", "常见症状"),  
            ("糖尿病", "体重下降", "典型症状"),
            ("糖尿病", "乏力", "常见症状"),
            ("高血压", "头痛", "常见症状"),
            ("高血压", "胸闷", "常见症状"),
            ("感冒", "发热", "典型症状"),
            ("感冒", "咳嗽", "常见症状"),
            ("感冒", "头痛", "常见症状"),
            ("感冒", "乏力", "常见症状"),
            ("肺炎", "发热", "典型症状"),
            ("肺炎", "咳嗽", "典型症状"),
            ("肺炎", "胸闷", "常见症状"),
            ("胃炎", "腹痛", "典型症状"),
            ("胃炎", "恶心", "常见症状")
        ]
        
        for disease_name, symptom_name, relation_type in disease_symptom_relations:
            session.run("""
                MATCH (d:Disease {name: $disease_name})
                MATCH (s:Symptom {name: $symptom_name})
                MERGE (d)-[:HAS_SYMPTOM {type: $relation_type}]->(s)
            """, disease_name=disease_name, symptom_name=symptom_name, relation_type=relation_type)
        print("✓ 疾病-症状关系已创建")
        
        # 创建药物-疾病关系（治疗关系）
        drug_disease_relations = [
            ("二甲双胍", "糖尿病", "一线治疗"),
            ("硝苯地平", "高血压", "常用治疗"),
            ("阿莫西林", "肺炎", "抗感染治疗"),
            ("布洛芬", "感冒", "对症治疗"),
            ("奥美拉唑", "胃炎", "抑酸治疗"),
            ("止咳糖浆", "感冒", "对症治疗"),
            ("止咳糖浆", "肺炎", "辅助治疗")
        ]
        
        for drug_name, disease_name, treatment_type in drug_disease_relations:
            session.run("""
                MATCH (drug:Drug {name: $drug_name})
                MATCH (d:Disease {name: $disease_name})
                MERGE (drug)-[:TREATS {type: $treatment_type}]->(d)
            """, drug_name=drug_name, disease_name=disease_name, treatment_type=treatment_type)
        print("✓ 药物-疾病治疗关系已创建")

def verify_data():
    """验证数据创建情况"""
    with driver.session() as session:
        # 统计各类型节点数量
        result = session.run("MATCH (d:Disease) RETURN count(d) as disease_count")
        disease_count = result.single()["disease_count"]
        
        result = session.run("MATCH (s:Symptom) RETURN count(s) as symptom_count")
        symptom_count = result.single()["symptom_count"]
        
        result = session.run("MATCH (drug:Drug) RETURN count(drug) as drug_count")
        drug_count = result.single()["drug_count"]
        
        # 统计关系数量
        result = session.run("MATCH ()-[r:HAS_SYMPTOM]->() RETURN count(r) as symptom_relations")
        symptom_relations = result.single()["symptom_relations"]
        
        result = session.run("MATCH ()-[r:TREATS]->() RETURN count(r) as treatment_relations")
        treatment_relations = result.single()["treatment_relations"]
        
        print("\n" + "="*50)
        print("数据验证结果：")
        print(f"疾病节点数量: {disease_count}")
        print(f"症状节点数量: {symptom_count}")
        print(f"药物节点数量: {drug_count}")
        print(f"疾病-症状关系数量: {symptom_relations}")
        print(f"药物-疾病治疗关系数量: {treatment_relations}")
        print("="*50)

def get_sample_queries():
    """返回示例查询语句"""
    sample_queries = [
        "查找糖尿病的所有症状",
        "什么药物可以治疗高血压",
        "发热可能是哪些疾病的症状",
        "查找所有呼吸系统疾病",
        "阿莫西林可以治疗什么疾病",
        "查找糖尿病的治疗药物",
        "哪些疾病会导致头痛",
        "查找所有抗生素类药物"
    ]
    
    print("\n建议测试的查询语句：")
    for i, query in enumerate(sample_queries, 1):
        print(f"{i}. {query}")

if __name__ == "__main__":
    try:
        print("开始初始化Neo4j测试数据...")
        
        # 测试连接
        with driver.session() as session:
            result = session.run("RETURN 1")
            result.single()
        print("✓ Neo4j连接成功")
        
        # 清空数据库
        clear_database()
        
        # 创建约束
        create_constraints()
        
        # 创建测试数据
        create_test_data()
        
        # 验证数据
        verify_data()
        
        # 显示示例查询
        get_sample_queries()
        
        print("\n✓ 测试数据初始化完成！")
        
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
    finally:
        driver.close()