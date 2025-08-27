#!/usr/bin/env python3
"""
Text-to-Cypher 自然语言转Cypher查询系统
支持医疗知识图谱的自然语言查询转换
"""

from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
from openai import OpenAI
import re
import json
from typing import Dict, List, Any, Tuple

# 加载环境变量
load_dotenv()

class TextToCypherConverter:
    def __init__(self):
        # Neo4j连接配置
        self.neo4j_uri = "bolt://localhost:7687"
        self.neo4j_username = "neo4j"
        self.neo4j_password = "password123"
        
        # 初始化Neo4j驱动
        self.driver = GraphDatabase.driver(
            self.neo4j_uri, 
            auth=(self.neo4j_username, self.neo4j_password)
        )
        
        # 初始化OpenAI客户端（DeepSeek API）
        self.openai_client = OpenAI(
            base_url="https://api.deepseek.com",
            api_key=os.getenv("DEEPSEEK_API_KEY")
        )
        
        # 获取数据库模式信息
        self.schema_info = self.get_database_schema()
        self.schema_description = self.build_schema_description()
        
    def __del__(self):
        """析构函数，关闭数据库连接"""
        if hasattr(self, 'driver'):
            self.driver.close()
    
    def get_database_schema(self) -> Dict[str, Any]:
        """获取数据库模式信息"""
        with self.driver.session() as session:
            # 查询节点标签
            node_labels_query = "CALL db.labels() YIELD label RETURN label"
            node_labels = [record["label"] for record in session.run(node_labels_query)]
            
            # 查询关系类型
            relationship_types_query = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
            relationship_types = [record["relationshipType"] for record in session.run(relationship_types_query)]
            
            # 查询每个标签的属性
            properties_by_label = {}
            for label in node_labels:
                try:
                    properties_query = f"""
                    MATCH (n:{label})
                    WITH n LIMIT 1
                    RETURN keys(n) as properties
                    """
                    result = session.run(properties_query)
                    record = result.single()
                    if record and record["properties"]:
                        properties_by_label[label] = record["properties"]
                except Exception as e:
                    print(f"获取{label}属性时出错: {e}")
            
            return {
                "node_labels": node_labels,
                "relationship_types": relationship_types,
                "properties_by_label": properties_by_label
            }
    
    def build_schema_description(self) -> str:
        """构建数据库模式的文本描述"""
        schema_desc = "医疗知识图谱数据库结构：\n\n"
        
        schema_desc += "节点类型：\n"
        for label in self.schema_info["node_labels"]:
            properties = self.schema_info["properties_by_label"].get(label, [])
            schema_desc += f"- {label}: {', '.join(properties)}\n"
        
        schema_desc += f"\n关系类型：\n"
        for rel_type in self.schema_info["relationship_types"]:
            schema_desc += f"- {rel_type}\n"
            
        schema_desc += """
常见查询模式：
1. 疾病症状查询：(d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
2. 药物治疗查询：(drug:Drug)-[:TREATS]->(d:Disease)
3. 根据症状查找疾病：(d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
4. 根据疾病查找治疗药物：(drug:Drug)-[:TREATS]->(d:Disease)

注意事项：
- 使用toLower()函数进行不区分大小写的文本匹配
- 使用CONTAINS或正则表达式进行模糊匹配
- 关系方向要正确
        """
        
        return schema_desc
    
    def generate_cypher_query(self, user_query: str) -> str:
        """使用LLM生成Cypher查询语句"""
        
        # 构建提示词
        prompt = f"""
{self.schema_description}

用户的自然语言问题："{user_query}"

请生成对应的Cypher查询语句。要求：

1. 根据用户问题确定需要查询的节点类型和关系
2. 使用正确的关系方向
3. 对于文本匹配，使用toLower()和CONTAINS进行不区分大小写的模糊匹配
4. 返回有意义的结果，包含必要的属性信息
5. 查询语句要高效且准确

示例：
- "查找糖尿病的症状" → MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom) WHERE toLower(d.name) CONTAINS '糖尿病' RETURN d.name, s.name, s.description
- "什么药能治疗高血压" → MATCH (drug:Drug)-[:TREATS]->(d:Disease) WHERE toLower(d.name) CONTAINS '高血压' RETURN drug.name, drug.type, drug.dosage

请只返回Cypher查询语句，不要包含任何解释或格式标记。
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system", 
                        "content": "你是一个专业的Cypher查询专家。请根据用户的自然语言问题生成准确的Cypher查询语句。只返回查询语句，不要包含任何Markdown格式或其他说明。"
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            # 清理生成的Cypher语句
            cypher = response.choices[0].message.content.strip()
            cypher = re.sub(r'```cypher\n?', '', cypher)
            cypher = re.sub(r'```\n?', '', cypher)
            cypher = cypher.strip()
            
            return cypher
            
        except Exception as e:
            raise Exception(f"生成Cypher查询时出错: {e}")
    
    def execute_cypher_query(self, cypher: str) -> List[Dict[str, Any]]:
        """执行Cypher查询并返回结果"""
        try:
            with self.driver.session() as session:
                result = session.run(cypher)
                records = []
                for record in result:
                    records.append(dict(record))
                return records
        except Exception as e:
            raise Exception(f"执行Cypher查询时出错: {e}")
    
    def validate_cypher_syntax(self, cypher: str) -> Tuple[bool, str]:
        """验证Cypher语句语法"""
        try:
            with self.driver.session() as session:
                # 使用EXPLAIN来验证语法而不执行查询
                explain_cypher = f"EXPLAIN {cypher}"
                session.run(explain_cypher)
                return True, "语法正确"
        except Exception as e:
            return False, str(e)
    
    def format_results(self, results: List[Dict[str, Any]], user_query: str) -> str:
        """格式化查询结果"""
        if not results:
            return "未找到匹配的结果。"
        
        formatted_output = f"查询问题：{user_query}\n"
        formatted_output += f"找到 {len(results)} 条结果：\n\n"
        
        for i, record in enumerate(results, 1):
            formatted_output += f"{i}. "
            
            # 格式化每条记录
            formatted_record = []
            for key, value in record.items():
                if value is not None:
                    formatted_record.append(f"{key}: {value}")
            
            formatted_output += " | ".join(formatted_record) + "\n"
        
        return formatted_output
    
    def query(self, user_query: str, show_cypher: bool = True) -> Dict[str, Any]:
        """主要查询方法"""
        result = {
            "user_query": user_query,
            "cypher": None,
            "results": [],
            "formatted_output": "",
            "error": None,
            "execution_time": 0
        }
        
        try:
            import time
            start_time = time.time()
            
            # 生成Cypher查询
            cypher = self.generate_cypher_query(user_query)
            result["cypher"] = cypher
            
            if show_cypher:
                print(f"生成的Cypher查询：\n{cypher}\n")
            
            # 验证语法
            is_valid, validation_message = self.validate_cypher_syntax(cypher)
            if not is_valid:
                result["error"] = f"Cypher语法错误：{validation_message}"
                return result
            
            # 执行查询
            query_results = self.execute_cypher_query(cypher)
            result["results"] = query_results
            
            # 格式化输出
            formatted_output = self.format_results(query_results, user_query)
            result["formatted_output"] = formatted_output
            
            result["execution_time"] = time.time() - start_time
            
        except Exception as e:
            result["error"] = str(e)
        
        return result

def interactive_query_session():
    """交互式查询会话"""
    print("=" * 60)
    print("Text-to-Cypher 医疗知识图谱查询系统")
    print("=" * 60)
    print("输入自然语言问题，系统将自动转换为Cypher查询")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'help' 查看示例查询")
    print("=" * 60)
    
    converter = TextToCypherConverter()
    
    while True:
        try:
            user_input = input("\n请输入您的问题: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("感谢使用！再见！")
                break
            
            if user_input.lower() in ['help', '帮助']:
                print("""
示例查询：
1. 查找糖尿病的所有症状
2. 什么药物可以治疗高血压  
3. 发热可能是哪些疾病的症状
4. 查找所有呼吸系统疾病
5. 阿莫西林可以治疗什么疾病
6. 查找糖尿病的治疗药物
7. 哪些疾病会导致头痛
8. 查找所有抗生素类药物
                """)
                continue
            
            if not user_input:
                print("请输入有效的问题！")
                continue
            
            # 执行查询
            result = converter.query(user_input, show_cypher=True)
            
            if result["error"]:
                print(f"❌ 查询出错：{result['error']}")
            else:
                print(f"✅ 执行成功（耗时：{result['execution_time']:.2f}秒）")
                print("\n" + "─" * 50)
                print(result["formatted_output"])
                print("─" * 50)
                
        except KeyboardInterrupt:
            print("\n\n程序被用户中断。再见！")
            break
        except Exception as e:
            print(f"❌ 系统错误：{e}")

def run_batch_test():
    """运行批量测试"""
    print("=" * 60)
    print("运行批量测试")
    print("=" * 60)
    
    test_queries = [
        "查找糖尿病的所有症状",
        "什么药物可以治疗高血压",
        "发热可能是哪些疾病的症状", 
        "查找所有呼吸系统疾病",
        "阿莫西林可以治疗什么疾病"
    ]
    
    converter = TextToCypherConverter()
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*20} 测试 {i} {'='*20}")
        result = converter.query(query, show_cypher=True)
        
        if result["error"]:
            print(f"❌ {result['error']}")
        else:
            print(f"✅ 执行成功")
            print(result["formatted_output"])

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 运行批量测试
        run_batch_test()
    else:
        # 交互式查询模式
        interactive_query_session()