"""
LlamaIndex 摘要索引 (Summary Indexing) 示例

本脚本演示如何使用摘要索引策略来提升RAG系统的检索效率和准确性。

摘要索引的核心思想：
1. 索引阶段：为每个文档块创建精炼的摘要，同时保留原文块，建立摘要与原文的链接关系
2. 检索阶段：先在摘要层进行检索，利用摘要的简洁性和语义集中性快速定位相关内容
3. 获取原文：通过链接获取对应的完整原文块，为生成提供丰富的上下文信息

主要优势：
- 提升检索效率：摘要更短，语义更集中，检索速度更快
- 提高相关性：避免在冗长的原文中"迷路"，更精准地匹配用户查询
- 保持完整性：生成时仍可获取完整的原文上下文，不丢失信息
- 降低噪声：摘要过滤掉无关细节，减少检索结果中的噪声信息
"""

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, BaseNode, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.deepseek import DeepSeek
import uuid
from typing import List, Dict, Any
import os

# 初始化嵌入模型
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh")

# 初始化LLM用于生成摘要
# 使用DeepSeek API，如果没有API密钥则使用简单的文本截断
try:
    llm = DeepSeek(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.1
    )
    use_llm = True
    print("✅ 成功配置 DeepSeek API 用于摘要生成")
except Exception as e:
    print(f"[提示] 未配置DeepSeek API，将使用简单文本截断作为摘要: {e}")
    llm = None
    use_llm = False

def print_nodes(title, nodes, show_metadata=True):
    """用于清晰打印和可视化分块结果的辅助函数"""
    print("\n" + "=" * 60)
    print(f"  {title}  ")
    print("=" * 60)
    print(f"总块数：{len(nodes)}\n")
    
    for i, node in enumerate(nodes, 1):
        text = node.text.strip()
        summary = text[:80] + ("..." if len(text) > 80 else "")
        print(f"--- 第 {i} 个块 ---")
        print(f"节点ID: {node.node_id}")
        print(f"内容长度：{len(text)} 字")
        print(f"内容摘要：{summary}")
        
        if show_metadata and hasattr(node, 'metadata') and node.metadata:
            print(f"元数据键: {list(node.metadata.keys())}")
            
            # 显示摘要信息
            if 'summary' in node.metadata:
                summary_text = node.metadata['summary']
                print(f"节点摘要: {summary_text}")
                print(f"摘要长度: {len(summary_text)} 字")
                
            # 显示原文链接信息
            if 'original_node_id' in node.metadata:
                print(f"原文节点ID: {node.metadata['original_node_id']}")
                
            # 显示节点类型
            if 'node_type' in node.metadata:
                print(f"节点类型: {node.metadata['node_type']}")
                
        print(f"完整内容：\n{text}")
        print("-" * 50)

def generate_summary(text: str, max_length: int = 100) -> str:
    """为给定文本生成摘要"""
    if use_llm and llm is not None:
        prompt = f"""请为以下文本生成一个简洁的摘要，长度控制在{max_length}字以内。
摘要应该：
1. 准确概括文本的主要内容和核心观点
2. 保持语义连贯性和可读性
3. 去除冗余信息和细节描述
4. 使用简洁明了的语言

原文：
{text}

摘要："""
        
        try:
            response = llm.complete(prompt)
            summary = response.text.strip()
            # 确保摘要长度不超过限制
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
            return summary
        except Exception as e:
            print(f"[警告] 摘要生成失败: {e}")
            # 如果LLM失败，使用简单的截断作为备用方案
            return text[:max_length] + ("..." if len(text) > max_length else "")
    else:
        # 使用简单的文本截断和关键句提取作为摘要
        sentences = text.split('。')
        if len(sentences) > 1:
            # 取前两个句子作为摘要
            summary = sentences[0] + '。' + (sentences[1] + '。' if len(sentences[1]) > 0 else '')
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
            return summary
        else:
            return text[:max_length] + ("..." if len(text) > max_length else "")

class SummaryIndexNodeParser:
    """摘要索引节点解析器"""
    
    def __init__(self, 
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 summary_max_length: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.summary_max_length = summary_max_length
        self.base_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def get_nodes_from_documents(self, documents: List[Document]) -> List[BaseNode]:
        """从文档生成摘要索引节点"""
        # 第一步：使用基础分块器创建原文块
        original_nodes = self.base_splitter.get_nodes_from_documents(documents)
        print(f"原文分块完成，共生成 {len(original_nodes)} 个原文块")
        
        # 第二步：为每个原文块生成摘要节点
        summary_nodes = []
        node_mapping = {}  # 用于存储摘要节点和原文节点的映射关系
        
        for i, original_node in enumerate(original_nodes):
            print(f"正在生成第 {i+1}/{len(original_nodes)} 个摘要...")
            
            # 生成摘要
            summary_text = generate_summary(original_node.text, self.summary_max_length)
            
            # 创建摘要节点
            summary_node = TextNode(
                text=summary_text,
                id_=str(uuid.uuid4()),
                metadata={
                    'node_type': 'summary',
                    'original_node_id': original_node.node_id,
                    'summary': summary_text,
                    'original_length': len(original_node.text),
                    'summary_length': len(summary_text)
                }
            )
            
            # 更新原文节点的元数据
            original_node.metadata = original_node.metadata or {}
            original_node.metadata.update({
                'node_type': 'original',
                'summary_node_id': summary_node.node_id,
                'summary': summary_text,
                'original_length': len(original_node.text),
                'summary_length': len(summary_text)
            })
            
            summary_nodes.append(summary_node)
            node_mapping[summary_node.node_id] = original_node.node_id
        
        print(f"摘要生成完成，共生成 {len(summary_nodes)} 个摘要节点")
        
        # 第三步：返回摘要节点和原文节点，以及映射关系
        return summary_nodes, original_nodes, node_mapping

def create_summary_index(summary_nodes: List[BaseNode], 
                        original_nodes: List[BaseNode],
                        node_mapping: Dict[str, str]) -> Dict[str, Any]:
    """创建摘要索引系统"""
    
    # 创建摘要节点的向量索引
    summary_index = VectorStoreIndex(
        nodes=summary_nodes,
        embed_model=embed_model
    )
    
    # 创建原文节点的映射字典（用于快速查找）
    original_nodes_dict = {node.node_id: node for node in original_nodes}
    
    # 创建摘要索引系统
    summary_index_system = {
        'summary_index': summary_index,
        'original_nodes': original_nodes_dict,
        'node_mapping': node_mapping
    }
    
    return summary_index_system

def search_with_summary_index(query: str, 
                            summary_index_system: Dict[str, Any],
                            top_k: int = 3) -> List[BaseNode]:
    """使用摘要索引进行检索"""
    
    # 第一步：在摘要层进行检索
    print(f"\n🔍 第一步：在摘要层检索相关内容...")
    print(f"查询：{query}")
    
    summary_index = summary_index_system['summary_index']
    summary_retriever = summary_index.as_retriever(similarity_top_k=top_k)
    summary_results = summary_retriever.retrieve(query)
    
    print(f"摘要层检索结果：")
    for i, result in enumerate(summary_results, 1):
        print(f"  {i}. [相似度: {result.score:.3f}] {result.text}")
    
    # 第二步：通过映射获取对应的原文节点
    print(f"\n📄 第二步：获取对应的原文内容...")
    original_nodes = []
    node_mapping = summary_index_system['node_mapping']
    original_nodes_dict = summary_index_system['original_nodes']
    
    for i, summary_result in enumerate(summary_results, 1):
        summary_node_id = summary_result.node.node_id
        if summary_node_id in node_mapping:
            original_node_id = node_mapping[summary_node_id]
            if original_node_id in original_nodes_dict:
                original_node = original_nodes_dict[original_node_id]
                original_nodes.append(original_node)
                print(f"  {i}. 原文片段 (长度: {len(original_node.text)} 字): {original_node.text[:100]}...")
            else:
                print(f"  {i}. ❌ 未找到原文节点 {original_node_id}")
        else:
            print(f"  {i}. ❌ 未找到摘要节点映射 {summary_node_id}")
    
    return original_nodes

def analyze_summary_effectiveness(summary_nodes: List[BaseNode], 
                                original_nodes: List[BaseNode]):
    """分析摘要索引的效果"""
    print("\n" + "=" * 60)
    print("摘要索引效果分析")
    print("=" * 60)
    
    # 统计基本信息
    total_original_length = sum(len(node.text) for node in original_nodes)
    total_summary_length = sum(len(node.text) for node in summary_nodes)
    
    print(f"原文节点数量: {len(original_nodes)}")
    print(f"摘要节点数量: {len(summary_nodes)}")
    print(f"原文总长度: {total_original_length} 字")
    print(f"摘要总长度: {total_summary_length} 字")
    print(f"压缩比例: {total_summary_length / total_original_length:.2%}")
    
    # 统计长度分布
    original_lengths = [len(node.text) for node in original_nodes]
    summary_lengths = [len(node.text) for node in summary_nodes]
    
    print(f"\n原文块长度 - 平均: {sum(original_lengths) / len(original_lengths):.1f} 字")
    print(f"原文块长度 - 范围: {min(original_lengths)} - {max(original_lengths)} 字")
    print(f"摘要块长度 - 平均: {sum(summary_lengths) / len(summary_lengths):.1f} 字")
    print(f"摘要块长度 - 范围: {min(summary_lengths)} - {max(summary_lengths)} 字")
    
    # 分析压缩效果
    compression_ratios = []
    for i, (original_node, summary_node) in enumerate(zip(original_nodes, summary_nodes)):
        ratio = len(summary_node.text) / len(original_node.text)
        compression_ratios.append(ratio)
    
    avg_compression = sum(compression_ratios) / len(compression_ratios)
    print(f"\n平均压缩比例: {avg_compression:.2%}")
    print(f"最高压缩比例: {min(compression_ratios):.2%}")
    print(f"最低压缩比例: {max(compression_ratios):.2%}")

# 主演示流程
def main():
    print("=" * 60)
    print("摘要索引 (Summary Indexing) 演示")
    print("=" * 60)
    
    # 1. 读取示例文档
    try:
        documents = SimpleDirectoryReader(
            input_files=["data/txt/糖尿病.txt"]
        ).load_data()
        print(f"✅ 成功读取文档，共 {len(documents)} 个文档")
        print(f"文档总长度：{len(documents[0].text)} 字")
    except Exception as e:
        print(f"[错误] 糖尿病文档读取失败: {e}")
        try:
            documents = SimpleDirectoryReader(
                input_files=["data/txt/Qwen3.txt"]
            ).load_data()
            print(f"✅ 回退使用 Qwen3.txt，文档长度：{len(documents[0].text)} 字")
        except Exception as e2:
            print(f"[错误] 文档读取失败: {e2}")
            return
    
    # 2. 创建摘要索引节点解析器
    print("\n📝 创建摘要索引节点解析器...")
    summary_parser = SummaryIndexNodeParser(
        chunk_size=400,      # 原文块大小
        chunk_overlap=50,    # 原文块重叠
        summary_max_length=80  # 摘要最大长度
    )
    
    # 3. 生成摘要索引节点
    print("\n🔄 生成摘要索引节点...")
    summary_nodes, original_nodes, node_mapping = summary_parser.get_nodes_from_documents(documents)
    
    # 4. 展示摘要节点
    print_nodes("摘要节点展示", summary_nodes[:3])
    
    # 5. 展示原文节点
    print_nodes("原文节点展示", original_nodes[:3])
    
    # 6. 分析摘要效果
    analyze_summary_effectiveness(summary_nodes, original_nodes)
    
    # 7. 创建摘要索引系统
    print("\n🏗️ 创建摘要索引系统...")
    summary_index_system = create_summary_index(summary_nodes, original_nodes, node_mapping)
    print("✅ 摘要索引系统创建完成")
    
    # 8. 演示摘要索引检索
    print("\n" + "=" * 60)
    print("摘要索引检索演示")
    print("=" * 60)
    
    # 示例查询
    test_queries = [
        "糖尿病的症状有哪些？",
        "如何预防糖尿病？",
        "糖尿病的治疗方法"
    ]
    
    for query in test_queries:
        print(f"\n{'='*40}")
        print(f"查询: {query}")
        print('='*40)
        
        # 使用摘要索引检索
        retrieved_nodes = search_with_summary_index(
            query, 
            summary_index_system, 
            top_k=2
        )
        
        print(f"\n✅ 最终检索结果：{len(retrieved_nodes)} 个原文片段")
        for i, node in enumerate(retrieved_nodes, 1):
            print(f"\n片段 {i}:")
            print(f"长度: {len(node.text)} 字")
            print(f"内容: {node.text[:200]}...")
            if 'summary' in node.metadata:
                print(f"摘要: {node.metadata['summary']}")
    
    # 9. 对比传统检索
    print("\n" + "=" * 60)
    print("传统检索 vs 摘要索引检索对比")
    print("=" * 60)
    
    # 创建传统索引用于对比
    traditional_index = VectorStoreIndex(
        nodes=original_nodes,
        embed_model=embed_model
    )
    
    query = "糖尿病的主要症状"
    print(f"对比查询: {query}")
    
    # 传统检索
    print(f"\n📊 传统检索结果:")
    traditional_retriever = traditional_index.as_retriever(similarity_top_k=2)
    traditional_results = traditional_retriever.retrieve(query)
    for i, result in enumerate(traditional_results, 1):
        print(f"  {i}. [相似度: {result.score:.3f}] {result.text[:100]}...")
    
    # 摘要索引检索
    print(f"\n🎯 摘要索引检索结果:")
    summary_results = search_with_summary_index(query, summary_index_system, top_k=2)
    for i, node in enumerate(summary_results, 1):
        print(f"  {i}. [通过摘要检索] {node.text[:100]}...")

if __name__ == "__main__":
    main()