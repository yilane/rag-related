"""
LlamaIndex 句子窗口分块 (SentenceWindowNodeParser) 示例

本脚本演示如何使用 SentenceWindowNodeParser 进行文本分块。

SentenceWindowNodeParser 是一种专门为 RAG 应用设计的分块策略：
- 将文本分割为句子级别的节点
- 每个节点都包含上下文窗口信息，即前后几个句子的内容
- 在检索时只返回核心句子，但在生成时可以利用完整的上下文窗口
- 这种方式能够在保持检索精度的同时，为生成提供更丰富的上下文信息

主要特点：
1. 精确检索：检索时基于单个句子，提高相关性匹配
2. 丰富上下文：生成时可以访问前后句子，提供更完整的语义信息
3. 灵活配置：可以自定义窗口大小和句子分割策略
"""

# 导入LlamaIndex核心组件
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import (
    SentenceWindowNodeParser,  # 句子窗口分块器
    SentenceSplitter,          # 基础句子分块器（用于对比）
)
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.schema import NodeWithScore

def print_nodes(title, nodes):
    """用于清晰打印和可视化分块结果的辅助函数"""
    print("\n" + "=" * 50)
    print(f"  {title}  ")
    print("=" * 50)
    print(f"总块数：{len(nodes)}\n")
    
    for i, node in enumerate(nodes, 1):
        text = node.text.strip()
        summary = text[:50] + ("..." if len(text) > 50 else "")
        print(f"--- 第 {i} 个块 ---")
        print(f"节点ID: {node.node_id}")
        print(f"内容长度：{len(text)} 字")
        print(f"内容摘要：{summary}")
        print(f"完整内容：\n{text}")
        
        # 打印窗口上下文信息（如果存在）
        if hasattr(node, 'metadata') and node.metadata:
            print(f"节点metadata: {list(node.metadata.keys())}")
            if 'window' in node.metadata:
                window_text = node.metadata['window']
                window_summary = window_text[:100] + ("..." if len(window_text) > 100 else "")
                print(f"窗口上下文长度：{len(window_text)} 字")
                print(f"窗口上下文摘要：{window_summary}")
                
                # 比较核心句子和窗口上下文的差异
                if window_text != text:
                    print(f"【窗口扩展】: 窗口比核心句子多了 {len(window_text) - len(text)} 字")
                else:
                    print(f"【注意】: 窗口上下文与核心句子相同")
            else:
                print(f"❌ metadata中未找到window字段")
        else:
            print(f"❌ 节点无metadata或metadata为空")
        
        print("-" * 50)

# 1. 读取示例文档
try:
    documents = SimpleDirectoryReader(
        input_files=["data/txt/糖尿病.txt"]  # 换用更长的文档
    ).load_data()
    print(f"成功读取文档，共 {len(documents)} 个文档")
    print(f"文档总长度：{len(documents[0].text)} 字")
except Exception as e:
    print(f"[错误] 文档读取失败: {e}")
    # 如果糖尿病文档不存在，回退到原文档
    try:
        documents = SimpleDirectoryReader(
            input_files=["data/txt/Qwen3.txt"]
        ).load_data()
        print(f"回退使用 Qwen3.txt，文档长度：{len(documents[0].text)} 字")
    except:
        documents = []

if documents:
    # 2. 使用基础句子分块器作为对比
    print("\n" + "=" * 60)
    print("基础句子分块 vs 句子窗口分块对比")
    print("=" * 60)
    
    try:
        base_splitter = SentenceSplitter(
            chunk_size=100,  # 每个块的最大字符数
            chunk_overlap=10,  # 块之间的重叠字符数
        )
        base_nodes = base_splitter.get_nodes_from_documents(documents)
        print_nodes("基础句子分块结果 (SentenceSplitter)", base_nodes[:3])  # 只显示前3个块
    except Exception as e:
        print(f"[错误] 基础句子分块失败: {e}")

    # 3. 使用句子窗口分块器 - 使用官方推荐的方式但手动实现窗口逻辑
    try:
        # 第一步：使用递归分块策略创建句子级别的基础节点
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.text_splitter import SentenceSplitter as TextSplitter
        
        # 创建递归句子分割器，避免直接截断句子
        sentence_splitter = SentenceSplitter(
            chunk_size=200,         # 适中的chunk_size
            chunk_overlap=0,        # 句子间无重叠
            separator="。",         # 优先使用中文句号分割
            secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?",  # 句子级分割模式
            paragraph_separator="\n\n",  # 段落分隔符
        )
        
        # 执行递归分块
        sentence_nodes = sentence_splitter.get_nodes_from_documents(documents)
        print(f"递归句子分割器创建了 {len(sentence_nodes)} 个句子节点")
        
        # 第二步：创建SentenceWindowNodeParser配置
        sentence_window_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,          # 前后各3个句子作为窗口
            window_metadata_key="window",  # 窗口内容存储在metadata中的key
            original_text_metadata_key="original_text",  # 原始文本存储的key
        )
        
        # 第三步：手动实现窗口逻辑（按照SentenceWindowNodeParser的官方逻辑）
        sentence_window_nodes = []
        for i, node in enumerate(sentence_nodes):
            # 为每个句子节点创建窗口上下文
            start_idx = max(0, i - sentence_window_parser.window_size)
            end_idx = min(len(sentence_nodes), i + sentence_window_parser.window_size + 1)
            
            # 构建窗口内容
            window_texts = []
            for j in range(start_idx, end_idx):
                window_texts.append(sentence_nodes[j].text)
            window_content = "".join(window_texts)
            
            # 创建窗口节点
            window_node = node.model_copy()
            window_node.metadata = window_node.metadata or {}
            window_node.metadata[sentence_window_parser.window_metadata_key] = window_content
            window_node.metadata[sentence_window_parser.original_text_metadata_key] = node.text
            
            sentence_window_nodes.append(window_node)
        print(f"句子窗口分块生成了 {len(sentence_window_nodes)} 个节点")
        print_nodes("句子窗口分块结果 (SentenceWindowNodeParser)", sentence_window_nodes[:3])  # 只显示前3个块
        
        # 4. 展示句子窗口的特殊功能
        print("\n" + "=" * 60)
        print("句子窗口分块的特殊功能演示")
        print("=" * 60)
        
        if sentence_window_nodes and len(sentence_window_nodes) > 5:
            # 选择多个节点来演示窗口功能
            demo_indices = [0, len(sentence_window_nodes)//2, len(sentence_window_nodes)-1]
            for idx in demo_indices:
                demo_node = sentence_window_nodes[idx]
                position = ["第一个", "中间", "最后一个"][demo_indices.index(idx)]
                
                print(f"\n【{position}节点演示】 (索引: {idx})")
                print(f"节点ID: {demo_node.node_id}")
                print(f"核心句子: {demo_node.text}")
                print(f"核心句子长度: {len(demo_node.text)} 字")
                
                # 显示窗口上下文
                if hasattr(demo_node, 'metadata') and 'window' in demo_node.metadata:
                    window_context = demo_node.metadata['window']
                    print(f"窗口上下文: {window_context}")
                    print(f"窗口长度: {len(window_context)} 字")
                    
                    # 分析窗口效果
                    if len(window_context) > len(demo_node.text):
                        expansion = len(window_context) - len(demo_node.text)
                        ratio = len(window_context) / len(demo_node.text)
                        print(f"✅ 窗口扩展: +{expansion}字 (扩展比例: {ratio:.2f}x)")
                    else:
                        print(f"⚠️  窗口未扩展 (可能是边界节点)")
                else:
                    print("❌ 未找到窗口上下文")
                print("-" * 50)
        
        # 5. 演示使用MetadataReplacementPostProcessor
        print("\n" + "=" * 60)
        print("MetadataReplacementPostProcessor 演示")
        print("=" * 60)
        
        # 创建后处理器，用于在检索时替换节点内容
        postprocessor = MetadataReplacementPostProcessor(
            target_metadata_key="window"
        )
        
        # 模拟检索结果
        if sentence_window_nodes:
            # 选择几个节点作为模拟检索结果
            sample_nodes = sentence_window_nodes[:3]
            mock_retrieval_results = [
                NodeWithScore(node=node, score=0.8 - i*0.1) 
                for i, node in enumerate(sample_nodes)
            ]
            
            print("检索前（原始核心句子）:")
            for i, node_with_score in enumerate(mock_retrieval_results):
                print(f"节点{i+1}: {node_with_score.node.text[:100]}...")
            
            # 使用后处理器替换为窗口内容
            processed_results = postprocessor.postprocess_nodes(mock_retrieval_results)
            
            print("\n检索后（窗口上下文）:")
            for i, node_with_score in enumerate(processed_results):
                print(f"节点{i+1}: {node_with_score.node.text[:100]}...")
                if hasattr(node_with_score.node, 'metadata') and 'window' in node_with_score.node.metadata:
                    print(f"  窗口长度: {len(node_with_score.node.metadata['window'])} 字")
        
        # 6. 统计分析
        print("\n" + "=" * 60)
        print("分块统计分析")
        print("=" * 60)
        
        # 统计节点大小分布
        node_sizes = [len(node.text) for node in sentence_window_nodes]
        window_sizes = []
        for node in sentence_window_nodes:
            if hasattr(node, 'metadata') and 'window' in node.metadata:
                window_sizes.append(len(node.metadata['window']))
        
        print(f"节点总数: {len(sentence_window_nodes)}")
        print(f"核心句子平均长度: {sum(node_sizes) / len(node_sizes):.1f} 字")
        print(f"核心句子最小长度: {min(node_sizes)} 字")
        print(f"核心句子最大长度: {max(node_sizes)} 字")
        
        if window_sizes:
            print(f"窗口上下文平均长度: {sum(window_sizes) / len(window_sizes):.1f} 字")
            print(f"窗口上下文最小长度: {min(window_sizes)} 字")
            print(f"窗口上下文最大长度: {max(window_sizes)} 字")
            print(f"平均上下文扩展比例: {sum(window_sizes) / sum(node_sizes):.2f}x")
        
    except Exception as e:
        print(f"[错误] 句子窗口分块失败: {e}")
