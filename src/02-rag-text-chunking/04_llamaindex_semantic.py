"""
LlamaIndex 语义分块 (SemanticSplitterNodeParser) 示例

本脚本演示并对比了两种不同的文本分块策略：
1.  基础句子分块 (SentenceSplitter): 
    - 简单地根据句子结束符（如句号、问号）来切分文本。
    - 优点：速度快，实现简单。
    - 缺点：无法理解句子的语义关联，可能将紧密相关的句子切分到不同的块中。

2.  语义分块 (SemanticSplitterNodeParser):
    - 利用嵌入模型将句子转换为向量，通过计算句子之间的语义相似度来决定切分点。
    - 当相邻句子间的语义相似度低于某个阈值时，认为这里是一个自然的语义边界，从而进行切分。
    - 优点：能够根据内容的语义连贯性进行智能分块，保留完整的语义单元，对RAG的效果通常更好。
    - 缺点：需要加载嵌入模型，计算成本相对较高。
"""

# 导入LlamaIndex核心组件和分块工具
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import (
    SentenceSplitter,           # 基础句子分块器
    SemanticSplitterNodeParser, # 语义分块器
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. 初始化嵌入模型
# 嵌入模型用于将文本（句子）转换为数值向量（Embedding）。
# 语义分块器依赖这些向量来计算句子之间的语义相似度。
# 此处使用BGE（BAAI General Embedding）的中文小模型，专门针对中文文本进行了优化。
# (需确保已安装 `llama-index-embeddings-huggingface`)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh")

def print_nodes(title, nodes):
    """一个用于清晰打印和可视化分块结果的辅助函数。"""
    print("\n" + "=" * 30)
    print(f"  {title}  ")
    print("=" * 30)
    print(f"总块数：{len(nodes)}\n")
    for i, node in enumerate(nodes, 1):
        text = node.text.strip()
        summary = text[:50] + ("..." if len(text) > 50 else "")
        print(f"--- 第 {i} 个块 ---")
        print(f"内容长度：{len(text)} 字")
        print(f"内容摘要：{summary}")
        print(f"完整内容：\n{text}")
        print("-" * 50)

# 2. 读取示例文档
try:
    documents = SimpleDirectoryReader(
        input_files=["data/txt/Qwen3.txt"]
    ).load_data()
    # print(f"[调试] 文档数量: {len(documents)}")
    # print(f"[调试] 文档前200字: {documents[0].text[:200]}")
except Exception as e:
    print(f"[错误] 文档读取失败: {e}")
    documents = []

# 3. 使用基础句子分块器作为基准对比
try:
    base_splitter = SentenceSplitter()
    sentence_nodes = base_splitter.get_nodes_from_documents(documents)
    # print(f"[调试] 分句节点数: {len(sentence_nodes)}")
    # if sentence_nodes:
    #     print(f"[调试] 第一个句子节点内容: {sentence_nodes[0].text[:100]}")
    print_nodes("基础句子分块结果展示 (SentenceSplitter)", sentence_nodes)
except Exception as e:
    print(f"[错误] 基础句子分块失败: {e}")

# 4. 使用语义分块器
# 这是本示例的核心。它会根据句子之间的语义差异来决定切分点。
try:
    splitter = SemanticSplitterNodeParser(
        buffer_size=1,  # 缓冲区大小，用于在决策分割点时考虑前后句子的关系。
                        # 设置为1表示在当前句子和下一个句子之间进行比较。
        breakpoint_percentile_threshold=95,  # 分割点百分位阈值。
                                           # 决定了分割的敏感度。值越低，分割越频繁（块更小）；值越高，分割越不频繁（块更大）。
                                           # 例如，95意味着只有当句子间的余弦距离差异处于所有句子差异的前5%时，才会进行分割。
        embed_model=embed_model, # 指定用于计算句子向量的嵌入模型。
    )
    # 重要提示：当前版本的 `SemanticSplitterNodeParser` 只能直接处理 `Document` 对象。
    # 它不能像其他分块器一样接收 `Node` 对象作为输入进行二次分块。
    semantic_nodes = splitter.get_nodes_from_documents(documents)
    print_nodes("语义分块结果展示 (SemanticSplitterNodeParser)", semantic_nodes)
except Exception as e:
    print(f"[错误] 语义分块失败: {e}")
