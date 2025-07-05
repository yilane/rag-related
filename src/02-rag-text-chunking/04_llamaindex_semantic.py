"""
语义分块示例
"""

# 导入LlamaIndex核心组件和分块工具
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import (
    SentenceSplitter,           # 基础句子分块器
    SemanticSplitterNodeParser, # 语义分块器
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 初始化中文嵌入模型（需确保已安装llama-index-embeddings-huggingface）
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh")

def print_nodes(title, nodes):
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

# 读取文档
try:
    documents = SimpleDirectoryReader(
        input_files=["data/txt/xiaomi14Ultra.txt"]
    ).load_data()
    print(f"[调试] 文档数量: {len(documents)}")
    print(f"[调试] 文档前200字: {documents[0].text[:200]}")
except Exception as e:
    print(f"[错误] 文档读取失败: {e}")
    documents = []

# 基础句子分块
try:
    base_splitter = SentenceSplitter()
    sentence_nodes = base_splitter.get_nodes_from_documents(documents)
    print(f"[调试] 分句节点数: {len(sentence_nodes)}")
    if sentence_nodes:
        print(f"[调试] 第一个句子节点内容: {sentence_nodes[0].text[:100]}")
    print_nodes("基础句子分块结果展示", sentence_nodes)
except Exception as e:
    print(f"[错误] 基础句子分块失败: {e}")

# 语义分块（只能对Document对象分块，当前llama-index版本不支持Node输入）
try:
    splitter = SemanticSplitterNodeParser(
        buffer_size=2,
        breakpoint_percentile_threshold=50,
        embed_model=embed_model,
    )
    semantic_nodes = splitter.get_nodes_from_documents(documents)
    print_nodes("直接对文档语义分块结果展示", semantic_nodes)
except Exception as e:
    print(f"[错误] 语义分块失败: {e}")

# 说明：当前llama-index==0.12.46版本，SemanticSplitterNodeParser仅支持Document输入，
# 不支持Node链式分块。如需更灵活分块，请关注官方API更新或升级至更高版本。
