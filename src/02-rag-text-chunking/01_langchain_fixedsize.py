"""
固定大小分块示例

本脚本演示如何使用 LangChain 的 CharacterTextSplitter 对文本文件进行定长分块处理。
适用于需要将长文本按字符数均匀切片的场景。

主要流程：
1. 加载本地 txt 文档
2. 按指定 chunk_size 和 chunk_overlap 分块
3. 输出分块总数、每块内容、长度和元数据信息

参数说明：
- chunk_size: 每个文本块的最大字符数
- chunk_overlap: 相邻文本块之间的重叠字符数

文件路径、分块参数可根据实际需求调整。
"""
# 导入 LangChain 文档加载器和分块器
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
import json

# 步骤1：加载本地 txt 文档，返回 Document 对象列表
loader = TextLoader("data/txt/xiaomi14Ultra.txt")  # 指定待分块的文本文件路径
documents = loader.load()  # 读取文件内容

# 输出文档基本信息，便于调试
print("文档对象数量：", len(documents))
print("文档内容长度：", len(documents[0].page_content))
print("文档内容：", documents[0].page_content)

# 步骤2：初始化定长分块器
# chunk_size: 每块最大字符数；chunk_overlap: 块间重叠字符数
text_splitter = CharacterTextSplitter(
    chunk_size=50,  # 可根据实际需求调整
    chunk_overlap=10
)

# 步骤3：对文档进行分块，返回分块后的 Document 列表
chunks = text_splitter.split_documents(documents)

# 步骤4：格式化输出分块结果
print(f"\n=== 文档分块结果（共{len(chunks)}块） ===")
for i, chunk in enumerate(chunks, 1):
    # 输出每个分块的编号、内容（截断）、长度和元数据
    content = chunk.page_content if chunk.page_content else "内容为空"
    content_len = len(content)
    if content_len > 100:
        display_content = content[:100] + f"...（共{content_len}字）"
    else:
        display_content = content
    print(f"内容: {display_content}")
    print(f"长度: {content_len}")
    try:
        metadata_str = json.dumps(chunk.metadata, ensure_ascii=False, indent=2)
    except Exception as e:
        metadata_str = str(chunk.metadata)
    print(f"元数据:\n{metadata_str}")
    print("=" * 60)
