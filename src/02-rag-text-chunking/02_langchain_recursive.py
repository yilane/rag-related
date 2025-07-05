"""
递归分块示例

本脚本演示如何使用 LangChain 的 RecursiveCharacterTextSplitter 对文本文件进行递归分块处理。
适用于需要优先按语义分隔符（如段落、句子、标点、空格）切分文本的场景。

主要流程：
1. 加载本地 txt 文档
2. 按指定分隔符优先级、chunk_size 和 chunk_overlap 递归分块
3. 输出分块总数、每块内容、长度和元数据信息

参数说明：
- chunk_size: 每个文本块的最大字符数
- chunk_overlap: 相邻文本块之间的重叠字符数
- separators: 分割符优先级列表，依次尝试分割

文件路径、分块参数可根据实际需求调整。
"""
# 导入 LangChain 文档加载器和递归分块器
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 步骤1：加载本地 txt 文档，返回 Document 对象列表
loader = TextLoader("data/txt/xiaomi14Ultra.txt")  # 指定待分块的文本文件路径
documents = loader.load()  # 读取文件内容

# 步骤2：定义分割符优先级列表，优先按段落、句号、逗号、空格分割
separators = ["\n\n", ".", "，", " "]  # 可根据文本特点调整

# 步骤3：初始化递归分块器
# chunk_size: 每块最大字符数；chunk_overlap: 块间重叠字符数；separators: 分割符优先级
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, 
    chunk_overlap=10, 
    separators=separators
)

# 步骤4：对文档进行递归分块，返回分块后的 Document 列表
chunks = text_splitter.split_documents(documents)

# 步骤5：格式化输出分块结果
print("\n" + "="*60)
print("📄 文档分块结果")
print("="*60)

for i, chunk in enumerate(chunks, 1):
    # 输出每个分块的编号、内容长度、内容预览和元数据
    print(f"\n🔹 第 {i:2d} 个文档块")
    print("─" * 40)
    print(f"📝 内容长度: {len(chunk.page_content)} 字符")
    print(f"📋 内容预览: {chunk.page_content[:100]}{'...' if len(chunk.page_content) > 100 else ''}")
    print(f"🏷️  元数据: {chunk.metadata}")
    print("─" * 40)
