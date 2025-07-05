"""
代码分块示例

本脚本演示如何读取一个Python源码文件，并用LangChain的RecursiveCharacterTextSplitter按代码结构进行分块。
适用于代码理解、检索等场景。
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import Language
import os
import json

# 获取指定语言的分割符
separators = RecursiveCharacterTextSplitter.get_separators_for_language(Language.JS)
print(separators)

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

# 步骤1：读取目标Python源码文件内容
code_file_path = os.path.join(os.path.dirname(__file__), '../01-rag-doc-parsing/parse_pdf_marker.py')
with open(code_file_path, 'r', encoding='utf-8') as f:
    code_content = f.read()

# 步骤2：初始化针对Python的递归分块器
python_splitter = RecursiveCharacterTextSplitter.from_language(
   language=Language.PYTHON,  # 指定编程语言为Python
   chunk_size=1000,
   chunk_overlap=0
)

# 步骤3：对源码内容进行分块，create_documents参数为字符串列表
python_docs = python_splitter.create_documents([code_content])

# 步骤4：格式化输出分块结果
print(f"\n=== 代码分块结果（共{len(python_docs)}块） ===")
for i, chunk in enumerate(python_docs, 1):
    print(f"\n--- 第 {i} 个代码块 ---")
    content = chunk.page_content if chunk.page_content else "内容为空"
    content_len = len(content)
    if content_len > 120:
        display_content = content[:120] + f"...（共{content_len}字）"
    else:
        display_content = content
    print(f"代码内容: {display_content}")
    print(f"长度: {content_len}")
    try:
        metadata_str = json.dumps(chunk.metadata, ensure_ascii=False, indent=2)
    except Exception as e:
        metadata_str = str(chunk.metadata)
    print(f"元数据:\n{metadata_str}")
    print("=" * 60)