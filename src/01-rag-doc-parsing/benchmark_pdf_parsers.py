import os
import time
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# 工具脚本名与可读名称映射
TOOLS = {
    "parse_pdf_pypdf.py": "pypdf",
    "parse_pdf_pymupdf4llm.py": "pymupdf4llm",
    "parse_pdf_docling.py": "docling",
    "parse_pdf_marker.py": "marker",
    "parse_pdf_mineru.py": "mineru",
}

# 测试用PDF文件页数列表（假设已准备好不同页数的PDF，命名如 test_1.pdf, test_2.pdf ...）
PAGE_COUNTS = [1, 3, 7, 11, 33, 106]
PDF_DIR = "data/pdf"  # 你需要准备好这些PDF

# 结果存储
results = {tool: [] for tool in TOOLS.values()}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

for n_pages in PAGE_COUNTS:
    pdf_file = os.path.join(PDF_DIR, f"test_{n_pages}.pdf")
    if not os.path.isfile(pdf_file):
        logging.warning(f"PDF文件不存在: {pdf_file}")
        for tool in TOOLS.values():
            results[tool].append(None)
        continue
    for script, tool_name in TOOLS.items():
        script_path = os.path.join("src/01-rag-doc-parsing", script)
        cmd = f"python {script_path} --source_path {pdf_file} --output_path output/benchmark_{tool_name}"  # 需支持命令行参数
        start = time.time()
        exit_code = os.system(cmd)
        elapsed = time.time() - start
        if exit_code == 0:
            results[tool_name].append(elapsed)
            logging.info(f"{tool_name} 解析 {n_pages} 页 PDF 用时: {elapsed:.2f} 秒")
        else:
            results[tool_name].append(None)
            logging.error(f"{tool_name} 解析 {n_pages} 页 PDF 失败")

# 绘制柱状图
x = np.arange(len(PAGE_COUNTS))
bar_width = 0.12
plt.figure(figsize=(12, 6))
for i, (tool, times) in enumerate(results.items()):
    plt.bar(
        x + i * bar_width,
        [t if t is not None else 0 for t in times],
        width=bar_width,
        label=tool,
    )

plt.xlabel('PDF Pages')
plt.ylabel('Processing Time (seconds)')
plt.title('PDF Parsing Performance Comparison by Tools')
plt.xticks(x + bar_width * 1.5, PAGE_COUNTS)
plt.legend()
plt.tight_layout()
plt.savefig('output/pdf_processing_benchmark.png')
plt.show()

# 说明：
# 1. 你需要准备 data/benchmark_pdfs/test_1.pdf, test_2.pdf ... test_128.pdf
# 2. 各解析脚本需支持 --source_path 和 --output_path 命令行参数
# 3. 结果图片保存在 output/pdf_processing_benchmark.png
