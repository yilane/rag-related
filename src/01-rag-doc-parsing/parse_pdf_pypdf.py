import logging
import os
import sys
from pathlib import Path
import argparse

# 保证可以跨目录导入utils工具函数
sys.path.append(str(Path(__file__).resolve().parent.parent))
from langchain_community.document_loaders import PyPDFLoader
from utils.file_utils import ensure_output_dir, get_output_filename


def convert_pdf_to_markdown(source_path, output_path):
    """
    使用PyPDFLoader将PDF文件转换为Markdown格式。
    :param source_path: PDF文件路径
    :param output_path: Markdown输出目录
    """
    ensure_output_dir(output_path)
    output_file = os.path.join(output_path, get_output_filename(source_path))
    try:
        logging.info(f"Converting: {source_path}")
        loader = PyPDFLoader(source_path)
        pages = loader.load()
        print(f"加载了 {len(pages)} 页PDF文档")
        md_lines = []
        for i, page in enumerate(pages, 1):
            md_lines.append(page.page_content)
        md_content = "\n".join(md_lines)
        # 写入Markdown文件
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"已保存为 Markdown 文件：{output_file}")
    except Exception as e:
        # 捕获并记录所有异常，便于排查问题
        logging.error(f"Failed to convert {source_path}: {e}")


def main():
    """
    主入口，支持批量和单文件处理。
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if os.path.isdir(source_path):
        # 批量处理目录下所有PDF文件
        pdf_files = [str(p) for p in Path(source_path).glob("*.pdf")]
        if not pdf_files:
            logging.warning(f"No PDF files found in directory: {source_path}")
        for pdf_file in pdf_files:
            convert_pdf_to_markdown(pdf_file, output_path)
    else:
        # 单文件处理
        convert_pdf_to_markdown(source_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_path", type=str, default="data/pdf", help="PDF文件路径或目录"
    )
    parser.add_argument(
        "--output_path", type=str, default="output/pypdf", help="Markdown输出目录"
    )
    args = parser.parse_args()
    source_path = args.source_path
    output_path = args.output_path
    main()
