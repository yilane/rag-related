import logging
import os
import sys
from pathlib import Path
import argparse
import subprocess
import glob
# 保证可以跨目录导入utils工具函数
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.file_utils import ensure_output_dir

"""
安装 MinerU
使用 pip 或 uv 安装
```
pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple
pip install uv -i https://mirrors.aliyun.com/pypi/simple
uv pip install -U "mineru[core]" -i https://mirrors.aliyun.com/pypi/simple 
``` 
"""
def convert_pdf_to_markdown(source_path, output_path, device="cpu"):
    """
    使用MinerU命令行工具将PDF文件转换为Markdown格式。
    :param source_path: PDF文件路径
    :param output_path: Markdown输出目录
    :param device: 推理设备（如'cpu'、'cuda'等）
    """
    ensure_output_dir(output_path)
    try:
        logging.info(f"Converting: {source_path}")
        cmd = [
            "mineru",
            "-p", source_path,
            "-o", output_path,
            "-d", device
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"MinerU 已完成解析，结果保存在: {output_path}")
        else:
            logging.error(f"mineru failed: {result.stderr}")
    except Exception as e:
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
            convert_pdf_to_markdown(pdf_file, output_path, device=device)
    else:
        # 单文件处理
        convert_pdf_to_markdown(source_path, output_path, device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, default="data/pdf", help="PDF文件路径或目录")
    parser.add_argument("--output_path", type=str, default="output/mineru", help="Markdown输出目录")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备，如'cpu'、'cuda'、'cuda:0'")
    args = parser.parse_args()
    source_path = args.source_path
    output_path = args.output_path
    device = args.device
    main() 