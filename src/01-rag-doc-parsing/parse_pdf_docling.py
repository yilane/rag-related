import logging
import os
from pathlib import Path
import sys
import argparse

# 保证可以跨目录导入utils工具函数
sys.path.append(str(Path(__file__).resolve().parent.parent))
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
)
from docling.datamodel.base_models import InputFormat
from utils.file_utils import ensure_output_dir, get_output_filename


def convert_pdf_to_markdown(source_path, output_path, device="cuda"):
    """
    将PDF文件转换为Markdown格式，支持图片和表格结构提取。
    :param source_path: PDF文件路径
    :param output_path: Markdown输出目录
    :param device: 推理设备（如'cpu'、'cuda'、'cuda:0'）
    """
    ensure_output_dir(output_path)
    output_file = os.path.join(output_path, get_output_filename(source_path))
    try:
        logging.info(f"Converting: {source_path} (device={device})")
        # 配置PDF解析参数
        pipeline_options = PdfPipelineOptions(
            # generate_picture_images=True: 提取文档中的图片并嵌入Markdown
            generate_picture_images=True,
            # images_scale=2.0: 图片分辨率提升2倍
            images_scale=2.0,
            # do_table_structure=True: 尝试识别和导出表格结构
            do_table_structure=True,
            # accelerator_options: 配置推理设备和线程数（可选'cpu'或'cuda'）
            accelerator_options=AcceleratorOptions(device=device, num_threads=8),
        )
        # 初始化DocumentConverter
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        # 执行PDF到结构化文档的转换
        result = converter.convert(source_path)
        # 导出为Markdown格式
        md_content = result.document.export_to_markdown()
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
            convert_pdf_to_markdown(pdf_file, output_path, device=device)
    else:
        # 单文件处理
        convert_pdf_to_markdown(source_path, output_path, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_path", type=str, default="data/pdf", help="PDF文件路径或目录"
    )
    parser.add_argument(
        "--output_path", type=str, default="output/docling", help="Markdown输出目录"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="推理设备，如'cpu'、'cuda'、'cuda:0'"
    )
    args = parser.parse_args()
    source_path = args.source_path
    output_path = args.output_path
    device = args.device
    main()
