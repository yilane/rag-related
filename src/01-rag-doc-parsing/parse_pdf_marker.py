import logging
import os
import sys
from pathlib import Path
import argparse

# 保证可以跨目录导入utils工具函数
sys.path.append(str(Path(__file__).resolve().parent.parent))
from marker.converters.pdf import PdfConverter  # pip install marker-pdf[full]
from marker.config.parser import ConfigParser
from marker.models import create_model_dict
from utils.file_utils import ensure_output_dir, get_output_filename


# =====================
# 可选：全局环境变量控制（如需全局强制CPU，可取消注释）
# os.environ["TORCH_DEVICE"] = "cpu"


# =====================
# 主转换函数：调用PdfConverter实现，支持device参数
# =====================
def convert_pdf_to_markdown(source_path, output_path, device="cuda"):
    """
    使用Marker的PdfConverter将PDF文件转换为Markdown格式。
    :param source_path: PDF文件路径
    :param output_path: Markdown输出目录
    :param device: 推理设备（如'cpu'、'cuda'、'cuda:0'）
    """
    ensure_output_dir(output_path)
    output_file = os.path.join(output_path, get_output_filename(source_path))
    try:
        logging.info(f"Converting: {source_path} (device={device})")

        config = {
            "output_format": "markdown",
            "output_file": output_file,
            "source_file": source_path,
        }
        # 初始化配置解析器
        config_parser = ConfigParser(config)
        converter_config = config_parser.generate_config_dict()

        # 初始化PdfConverter，配置转换参数， device可选'cpu'、'cuda'等
        converter = PdfConverter(
            # config: 转换配置字典，包含输出格式、文件路径等设置
            config=converter_config,
            # artifact_dict: 模型字典，指定使用的AI模型和设备（如CUDA加速）
            artifact_dict=create_model_dict(device=device),
            # processor_list: 处理器列表，定义PDF解析的各个处理步骤
            processor_list=config_parser.get_processors(),
            # renderer: 渲染器，负责将解析结果转换为最终输出格式
            renderer=config_parser.get_renderer(),
        )
        result = converter(source_path)
        md_content = result.markdown if hasattr(result, "markdown") else str(result)
        # 写入Markdown文件
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"已保存为 Markdown 文件：{output_file}")
    except Exception as e:
        # 捕获并记录所有异常，便于排查问题
        logging.error(f"Failed to convert {source_path}: {e}")


# =====================
# 主入口：支持单文件和目录批量处理
# =====================
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
        "--output_path", type=str, default="output/marker", help="Markdown输出目录"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="推理设备，如'cpu'、'cuda'、'cuda:0'"
    )
    args = parser.parse_args()
    source_path = args.source_path
    output_path = args.output_path
    device = args.device
    main()
