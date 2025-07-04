# MonkeyOCR PDF解析工具

## 简介

MonkeyOCR是一个基于Structure-Recognition-Relation (SRR)三元范式的轻量级文档解析模型，专门用于高精度的PDF文档解析。它采用"在哪里？"（结构）、"是什么？"（识别）和"如何组织？"（关系）的方法来简化文档解析流程。

## 主要特性

- **高精度解析**：在公式识别方面提升15.0%，表格处理提升8.6%
- **快速处理**：每秒处理0.84页，比MinerU(0.65页/秒)和Qwen2.5-VL-7B(0.12页/秒)更快
- **轻量级模型**：3B参数模型可在单张RTX 3090 GPU上高效运行
- **支持多种文档**：书籍、幻灯片、财务报告、教科书、试卷、杂志、学术论文、笔记、报纸等9种文档类型

## 安装方法

### 1. 从GitHub安装

```bash
# 克隆MonkeyOCR仓库
git clone https://github.com/Yuliang-Liu/MonkeyOCR.git
cd MonkeyOCR

# 安装依赖
pip install -r requirements.txt

# 根据需要下载模型权重
# 具体下载链接请参考官方GitHub页面
```

### 2. 环境要求

- Python >= 3.8
- PyTorch >= 1.8
- CUDA支持（推荐，也支持CPU）
- 至少8GB GPU内存（对于3B模型）

## 使用方法

### 1. 基本用法

```bash
# 解析单个PDF文件
python parse_pdf_monkeyocr.py --source_path /path/to/your/document.pdf --output_path /path/to/output

# 批量解析整个目录的PDF文件
python parse_pdf_monkeyocr.py --source_path /path/to/pdf/directory --output_path /path/to/output

# 指定使用CPU
python parse_pdf_monkeyocr.py --source_path document.pdf --device cpu

# 指定特定GPU
python parse_pdf_monkeyocr.py --source_path document.pdf --device cuda:1
```

### 2. 命令行参数

- `--source_path`: PDF文件路径或包含PDF文件的目录
- `--output_path`: Markdown输出目录（默认：output/monkeyocr）
- `--device`: 推理设备，支持'cpu'、'cuda'、'cuda:0'等（默认：cuda）

### 3. Python脚本示例

```python
from parse_pdf_monkeyocr import convert_pdf_to_markdown

# 解析单个文件
convert_pdf_to_markdown(
    source_path="document.pdf",
    output_path="output/monkeyocr",
    device="cuda"
)
```

## 性能表现

### 基准测试结果

| 文档类型 | MonkeyOCR-3B | MinerU | 改进幅度 |
|---------|--------------|--------|----------|
| 书籍 | 0.046 | 0.055 | +16.4% |
| 幻灯片 | 0.120 | 0.124 | +3.2% |
| 财务报告 | 0.024 | 0.033 | +27.3% |
| 教科书 | 0.100 | 0.102 | +2.0% |
| 试卷 | 0.129 | 0.159 | +18.9% |
| 学术论文 | 0.024 | 0.025 | +4.0% |

### 与其他模型对比

| 模型 | 参数量 | 平均性能 | 处理速度 |
|------|--------|----------|----------|
| MonkeyOCR-3B | 3B | **最优** | 0.84页/秒 |
| Qwen2.5-VL-7B | 7B | 良好 | 0.12页/秒 |
| GPT-4o | 未知 | 良好 | 未公开 |
| MinerU | - | 基准 | 0.65页/秒 |

## 支持的文档类型

1. **书籍** - 小说、技术书籍等
2. **幻灯片** - PPT演示文稿
3. **财务报告** - 年报、季报等
4. **教科书** - 学术教材
5. **试卷** - 考试题目
6. **杂志** - 期刊杂志
7. **学术论文** - 研究论文
8. **笔记** - 手写或打印笔记
9. **报纸** - 新闻报纸

## 故障排除

### 常见问题

1. **模型未找到错误**
   ```
   解决方案：确保已正确安装MonkeyOCR并下载了模型权重
   ```

2. **内存不足错误**
   ```
   解决方案：
   - 使用CPU推理：--device cpu
   - 减少批处理大小
   - 使用更小的模型变体
   ```

3. **CUDA错误**
   ```
   解决方案：
   - 检查CUDA安装
   - 更新GPU驱动
   - 确认GPU内存充足
   ```

### 调试选项

```bash
# 启用详细日志
python parse_pdf_monkeyocr.py --source_path document.pdf --verbose

# 保留中间文件用于调试
python parse_pdf_monkeyocr.py --source_path document.pdf --keep-intermediate
```

## 注意事项

1. **GPU内存要求**：建议至少8GB显存用于3B模型
2. **文档质量**：高质量的PDF文档能获得更好的解析效果
3. **语言支持**：主要针对中英文文档优化
4. **商业使用**：请查看官方许可协议

## 相关链接

- [MonkeyOCR GitHub](https://github.com/Yuliang-Liu/MonkeyOCR)
- [在线演示](http://vlrlabmonkey.xyz:7685)
- [论文地址](https://arxiv.org/abs/2506.05218)
- [Hugging Face模型](https://huggingface.co/echo840/MonkeyOCR)

## 更新日志

- **v1.0.0** (2025-01-14): 初始版本发布
  - 支持基本PDF解析功能
  - 集成MonkeyOCR SRR范式
  - 支持批量处理
  - CPU/GPU设备选择

## 贡献

欢迎提交Issue和Pull Request来改进这个工具。

## 许可证

本工具遵循Apache 2.0许可证。MonkeyOCR模型请参考其官方许可协议。 