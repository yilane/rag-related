# ICD-10 RAG检索Demo项目

## 项目简介

基于ICD-10医保编码数据构建的RAG向量检索系统，通过NER命名实体识别和向量相似度检索，为用户主诉或医生诊断信息推荐最相似的ICD-10编码。

## 项目特点

- 🏥 **医学专用**: 针对ICD-10医疗编码优化的检索系统
- 🧠 **智能识别**: 集成NER实体识别，提取医学关键信息
- ⚡ **高效检索**: 基于Milvus向量数据库的毫秒级检索
- 📊 **层次分类**: 保持ICD-10完整的分类层次结构
- 🔍 **多策略**: 支持精确匹配、语义检索、分层过滤等多种策略

## 数据概况

- **数据源**: ICD-10医保1.0版.csv
- **数据量**: 35,877条医疗编码记录
- **分类层次**: 章(22个) → 节 → 三位码 → 四位码 → 疾病编码

## 技术架构

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Gradio UI     │───▶│    核心服务       │───▶│   Milvus向量库   │
│  - NER实体识别   │    │ - 实体识别服务     │    │ - ICD-10向量索引 │
│  - 诊断标准化    │    │ - 向量检索服务     │    │ - 相似度检索     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 核心组件

1. **数据构建模块** (`build_database.py`)
   - CSV数据加载和清洗
   - 文本向量化处理
   - Milvus数据库构建

2. **NER服务模块** (`medical_ner_service.py`)
   - 基于`lixin12345/chinese-medical-ner`的医学实体识别
   - 实体类型分类和过滤

3. **检索服务模块** (`search_service.py`)
   - 向量检索功能
   - 结果格式化

4. **Gradio UI模块** (`app.py`)
   - 🔍 实体识别界面：医学文本NER分析
   - 📋 诊断标准化界面：ICD-10编码检索
   - 交互式Web界面

## 快速开始

### 环境准备

**推荐使用conda环境**: embed_test

```bash
# Python环境要求
Python >= 3.10

# 核心依赖包版本
pymilvus==2.5.14              # Milvus向量数据库客户端
sentence-transformers==5.0.0   # 文本向量化模型
torch==2.7.1                  # PyTorch深度学习框架
transformers==4.53.2          # HuggingFace模型库
gradio==5.42.0                # Web UI界面框架
pandas==2.3.1                 # 数据处理库
numpy==2.2.6                  # 数值计算库
tqdm==4.67.1                  # 进度条显示

# 环境激活命令
source /home/ylins/miniconda3/etc/profile.d/conda.sh
conda activate embed_test

# 启动Milvus服务
docker-compose up -d milvus
```

### 数据构建

```bash
# 第一步：构建向量数据库
python build_database.py

# 预期输出：
# ✅ 数据加载完成: 35,877条记录
# ✅ 向量化处理完成: 1024维向量
# ✅ Milvus入库完成: icd10_diseases集合
# ✅ 索引构建完成: IVF_FLAT索引
```

### 启动Gradio界面

```bash
# 启动Gradio Web界面
python app.py

# 访问地址: http://localhost:7860
# 包含两个功能模块：
# 1. 命名实体识别
# 2. 诊断标准化检索
```

## 界面使用指南

### 功能模块一：命名实体识别 🔍

**功能说明**: 从医学文本中识别疾病、症状、身体部位等医学实体

**使用步骤**:
1. 在左侧输入框输入医学文本
2. 调整置信度阈值(推荐0.7)
3. 点击"识别实体"按钮
4. 在右侧查看识别结果

**输入示例**:
```
患者主诉胸痛3天，伴有呼吸困难和心悸，既往有高血压病史
```

**输出结果**:
- **实体识别表格**: 显示识别的实体、类型、置信度
- **实体统计**: 各类型实体的数量统计  
- **高亮显示**: 原文中实体的彩色高亮标注

### 功能模块二：诊断标准化 📋

**功能说明**: 基于诊断描述，智能匹配相应的ICD-10标准编码

**使用步骤**:
1. 在左侧输入诊断信息或疾病名称
2. 设置返回结果数量和相似度阈值
3. 点击"检索ICD-10编码"按钮
4. 查看匹配的编码结果

**输入示例**:
```
急性心肌梗死
```

**输出结果**:
- **检索结果表格**: ICD-10编码、疾病名称、相似度分数
- **实体分析**: 查询文本中识别的医学实体
- **详细信息**: 最佳匹配编码的完整分类信息

### 界面特色功能

- ✨ **实时处理**: 即时显示识别和检索结果
- 🎯 **参数调节**: 支持置信度、相似度阈值调整
- 📝 **示例输入**: 提供常见医学文本示例
- 🌈 **可视化**: 实体高亮显示，结果表格化展示
- 📱 **响应式**: 适配不同设备屏幕

## 项目结构

```
icd10-vectors-demo/
├── data/                               # 数据文件
│   └── ICD-10医保1.0版.csv
├── 核心服务模块                          # 源代码
│   ├── build_database.py               # 数据库构建
│   ├── milvus_service.py               # Milvus向量服务
│   ├── medical_ner_service.py          # 医学NER服务
│   ├── search_service.py               # 检索服务
│   └── config.py                       # 配置文件
├── UI界面和应用                          # 应用程序
│   ├── app.py                          # Gradio主应用
└── README.md                           # 项目说明文档
```

## 配置说明

### 简化配置 (`config.py`)

```python
# 基础配置
DATABASE_CONFIG = {
    'milvus_host': "localhost",
    'milvus_port': 19530,
    'collection_name': "icd10_diseases"
}

EMBEDDING_CONFIG = {
    'model_name': "BAAI/bge-large-zh-v1.5",
    'dimension': 1024,
    'batch_size': 32
}

NER_CONFIG = {
    'model_name': "lixin12345/chinese-medical-ner",
    'confidence_threshold': 0.7
}

GRADIO_CONFIG = {
    'server_port': 7860,
    'server_name': "0.0.0.0",
    'share': False
}
```

## 系统要求

### 基础环境
- **Python**: >= 3.10
- **内存**: >= 4GB
- **存储**: >= 5GB
- **Milvus**: >= 2.3.0

### 性能表现
- **NER识别速度**: < 100ms (单次)
- **向量检索速度**: < 200ms (Top-10)
- **界面响应**: 实时处理
- **支持并发**: 多用户同时使用
