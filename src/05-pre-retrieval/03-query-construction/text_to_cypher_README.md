# Text-to-Cypher 医疗知识图谱查询系统

这是一个将自然语言问题转换为Cypher查询语句的系统，专门针对医疗知识图谱进行优化。

## 项目文件结构

本项目包含以下核心文件，按功能模块组织：

```
03-query-construction/
├── 02_build_neo4j_testdata.py      # Neo4j测试数据构建脚本（Text-to-Cypher前置准备）
├── 02_text_to_cypher.py            # Text-to-Cypher核心实现（主程序）
├── test_additional_queries.py      # Text-to-Cypher扩展测试脚本
├── text_to_cypher_test_report.md   # Text-to-Cypher完整测试报告
└── README.md                       # 本说明文档
```

**文件命名说明**：
- `02_build_neo4j_testdata.py`：为text_to_cypher系统构建Neo4j测试数据的脚本
- `02_text_to_cypher.py`：text_to_cypher系统的主要实现文件
- 所有相关测试和文档文件都以"text_to_cypher"命名，便于识别归属

## 功能特性

- 🔄 自然语言到Cypher查询的自动转换
- 🏥 专门优化的医疗知识图谱结构
- 🤖 基于DeepSeek大语言模型的智能查询生成
- ✅ Cypher语法验证和错误处理
- 📊 查询结果格式化输出
- 🎯 交互式查询和批量测试模式

## 环境要求

- Python 3.8+
- Neo4j数据库
- DeepSeek API密钥

## 快速开始

### 1. 启动Neo4j数据库

使用Docker快速启动Neo4j：

```bash
docker run -d --name neo4j-text2cypher -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password123 neo4j:latest
```

### 2. 安装Python依赖

```bash
pip install neo4j python-dotenv openai
```

### 3. 配置环境变量

编辑 `.env` 文件，添加你的DeepSeek API密钥：

```
DEEPSEEK_API_KEY=your_actual_deepseek_api_key_here
```

### 4. 初始化测试数据（Text-to-Cypher前置准备）

```bash
python 02_build_neo4j_testdata.py
```

这将为Text-to-Cypher系统创建一个包含疾病、症状、药物及其关系的医疗知识图谱测试数据。

### 5. 运行Text-to-Cypher查询系统

#### 交互式模式：
```bash
python 02_text_to_cypher.py
```

#### 批量测试模式：
```bash
python 02_text_to_cypher.py test
```

#### 扩展功能测试：
```bash
python test_additional_queries.py
```

## 数据结构

### 节点类型

- **Disease（疾病）**: name, code, type, description
- **Symptom（症状）**: name, severity, description  
- **Drug（药物）**: name, type, indication, dosage

### 关系类型

- **HAS_SYMPTOM**: 疾病与症状的关系
- **TREATS**: 药物治疗疾病的关系

## 示例查询

系统支持以下类型的自然语言查询：

1. **疾病症状查询**
   - "查找糖尿病的所有症状"
   - "糖尿病有什么症状？"

2. **药物治疗查询**
   - "什么药物可以治疗高血压"
   - "治疗糖尿病用什么药"

3. **症状疾病查询**
   - "发热可能是哪些疾病的症状"
   - "头痛是什么病的症状"

4. **疾病分类查询**
   - "查找所有呼吸系统疾病"
   - "有哪些内分泌疾病"

5. **药物信息查询**
   - "阿莫西林可以治疗什么疾病"
   - "查找所有抗生素类药物"

## 系统架构

### Text-to-Cypher工作流程
```
用户自然语言问题
        ↓
   LLM智能理解
        ↓
   生成Cypher查询
        ↓
   语法验证
        ↓
   执行查询
        ↓
   结果格式化输出
```

### 文件协作关系
```
02_build_neo4j_testdata.py (数据准备)
            ↓
      Neo4j数据库 (医疗知识图谱)
            ↓
02_text_to_cypher.py (核心系统) ← test_additional_queries.py (扩展测试)
            ↓
text_to_cypher_test_report.md (测试结果文档)
```

## 核心功能

### TextToCypherConverter 类
**位置**: `02_text_to_cypher.py`

主要方法：

- `get_database_schema()`: 获取数据库模式信息
- `generate_cypher_query(user_query)`: 生成Cypher查询语句
- `execute_cypher_query(cypher)`: 执行Cypher查询
- `validate_cypher_syntax(cypher)`: 验证Cypher语法
- `format_results(results)`: 格式化查询结果
- `query(user_query)`: 主要查询方法

### 辅助功能

- **测试数据构建**: `02_build_neo4j_testdata.py` - 自动创建医疗知识图谱测试环境
- **扩展测试**: `test_additional_queries.py` - 提供额外的复杂查询测试用例
- **测试报告**: `text_to_cypher_test_report.md` - 详细的功能测试和性能分析报告

## Neo4j Web界面

启动后可以通过浏览器访问Neo4j：
- URL: http://localhost:7474
- 用户名: neo4j
- 密码: password123

## 注意事项

1. 确保Neo4j容器正常运行
2. 配置正确的DeepSeek API密钥
3. 初始化测试数据后再运行查询
4. 查询语句区分大小写，系统会自动处理模糊匹配

## 扩展功能

- 支持更复杂的医疗知识图谱结构
- 添加更多医疗实体类型（如医生、医院、检查项目等）
- 支持多跳查询和复杂关系推理
- 添加查询历史和缓存机制

## 问题排查

### 常见问题

1. **连接Neo4j失败**
   - 检查Docker容器是否正常运行
   - 确认端口7687未被占用

2. **API调用失败**
   - 验证DeepSeek API密钥是否正确
   - 检查网络连接

3. **查询结果为空**
   - 确认测试数据已正确导入
   - 检查查询语句是否符合数据结构

## 开发计划

- [ ] 支持更多LLM模型
- [ ] 添加查询优化建议
- [ ] 实现查询结果可视化
- [ ] 支持批量数据导入

## 文件规范说明

本Text-to-Cypher项目采用以下命名规范：

1. **核心文件**: 以`02_text_to_cypher.py`为主程序文件
2. **数据准备**: `02_build_neo4j_testdata.py`负责构建测试环境
3. **测试文件**: 所有测试相关文件包含`text_to_cypher`关键词
4. **文档文件**: 所有说明文档明确标注与text_to_cypher的关联性

这种命名方式确保了项目文件的清晰组织和易于维护。