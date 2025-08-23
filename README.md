# 大模型RAG的学习笔记

## 目录

### 1. **什么是RAG？**
- 1.1 [给AI装个“超级大脑”！5分钟带你搞懂RAG，原来这么简单！](https://mp.weixin.qq.com/s/nAEEkEvrb-WR_MzvuuvYXQ)
  
### 2. **初识RAG应用开发**
- 2.1 [RAG开发环境准备](RAG开发环境准备.md)
- 2.2 [开发简单RAG应用](src/00-simple-rag)
- 2.3 [手工制作一个RAG框架](https://github.com/yilane/rag-framework)

### 3. **RAG核心流程**

#### 3.1 索引阶段
 - 3.1.1 [文档解析](src/01-rag-doc-parsing)
 - 3.1.2 [文本分块](src/02-rag-text-chunking)
 - 3.1.3 [信息嵌入](src/03-rag-embedding)
 - 3.1.4 [向量存储](src/04-rag-vector-storage/)
 - 3.1.5 [实战：ICD-10 RAG检索项目Demo](src/04-rag-vector-storage/icd10-vectors-demo)
#### 3.2 检索阶段
 - 3.2.1 预检索优化
   - [查询翻译](src/05-pre-retrieval/01-query-translation)
   - [查询路由](src/05-pre-retrieval/02-query-routing)
 - 3.2.2 索引优化
 - 3.2.3 后检索优化

#### 3.3 生成阶段
 - 3.3.1 响应生成

### 4. RAG系统评估
...

### 5. 高级RAG
...

