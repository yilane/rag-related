# -*- coding: utf-8 -*-
"""
小红书热点新闻Self-Query Retriever示例
使用模拟的小红书热点新闻数据演示自查询检索功能
"""

from langchain_deepseek import ChatDeepSeek
from langchain_core.documents import Document
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 定义新闻元数据模型
class NewsMetadata(BaseModel):
    """小红书新闻元数据模型，定义了需要提取的新闻属性"""

    title: str = Field(description="新闻标题")
    category: str = Field(description="新闻分类")
    author: str = Field(description="作者昵称")
    likes_count: int = Field(description="点赞数")
    comments_count: int = Field(description="评论数")
    shares_count: int = Field(description="分享数")
    publish_date: str = Field(description="发布日期")
    tags: str = Field(description="标签，多个标签用逗号分隔")
    region: str = Field(description="地区")


# 创建模拟的小红书热点新闻数据
def create_mock_news_data():
    """创建模拟的小红书热点新闻数据"""

    # 模拟新闻数据
    mock_news = [
        {
            "title": "2024年最火爆的护肤品推荐，敏感肌必备！",
            "content": "今年最受欢迎的护肤品盘点，包括神仙水、SK-II精华、兰蔻小黑瓶等，特别适合敏感肌使用，亲测有效！",
            "category": "美妆护肤",
            "author": "美妆小达人Amy",
            "likes_count": 15680,
            "comments_count": 890,
            "shares_count": 456,
            "publish_date": "2024-03-15",
            "publish_day": 15,
            "tags": "护肤,敏感肌,美妆推荐",
            "region": "上海",
        },
        {
            "title": "成都必吃火锅店TOP10，本地人推荐！",
            "content": "成都火锅店深度测评，从老牌火锅到网红新店，包括海底捞、蜀大侠、小龙坎等，每一家都有独特风味。",
            "category": "美食探店",
            "author": "成都吃货王",
            "likes_count": 28900,
            "comments_count": 1245,
            "shares_count": 789,
            "publish_date": "2024-03-10",
            "publish_day": 10,
            "tags": "成都美食,火锅,探店推荐",
            "region": "成都",
        },
        {
            "title": "春季穿搭指南：温柔系女生必看搭配技巧",
            "content": "春天来了，分享一些温柔系穿搭技巧，包括色彩搭配、单品选择、配饰运用等，让你轻松穿出优雅气质。",
            "category": "时尚穿搭",
            "author": "时尚博主Lily",
            "likes_count": 12450,
            "comments_count": 567,
            "shares_count": 334,
            "publish_date": "2024-03-12",
            "publish_day": 12,
            "tags": "穿搭,春季搭配,温柔风",
            "region": "北京",
        },
        {
            "title": "日本樱花季旅游攻略，最佳赏樱地点推荐",
            "content": "2024年日本樱花季即将到来，分享最佳赏樱地点、交通攻略、住宿建议，让你的日本之行完美无缺。",
            "category": "旅游攻略",
            "author": "旅行达人小王",
            "likes_count": 45600,
            "comments_count": 2100,
            "shares_count": 1560,
            "publish_date": "2024-03-08",
            "publish_day": 8,
            "tags": "日本旅游,樱花季,旅游攻略",
            "region": "广州",
        },
        {
            "title": "居家健身必备器材推荐，小空间大效果",
            "content": "疫情时代居家健身成为趋势，推荐几款实用的健身器材，包括瑜伽垫、哑铃、弹力带等，适合小户型使用。",
            "category": "健身运动",
            "author": "健身教练Mark",
            "likes_count": 8760,
            "comments_count": 432,
            "shares_count": 198,
            "publish_date": "2024-03-13",
            "publish_day": 13,
            "tags": "健身,居家运动,器材推荐",
            "region": "深圳",
        },
        {
            "title": "大学生必备数码产品清单，性价比之选",
            "content": "为大学生推荐性价比超高的数码产品，包括笔记本电脑、平板、耳机、充电宝等，学习生活两不误。",
            "category": "数码科技",
            "author": "数码评测师Leo",
            "likes_count": 19800,
            "comments_count": 756,
            "shares_count": 445,
            "publish_date": "2024-03-11",
            "publish_day": 11,
            "tags": "数码产品,大学生,性价比",
            "region": "杭州",
        },
        {
            "title": "新手养猫完全指南，从选猫到养护全攻略",
            "content": "第一次养猫需要注意什么？从猫咪品种选择、必备用品、喂养方法到健康护理，新手铲屎官必看指南。",
            "category": "宠物生活",
            "author": "猫咪专家Sarah",
            "likes_count": 13560,
            "comments_count": 689,
            "shares_count": 287,
            "publish_date": "2024-03-14",
            "publish_day": 14,
            "tags": "养猫,宠物护理,新手指南",
            "region": "南京",
        },
        {
            "title": "职场新人必备技能，如何快速适应工作环境",
            "content": "初入职场如何快速融入团队？分享职场沟通技巧、工作效率提升方法、职业形象塑造等实用建议。",
            "category": "职场干货",
            "author": "职场导师Anna",
            "likes_count": 22100,
            "comments_count": 980,
            "shares_count": 567,
            "publish_date": "2024-03-09",
            "publish_day": 9,
            "tags": "职场技能,新人指南,工作经验",
            "region": "苏州",
        },
    ]

    # 将数据转换为Document对象
    documents = []
    for news in mock_news:
        metadata = {key: value for key, value in news.items() if key != "content"}
        doc = Document(page_content=news["content"], metadata=metadata)
        documents.append(doc)

    return documents


# 主程序
def main():
    print("=== 小红书热点新闻 Self-Query Retriever 示例 ===\n")

    # 创建模拟新闻数据
    print("📰 正在加载小红书热点新闻数据...")
    news_docs = create_mock_news_data()
    print(f"✅ 成功加载 {len(news_docs)} 条新闻数据\n")

    # 显示加载的新闻标题
    print("📋 已加载的新闻列表：")
    for i, doc in enumerate(news_docs, 1):
        print(f"{i}. {doc.metadata['title']}")
    print()

    # 创建向量存储
    print("🔄 正在创建向量存储...")
    embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
    vectorstore = Chroma.from_documents(news_docs, embed_model)
    print("✅ 向量存储创建完成\n")

    # 配置检索器的元数据字段
    metadata_field_info = [
        AttributeInfo(
            name="title",
            description="新闻标题（字符串）",
            type="string",
        ),
        AttributeInfo(
            name="category",
            description="新闻分类，如：美妆护肤、美食探店、时尚穿搭、旅游攻略等（字符串）",
            type="string",
        ),
        AttributeInfo(
            name="author",
            description="新闻作者昵称（字符串）",
            type="string",
        ),
        AttributeInfo(
            name="likes_count",
            description="点赞数量（整数）",
            type="integer",
        ),
        AttributeInfo(
            name="comments_count",
            description="评论数量（整数）",
            type="integer",
        ),
        AttributeInfo(
            name="shares_count",
            description="分享数量（整数）",
            type="integer",
        ),
        AttributeInfo(
            name="publish_date",
            description="发布日期，格式为YYYY-MM-DD的字符串",
            type="string",
        ),
        AttributeInfo(
            name="publish_day",
            description="发布日期的天数（整数，1-31）",
            type="integer",
        ),
        AttributeInfo(
            name="tags",
            description="新闻标签，多个标签用逗号分隔的字符串",
            type="string",
        ),
        AttributeInfo(
            name="region",
            description="发布地区（字符串）",
            type="string",
        ),
    ]

    # 创建自查询检索器
    print("🔧 正在初始化Self-Query Retriever...")

    llm = ChatDeepSeek(
        model="deepseek-chat", 
        temperature=0,
        api_key=os.getenv("DEEPSEEK_API_KEY")
    )

    retriever = SelfQueryRetriever.from_llm(
        llm=llm,                                    # 语言模型：用于理解查询意图并生成结构化查询
        vectorstore=vectorstore,                    # 向量存储：存储文档向量和元数据的数据库
        document_contents="包含小红书热点新闻的标题、分类、作者、互动数据等信息",  # 文档内容描述：帮助LLM理解文档结构
        metadata_field_info=metadata_field_info,   # 元数据字段信息：定义可查询的元数据字段及其类型
        enable_limit=True,                          # 启用限制：允许在查询中指定返回结果数量
        verbose=True,                               # 详细输出：显示查询处理的详细过程信息
    )
    print("✅ Self-Query Retriever 初始化完成\n")

    # 执行示例查询
    queries = [
        "找出点赞数超过20000的热门新闻",
        "显示美妆护肤类别的新闻",
        "查找来自成都地区的美食相关内容",
        "找出发布日期天数大于13的新闻",
        "显示评论数最多的旅游攻略",
        "查找包含'推荐'标签的新闻",
    ]

    # 执行查询并输出结果
    for query in queries:
        print(f"\n{'='*50}")
        print(f"🔍 查询：{query}")
        print(f"{'='*50}")

        try:
            results = retriever.invoke(query)

            if not results:
                print("❌ 未找到匹配的新闻")
                continue

            print(f"✅ 找到 {len(results)} 条匹配的新闻：\n")

            for i, doc in enumerate(results, 1):
                print(f"📰 新闻 {i}:")
                print(f"   标题：{doc.metadata['title']}")
                print(f"   分类：{doc.metadata['category']}")
                print(f"   作者：{doc.metadata['author']}")
                print(f"   地区：{doc.metadata['region']}")
                print(f"   发布日期：{doc.metadata['publish_date']}")
                print(
                    f"   点赞：{doc.metadata['likes_count']} | 评论：{doc.metadata['comments_count']} | 分享：{doc.metadata['shares_count']}"
                )
                print(f"   标签：{doc.metadata['tags']}")
                print(f"   内容预览：{doc.page_content[:50]}...")
                print(f"   {'-'*40}")

        except Exception as e:
            print(f"❌ 查询出错：{str(e)}")
            continue

    print(f"\n{'='*50}")
    print("🎉 Self-Query Retriever 示例演示完成！")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
