"""
Text-to-SQL查询构建系统 - DeepSeek版本
使用DeepSeek大语言模型将自然语言查询转换为SQL语句
实现智能化的数据库查询构建，支持复杂查询场景
"""

import sqlite3
import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate

# 加载环境变量
load_dotenv()


class TextToSQLConverter:
    """文本到SQL转换器"""

    def __init__(self, db_path="sales_database.db"):
        """
        初始化转换器

        Args:
            db_path (str): SQLite数据库文件路径
        """
        self.db_path = db_path
        self.conn = None
        self.cursor = None

    def connect_database(self):
        """连接数据库"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            print(f"✅ 成功连接数据库: {self.db_path}")
        except Exception as e:
            print(f"❌ 数据库连接失败: {e}")
            raise

    def create_sample_database(self):
        """创建示例数据库和数据"""
        try:
            # 创建销售数据表
            self.cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS sales_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_name TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                sale_date TEXT NOT NULL,
                revenue REAL NOT NULL,
                region TEXT NOT NULL,
                customer_type TEXT NOT NULL
            )
            """
            )

            # 清空现有数据
            self.cursor.execute("DELETE FROM sales_data")

            # 插入示例数据
            sample_data = [
                ("iPhone 15", 10, "2023-07-15", 15000.0, "北京", "个人"),
                ("iPhone 15", 5, "2023-08-10", 7500.0, "上海", "个人"),
                ("MacBook Pro", 20, "2023-07-20", 40000.0, "广州", "企业"),
                ("iPhone 15", 15, "2023-09-01", 22500.0, "深圳", "个人"),
                ("iPad Air", 7, "2023-09-15", 3500.0, "北京", "个人"),
                ("MacBook Air", 12, "2023-08-25", 12000.0, "上海", "企业"),
                ("iPhone 14", 8, "2023-07-30", 8000.0, "成都", "个人"),
                ("iPad Pro", 6, "2023-09-10", 7200.0, "杭州", "企业"),
                ("MacBook Pro", 3, "2023-08-05", 6000.0, "武汉", "个人"),
                ("iPhone 15 Pro", 25, "2023-09-20", 37500.0, "北京", "企业"),
            ]

            self.cursor.executemany(
                """
            INSERT INTO sales_data (product_name, quantity, sale_date, revenue, region, customer_type)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
                sample_data,
            )

            self.conn.commit()
            print("✅ 示例数据库和数据创建成功")

            # 显示数据统计
            self.cursor.execute("SELECT COUNT(*) FROM sales_data")
            count = self.cursor.fetchone()[0]
            print(f"📊 数据库中共有 {count} 条销售记录")

        except Exception as e:
            print(f"❌ 数据库创建失败: {e}")
            raise

    def generate_sql(self, natural_query):
        """
        使用DeepSeek模型生成SQL查询
        将所有SQL生成相关的初始化和配置集中在此函数中，便于理解整个生成流程

        Args:
            natural_query (str): 自然语言查询

        Returns:
            str: 生成的SQL语句
        """
        try:
            print("🔧 步骤1: 初始化DeepSeek模型...")
            # 初始化DeepSeek模型
            llm = ChatDeepSeek(
                model="deepseek-chat",
                temperature=0.1,  # 设置较低温度确保SQL生成的准确性
                max_tokens=1024,
                api_key=os.getenv("DEEPSEEK_API_KEY"),
            )

            print("📋 步骤2: 准备数据库Schema信息...")
            # 数据库schema信息
            schema_info = """
                数据库Schema信息：
                
                表名: sales_data
                字段说明:
                - id: INTEGER PRIMARY KEY (主键，自增)
                - product_name: TEXT (产品名称)
                - quantity: INTEGER (销售数量)
                - sale_date: TEXT (销售日期，格式：YYYY-MM-DD)
                - revenue: REAL (销售收入)
                - region: TEXT (销售区域)
                - customer_type: TEXT (客户类型：个人/企业)
            """

            print("📝 步骤3: 创建SQL生成提示模板...")
            # 创建SQL生成提示模板
            sql_prompt = ChatPromptTemplate.from_template(
                """
你是一个专业的SQL查询生成专家。基于以下数据库schema，将用户的自然语言查询转换为准确的SQL语句。
{schema}
规则要求：
1. 只生成SQL语句，不要包含任何解释文字
2. 确保SQL语法正确且符合SQLite标准
3. 使用适当的聚合函数、WHERE条件、GROUP BY、ORDER BY等
4. 日期比较请使用字符串比较（如：sale_date >= '2023-07-01'）
5. 产品名称匹配请使用LIKE操作符支持模糊查询
6. 如果查询涉及时间范围，请合理解释季度、月份等时间概念
用户查询: {query}
SQL语句:"""
            )

            print("⚙️ 步骤4: 格式化提示词...")
            # 格式化提示词
            formatted_prompt = sql_prompt.format(
                schema=schema_info, query=natural_query
            )

            print("🤖 步骤5: 调用DeepSeek模型生成SQL...")
            # 调用DeepSeek模型生成SQL
            response = llm.invoke(formatted_prompt)
            sql_query = response.content.strip()

            print("✨ 步骤6: 清理和格式化SQL语句...")
            # 清理SQL语句（移除可能的markdown格式）
            if sql_query.startswith("```sql"):
                sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            elif sql_query.startswith("```"):
                sql_query = sql_query.replace("```", "").strip()

            print("✅ SQL生成完成!")
            return sql_query

        except Exception as e:
            print(f"❌ SQL生成失败: {e}")
            return None

    def execute_sql(self, sql_query):
        """
        执行SQL查询

        Args:
            sql_query (str): SQL查询语句

        Returns:
            list: 查询结果
        """
        try:
            self.cursor.execute(sql_query)
            results = self.cursor.fetchall()

            # 获取列名
            column_names = [description[0] for description in self.cursor.description]

            return results, column_names

        except Exception as e:
            print(f"❌ SQL执行失败: {e}")
            return None, None

    def format_results(self, results, column_names):
        """
        格式化查询结果

        Args:
            results (list): 查询结果
            column_names (list): 列名列表

        Returns:
            str: 格式化后的结果字符串
        """
        if not results:
            return "查询结果为空"

        # 计算每列的最大宽度
        col_widths = []
        for i, col_name in enumerate(column_names):
            max_width = len(col_name)
            for row in results:
                if i < len(row) and row[i] is not None:
                    max_width = max(max_width, len(str(row[i])))
            col_widths.append(max_width + 2)

        # 构建格式化字符串
        formatted_output = []

        # 表头
        header = "|".join(
            f"{col_name:<{width}}" for col_name, width in zip(column_names, col_widths)
        )
        formatted_output.append(header)
        formatted_output.append("-" * len(header))

        # 数据行
        for row in results:
            row_str = "|".join(
                f"{str(cell) if cell is not None else 'NULL':<{width}}"
                for cell, width in zip(row, col_widths)
            )
            formatted_output.append(row_str)

        return "\n".join(formatted_output)

    def process_query(self, natural_query):
        """
        处理完整的查询流程

        Args:
            natural_query (str): 自然语言查询
        """
        print("=" * 80)
        print("🤖 Text-to-SQL 智能查询系统 (DeepSeek版本)")
        print("=" * 80)
        print(f"📝 用户查询: {natural_query}")
        print("-" * 80)

        # 生成SQL
        print("🔄 正在生成SQL查询...")
        sql_query = self.generate_sql(natural_query)

        if not sql_query:
            print("❌ SQL生成失败")
            return

        print(f"🔍 生成的SQL: {sql_query}")
        print("-" * 80)

        # 执行SQL
        print("⚡ 正在执行查询...")
        results, column_names = self.execute_sql(sql_query)

        if results is None:
            print("❌ 查询执行失败")
            return

        # 显示结果
        print("📊 查询结果:")
        if results:
            formatted_results = self.format_results(results, column_names)
            print(formatted_results)
            print(f"\n📈 共返回 {len(results)} 条记录")
        else:
            print("查询结果为空")

        print("=" * 80)

    def close_connection(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            print("✅ 数据库连接已关闭")


def main():
    """主函数 - 演示Text-to-SQL功能"""

    # 初始化转换器
    converter = TextToSQLConverter()

    try:
        # 连接数据库
        converter.connect_database()

        # 创建示例数据
        converter.create_sample_database()

        # 测试查询示例
        test_queries = [
            "2023年第三季度iPhone 15的总销售收入是多少？",
            "哪个产品在北京地区销量最高？",
            "企业客户购买了哪些产品，总金额是多少？",
            "按地区统计2023年8月的销售情况",
            "销售收入超过10000元的订单有哪些？",
        ]

        print("\n🚀 开始演示Text-to-SQL查询转换...")

        for i, query in enumerate(test_queries, 1):
            print(f"\n\n🔸 测试查询 {i}:")
            converter.process_query(query)

            if i < len(test_queries):
                input("\n按Enter键继续下一个查询...")

        # 交互式查询
        print("\n\n🎯 进入交互模式，请输入您的查询（输入'quit'退出）:")
        while True:
            user_input = input("\n请输入查询: ").strip()
            if user_input.lower() in ["quit", "exit", "退出"]:
                break
            if user_input:
                converter.process_query(user_input)

    except Exception as e:
        print(f"❌ 程序执行出错: {e}")

    finally:
        # 清理资源
        converter.close_connection()


if __name__ == "__main__":
    main()
