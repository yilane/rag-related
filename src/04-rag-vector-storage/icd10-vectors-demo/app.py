# -*- coding: utf-8 -*-
"""
ICD-10 RAG检索系统 - Gradio Web界面
集成医学NER实体识别和智能ICD-10编码检索功能
"""

import gradio as gr
import pandas as pd
import logging
import signal
import sys
import atexit
import threading
import time
from typing import List, Dict, Any, Tuple

from medical_ner_service import MedicalNERService
from search_service import SearchService
from config import GRADIO_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局服务实例
ner_service = None
search_service = None

# 资源清理标志
cleanup_in_progress = False
cleanup_lock = threading.Lock()

def initialize_services():
    """初始化服务实例"""
    global ner_service, search_service
    try:
        logger.info("正在初始化服务...")
        ner_service = MedicalNERService()
        search_service = SearchService()
        logger.info("服务初始化完成")
        return True
    except Exception as e:
        logger.error(f"服务初始化失败: {e}")
        return False

def cleanup_all_resources():
    """清理所有服务资源"""
    global ner_service, search_service, cleanup_in_progress
    
    with cleanup_lock:
        if cleanup_in_progress:
            return
        cleanup_in_progress = True
    
    try:
        logger.info("🔄 开始清理所有服务资源...")
        
        # 清理搜索服务（会连锁清理其他服务）
        if search_service is not None:
            logger.info("清理搜索服务...")
            search_service.cleanup_resources()
            search_service = None
        
        # 如果NER服务是独立的，也清理它
        if ner_service is not None:
            logger.info("清理NER服务...")
            ner_service.cleanup_resources()
            ner_service = None
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        # 最后清理CUDA缓存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("全局CUDA缓存已清理")
        except ImportError:
            pass
        
        logger.info("✅ 所有服务资源清理完成")
        
    except Exception as e:
        logger.error(f"❌ 资源清理过程中出现错误: {e}")
    finally:
        cleanup_in_progress = False

def signal_handler(signum, frame):
    """信号处理器：处理Ctrl+C等中断信号"""
    signal_name = signal.Signals(signum).name
    logger.info(f"🛑 接收到信号 {signal_name}，正在优雅关闭服务...")
    
    # 清理资源
    cleanup_all_resources()
    
    # 给一点时间让清理完成
    time.sleep(2)
    
    logger.info("👋 服务已优雅关闭")
    sys.exit(0)

def setup_signal_handlers():
    """设置信号处理器"""
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # 终止信号
    
    # 注册退出时的清理函数
    atexit.register(cleanup_all_resources)
    
    logger.info("🔧 信号处理器已设置")

def graceful_shutdown():
    """优雅关闭函数"""
    logger.info("🔄 执行优雅关闭...")
    cleanup_all_resources()

def extract_entities_ui(text: str, confidence_threshold: float) -> Tuple[pd.DataFrame, Dict, str]:
    """UI界面的实体识别处理函数"""
    try:
        if not text.strip():
            return pd.DataFrame(), {}, ""
        
        # 调用NER服务
        entities = ner_service.extract_entities(text)
        
        # 过滤低置信度实体
        filtered_entities = [e for e in entities if e['confidence'] >= confidence_threshold]
        
        # 准备表格数据
        if filtered_entities:
            table_data = []
            for entity in filtered_entities:
                table_data.append([
                    entity['text'],
                    entity['label'],
                    f"{entity['confidence']:.3f}",
                    f"{entity['start']}-{entity['end']}"
                ])
            
            df = pd.DataFrame(table_data, columns=["实体文本", "实体类型", "置信度", "位置"])
            
            # 生成统计信息
            stats = ner_service.analyze_entities(filtered_entities)
            
            # 生成高亮文本
            highlighted = ner_service.highlight_entities(text, filtered_entities)
            
            return df, stats, highlighted
        else:
            return pd.DataFrame(), {"提示": "未识别到满足置信度要求的实体"}, text
            
    except Exception as e:
        logger.error(f"实体识别处理失败: {e}")
        error_msg = f"处理失败: {str(e)}"
        return pd.DataFrame(), {"错误": error_msg}, text

def search_icd_codes_ui(query_text: str, top_k: int, score_threshold: float, use_ner: bool) -> Tuple[pd.DataFrame, str, str]:
    """UI界面的ICD-10编码检索处理函数"""
    try:
        if not query_text.strip():
            return pd.DataFrame(), "", ""
        
        # 调用检索服务
        result = search_service.search_icd_codes(
            query_text=query_text,
            top_k=top_k,
            score_threshold=score_threshold,
            use_ner=use_ner
        )
        
        if result['success']:
            # 准备检索结果表格
            if result['results']:
                table_data = []
                for res in result['results']:
                    table_data.append([
                        res['rank'],
                        res['disease_code'],
                        res['disease_name'],
                        f"{res['similarity_score']:.4f}",
                        res['chapter_name'],
                        res['section_name']
                    ])
                
                df = pd.DataFrame(table_data, columns=[
                    "排名", "疾病编码", "疾病名称", "相似度", "章名称", "节名称"
                ])
                
                # NER分析结果
                ner_info = result['ner_analysis']
                ner_summary = f"识别到 {ner_info['entities_found']} 个医学实体"
                if ner_info['entities']:
                    ner_summary += "：\n"
                    for entity in ner_info['entities']:
                        ner_summary += f"• {entity['text']} ({entity['label']}, 置信度: {entity['confidence']:.3f})\n"
                
                # 最佳匹配详情
                best_match = result['best_match']
                best_info = f"""
**最佳匹配详情**
• 疾病编码: {best_match['disease_code']}
• 疾病名称: {best_match['disease_name']}
• 相似度分数: {best_match['similarity_score']:.4f}
• 分类路径: {best_match['chapter_name']} > {best_match['section_name']}

**检索统计**
• 查询文本: {result['query_text']}
• 返回结果数: {result['search_params']['total_found']}
• 平均相似度: {result['summary']['avg_score']:.4f}
• 最高相似度: {result['summary']['best_score']:.4f}
"""
                
                return df, ner_summary, best_info
            else:
                empty_msg = f"未找到相似度大于 {score_threshold} 的匹配结果"
                return pd.DataFrame(), empty_msg, ""
        else:
            error_msg = f"检索失败: {result.get('error', '未知错误')}"
            return pd.DataFrame(), error_msg, ""
            
    except Exception as e:
        logger.error(f"ICD-10编码检索失败: {e}")
        error_msg = f"检索失败: {str(e)}"
        return pd.DataFrame(), error_msg, ""

def create_ner_interface():
    """创建NER实体识别界面"""
    with gr.Row():
        with gr.Column(scale=2):
            # 输入区域
            input_text = gr.Textbox(
                label="📝 输入医学文本",
                placeholder="请输入患者主诉、症状描述或诊断信息...\n例如：患者主诉胸痛3天，伴有呼吸困难和心悸，既往有高血压病史",
                lines=5,
                max_lines=10
            )
            
            # 控制参数
            with gr.Row():
                confidence_threshold = gr.Slider(
                    minimum=0.5, 
                    maximum=1.0, 
                    value=0.7, 
                    step=0.05,
                    label="🎯 置信度阈值"
                )
                
                extract_btn = gr.Button("🔍 识别实体", variant="primary")
            
            # 示例输入
            gr.Examples(
                examples=[
                    "患者主诉胸痛3天，伴有呼吸困难和心悸，既往有高血压病史",
                    "急性阑尾炎，建议手术治疗",
                    "慢性支气管炎急性发作，给予抗炎治疗",
                    "2型糖尿病，血糖控制不佳，调整胰岛素用量",
                    "腰椎间盘突出症，腰痛伴下肢放射痛",
                    "患者确诊为急性心肌梗死，需要立即进行心电图检查"
                ],
                inputs=input_text,
                label="💡 示例文本"
            )
        
        with gr.Column(scale=3):
            # 输出区域
            with gr.Tab("📊 实体识别结果"):
                entities_table = gr.DataFrame(
                    headers=["实体文本", "实体类型", "置信度", "位置"],
                    label="识别到的医学实体",
                    interactive=False
                )
            
            with gr.Tab("📈 实体统计"):
                entity_stats = gr.JSON(label="实体类型统计和置信度分析")
            
            with gr.Tab("🌈 高亮显示"):
                highlighted_text = gr.HTML(label="实体高亮文本")
    
    # 绑定事件
    extract_btn.click(
        fn=extract_entities_ui,
        inputs=[input_text, confidence_threshold],
        outputs=[entities_table, entity_stats, highlighted_text]
    )
    
    return input_text, confidence_threshold, extract_btn, entities_table, entity_stats, highlighted_text

def create_search_interface():
    """创建ICD-10编码检索界面"""
    with gr.Row():
        with gr.Column(scale=2):
            # 输入区域
            query_text = gr.Textbox(
                label="🔍 输入诊断查询",
                placeholder="请输入疾病名称、症状描述或诊断信息...\n例如：急性心肌梗死",
                lines=3,
                max_lines=5
            )
            
            # 控制参数
            with gr.Row():
                top_k = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=10,
                    step=1,
                    label="📋 返回结果数量"
                )
                
                score_threshold = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.3,
                    step=0.05,
                    label="🎯 相似度阈值"
                )
            
            use_ner = gr.Checkbox(
                value=True,
                label="🧠 启用NER实体识别增强"
            )
            
            search_btn = gr.Button("🔍 检索ICD-10编码", variant="primary")
            
            # 示例输入
            gr.Examples(
                examples=[
                    "急性心肌梗死",
                    "2型糖尿病伴肾病",
                    "高血压性心脏病",
                    "慢性阻塞性肺疾病急性加重",
                    "患者主诉胸痛3天，伴有呼吸困难和心悸",
                    "糖尿病并发症",
                    "脑血管意外",
                    "慢性肾功能不全"
                ],
                inputs=query_text,
                label="💡 示例查询"
            )
        
        with gr.Column(scale=3):
            # 输出区域
            with gr.Tab("🎯 检索结果"):
                results_table = gr.DataFrame(
                    headers=["排名", "疾病编码", "疾病名称", "相似度", "章名称", "节名称"],
                    label="ICD-10编码匹配结果",
                    interactive=False
                )
            
            with gr.Tab("🧠 实体分析"):
                ner_analysis = gr.Textbox(
                    label="NER实体识别分析",
                    lines=8,
                    interactive=False
                )
            
            with gr.Tab("⭐ 最佳匹配"):
                best_match_info = gr.Markdown(
                    label="最佳匹配详细信息"
                )
    
    # 绑定事件
    search_btn.click(
        fn=search_icd_codes_ui,
        inputs=[query_text, top_k, score_threshold, use_ner],
        outputs=[results_table, ner_analysis, best_match_info]
    )
    
    return query_text, top_k, score_threshold, use_ner, search_btn, results_table, ner_analysis, best_match_info

def create_main_interface():
    """创建主界面"""
    # 初始化服务
    init_success = initialize_services()
    
    with gr.Blocks(
        title="ICD-10 RAG检索系统",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            font-family: 'Arial', sans-serif;
        }
        .tab-nav {
            font-weight: bold;
        }
        .highlight-entity {
            padding: 2px 4px;
            border-radius: 3px;
            margin: 0 2px;
        }
        """
    ) as demo:
        
        # 页面标题和说明
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🏥 ICD-10 RAG检索系统</h1>
            <p style="font-size: 18px; color: #666;">
                基于NER实体识别和向量检索的智能医疗编码匹配系统
            </p>
            <hr style="margin: 20px 0;">
        </div>
        """)
        
        if not init_success:
            gr.HTML("""
            <div style="background-color: #ffebee; padding: 20px; border-radius: 10px; margin: 20px;">
                <h3 style="color: #c62828;">⚠️ 系统初始化失败</h3>
                <p>请检查以下问题：</p>
                <ul>
                    <li>Milvus数据库是否正常运行</li>
                    <li>模型文件是否正确下载</li>
                    <li>网络连接是否正常</li>
                </ul>
            </div>
            """)
        else:
            gr.HTML("""
            <div style="background-color: #e8f5e8; padding: 15px; border-radius: 10px; margin: 20px;">
                <p style="color: #2e7d32; margin: 0;">
                    ✅ 系统初始化成功！数据库已连接，模型已加载。
                </p>
            </div>
            """)
        
        # 主功能Tab界面
        with gr.Tabs():
            with gr.Tab("🔍 命名实体识别", id="ner_tab"):
                gr.HTML("""
                <div style="padding: 15px; background-color: #f5f5f5; border-radius: 8px; margin-bottom: 20px;">
                    <h3>📋 功能说明</h3>
                    <p>从医学文本中识别疾病、症状、身体部位、检查项目等医学实体，支持置信度调节和可视化显示。</p>
                </div>
                """)
                create_ner_interface()
            
            with gr.Tab("🎯 诊断标准化", id="search_tab"):
                gr.HTML("""
                <div style="padding: 15px; background-color: #f5f5f5; border-radius: 8px; margin-bottom: 20px;">
                    <h3>📋 功能说明</h3>
                    <p>基于诊断描述智能匹配对应的ICD-10标准编码，结合NER实体识别和向量语义检索技术。</p>
                </div>
                """)
                create_search_interface()
        
        # 页脚信息
        gr.HTML("""
        <div style="text-align: center; padding: 20px; margin-top: 30px; border-top: 1px solid #ddd;">
            <p style="color: #888; font-size: 14px;">
                🏥 ICD-10 RAG检索系统 | 基于 Milvus + BGE + chinese-medical-ner 构建
            </p>
        </div>
        """)
    
    return demo

def main():
    """主函数：启动Gradio应用"""
    try:
        # 设置信号处理器
        setup_signal_handlers()
        
        logger.info("🚀 正在启动ICD-10 RAG检索系统...")
        
        # 创建界面
        demo = create_main_interface()
        
        # 启动应用
        logger.info(f"🌐 启动Gradio应用，访问地址: http://localhost:{GRADIO_CONFIG['server_port']}")
        demo.launch(
            server_name=GRADIO_CONFIG['server_name'],
            server_port=GRADIO_CONFIG['server_port'],
            share=GRADIO_CONFIG['share'],
            show_error=True,
            debug=True,
            prevent_thread_lock=False
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 用户中断，正在优雅关闭...")
        cleanup_all_resources()
    except Exception as e:
        logger.error(f"启动Gradio应用失败: {e}")
        print(f"❌ 启动失败: {e}")
        cleanup_all_resources()
    finally:
        # 确保资源被清理
        cleanup_all_resources()

if __name__ == "__main__":
    main()