"""
多模态检索器
使用BGE-VL模型对图片进行编码，并计算查询与图片的余弦相似度
"""

# pip install transformers Pillow torch
import os
import torch
from PIL import Image
from typing import List, Tuple
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
import warnings
warnings.filterwarnings("ignore")

class MultimodalRetriever:
    
    def __init__(self, image_dir: str, model_name: str = "BAAI/bge-vl-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")

        # 加载模型和对应的处理器
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)

        # 加载并编码图片
        self.image_paths, self.image_embeddings = self._load_and_encode_images(image_dir)

    def _load_and_encode_images(self, image_dir: str):
        if not os.path.isdir(image_dir):
            print(f"错误: 目录 '{image_dir}' 不存在。")
            return [], None

        image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_paths:
            print(f"警告: 在 '{image_dir}' 中未找到图片。")
            return [], None

        print(f"找到 {len(image_paths)} 张图片，正在编码...")
        images = [Image.open(p).convert("RGB") for p in image_paths]
        
        with torch.no_grad():
            image_inputs = self.image_processor(images, return_tensors="pt").to(self.device)
            image_embeddings = self.model.get_image_features(**image_inputs)
            # L2归一化
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        
        print("图片编码完成。")
        return image_paths, image_embeddings.cpu()

    def _encode_text(self, text: str):
        with torch.no_grad():
            text_inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            text_embeddings = self.model.get_text_features(**text_inputs)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings.cpu()

    def search_by_text(self, query_text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        if self.image_embeddings is None: return []
        
        query_embedding = self._encode_text(query_text)
        similarities = (query_embedding @ self.image_embeddings.T).squeeze(0)
        
        top_indices = similarities.argsort(descending=True)[:top_k]
        
        return [(self.image_paths[i], similarities[i].item()) for i in top_indices]

# --- 主函数 ---
def main():
    # 获取脚本所在目录，构建正确的图片路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_directory = os.path.join(script_dir, "../../data/images")
    
    if not os.path.exists(image_directory):
        print(f"错误: 图片目录 '{image_directory}' 不存在。")
        print("请确保 data/images 目录存在并包含图片文件。")
        return

    retriever = MultimodalRetriever(image_dir=image_directory)
    
    query = "一只可爱的猫咪"
    results = retriever.search_by_text(query, top_k=2)

    print(f"\n文本查询: '{query}'")
    print("多模态搜索结果:")
    for path, score in results:
        print(f"  - 图片: {os.path.basename(path)} [相似度: {score:.4f}]")

if __name__ == "__main__":
    main()