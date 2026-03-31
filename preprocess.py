import os
import json
import random
import argparse
import glob
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

# 解决超大图片报错
Image.MAX_IMAGE_PIXELS = None

def prepare_chartqa(dataset_path, output_dir):
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # 1. 强化路径搜索
    # 尝试在当前目录及所有子目录下寻找 .parquet 文件
    search_path = os.path.join(dataset_path, "**", "*.parquet")
    parquet_files = glob.glob(search_path, recursive=True)
    
    print(f"🔍 搜索路径: {search_path}")
    print(f"📂 找到文件总数: {len(parquet_files)}")
    
    if not parquet_files:
        print(f"❌ 错误：没找到任何 .parquet 文件！")
        print(f"检查一下路径：{os.path.abspath(dataset_path)} 是否正确？")
        return

    # 打印前两个文件看看路径对不对
    for f in parquet_files[:2]:
        print(f"📄 准备加载: {f}")

    # 2. 尝试加载
    try:
        # 不指定 split="train"，先让它加载所有发现的文件
        raw_dataset = load_dataset("parquet", data_files=parquet_files)
        # 即使只有一个 split，load_dataset 也会返回一个 DatasetDict
        # 我们取第一个 split（通常是 'train'）
        split_name = list(raw_dataset.keys())[0]
        dataset = raw_dataset[split_name]
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    total_count = len(dataset)
    print(f"✅ 成功加载！共 {total_count} 条数据。")

    # 3. 提取 6000 条
    num_to_extract = min(total_count, 6000)
    indices = list(range(total_count))
    random.shuffle(indices)
    
    train_count = 4000 if num_to_extract >= 6000 else int(num_to_extract * 0.67)
    train_indices = indices[:train_count]
    val_indices = indices[train_count:num_to_extract]

    def process_and_save(idx_list, split_name):
        if not idx_list: return 0
        subset = dataset.select(idx_list)
        formatted_data = []
        
        for idx, item in enumerate(tqdm(subset, desc=f"处理 {split_name}")):
            # ChartQA 常见的 key 是 'image', 'query', 'label'
            img = item.get('image')
            question = item.get('query', '')
            answer = item.get('label', '')

            if isinstance(answer, list):
                answer = answer[0] if len(answer) > 0 else ""

            if img is None: continue

            img_filename = f"{split_name}_{idx}.jpg"
            img_path = os.path.join(images_dir, img_filename)
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 保持 1200 分辨率以看清图表文字
            img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
            img.save(img_path)

            # === 主要修改点在这里 ===
            # 使用 os.path.abspath 获取保存图片的绝对路径
            absolute_img_path = os.path.abspath(img_path)

            formatted_data.append({
                "id": f"chart_{split_name}_{idx}",
                "images": [absolute_img_path],  # <-- 替换为绝对路径
                "conversations": [
                    {"from": "human", "value": f"<image>\n{question}"},
                    {"from": "gpt", "value": str(answer)}
                ]
            })
            
        with open(os.path.join(output_dir, f"{split_name}.json"), 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)
        return len(formatted_data)

    final_train = process_and_save(train_indices, "train")
    final_val = process_and_save(val_indices, "val")
    
    print(f"\n🎉 运行结束！训练集: {final_train}, 验证集: {final_val}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/root/datasets/chartqa")
    parser.add_argument("--output_dir", type=str, default="/root/llava_chart_data")
    args = parser.parse_args()
    prepare_chartqa(args.dataset_path, args.output_dir)