import os
import sys
import io
import argparse
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import CalvinFilteredABCDataset, CalvinFilteredBDataset, CalvinFilteredDDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Batch Visualize CALVIN Filtered Tasks")
    parser.add_argument("--mode", type=str, default="ABC", choices=["B", "ABC", "D"], help="Which dataset to visualize")
    parser.add_argument("--output_dir", type=str, default="video", help="Output directory")
    parser.add_argument("--fps", type=int, default=25, help="FPS")
    parser.add_argument("--max_episodes", type=int, default=3, help="Max videos per env")
    return parser.parse_args()

def parse_image(val):
    """解析 parquet 中的图片格式"""
    if isinstance(val, dict):
        if 'bytes' in val and val['bytes'] is not None:
            return Image.open(io.BytesIO(val['bytes'])).convert("RGB")
        elif 'path' in val and val['path'] is not None:
            return Image.open(val['path']).convert("RGB")
    if isinstance(val, (bytes, bytearray)):
        return Image.open(io.BytesIO(val)).convert("RGB")
    elif isinstance(val, np.ndarray):
        return Image.fromarray(val).convert("RGB")
    return None

def render_video(file_path, env_name, output_path, fps):
    df = pd.read_parquet(file_path)
    h, w = 400, 400
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w * 2, h))
    
    for i in range(len(df)):
        img1 = cv2.resize(np.array(parse_image(df.iloc[i]["image"])), (w, h))
        img2 = cv2.resize(np.array(parse_image(df.iloc[i]["wrist_image"])), (w, h))
        frame = cv2.cvtColor(np.hstack((img1, img2)), cv2.COLOR_RGB2BGR)
        
        # 写入环境前缀和帧信息
        text = f"[{env_name}] Frame: {i}/{len(df)}"
        cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        writer.write(frame)
    writer.release()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 实例化数据集并获取对应关系
    if args.mode == "ABC":
        ds = CalvinFilteredABCDataset(val_ratio=0.0)
        env_pairs = zip(ds.env_names, ds.datasets)
    elif args.mode == "B":
        ds = CalvinFilteredBDataset(val_ratio=0.0)
        env_pairs = [("B", ds)]
    else:
        ds = CalvinFilteredDDataset(val_ratio=0.0)
        env_pairs = [("D", ds)]
        
    for env_name, sub_ds in env_pairs:
        # 获取该数据集下的所有文件路径 (CalvinDataset 继承自 Dataset，其内部 parquet 文件通常存储在 self.episode_files 中)
        files = sub_ds.episode_files[:args.max_episodes]
        print(f"--- 正在处理环境 {env_name}，共 {len(files)} 个片段 ---")
        
        for f in tqdm(files):
            # 获取文件名并加上环境前缀
            base_name = os.path.basename(f).replace(".parquet", ".mp4")
            out_path = os.path.join(args.output_dir, f"{env_name}_{base_name}")
            render_video(f, env_name, out_path, args.fps)

if __name__ == "__main__":
    main()
