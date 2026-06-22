import os
import sys
import io
import argparse
import glob
import torch
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

# 确保能正确引入你本地的 data.dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset import CalvinABCDataset, CalvinBDataset, CalvinDDataset

def parse_args():
    parser = argparse.ArgumentParser(description="按 Task Index 分组可视化 CALVIN 视频")
    parser.add_argument("--mode", type=str, default="ABC", choices=["B", "ABC", "D"], help="要可视化的数据集环境")
    parser.add_argument("--output_dir", type=str, default="video_tasks", help="视频输出主目录")
    parser.add_argument("--fps", type=int, default=25, help="视频帧率")
    parser.add_argument("--max_episodes", type=int, default=3, help="每个 Task ID 最多生成的视频数")
    return parser.parse_args()

def parse_image(val):
    """解析 parquet 中的图片数据"""
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

def render_video(file_path, env_name, task_idx, sample_num, output_dir, fps):
    """渲染单个 Parquet 文件为双路拼接视频"""
    df = pd.read_parquet(file_path)
    h, w = 400, 400
    
    # 命名格式：环境_TaskID_样本序号_原始Episode号.mp4
    ep_idx = df.iloc[0]["episode_index"] if "episode_index" in df.columns else "unknown"
    out_name = f"{env_name}_task_{task_idx}_sample_{sample_num}_ep_{ep_idx}.mp4"
    out_path = os.path.join(output_dir, out_name)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w * 2, h))
    
    for i in range(len(df)):
        img1 = cv2.resize(np.array(parse_image(df.iloc[i]["image"])), (w, h))
        img2 = cv2.resize(np.array(parse_image(df.iloc[i]["wrist_image"])), (w, h))
        frame = cv2.cvtColor(np.hstack((img1, img2)), cv2.COLOR_RGB2BGR)
        
        # 在左上角强显当前环境、Task ID 核心元数据，方便你肉眼辨识
        text = f"[{env_name}] TASK ID: {task_idx} | Frame: {i}/{len(df)}"
        cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        writer.write(frame)
        
    writer.release()

def main():
    args = parse_args()
    
    # 1. 实例化你指定的原始 CalvinDataset 底座
    print(f"⏳ 正在加载环境模板 [{args.mode}] 的原始 Parquet 文件列表...")
    if args.mode == "ABC":
        ds = CalvinABCDataset(val_ratio=0.0)
        # CalvinMultiEnvDataset 内部包含多个子 dataset 对象
        env_pairs = zip(ds.env_names, ds.datasets)
    elif args.mode == "B":
        ds = CalvinBDataset(val_ratio=0.0)
        env_pairs = [("B", ds)]
    else:
        ds = CalvinDDataset(val_ratio=0.0)
        env_pairs = [("D", ds)]
        
    for env_name, sub_ds in env_pairs:
        print(f"\n🔍 正在快速扫描 [环境-{env_name}] 的 Task ID 分布情况 (仅读取索引列)...")
        
        # 使用字典将 file_path 归类到对应的 task_index
        task_to_files = defaultdict(list)
        
        # 用 tqdm 包装扫描过程
        for f_path in tqdm(sub_ds.episode_files, desc=f"扫描环境 {env_name}"):
            try:
                # 💡 核心优化：只读取 'task_index' 列，不读图片，速度极快
                df_meta = pd.read_parquet(f_path, columns=["task_index"])
                if df_meta.empty:
                    continue
                t_idx = int(df_meta.iloc[0]["task_index"])
                task_to_files[t_idx].append(f_path)
            except Exception as e:
                print(f"警告: 读取文件 {f_path} 失败, 错误: {e}")
                continue
        
        print(f"✅ 扫描完毕。环境-{env_name} 中总共发现了 {len(task_to_files)} 个不同的 Task Index。")
        
        # 为当前环境创建独立文件夹，防止混乱
        env_output_dir = os.path.join(args.output_dir, f"env_{env_name}")
        os.makedirs(env_output_dir, exist_ok=True)
        
        # 2. 遍历每个 Task 渲染特定数量的视频
        print(f"🎬 开始为每个 Task Index 渲染前 {args.max_episodes} 个演示视频...")
        for t_idx, file_list in task_to_files.items():
            samples_to_render = file_list[:args.max_episodes]
            
            for idx, f_path in enumerate(samples_to_render):
                render_video(
                    file_path=f_path,
                    env_name=env_name,
                    task_idx=t_idx,
                    sample_num=idx + 1,
                    output_dir=env_output_dir,
                    fps=args.fps
                )
                
        print(f"🎉 环境-{env_name} 的全部任务视频已保存在: {env_output_dir}")

if __name__ == "__main__":
    main()
