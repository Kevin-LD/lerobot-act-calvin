import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import SingleEpisodeDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Qualitative Visual Distribution Shift Analysis across CALVIN Splits")
    # 分别接收 4 个环境的单轨迹 Parquet 文件路径
    parser.add_argument("--parquet_A", type=str, required=True, help="Path to a parquet from Split A")
    parser.add_argument("--parquet_B", type=str, required=True, help="Path to a parquet from Split B")
    parser.add_argument("--parquet_C", type=str, required=True, help="Path to a parquet from Split C")
    parser.add_argument("--parquet_D", type=str, required=True, help="Path to a parquet from Split D")
    
    parser.add_argument("--frame_idx", type=int, default=0, help="The frame index to extract within the episode")
    parser.add_argument("--output_dir", type=str, default="figure", help="Directory to save the analysis figure")
    return parser.parse_args()

def preprocess_tensor_to_rgb(img_tensor):
    """
    将 Dataset 返回的 PyTorch Tensor 安全转换为 matplotlib 可渲染的 RGB NumPy 数组
    """
    # 调整通道顺序: (C, H, W) -> (H, W, C)
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    
    # 稳健性检查：判断数值区间是否需要反向缩放或裁剪
    if img.max() > 1.0:
        img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img = np.clip(img, 0, 1.0)
    return img

def main():
    args = parse_args()
    
    # 1. 组装路径字典以进行批处理循环加载
    target_files = {
        "Split A (Train)": args.parquet_A,
        "Split B (Train)": args.parquet_B,
        "Split C (Train)": args.parquet_C,
        "Split D (Unseen-Test)": args.parquet_D
    }
    
    # 2. 提取各环境指定帧的静止与手眼图像
    visual_data = {}
    print("Extracting frame cross-comparison arrays from datasets...")
    
    for split_name, file_path in target_files.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"未找到该环境的目标 Parquet 文件: {file_path}")
            
        # 建立轻量化单文件数据流 (不带 Action Horizon 约束，仅抽图)
        dataset = SingleEpisodeDataset(file_path=file_path, action_horizon=1)
        
        # 边界防错
        idx = min(args.frame_idx, len(dataset) - 1)
        sample = dataset[idx]
        
        # 提取双路相机特征
        static_rgb = preprocess_tensor_to_rgb(sample["images"]["image"])
        wrist_rgb = preprocess_tensor_to_rgb(sample["images"]["wrist_image"])
        
        visual_data[split_name] = {
            "static": static_rgb,
            "wrist": wrist_rgb
        }
        print(f"{split_name} [Frame {idx:03d}] Loaded. (Static: {static_rgb.shape}, Wrist: {wrist_rgb.shape})")

    # 3. 画布构建与渲染设置
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 11
    
    # 创建 2x4 的图像对比网格
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=300)
    splits = list(visual_data.keys())
    
    for col_idx, split_name in enumerate(splits):
        # 渲染第 1 行：全局静态相机
        ax_static = axes[0, col_idx]
        ax_static.imshow(visual_data[split_name]["static"])
        ax_static.set_title(split_name, fontsize=12, fontweight="bold", pad=10)
        ax_static.get_xaxis().set_visible(False)
        ax_static.get_yaxis().set_visible(False)
        
        # 渲染第 2 行：手眼嵌入相机
        ax_wrist = axes[1, col_idx]
        ax_wrist.imshow(visual_data[split_name]["wrist"])
        ax_wrist.get_xaxis().set_visible(False)
        ax_wrist.get_yaxis().set_visible(False)
        
        # 仅在最左侧添加行级标注（Row Labels）
        if col_idx == 0:
            ax_static.set_ylabel("Static Camera\n", fontsize=12, fontweight="bold", labelpad=15)
            ax_wrist.set_ylabel("Wrist Camera\n", fontsize=12, fontweight="bold", labelpad=15)
            # 恢复 y 轴以便显示行标，但隐藏刻度线
            ax_static.get_yaxis().set_visible(True)
            ax_wrist.get_yaxis().set_visible(True)
            ax_static.set_xticks([])
            ax_static.set_yticks([])
            ax_wrist.set_xticks([])
            ax_wrist.set_yticks([])

    # 4. 调整紧凑度并导出持久化
    plt.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "visual_distribution_shift.png")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"\n跨环境视觉分布偏移定性分析已成功保存至: {save_path}")

if __name__ == "__main__":
    main()
