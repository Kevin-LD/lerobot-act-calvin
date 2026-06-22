import os
import sys
import yaml
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import SingleEpisodeDataset
from utils.utils import create_policy_feature, collect_episode_trajectory
from lerobot.policies.act.modeling_act import ACTPolicy, ACTConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Dimension-wise Action Tracking via Config-driven Reconstructed ACT")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained checkpoint (e.g., runs/run_xxx/best_act_policy.pt)")
    parser.add_argument("--config", type=str, default="configs/train_B.yaml", help="Path to the training config file to reconstruct architecture")
    parser.add_argument("--target_parquet", type=str, required=True, help="Path to the specific single parquet file to analyze (e.g., data/splitD/data/chunk-000/episode_00000.parquet)")
    parser.add_argument("--output_dir", type=str, default="figure", help="Directory to save the generated figure")
    return parser.parse_args()

def plot_action_matrix(gt_actions, pred_actions, output_dir, file_name):
    """
    生成 7x2 矩阵图
    """
    num_samples, num_dims = gt_actions.shape
    dim_names = [
        "X (Translation)", "Y (Translation)", "Z (Translation)",
        "Roll (Rotation)", "Pitch (Rotation)", "Yaw (Rotation)",
        "Gripper (Open/Close)"
    ]
    
    absolute_errors = np.abs(gt_actions - pred_actions)
    time_steps = np.arange(num_samples)
    
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.linewidth"] = 1.0
    
    fig, axes = plt.subplots(num_dims, 2, figsize=(14, 2.2 * num_dims), dpi=300, sharex="col")
    
    for i in range(num_dims):
        # 左侧子图：Ground Truth vs Predicted 连续轨迹比对
        ax_left = axes[i, 0]
        ax_left.plot(time_steps, gt_actions[:, i], color="#1F77B4", linewidth=1.6, label="Ground Truth")
        ax_left.plot(time_steps, pred_actions[:, i], color="#FF7F0E", linewidth=1.4, linestyle="--", label="Predicted")
        ax_left.set_ylabel(dim_names[i], fontsize=10, fontweight="bold")
        ax_left.grid(True, linestyle="--", alpha=0.3)
        
        if i == 0:
            ax_left.set_title("Continuous Trajectory Tracking", fontsize=12, fontweight="bold", pad=12)
            ax_left.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="#E0E0E0")
            
        # 右侧子图：单维度时序绝对误差填充
        ax_right = axes[i, 1]
        ax_right.plot(time_steps, absolute_errors[:, i], color="#D62728", linewidth=1.1, alpha=0.8)
        ax_right.fill_between(time_steps, absolute_errors[:, i], color="#D62728", alpha=0.1)
        ax_right.set_ylabel("Abs Error", fontsize=10)
        ax_right.grid(True, linestyle="--", alpha=0.3)
        
        if i == 0:
            ax_right.set_title("Dimension-wise Absolute Error", fontsize=12, fontweight="bold", pad=12)

    axes[-1, 0].set_xlabel("Time Steps (Frames)", fontsize=11, fontweight="bold", labelpad=8)
    axes[-1, 1].set_xlabel("Time Steps (Frames)", fontsize=11, fontweight="bold", labelpad=8)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, file_name)
    plt.savefig(save_path, bbox_inches="tight")
    print(f"\n图表已保存至: {save_path}")

def main():
    args = parse_args()
    
    # 1. 解析模型基础配置
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config["infrastructure"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. 仅针对单个 parquet 文件建立轻量化数据流
    dataset_single = SingleEpisodeDataset(
        file_path=args.target_parquet,
        action_horizon=config["dataset"]["action_horizon"]
    )
    
    # 3. 动态重建 ACT 模型架构
    print("\nReconstructing LeRobot ACT Policy Architecture...")
    input_features = {
        "observation.state": create_policy_feature("state", [15]),
        "observation.images.image": create_policy_feature("image", [3, 200, 200]),
        "observation.images.wrist_image": create_policy_feature("image", [3, 84, 84]),
    }
    output_features = {
        "action": create_policy_feature("action", [7]),
    }
    
    act_config = ACTConfig(
        n_action_steps=config["dataset"]["action_horizon"],
        chunk_size=config["dataset"]["action_horizon"],
        vision_backbone=config["model"]["backbone"],
        dim_model=config["model"]["hidden_dim"],
        dim_feedforward=config["model"]["dim_feedforward"],
        kl_weight=config["model"]["kl_weight"],
        input_features=input_features,
        output_features=output_features,
        optimizer_lr=float(config["model"]["lr"]),
        optimizer_weight_decay=float(config["model"]["weight_decay"]),
        optimizer_lr_backbone=float(config["model"]["lr_backbone"]),
    )
    
    policy = ACTPolicy(act_config)
    
    # 4. 导入已训练完成的 Checkpoint 权重（完全复刻 eval.py 的兼容层）
    print(f"\nLoading trained weights from: {args.checkpoint}")
    checkpoint_data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    
    if "model_state_dict" in checkpoint_data:
        policy.load_state_dict(checkpoint_data["model_state_dict"])
        orig_epoch = checkpoint_data.get("epoch", "Unknown")
        orig_step = checkpoint_data.get("global_step", "Unknown")
        print(f"成功恢复权重！Checkpoint 产自训练 Epoch {orig_epoch} (Step {orig_step})")
    else:
        policy.load_state_dict(checkpoint_data)
        print("以纯状态字典模式注入权重。")
        
    policy.to(device)
    policy.eval()

    # 5. 进行单轨迹推演
    print(f"\n启动单轨迹推演: {os.path.basename(args.target_parquet)}")
    
    # 直接调用修改后的函数
    gt_actions, pred_actions = collect_episode_trajectory(
        policy=policy, 
        dataset=dataset_single, 
        device=device
    )
    
    # 6. 送入画布组件
    pure_file_name = os.path.splitext(os.path.basename(args.target_parquet))[0]
    
    split_tag = "unknown"
    for s in ["splitA", "splitB", "splitC", "splitD"]:
        if s in args.target_parquet:
            split_tag = s
            break
            
    output_png_name = f"{split_tag}_{pure_file_name}_dim_analysis.png"
    
    plot_action_matrix(
        gt_actions=gt_actions,
        pred_actions=pred_actions,
        output_dir=args.output_dir,
        file_name=output_png_name
    )

if __name__ == "__main__":
    main()
