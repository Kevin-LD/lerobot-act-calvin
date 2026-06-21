import os
import sys
import matplotlib
# 在无显示器的云服务器上，强制使用 Agg 后端以防止渲染崩溃
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 确保能够正确引入同级目录下的 dataset.py
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from dataset import CalvinBDataset, CalvinDDataset

def main():
    print("Initializing CalvinBDataset for data exploration...")
    # 实例化数据集（action_horizon 设为 16 以对齐 ACT 算法）
    dataset = CalvinDDataset(action_horizon=16)
    
    # 抽取特定帧进行可视化验证
    sample_idx = 0
    print(f"Extracting data for global frame index {sample_idx}...")
    sample = dataset[sample_idx]
    
    qpos = sample["qpos"].numpy()          # 形状: [15]
    actions = sample["actions"].numpy()    # 形状: [16, 7]
    img_static = sample["images"]["image"].permute(1, 2, 0).numpy()       # 形状: [200, 200, 3]
    img_wrist = sample["images"]["wrist_image"].permute(1, 2, 0).numpy() # 形状: [84, 84, 3]

    print("Rendering data visualization panel...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"CALVIN Dataset Exploration (Frame Index: {sample_idx})", fontsize=16, fontweight='bold')

    # 1. Top-Left: 静态相机主视角
    axes[0, 0].imshow(img_static)
    axes[0, 0].set_title(f"Static Camera View {img_static.shape[:2]}", fontsize=12)
    axes[0, 0].axis('off')

    # 2. Top-Right: 腕部相机视角
    axes[0, 1].imshow(img_wrist)
    axes[0, 1].set_title(f"Wrist Camera View {img_wrist.shape[:2]}", fontsize=12)
    axes[0, 1].axis('off')

    # 3. Bottom-Left: 16步动作序列轨迹
    steps = np.arange(actions.shape[0])
    action_labels = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    for i in range(actions.shape[1]):
        axes[1, 0].plot(steps, actions[:, i], label=action_labels[i], alpha=0.8)
    axes[1, 0].set_title("Action Trajectory over Horizon (16 steps)", fontsize=12)
    axes[1, 0].set_xlabel("Future Steps")
    axes[1, 0].set_ylabel("Action Value")
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)
    axes[1, 0].legend(loc="upper right", ncol=2)

    # 4. Bottom-Right: 当前帧 15 维机器人状态 (qpos)
    state_indices = np.arange(len(qpos))
    axes[1, 1].bar(state_indices, qpos, color='steelblue', alpha=0.8)
    axes[1, 1].set_title(f"Robot Current State Vector ({len(qpos)} Dim)", fontsize=12)
    axes[1, 1].set_xlabel("State Dimensions")
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].set_xticks(state_indices)
    axes[1, 1].grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    
    # 定位根目录并创建 figure 文件夹
    root_dir = os.path.dirname(current_dir)
    figure_dir = os.path.join(root_dir, "figure")
    os.makedirs(figure_dir, exist_ok=True)
    
    output_path = os.path.join(figure_dir, "data_exploration_output_D.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Visualization panel saved successfully to: {output_path}")

if __name__ == "__main__":
    main()
