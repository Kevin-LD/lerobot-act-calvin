import os
import yaml
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import CalvinDDataset, CalvinFilteredDDataset
from utils.utils import create_policy_feature, evaluate_policy_ema, evaluate_policy_dimension_wise
from lerobot.policies.act.modeling_act import ACTPolicy, ACTConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script for ACT on CALVIN Environment D (Zero-Shot Generalization)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained checkpoint (e.g., runs/run_xxx/best_act_policy.pt)")
    parser.add_argument("--config", type=str, default=None, help="Optional fallback config path if checkpoint doesn't contain it")
    parser.add_argument("--data_root_D", type=str, default=None, help="Overrides the data root for Environment D if necessary")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers for DataLoader")
    
    parser.add_argument("--mode", type=str, default="none", choices=["none", "ema"], 
                        help="Evaluation mode: 'none' (baseline) or 'ema' (temporal ensembling)")
    parser.add_argument("--k", type=float, default=0.1, 
                        help="Temporal ensembling exponential decay constant")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 载入 Checkpoint 权重
    print(f"正在读取 Checkpoint 核心数据: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"未找到指定的权重文件: {args.checkpoint}")
        
    checkpoint_data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    
    # 2. 智能化提取训练时的 Config 配置
    config = None
    if isinstance(checkpoint_data, dict) and "config" in checkpoint_data:
        config = checkpoint_data["config"]
        print(" 🎯 [Auto Config] 成功从 Checkpoint 中检索到训练期原始配置，无需手动对齐！")
    else:
        # 兼容老版本没有保存 config 的 checkpoint
        if args.config is None:
            raise ValueError("该 Checkpoint 属于历史遗留权重，未内嵌 config，请通过 --config 手动指定其训练配置文件！")
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            
    device = torch.device(config["infrastructure"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 3. 自动化路由：根据训练模式自动判定是用完整 D 还是纯净过滤单任务 D
    env_mode = config["dataset"].get("env_mode", "B").upper()
    if "FILTERED" in env_mode:
        dataset_cls = CalvinFilteredDDataset
        print(" 🎯 [Eval Mode: FILTERED] 检测到模型基于【单任务过滤】训练，自动挂载 CalvinFilteredDDataset (filtered_splitD)...")
    else:
        dataset_cls = CalvinDDataset
        print(" [Eval Mode: STANDARD] 检测到模型基于【标准全任务】训练，自动挂载 CalvinDDataset (splitD)...")
        
    # 4. 载入环境 D 测试集
    print("Loading Environment D Test Dataset...")
    test_dataset_D = dataset_cls(
        data_root=args.data_root_D,  # 如果手动传了则覆盖，不传则由对应的 Dataset 类自动寻找标准路径
        action_horizon=config["dataset"]["action_horizon"],
        split="train",               # val_ratio=0 时，所有数据均存在于默认的轨迹池中
        val_ratio=0.0,
        seed=config["infrastructure"]["seed"]
    )
    
    test_dataloader = DataLoader(
        test_dataset_D,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True
    )
    
    # 5. 按照训练配置动态重建 ACT 策略网络结构
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
    
    # 6. 注入模型权重
    if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
        policy.load_state_dict(checkpoint_data["model_state_dict"])
        orig_epoch = checkpoint_data.get("epoch", "Unknown")
        orig_step = checkpoint_data.get("global_step", "Unknown")
        orig_err = checkpoint_data.get("val_action_error", float("inf"))
        print(f"成功恢复权重！该 Checkpoint 产自训练阶段的 Epoch {orig_epoch} (Global Step {orig_step})")
        print(f"当时在环境 B 验证集上的最好误差为: {orig_err:.4f}")
    else:
        policy.load_state_dict(checkpoint_data)
        print("以纯状态字典模式成功注入权重。")
        
    policy.to(device)
    
    # 7. 执行跨环境测试集评估
    print(f"\n 启动零样本环境迁移验证 (Zero-Shot Cross-Env Evaluation on Split D) ...")
    print(f"评估模式配置: 【{args.mode.upper()}】 | 总计待测试帧数: {len(test_dataset_D)} | 总 Batch 步数: {len(test_dataloader)}")
    
    dim_errors_D = None # 初始化，防止后续 NameError
    if args.mode == "none":
        # 模式一：基础单步前向评估
        avg_action_error_D, dim_errors_D = evaluate_policy_dimension_wise(
            policy=policy,
            dataloader=test_dataloader,
            device=device,
            desc="Cross-Eval Split D (None Mode)"
        )
    else:
        # 模式二：时序滑动平均集成评估
        avg_action_error_D = evaluate_policy_ema(
            policy=policy,
            dataloader=test_dataloader,
            device=device,
            k=args.k,
            desc="Cross-Eval Split D (EMA)"
        )

    dim_names = [
        "X (Translation)", "Y (Translation)", "Z (Translation)",
        "Roll (Rotation)", "Pitch (Rotation)", "Yaw (Rotation)",
        "Gripper (Open/Close)"
    ]

    # 8. 打印与保存核心结果
    print("\n" + "="*60)
    print("                    📊 EVALUATION REPORT                    ")
    print("="*60)
    print(f" 测试环境模式:   {env_mode}")
    print(f" 测试数据集类:   {dataset_cls.__name__}")
    print(f" 评估模型源自:   {os.path.basename(args.checkpoint)}")
    print(f" 动作集成模式:   {args.mode.upper()} (k={args.k if args.mode=='ema' else 'N/A'})")
    print(f" 测试样本总量:   {len(test_dataset_D)} 帧")
    print(f" 🎯 泛化平均动作误差 (Avg Action L1 Error): {avg_action_error_D:.5f}")
    
    if dim_errors_D is not None:
        print("-"*60)
        print("            📊 DIMENSION-WISE ACTION ERROR REPORT           ")
        print("-"*60)
        for name, err in zip(dim_names, dim_errors_D):
            print(f" 📍 {name:<22} 平均绝对误差 (L1): {err:.5f}")
        print("-"*60)
        print(f" 🎯 基础前向综合平均误差 (Base L1 Check): {dim_errors_D.mean():.5f}")
    print("="*60)
    
    # 将测试结果备份
    report_path = os.path.join(os.path.dirname(args.checkpoint), f"eval_D_report_{args.mode}.txt")
    with open(report_path, "w") as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Training Env Mode: {env_mode}\n")
        f.write(f"Dataset Class: {dataset_cls.__name__}\n")
        f.write(f"Eval Mode: {args.mode} (k={args.k})\n")
        f.write(f"Dataset D Frames: {len(test_dataset_D)}\n")
        f.write(f"Avg Action L1 Error on D: {avg_action_error_D:.5f}\n\n")
        
        if dim_errors_D is not None:
            f.write("--- Dimension-wise Error Detail ---\n")
            for name, err in zip(dim_names, dim_errors_D):
                f.write(f"{name}: {err:.5f}\n")
            f.write(f"Base L1 Check: {dim_errors_D.mean():.5f}\n")
        else:
            f.write("Note: Dimension-wise errors are not available in EMA mode.\n")
        
    print(f"结果评估报告已自动同步保存至: {report_path}\n")

if __name__ == "__main__":
    main()
