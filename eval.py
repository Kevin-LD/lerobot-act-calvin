import os
import yaml
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# 导入共享的组件
from data.dataset import CalvinDDataset
from utils.utils import evaluate_policy, create_policy_feature, evaluate_policy_ema
from lerobot.policies.act.modeling_act import ACTPolicy, ACTConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script for ACT on CALVIN Environment D (Zero-Shot Generalization)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained checkpoint (e.g., runs/run_xxx/best_act_policy.pt)")
    parser.add_argument("--config", type=str, default="configs/train_B.yaml", help="Path to the training config file to reconstruct architecture")
    parser.add_argument("--data_root_D", type=str, default=None, help="Overrides the data root for Environment D if necessary")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers for DataLoader")
    
    parser.add_argument("--mode", type=str, default="none", choices=["none", "ema"], 
                        help="Evaluation mode: 'none' (baseline, first step only) or 'ema' (temporal ensembling)")
    parser.add_argument("--k", type=float, default=0.1, 
                        help="Temporal ensembling exponential decay constant (smaller k means history predictions have more weight)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 解析模型配置
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config["infrastructure"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. 载入环境 D 测试集
    print("\nLoading Environment D Test Dataset...")
    test_dataset_D = CalvinDDataset(
        data_root=args.data_root_D,  # 若为 None 会自动匹配到 data/splitD
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
    
    # 3. 按照训练配置动态重建 ACT 策略网络结构
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
    
    # 4. 导入已训练完成的 Checkpoint 权重
    print(f"Loading trained weights from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"❌ 未找到指定的权重文件: {args.checkpoint}")
        
    checkpoint_data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    
    # 提取状态字典并注入模型
    if "model_state_dict" in checkpoint_data:
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
    
    # 5. 执行跨环境测试集评估
    print(f"\n 启动零样本环境迁移验证 (Zero-Shot Cross-Env Evaluation on Split D) ...")
    print(f"评估模式配置: 【{args.mode.upper()}】 | 总计待测试帧数: {len(test_dataset_D)} | 总 Batch 步数: {len(test_dataloader)}")
    
    if args.mode == "none":
        # 模式一：复用原有的标准单步离线评估流
        avg_action_error_D = evaluate_policy(
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

    # 6. 打印核心结果
    print("\n" + "="*60)
    print("                    📊 EVALUATION REPORT                    ")
    print("="*60)
    print(f" 测试环境名称:   CALVIN Split D (Unseen Deployment Env)")
    print(f" 评估模型源自:   {os.path.basename(args.checkpoint)}")
    print(f" 动作集成模式:   {args.mode.upper()} (k={args.k if args.mode=='ema' else 'N/A'})")
    print(f" 测试样本总量:   {len(test_dataset_D)} 帧")
    print(f" 🎯 泛化平均动作误差 (Avg Action L1 Error): {avg_action_error_D:.5f}")
    print("="*60)
    
    # 将测试结果备份
    report_path = os.path.join(os.path.dirname(args.checkpoint), f"eval_D_report_{args.mode}.txt")
    with open(report_path, "w") as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Mode: {args.mode} (k={args.k})\n")
        f.write(f"Dataset D Frames: {len(test_dataset_D)}\n")
        f.write(f"Avg Action L1 Error on D: {avg_action_error_D:.5f}\n")
    print(f"💡 结果评估报告已自动同步留存至: {report_path}\n")

if __name__ == "__main__":
    main()
