import os
import yaml
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from datetime import datetime
from tqdm import tqdm

from data.dataset import (
    CalvinBDataset, 
    CalvinABCDataset, 
    CalvinFilteredBDataset, 
    CalvinFilteredABCDataset
)
from lerobot.policies.act.modeling_act import ACTPolicy, ACTConfig
from utils.utils import build_lerobot_batch, evaluate_policy, evaluate_policy_ema, create_policy_feature

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for ACT on CALVIN Dataset")
    parser.add_argument("--config", type=str, default="configs/train_B.yaml", help="Path to the config file")
    return parser.parse_args()


def main():
    args = parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    set_seed(config["infrastructure"]["seed"])
    device = torch.device(config["infrastructure"]["device"] if torch.cuda.is_available() else "cpu")
    
    val_cfg = config.get("val", {})
    val_mode = val_cfg.get("mode", "none").lower()  # 'none' 或 'ema'
    val_k = val_cfg.get("k", 0.01)
    
    # 生成时间戳并动态拼接 save_dir 与 wandb run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["infrastructure"]["save_dir"] = f"{config['infrastructure']['save_dir']}_{timestamp}"
    
    base_run_name = config["wandb"].get("name", "act_train")
    wandb_run_name = f"{base_run_name}_{timestamp}"
    
    wandb.init(
        project=config["wandb"]["project"],
        name=wandb_run_name,
        mode=config["wandb"]["mode"],
        config=config
    )
    
    # 分流加载不同的数据集类以适配多环境联合训练或单环境过滤训练的需求
    env_mode = config["dataset"].get("env_mode", "B").upper()
    if env_mode == "ABC":
        dataset_cls = CalvinABCDataset
        print(" [Mode: ABC] 检测到多环境联合训练配置，正在组装 A + B + C 原始数据流...")
    elif env_mode == "B":
        dataset_cls = CalvinBDataset
        print(" [Mode: B] 检测到单一环境训练配置，正在加载环境 B 原始数据流...")
    elif env_mode == "FILTERED_B":
        dataset_cls = CalvinFilteredBDataset
        print(" 🎯 [Mode: FILTERED_B] 检测到【单任务过滤】配置，正在加载环境 B 纯净单任务数据流...")
    elif env_mode == "FILTERED_ABC":
        dataset_cls = CalvinFilteredABCDataset
        print(" 🎯 [Mode: FILTERED_ABC] 检测到【单任务过滤】多环境联合配置，正在组装 A + B + C 纯净单任务数据流...")
    else:
        raise ValueError(f" 无法识别的 env_mode: '{env_mode}'。合法的选项为 ['B', 'ABC', 'FILTERED_B', 'FILTERED_ABC']")

    print("Loading dataset...")
    base_train_dataset = dataset_cls(
        data_root=config["dataset"].get("data_root"),
        action_horizon=config["dataset"]["action_horizon"],
        split="train",
        val_ratio=0.1,
        seed=config["infrastructure"]["seed"]
    )
    
    overfit_cfg = config.get("training_mode", {})
    if overfit_cfg.get("overfit_single_batch", False):
        batch_size = config["dataset"]["batch_size"]
        train_dataset = Subset(base_train_dataset, list(range(batch_size)))
        val_dataset = train_dataset
        shuffle = False
        drop_last = False
        print(f" Sanity Check: 过拟合模式已开启。训练集与验证集均指向前 {batch_size} 个相同样本。")
    else:
        base_val_dataset = dataset_cls(
            data_root=config["dataset"].get("data_root"),
            action_horizon=config["dataset"]["action_horizon"],
            split="val",
            val_ratio=0.1,
            seed=config["infrastructure"]["seed"]
        )
        train_dataset = base_train_dataset
        val_dataset = base_val_dataset
        shuffle = True
        drop_last = True
        
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=shuffle,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=True
    )
    
    # 注意：进行时序滑动集成(EMA)时，测试集DataLoader的shuffle必须为False
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=True,
        drop_last=False,
        persistent_workers=True
    )
    
    print("Initializing LeRobot ACT Policy with current API schema...")
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
    policy.to(device)
    
    optimizer_groups = [
        {
            "params": [p for n, p in policy.named_parameters() if "backbone" not in n and p.requires_grad],
            "lr": float(config["model"]["lr"]),
        },
        {
            "params": [p for n, p in policy.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": float(config["model"]["lr_backbone"]),
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=float(config["model"]["weight_decay"]))
    
    num_epochs = config["model"]["num_epochs"]
    steps_per_epoch = len(train_dataloader)
    total_steps = num_epochs * steps_per_epoch
    
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    eval_freq = config["model"].get("eval_freq", 2000)
    
    save_dir = config["infrastructure"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {save_dir}")
    print(f"Total Expected Global Steps: {total_steps} ({steps_per_epoch} steps/epoch)")
    print(f"Validation Mode Enabled: 【{val_mode.upper()}】 (k={val_k if val_mode=='ema' else 'N/A'}) | Interval: Every {eval_freq} steps")
    
    print(f"Starting training pipeline for {num_epochs} epochs...")
    global_step = 0
    best_val_action_error = float("inf")
    
    for epoch in range(1, num_epochs + 1):
        # 训练
        policy.train()
        
        train_pbar = tqdm(train_dataloader, desc=f"Epoch [{epoch}/{num_epochs}] Train", leave=True)
        for batch in train_pbar:
            lerobot_batch = build_lerobot_batch(batch, device)
            
            total_loss, loss_dict = policy(lerobot_batch)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # 每个 Batch 更新 scheduler
            scheduler.step()
            
            loss_val = total_loss.item()
            global_step += 1
            
            # 动态刷新进度条右侧的实时 Loss 和当前的学习率
            current_lr_tf = optimizer.param_groups[0]["lr"]
            train_pbar.set_postfix({
                "loss": f"{loss_val:.4f}", 
                "step": global_step, 
                "lr_tf": f"{current_lr_tf:.2e}"
            })
            
            # 每个 Batch 上传 wandb
            log_data = {f"train/{k}": v for k, v in loss_dict.items()}
            log_data["train/loss"] = loss_val
            log_data["train/lr_transformer"] = current_lr_tf
            log_data["train/lr_backbone"] = optimizer.param_groups[1]["lr"]
            log_data["train/global_step"] = global_step
            log_data["train/epoch"] = epoch
            wandb.log(log_data)
            
            # 基于 Step 触发验证
            if global_step % eval_freq == 0:
                train_pbar.write(f"\n Step [{global_step}/{total_steps}] 触发定点验证演练中... 模式: 【{val_mode.upper()}】")

                # 根据配置分支选择不同的验证评估函数
                if val_mode == "ema":
                    avg_val_action_error = evaluate_policy_ema(
                        policy, val_dataloader, device, k=val_k, desc=f"Eval at Step {global_step} (EMA)"
                    )
                else:
                    avg_val_action_error = evaluate_policy(
                        policy, val_dataloader, device, desc=f"Eval at Step {global_step}"
                    )
                
                # 记录基于精准全局步数的验证结果
                wandb.log({"val/step_avg_action_error": avg_val_action_error, "train/global_step": global_step})
                train_pbar.write(f"📊 Step [{global_step}] 验证完毕 -> Avg Val Action Error: {avg_val_action_error:.4f}")
                
                # 最优权重捕获保存逻辑
                if avg_val_action_error < best_val_action_error:
                    best_val_action_error = avg_val_action_error
                    best_checkpoint_path = os.path.join(save_dir, "best_act_policy.pt")
                    torch.save({
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": policy.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_action_error": best_val_action_error,
                        "config": config,
                    }, best_checkpoint_path)
                    train_pbar.write(f" Best Checkpoint 已更新并成功保存至: {best_checkpoint_path}")
                
                # 极其关键：将模型切回训练模式
                policy.train()
        
        print(f" -> Epoch [{epoch}/{num_epochs}] 归档，当前已推进至 Global Step: {global_step} ")

    # 如果总步数不是评估频率的整数倍，在整个训练完全结束时，无条件执行最后一次强行验证
    if global_step % eval_freq != 0:
        print(f"\n 训练结束。执行最终验证 (Final Step: {global_step})... 模式: 【{val_mode.upper()}】")
        
        if val_mode == "ema":
            avg_val_action_error = evaluate_policy_ema(
                policy, val_dataloader, device, k=val_k, desc=f"Final Eval at Step {global_step} (EMA)"
            )
        else:
            avg_val_action_error = evaluate_policy(
                policy, val_dataloader, device, desc=f"Final Eval at Step {global_step}"
            )
        
        wandb.log({"val/step_avg_action_error": avg_val_action_error, "train/global_step": global_step})
        print(f" 最终验证完毕 -> Avg Val Action Error: {avg_val_action_error:.4f}")
        
        if avg_val_action_error < best_val_action_error:
            best_val_action_error = avg_val_action_error
            best_checkpoint_path = os.path.join(save_dir, "best_act_policy.pt")
            torch.save({
                "epoch": num_epochs,
                "global_step": global_step,
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_action_error": best_val_action_error,
                "config": config,
            }, best_checkpoint_path)
            print(f" 成功打破历史纪录！Best Checkpoint 保存至: {best_checkpoint_path}")
            
    wandb.finish()
    print("Training pipeline finished successfully.")

if __name__ == "__main__":
    main()
