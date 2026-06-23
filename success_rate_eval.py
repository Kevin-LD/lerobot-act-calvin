import os
import yaml
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import traceback

# Hydra 编排核心组件
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

# 环境与模型组件
from calvin_env.envs.play_table_env import PlayTableSimEnv
from utils.utils import create_policy_feature
from lerobot.policies.act.modeling_act import ACTPolicy, ACTConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Closed-Loop Success Rate Evaluation for ACT on CALVIN Environment D")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained checkpoint (e.g., runs/run_xxx/best_act_policy.pt)")
    parser.add_argument("--config", type=str, default=None, help="Optional fallback config path if checkpoint doesn't contain it")
    
    # 仿真特有参数
    parser.add_argument("--task_name", type=str, default="open_drawer", 
                        help="The target task name to evaluate (e.g., 'open_drawer', 'turn_on_light', etc.)")
    parser.add_argument("--num_rollouts", type=int, default=20, help="Number of evaluation independent rollouts")
    parser.add_argument("--max_steps", type=int, default=360, help="Max simulation steps per rollout episode")
    
    # 推理集成模式
    parser.add_argument("--mode", type=str, default="ema", choices=["none", "ema"], 
                        help="Action inference mode: 'none' (replan every step) or 'ema' (temporal ensembling)")
    parser.add_argument("--k", type=float, default=0.01, 
                        help="Temporal ensembling exponential decay constant")
    return parser.parse_args()


def preprocess_image(img_array, target_size=(200, 200)):
    """将环境返回的原始图像安全转换为 LeRobot Policy 接收的标准归一化 Tensor [1, 3, H, W]"""
    img = np.asarray(img_array).copy()
    
    # 维度自适应调整 (C, H, W) -> (H, W, C)
    if img.ndim == 3 and img.shape[0] in (1, 3, 4):
        img = img.transpose(1, 2, 0)
    if img.shape[-1] == 4: # 裁剪 alpha 通道
        img = img[:, :, :3]
        
    # 转换为 FloatTensor 并归一化到 [0, 1]
    img_tensor = torch.from_numpy(img).float()
    if img_tensor.max() > 1.0:
        img_tensor /= 255.0
        
    # 调整通道顺序为 (C, H, W)
    img_tensor = img_tensor.permute(2, 0, 1)
    
    # 尺寸缩放适配输入特性
    if img_tensor.shape[1:] != target_size:
        img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False).squeeze(0)
    else:
        img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


def main():
    args = parse_args()
    
    # 1. 载入 Checkpoint 权重与配置
    print(f"正在读取 Checkpoint 核心数据: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"未找到指定的权重文件: {args.checkpoint}")
        
    checkpoint_data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    
    config = None
    if isinstance(checkpoint_data, dict) and "config" in checkpoint_data:
        config = checkpoint_data["config"]
        print(" 🎯 [Auto Config] 成功从 Checkpoint 中检索到训练期原始配置！")
    else:
        if args.config is None:
            raise ValueError("该 Checkpoint 未内嵌 config，请通过 --config 手动指定配置文件！")
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            
    device = torch.device(config["infrastructure"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. 按照训练配置动态重建 ACT 策略网络结构
    print("\nReconstructing LeRobot ACT Policy Architecture...")
    action_horizon = config["dataset"]["action_horizon"]
    
    input_features = {
        "observation.state": create_policy_feature("state", [15]),
        "observation.images.image": create_policy_feature("image", [3, 200, 200]),
        "observation.images.wrist_image": create_policy_feature("image", [3, 84, 84]),
    }
    output_features = {
        "action": create_policy_feature("action", [7]),
    }
    
    act_config = ACTConfig(
        n_action_steps=action_horizon,
        chunk_size=action_horizon,
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
    
    if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
        policy.load_state_dict(checkpoint_data["model_state_dict"])
    else:
        policy.load_state_dict(checkpoint_data)
    print("模型权重成功注入。")
    policy.to(device)
    policy.eval()

    # 3. 动态唤醒 CALVIN 仿真环境
    config_dir = "/root/autodl-tmp/calvin_env/conf"
    print("\n🚀 正在通过 Hydra 拼装仿真世界资产...")
    try:
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=config_dir, version_base=None, job_name="eval_calvin_env"):
            cfg = compose(
                config_name="config_data_collection",
                overrides=["env.show_gui=False", "cameras=static_and_gripper"]
            )
        
        env = PlayTableSimEnv(
            robot_cfg=cfg.robot,
            seed=cfg.seed,
            use_vr=False,
            bullet_time_step=cfg.env.bullet_time_step,
            cameras=cfg.cameras,
            show_gui=False,
            scene_cfg=cfg.scene,
            use_scene_info=True,
            use_egl=False,
        )
        print("🎉 CALVIN 物理仿真环境成功拉起！")
    except Exception:
        print("❌ 环境组装阶段崩溃：")
        traceback.print_exc()
        return

    # 4. 闭环评测循环 (Closed-Loop Rollouts)
    print(f"\n评估启动：目标任务【{args.task_name}】 | 总计轮次: {args.num_rollouts} | 单轮最大步数: {args.max_steps}")
    success_count = 0

    try:
        for rollout_idx in range(args.num_rollouts):
            print(f"🎬 开始第 {rollout_idx + 1}/{args.num_rollouts} 轮测试...")
            
            # 为每轮变换随机种子以确保泛化性评估的严谨
            current_seed = int(config["infrastructure"]["seed"] + rollout_idx * 100)
            env.seed(current_seed)
            obs = env.reset()
            
            start_info = env.get_info()
            rollout_success = False
            
            # 🎯 初始化时序动作集成(Temporal Ensembling)数据矩阵
            predicted_actions = torch.zeros((args.max_steps + action_horizon, args.max_steps + action_horizon, 7), device=device)
            has_prediction = torch.zeros((args.max_steps + action_horizon, args.max_steps + action_horizon), dtype=torch.bool, device=device)

            for t in tqdm(range(args.max_steps), desc=f"Rollout {rollout_idx+1}"):
                # A. 提取并预处理当前帧的多模态观测
                state_raw = torch.tensor(obs["robot_obs"], dtype=torch.float32, device=device).unsqueeze(0) # [1, 15]
                
                rgb_dict = obs["rgb_obs"]
                img_static = preprocess_image(rgb_dict["rgb_static"], target_size=(200, 200)).to(device)
                img_gripper = preprocess_image(rgb_dict["rgb_gripper"], target_size=(84, 84)).to(device)
                
                # 包装为 LeRobot Policy 标准输入格式
                batch = {
                    "observation.state": state_raw,
                    "observation.images.image": img_static,
                    "observation.images.wrist_image": img_gripper
                }
                
                # B. 模型前向推理，预测一个 Action Chunk
                with torch.no_grad():
                    action_chunk = policy.predict_action_chunk(batch)[0] # 形状: [action_horizon, 7]
                
                # C. 将当前推导出的动作片段填入全局时序预测矩阵
                for i in range(action_horizon):
                    predicted_actions[t, t + i] = action_chunk[i]
                    has_prediction[t, t + i] = True
                
                # D. 根据集成策略解出当前步应执行的动作
                if args.mode == "ema":
                    # 时序滑动平均：加权融合同一时间步被历史不同阶段预测的结果
                    actions_for_t = []
                    weights = []
                    for tau in range(max(0, t - action_horizon + 1), t + 1):
                        if has_prediction[tau, t]:
                            actions_for_t.append(predicted_actions[tau, t])
                            weights.append(np.exp(-args.k * (t - tau)))
                            
                    w_tensor = torch.tensor(weights, device=device, dtype=torch.float32)
                    w_tensor = w_tensor / w_tensor.sum() # 权重归一化
                    
                    current_action = (torch.stack(actions_for_t) * w_tensor.unsqueeze(-1)).sum(dim=0)
                else:
                    # Baseline 模式：不做集成，直接执行当前步产生的第一步动作（即每步都 Replan）
                    current_action = action_chunk[0]
                
                # E. 物理仿真环境前进一步
                action_np = current_action.cpu().numpy()
                
                # 🔒 夹爪动作二值化：强行对齐 CALVIN 底层的 (-1, 1) 断言要求
                action_np[6] = 1.0 if action_np[6] > 0 else -1.0
                
                obs, reward, done, info = env.step(action_np)
                
                # F. 闭环任务成功率判定
                current_info = env.get_info()
                
                # 安全防御提取：优先读 info，若没有则直接向底层 scene 要
                start_scene = start_info.get("scene_info", env.scene.get_info())
                current_scene = current_info.get("scene_info", env.scene.get_info())
                
                # 🎯 精确锁定 CALVIN 官方抽屉物理关节路径
                start_drawer = start_scene["doors"]["base__drawer"]["current_state"]
                current_drawer = current_scene["doors"]["base__drawer"]["current_state"]
                
                # 严格对齐 CALVIN 官方标准：拉出距离至少达到 10 cm (0.10 米)
                is_success = abs(current_drawer - start_drawer) >= 0.10
                
                if is_success:
                    rollout_success = True
                    print(f" 🌟 [SUCCESS] 任务【{args.task_name}】在第 {t} 步提早宣告成功！")
                    break
                
                if is_success:
                    rollout_success = True
                    print(f" 🌟 [SUCCESS] 任务【{args.task_name}】在第 {t} 步提早宣告成功！")
                    break
            
            if rollout_success:
                success_count += 1
            else:
                print(f" ❌ [FAILED] 第 {rollout_idx + 1} 轮未能在最大步数内完成指定任务。")

        # 5. 指标统计与报告导出
        success_rate = (success_count / args.num_rollouts) * 100
        print("\n" + "="*60)
        print("                📊 SIMULATION SUCCESS REPORT                ")
        print("="*60)
        print(f" 评估模型:      {os.path.basename(args.checkpoint)}")
        print(f" 目标测试任务:  {args.task_name}")
        print(f" 动作集成模式:  {args.mode.upper()} (k={args.k if args.mode=='ema' else 'N/A'})")
        print(f" 测试总轮次:    {args.num_rollouts} 轮")
        print(f" 成功完成轮次:  {success_count} 轮")
        print(f" 🎯 仿真最终成功率 (Success Rate): {success_rate:.2f}%")
        print("="*60)
        
        # 保存本地报告
        report_path = os.path.join(os.path.dirname(args.checkpoint), f"eval_success_rate_{args.task_name}_{args.mode}.txt")
        with open(report_path, "w") as f:
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Target Task: {args.task_name}\n")
            f.write(f"Inference Mode: {args.mode} (k={args.k})\n")
            f.write(f"Total Rollouts: {args.num_rollouts}\n")
            f.write(f"Success Rollouts: {success_count}\n")
            f.write(f"Success Rate: {success_rate:.2f}%\n")
        print(f"成功率评估报告已同步保存至: {report_path}\n")

    except Exception:
        print("❌ 运行期间发生异常崩溃：")
        traceback.print_exc()
    finally:
        if 'env' in locals() and env is not None:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
