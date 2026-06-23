import os
import sys
import yaml
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2  # 用于生成视频
from tqdm import tqdm
import traceback

# Hydra 编排核心组件
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 环境与模型组件
from calvin_env.envs.play_table_env import PlayTableSimEnv
from utils.utils import create_policy_feature
from lerobot.policies.act.modeling_act import ACTPolicy, ACTConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Record Success and Failure Videos for ACT on CALVIN")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Optional fallback config path")
    
    # 环境场景选择参数
    parser.add_argument("--scene", type=str, default="D", choices=["A", "B", "C", "D", "a", "b", "c", "d"],
                        help="The evaluation scene environment: A, B, C, or D (default: D)")
    
    parser.add_argument("--max_steps", type=int, default=360, help="Max simulation steps per rollout")
    parser.add_argument("--mode", type=str, default="ema", choices=["none", "ema"])
    parser.add_argument("--k", type=float, default=0.01)
    return parser.parse_args()


def preprocess_image(img_array, target_size=(200, 200)):
    img = np.asarray(img_array).copy()
    if img.ndim == 3 and img.shape[0] in (1, 3, 4):
        img = img.transpose(1, 2, 0)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    img_tensor = torch.from_numpy(img).float()
    if img_tensor.max() > 1.0:
        img_tensor /= 255.0
    img_tensor = img_tensor.permute(2, 0, 1)
    if img_tensor.shape[1:] != target_size:
        img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False).squeeze(0)
    else:
        img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


def save_video(frames, save_path, fps=30):
    """将 RGB 帧列表安全导出为 MP4 视频"""
    if not frames:
        return
    # 确保维度为 (H, W, C)
    sample_frame = frames[0]
    if sample_frame.shape[0] == 3:
        sample_frame = sample_frame.transpose(1, 2, 0)
    
    h, w, _ = sample_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    
    for frame in frames:
        if frame.shape[0] == 3:
            frame = frame.transpose(1, 2, 0)
        # OpenCV 需要 BGR 格式
        bgr_frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
    out.release()
    print(f"🎥 视频已成功保存至: {save_path}")


def main():
    args = parse_args()
    target_scene = args.scene.upper()
    
    # 1. 加载模型与配置
    checkpoint_data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint_data["config"] if isinstance(checkpoint_data, dict) and "config" in checkpoint_data else None
    if config is None:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            
    device = torch.device(config["infrastructure"]["device"] if torch.cuda.is_available() else "cpu")
    action_horizon = config["dataset"]["action_horizon"]
    
    # 2. 重建策略
    input_features = {
        "observation.state": create_policy_feature("state", [15]),
        "observation.images.image": create_policy_feature("image", [3, 200, 200]),
        "observation.images.wrist_image": create_policy_feature("image", [3, 84, 84]),
    }
    output_features = {"action": create_policy_feature("action", [7])}
    
    act_config = ACTConfig(
        n_action_steps=action_horizon, chunk_size=action_horizon,
        vision_backbone=config["model"]["backbone"], dim_model=config["model"]["hidden_dim"],
        dim_feedforward=config["model"]["dim_feedforward"], kl_weight=config["model"]["kl_weight"],
        input_features=input_features, output_features=output_features,
        optimizer_lr=float(config["model"]["lr"]), optimizer_weight_decay=float(config["model"]["weight_decay"]),
        optimizer_lr_backbone=float(config["model"]["lr_backbone"]),
    )
    policy = ACTPolicy(act_config)
    policy.load_state_dict(checkpoint_data["model_state_dict"] if "model_state_dict" in checkpoint_data else checkpoint_data)
    policy.to(device).eval()

    # 3. 唤醒仿真环境 (注入动态 Scene 资产)
    config_dir = "/root/autodl-tmp/calvin_env/conf"
    print(f"\n🚀 正在通过 Hydra 拼装仿真世界资产，目标测试环境: Scene {target_scene} ...")
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None, job_name="eval_calvin_video"):
        cfg = compose(
            config_name="config_data_collection", 
            overrides=[
                "env.show_gui=False", 
                "cameras=static_and_gripper",
                f"scene=calvin_scene_{target_scene}"  # 🎯 动态注入命令行指定的场景
            ]
        )
    
    env = PlayTableSimEnv(
        robot_cfg=cfg.robot, seed=cfg.seed, use_vr=False, bullet_time_step=cfg.env.bullet_time_step,
        cameras=cfg.cameras, show_gui=False, scene_cfg=cfg.scene, use_scene_info=True, use_egl=False,
    )

    # 创建视频保存目录
    video_dir = os.path.join(os.path.dirname(args.checkpoint), "videos")
    os.makedirs(video_dir, exist_ok=True)

    saved_success = False
    saved_failure = False
    rollout_idx = 0

    print(f"\n🎬 视频录制启动！测试场景【Scene {target_scene}】| 目标：成功与失败视频各一份。")
    
    try:
        # 循环跑 rollout，直到集齐成功和失败视频，或者达到最大尝试次数（如 30 次）
        while (not saved_success or not saved_failure) and rollout_idx < 30:
            print(f"\n进展 -> 成功视频: {'[已捕获]' if saved_success else '[寻找中]'} | 失败视频: {'[已捕获]' if saved_failure else '[寻找中]'}")
            print(f"🏃 正在测试环境 Scene {target_scene} 下运行第 {rollout_idx + 1} 轮尝试...")
            
            current_seed = int(config["infrastructure"]["seed"] + rollout_idx * 100)
            env.seed(current_seed)
            obs = env.reset()
            
            start_info = env.get_info()
            rollout_success = False
            
            # 用于存储当前轮次所有帧的列表
            current_rollout_frames = []
            
            predicted_actions = torch.zeros((args.max_steps + action_horizon, args.max_steps + action_horizon, 7), device=device)
            has_prediction = torch.zeros((args.max_steps + action_horizon, args.max_steps + action_horizon), dtype=torch.bool, device=device)

            for t in tqdm(range(args.max_steps), desc=f"Rollout {rollout_idx+1}"):
                # 📸 捕获当前步的全局静态相机画面并缓存
                current_rollout_frames.append(obs["rgb_obs"]["rgb_static"])
                
                # A. 提取观测
                state_raw = torch.tensor(obs["robot_obs"], dtype=torch.float32, device=device).unsqueeze(0)
                rgb_dict = obs["rgb_obs"]
                img_static = preprocess_image(rgb_dict["rgb_static"], target_size=(200, 200)).to(device)
                img_gripper = preprocess_image(rgb_dict["rgb_gripper"], target_size=(84, 84)).to(device)
                
                batch = {
                    "observation.state": state_raw,
                    "observation.images.image": img_static,
                    "observation.images.wrist_image": img_gripper
                }
                
                # B. 模型推理
                with torch.no_grad():
                    action_chunk = policy.predict_action_chunk(batch)[0]
                
                for i in range(action_horizon):
                    predicted_actions[t, t + i] = action_chunk[i]
                    has_prediction[t, t + i] = True
                
                # C. 动作集成
                if args.mode == "ema":
                    actions_for_t, weights = [], []
                    for tau in range(max(0, t - action_horizon + 1), t + 1):
                        if has_prediction[tau, t]:
                            actions_for_t.append(predicted_actions[tau, t])
                            weights.append(np.exp(-args.k * (t - tau)))
                    w_tensor = torch.tensor(weights, device=device, dtype=torch.float32)
                    w_tensor = w_tensor / w_tensor.sum()
                    current_action = (torch.stack(actions_for_t) * w_tensor.unsqueeze(-1)).sum(dim=0)
                else:
                    current_action = action_chunk[0]
                
                # D. 环境步进
                action_np = current_action.cpu().numpy()
                action_np[6] = 1.0 if action_np[6] > 0 else -1.0
                obs, reward, done, info = env.step(action_np)
                
                # E. 严格对齐官方抽屉物理标准判定
                current_info = env.get_info()
                start_scene = start_info.get("scene_info", env.scene.get_info())
                current_scene = current_info.get("scene_info", env.scene.get_info())
                
                is_success = abs(current_scene["doors"]["base__drawer"]["current_state"] - 
                                 start_scene["doors"]["base__drawer"]["current_state"]) >= 0.10
                
                if is_success:
                    rollout_success = True
                    # 成功时把最后一帧也加上
                    current_rollout_frames.append(obs["rgb_obs"]["rgb_static"])
                    print(f" 🌟 [SUCCESS] 第 {rollout_idx + 1} 轮成功完成！")
                    break
            
            # 4. 分流保存逻辑 (视频文件名后缀携带 target_scene)
            if rollout_success and not saved_success:
                video_path = os.path.join(video_dir, f"success_rollout_scene_{target_scene}.mp4")
                save_video(current_rollout_frames, video_path, fps=30)
                saved_success = True
            elif not rollout_success and not saved_failure:
                video_path = os.path.join(video_dir, f"failed_rollout_scene_{target_scene}.mp4")
                save_video(current_rollout_frames, video_path, fps=30)
                saved_failure = True
                
            rollout_idx += 1

        print("\n🎉 录制任务结束！请去以下路径查看生成的视频：")
        print(f"📂 视频目录: {video_dir}")

    except Exception:
        print("❌ 运行期间发生异常崩溃：")
        traceback.print_exc()
    finally:
        if 'env' in locals() and env is not None:
            try: env.close()
            except Exception: pass


if __name__ == "__main__":
    main()
