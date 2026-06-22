import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import inspect
from lerobot.configs.types import PolicyFeature, FeatureType

def build_lerobot_batch(batch, device):
    """
    将 Dataset 产生的数据字典打包
    """
    lerobot_batch = {
        "observation.state": batch["qpos"].to(device),
        "observation.images.image": batch["images"]["image"].to(device),
        "observation.images.wrist_image": batch["images"]["wrist_image"].to(device),
        "action": batch["actions"].to(device),
        "action_is_pad": batch["action_is_pad"].to(device) 
    }
    return lerobot_batch

def evaluate_policy(policy, dataloader, device, desc="Evaluation"):
    """
    策略网络评估核心函数
    
    Args:
        policy: 已初始化的 ACTPolicy 模型
        dataloader: 评估集 DataLoader
        device: torch.device
        desc (str): tqdm 进度条的描述文本
        
    Returns:
        float: 全局平均动作误差 (L1 Loss)
    """
    policy.eval()
    step_val_action_errors = []
    
    # 使用 leave=False 确保多轮调用时进度条不会在终端刷屏
    val_pbar = tqdm(dataloader, desc=f"  [{desc}]", leave=False)
    
    with torch.no_grad():
        for val_batch in val_pbar:
            # 1. 组装基础 Batch
            lerobot_batch_val = build_lerobot_batch(val_batch, device)
            
            # 2. 对齐 LeRobot 内部的图像多路输入容器命名架构
            if policy.config.image_features:
                lerobot_batch_val = dict(lerobot_batch_val)
                try:
                    from lerobot.policies.act.modeling_act import OBS_IMAGES
                except ImportError:
                    OBS_IMAGES = "observation.images"
                lerobot_batch_val[OBS_IMAGES] = [lerobot_batch_val[key] for key in policy.config.image_features]
            
            # 3. 前向推理预测动作
            actions_hat, _ = policy.model(lerobot_batch_val)
            
            # 4. 提取当前第一步步长的预测与真值
            pred_action = actions_hat[:, 0, :]
            gt_action = lerobot_batch_val["action"][:, 0, :]
            
            # 5. 计算 L1 误差
            val_action_error = F.l1_loss(pred_action, gt_action, reduction="mean")
            error_val = val_action_error.item()
            step_val_action_errors.append(error_val)
            
            val_pbar.set_postfix({"act_err": f"{error_val:.4f}"})
            
    # 计算整体平均误差
    avg_val_action_error = np.mean(step_val_action_errors)
    return float(avg_val_action_error)

def evaluate_policy_ema(policy, dataloader, device, k=0.1, desc="Evaluation (EMA)"):
    """
   滑动平均评估函数
    """
    import torch.nn.functional as F
    import numpy as np
    from torch.utils.data import Subset
    from tqdm import tqdm
    
    policy.eval()
    total_frames = len(dataloader.dataset)
    
    dataset = dataloader.dataset
    if isinstance(dataset, Subset):
        indices = dataset.indices
        base_dataset = dataset.dataset
    else:
        indices = list(range(total_frames))
        base_dataset = dataset

    if hasattr(base_dataset, "global_indices"):
        # 将文件路径映射为唯一的整数 ID
        unique_files = base_dataset.episode_files
        file_to_id = {path: i for i, path in enumerate(unique_files)}
        
        # 提取当前 split 序列中每一帧对应的 Episode ID
        ep_ids = [file_to_id[base_dataset.global_indices[idx][0]] for idx in indices]
        episode_canvas = torch.tensor(ep_ids, dtype=torch.long, device=device)
        print(f"\n[EMA 验证] 检测到 {len(unique_files)} 个独立轨迹文件。")
    else:
        print(f"\n[EMA 验证] 未能检测到 global_indices 属性，降级为标准流式计算（存在跨序列污染）。")
        episode_canvas = torch.zeros((total_frames,), dtype=torch.long, device=device)

    pred_canvas = None
    gt_canvas = None
    start_idx = 0
    val_pbar = tqdm(dataloader, desc=f"  [{desc}]", leave=False)
    
    # 2. 正常高并发收集预测结果
    with torch.no_grad():
        for batch in val_pbar:
            lerobot_batch_val = build_lerobot_batch(batch, device)
            if policy.config.image_features:
                lerobot_batch_val = dict(lerobot_batch_val)
                try:
                    from lerobot.policies.act.modeling_act import OBS_IMAGES
                except ImportError:
                    OBS_IMAGES = "observation.images"
                lerobot_batch_val[OBS_IMAGES] = [lerobot_batch_val[key] for key in policy.config.image_features]
            
            actions_hat, _ = policy.model(lerobot_batch_val)
            
            if pred_canvas is None:
                action_dim = actions_hat.shape[-1]
                chunk_size = actions_hat.shape[1]
                pred_canvas = torch.zeros((total_frames, chunk_size, action_dim), device=device)
                gt_canvas = torch.zeros((total_frames, action_dim), device=device)
            
            B = actions_hat.shape[0]
            end_idx = start_idx + B
            
            pred_canvas[start_idx:end_idx] = actions_hat
            # 对应你 Dataset 返回的键名 "actions"
            gt_canvas[start_idx:end_idx] = lerobot_batch_val["action"][:, 0, :]
            
            start_idx = end_idx

    # 3. 带有时序隔离的滑动平滑集成
    print(f"\n 正在执行严格的时序边界隔离与滑动集成 (Total frames: {total_frames})...")
    compiled_actions = torch.zeros((total_frames, action_dim), device=device)
    compiled_weights = torch.zeros((total_frames, 1), device=device)
    
    for step in range(chunk_size):
        weight = np.exp(-k * step)
        s_indices = torch.arange(0, total_frames - step, device=device)
        t_indices = s_indices + step
        
        # 核心逻辑：只有当两个指针指向同一个 parquet 文件时，才允许进行时序集成叠加
        same_episode_mask = (episode_canvas[s_indices] == episode_canvas[t_indices])
        if not same_episode_mask.any():
            continue
            
        valid_s = s_indices[same_episode_mask]
        valid_t = t_indices[same_episode_mask]
        
        compiled_actions[valid_t] += pred_canvas[valid_s, step] * weight
        compiled_weights[valid_t] += weight
        
    final_actions = compiled_actions / compiled_weights
    val_action_error = F.l1_loss(final_actions, gt_canvas, reduction="mean")
    
    return float(val_action_error.item())


def create_policy_feature(feature_type_str, shape):
    """自适应构建 PolicyFeature 强类型枚举容器"""
    type_mapping = {
        "state": FeatureType.STATE,
        "image": FeatureType.VISUAL,
        "action": FeatureType.ACTION
    }
    ft_enum = type_mapping.get(feature_type_str.lower(), feature_type_str)

    sig = inspect.signature(PolicyFeature)
    params = sig.parameters
    kwargs = {}
    
    if "shape" in params:
        kwargs["shape"] = tuple(shape)
    if "type" in params:
        kwargs["type"] = ft_enum
        
    return PolicyFeature(**kwargs)

def collect_episode_trajectory(policy, dataset, device):
    """
    提取一条轨迹的预测值与真实值
    """
    policy.eval()
    all_gt = []
    all_pred = []
    
    with torch.no_grad():
        # 顺着单文件 Dataset 的长度从头走到尾
        for idx in tqdm(range(len(dataset)), desc=" 轨 迹 顺 序 推 理 中 "):
            sample = dataset[idx]
            
            val_batch = {}
            
            # 1. 状态映射：将 "qpos" 精准平铺映射为 "observation.state"
            val_batch["observation.state"] = sample["qpos"].unsqueeze(0).to(device)
            
            # 2. 图像映射：将多路图像直接平铺到根目录（形如 "observation.images.image"）
            for k, v in sample["images"].items():
                val_batch[f"observation.images.{k}"] = v.unsqueeze(0).to(device)
            
            # 3. 容器兼容层：对齐 LeRobot 内部可能访问的 OBS_IMAGES 列表句柄
            if hasattr(policy, "config") and hasattr(policy.config, "image_features"):
                try:
                    from lerobot.policies.act.modeling_act import OBS_IMAGES
                except ImportError:
                    OBS_IMAGES = "observation.images"
                val_batch[OBS_IMAGES] = [val_batch[key] for key in policy.config.image_features if key in val_batch]
            
            # 4. 动作与 Padding 映射（提供双重命名以确保稳健性）
            val_batch["action"] = sample["actions"].unsqueeze(0).to(device)
            val_batch["actions"] = sample["actions"].unsqueeze(0).to(device)
            if "action_is_pad" in sample:
                val_batch["action_is_pad"] = sample["action_is_pad"].unsqueeze(0).to(device)
            
            # 前向预测
            actions_hat, _ = policy.model(val_batch)
            
            # 剥离出当前第 0 步的物理动作
            pred_action = actions_hat[:, 0, :].cpu().numpy()[0]
            gt_action = val_batch["actions"][:, 0, :].cpu().numpy()[0]
            
            all_pred.append(pred_action)
            all_gt.append(gt_action)
            
    return np.array(all_gt), np.array(all_pred)

def evaluate_policy_dimension_wise(policy, dataloader, device, desc="Evaluation"):
    """
    策略网络评估核心函数（支持总体误差与分维度诊断单次推理同步输出）
    
    Args:
        policy: 已初始化的 ACTPolicy 模型
        dataloader: 评估集 DataLoader
        device: torch.device
        desc (str): tqdm 进度条的描述文本
        
    Returns:
        tuple: (
            float: 全局平均动作误差 (Overall L1 Loss),
            np.ndarray: 包含 7个 元素的 NumPy 数组，对应每个维度的平均动作误差
        )
    """
    policy.eval()
    step_val_action_errors = []
    step_dim_action_errors = []
    
    val_pbar = tqdm(dataloader, desc=f"  [{desc}]", leave=False)
    
    with torch.no_grad():
        for val_batch in val_pbar:
            # 1. 组装基础 Batch
            lerobot_batch_val = build_lerobot_batch(val_batch, device)
            
            # 2. 对齐 LeRobot 内部的图像多路输入容器命名架构
            if policy.config.image_features:
                lerobot_batch_val = dict(lerobot_batch_val)
                try:
                    from lerobot.policies.act.modeling_act import OBS_IMAGES
                except ImportError:
                    OBS_IMAGES = "observation.images"
                lerobot_batch_val[OBS_IMAGES] = [lerobot_batch_val[key] for key in policy.config.image_features]
            
            # 3. 前向推理
            actions_hat, _ = policy.model(lerobot_batch_val)
            
            # 4. 提取当前第一步步长的预测与真值
            pred_action = actions_hat[:, 0, :]
            gt_action = lerobot_batch_val["action"][:, 0, :]
            
            # 5. 计算分维度 L1 误差矩阵 (Batch_Size, 7)，保持空间不坍缩
            batch_dim_errors = F.l1_loss(pred_action, gt_action, reduction="none")
            
            # 指标一：当前 Batch 的总体标量误差（全部元素的平均值）
            error_val = batch_dim_errors.mean().item()
            step_val_action_errors.append(error_val)
            
            # 指标二：当前 Batch 的分维度误差向量（在 Batch 轴求平均，降维至 7）
            error_dim = batch_dim_errors.mean(dim=0).cpu().numpy()
            step_dim_action_errors.append(error_dim)
            
            val_pbar.set_postfix({"act_err": f"{error_val:.4f}"})
            
    # 沿着 Batch 轴计算全数据集的最终平均值
    avg_val_action_error = np.mean(step_val_action_errors)
    avg_dim_action_errors = np.mean(step_dim_action_errors, axis=0)
    
    return float(avg_val_action_error), avg_dim_action_errors
