import os
import glob
import io
import random
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
import torchvision.transforms as transforms

class CalvinDataset(Dataset):
    def __init__(self, data_root=None, action_horizon=16, cache_size=16, transform=None, split="train", val_ratio=0.1, seed=42, env_name="B"):
        """
        CALVIN 数据集通用底座
        
        Args:
            data_root (str): 数据根目录。若为 None 则根据 env_name 自动拼接
            action_horizon (int): 动作预测时间步长
            cache_size (int): 缓存的 Parquet 文件数量
            transform (callable): 图像预处理变换
            split (str): "train" 或 "val" 模式
            val_ratio (float): 验证集所占 Episode 的比例
            seed (int): 确保划分可复现的随机种子
            env_name (str): 环境标识符，如 "B" 或 "D"
        """
        self.env_name = env_name.upper()
        
        if data_root is None:
            data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"split{self.env_name}")
            
        self.data_root = data_root
        self.action_horizon = action_horizon
        self.cache_size = cache_size
        self._file_cache = {} 
        
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.ToTensor(),
        ])

        # 1. 检索所有 chunk 下的子 parquet 文件
        search_path = os.path.join(self.data_root, "data", "chunk-*", "*.parquet")
        all_episode_files = sorted(glob.glob(search_path))
        
        if not all_episode_files:
            raise FileNotFoundError(f" 未能在路径【{search_path}】下找到任何数据文件，请检查环境 {self.env_name} 的路径与解压状态！")
            
        # 2. 轨迹级别（Episode-level）划分训练/验证集，防止时序数据泄露
        if val_ratio > 0:
            rng = random.Random(seed)
            shuffled_files = list(all_episode_files)
            rng.shuffle(shuffled_files)
            
            val_size = int(len(shuffled_files) * val_ratio)
            if split == "train":
                self.episode_files = shuffled_files[val_size:]
            elif split == "val":
                self.episode_files = shuffled_files[:val_size]
            else:
                raise ValueError("split 参数必须为 'train' 或 'val'")
        else:
            self.episode_files = all_episode_files
            
        print(f" [CalvinDataset] 环境: 环境-{self.env_name} | 模式: [{split.upper()}] 成功加载 {len(self.episode_files)} 个 Episode 数据文件。")
        
        self.image_cols = ["image", "wrist_image"]
        self.state_col = "state"
        self.action_col = "actions"

        # 3. 构建当前 Split 的全局帧索引
        print(f" 正在扫描 [环境-{self.env_name} | {split.upper()}] 数据帧，构建索引...")
        self.global_indices = [] 
        
        for f_path in self.episode_files:
            meta = pq.read_metadata(f_path)
            num_rows = meta.num_rows 
            for f_idx in range(num_rows):
                self.global_indices.append((f_path, f_idx))
                
        print(f" 全局索引构建完成！当前环境-{self.env_name} 可用数据帧数: {len(self.global_indices)}")

    def _get_dataframe(self, file_path):
        """FIFO 缓存机制"""
        if file_path in self._file_cache:
            return self._file_cache[file_path]
        
        df = pd.read_parquet(file_path)
        
        if len(self._file_cache) >= self.cache_size:
            first_key = next(iter(self._file_cache))
            self._file_cache.pop(first_key)
            
        self._file_cache[file_path] = df
        return df

    def _parse_vector_data(self, df, col_name, frame_idx):
        """从 Object 列中稳定提取向量"""
        val = df.iloc[frame_idx][col_name]
        return np.array(val, dtype=np.float32)

    def _parse_image_data(self, df, col_name, frame_idx):
        """解析 Hugging Face 格式的图片字典"""
        val = df.iloc[frame_idx][col_name]
        
        if isinstance(val, dict):
            if 'bytes' in val and val['bytes'] is not None:
                return Image.open(io.BytesIO(val['bytes'])).convert("RGB")
            elif 'path' in val and val['path'] is not None:
                return Image.open(val['path']).convert("RGB")
                
        if isinstance(val, (bytes, bytearray)):
            return Image.open(io.BytesIO(val)).convert("RGB")
        elif isinstance(val, np.ndarray):
            return Image.fromarray(val).convert("RGB")
        else:
            raise TypeError(f"无法解析的图像格式: {type(val)}")

    def __len__(self):
        return len(self.global_indices)

    def __getitem__(self, idx):
        file_path, frame_idx = self.global_indices[idx]
        df = self._get_dataframe(file_path)
        total_frames = len(df)
        
        # 1. 当前时刻的状态 (qpos)
        qpos = self._parse_vector_data(df, self.state_col, frame_idx)
        
        # 2. 当前时刻的图像
        images_dict = {}
        for img_col in self.image_cols:
            pil_img = self._parse_image_data(df, img_col, frame_idx)
            images_dict[img_col] = self.transform(pil_img)
            
        # 3. 规范化动作分片提取（引入零填充与 Padding 掩码）
        sample_act = self._parse_vector_data(df, self.action_col, 0)
        action_dim = sample_act.shape[0]
        
        actions_list = []
        action_is_pad_list = []
        
        for step in range(self.action_horizon):
            target_frame = frame_idx + step
            if target_frame < total_frames:
                # 处于正常轨迹内
                act = self._parse_vector_data(df, self.action_col, target_frame)
                action_is_pad_list.append(False)
            else:
                # 超出轨迹范围：进行零填充，并打上 Padding 标记
                act = np.zeros(action_dim, dtype=np.float32)
                action_is_pad_list.append(True)
            actions_list.append(act)
            
        actions = np.stack(actions_list, axis=0) 
        action_is_pad = np.array(action_is_pad_list, dtype=bool)
        
        return {
            "qpos": torch.tensor(qpos, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.float32),
            "action_is_pad": torch.tensor(action_is_pad, dtype=torch.bool),
            "images": images_dict
        }


class CalvinBDataset(CalvinDataset):
    """专门面向环境 B (splitB) 的数据加载子类"""
    def __init__(self, data_root=None, action_horizon=16, cache_size=16, transform=None, split="train", val_ratio=0.1, seed=42):
        super().__init__(
            data_root=data_root, 
            action_horizon=action_horizon, 
            cache_size=cache_size, 
            transform=transform, 
            split=split, 
            val_ratio=val_ratio, 
            seed=seed, 
            env_name="B"
        )


class CalvinDDataset(CalvinDataset):
    """专门面向环境 D (splitD) 的数据加载子类"""
    def __init__(self, data_root=None, action_horizon=16, cache_size=16, transform=None, split="train", val_ratio=0.1, seed=42):
        super().__init__(
            data_root=data_root, 
            action_horizon=action_horizon, 
            cache_size=cache_size, 
            transform=transform, 
            split=split, 
            val_ratio=val_ratio, 
            seed=seed, 
            env_name="D"
        )


if __name__ == "__main__":
    print("====== 开始对重构后的通用架构进行多环境交叉测试 ======")
    
    try:
        # 测试 1: 验证环境 B 模块
        print("\n--- 正在测试环境 B (CalvinBDataset) ---")
        train_dataset_B = CalvinBDataset(action_horizon=16, split="train", val_ratio=0.1)
        print(f"环境 B 成功建立！总帧数: {len(train_dataset_B)}")
        
        # 测试 2: 验证环境 D 模块
        print("\n--- 正在测试新扩展的环境 D (CalvinDDataset) ---")
        # 提示：如果测试时本地还没有下载或配置好环境 D，这里会精准捕获错误并打印异常提示
        train_dataset_D = CalvinDDataset(action_horizon=16, split="train", val_ratio=0.1)
        print(f"环境 D 成功建立！总帧数: {len(train_dataset_D)}")
        
        dataloader_D = DataLoader(train_dataset_D, batch_size=4, shuffle=True, num_workers=2)
        for batch in dataloader_D:
            print("\n【测试成功】环境 D 成功产出标准规范化数据包！")
            print(f"  qpos 形状:         {batch['qpos'].shape}")
            print(f"  actions 形状:      {batch['actions'].shape}")
            print(f"  action_is_pad 形状:{batch['action_is_pad'].shape}")
            break
            
    except Exception as e:
        print(f"\n 测试流程拦截说明:\n")
        import traceback
        traceback.print_exc()
