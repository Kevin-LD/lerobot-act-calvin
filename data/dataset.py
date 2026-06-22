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

class CalvinMultiEnvDataset(Dataset):
    """
    通用多环境联合数据集包装器
    """
    def __init__(self, env_names=["A", "B", "C"], data_root=None, **kwargs):
        """
        Args:
            env_names (list): 需要合并的环境标识符列表，如 ["A", "B", "C"]
            data_root (str): 若不指定，子数据集会自动寻找各自对应的 splitA, splitB, splitC 文件夹
            **kwargs: 透传给子数据集的通用参数 (action_horizon, split, val_ratio, seed 等)
        """
        self.env_names = [env.upper() for env in env_names]
        self.datasets = []
        
        print(f"\n================ 开始构建多环境联合数据集: {self.env_names} ================")
        for env in self.env_names:
            # 允许为不同环境设定独立的根目录，如果不给，单环境类内部会自动拼接出 splitA/splitB/splitC
            env_data_root = None if data_root is None else os.path.join(data_root, f"split{env}")
            
            sub_dataset = CalvinDataset(
                data_root=env_data_root,
                env_name=env,
                **kwargs
            )
            self.datasets.append(sub_dataset)
            
        self.lengths = [len(ds) for ds in self.datasets]
        self.total_frames = sum(self.lengths)
        print(f" [MultiEnv] 联合数据集构建完毕！各环境帧数分布: {dict(zip(self.env_names, self.lengths))} | 总计可用帧数: {self.total_frames}\n")

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_frames:
            raise IndexError("Dataset index out of range")
        
        # 依靠步长区间进行动态路由寻址
        target_idx = idx
        for sub_dataset in self.datasets:
            if target_idx < len(sub_dataset):
                return sub_dataset[target_idx]
            target_idx -= len(sub_dataset)
            
        raise IndexError("Dataset index out of range")

    # ----- 兼容层：将底层所有子数据集的核心元数据铺平暴露 -----
    @property
    def global_indices(self):
        """平铺合并所有子环境的全局帧映射"""
        combined = []
        for ds in self.datasets:
            combined.extend(ds.global_indices)
        return combined

    @property
    def episode_files(self):
        """平铺合并所有子环境的文件追踪列表"""
        combined = []
        for ds in self.datasets:
            combined.extend(ds.episode_files)
        return combined


class CalvinABCDataset(CalvinMultiEnvDataset):
    """专门面向 A, B, C 三环境联合训练的数据加载子类"""
    def __init__(self, data_root=None, action_horizon=16, cache_size=16, transform=None, split="train", val_ratio=0.1, seed=42):
        super().__init__(
            env_names=["A", "B", "C"],
            data_root=data_root,
            action_horizon=action_horizon,
            cache_size=cache_size,
            transform=transform,
            split=split,
            val_ratio=val_ratio,
            seed=seed
        )

class SingleEpisodeDataset(Dataset):
    """
    轻量化单轨迹数据集：仅加载单个指定的 .parquet 文件
    """
    def __init__(self, file_path, action_horizon=16, transform=None):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"未找到指定的 Parquet 文件: {file_path}")
            
        self.file_path = file_path
        self.action_horizon = action_horizon
        
        self.df = pd.read_parquet(file_path)
        self.total_frames = len(self.df)
        
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.image_cols = ["image", "wrist_image"]
        self.state_col = "state"
        self.action_col = "actions"
        
        print(f"[SingleEpisodeDataset] 成功单点加载: {os.path.basename(file_path)} | 总帧数: {self.total_frames}")

    def __len__(self):
        return self.total_frames

    def _parse_vector_data(self, col_name, frame_idx):
        val = self.df.iloc[frame_idx][col_name]
        return np.array(val, dtype=np.float32)

    def _parse_image_data(self, col_name, frame_idx):
        val = self.df.iloc[frame_idx][col_name]
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

    def __getitem__(self, idx):
        # 帧索引与 Dataset 的 idx 完美等价 1:1 映射
        frame_idx = idx
        
        # 1. 提取当前帧状态 (qpos)
        qpos = self._parse_vector_data(self.state_col, frame_idx)
        
        # 2. 提取当前帧双路图像
        images_dict = {}
        for img_col in self.image_cols:
            pil_img = self._parse_image_data(img_col, frame_idx)
            images_dict[img_col] = self.transform(pil_img)
            
        # 3. 规范化动作分片提取（带时序自适应边界 Padding）
        sample_act = self._parse_vector_data(self.action_col, 0)
        action_dim = sample_act.shape[0]
        
        actions_list = []
        action_is_pad_list = []
        
        for step in range(self.action_horizon):
            target_frame = frame_idx + step
            if target_frame < self.total_frames:
                act = self._parse_vector_data(self.action_col, target_frame)
                action_is_pad_list.append(False)
            else:
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


class CalvinFilteredDataset(CalvinDataset):
    """
    针对过滤后单任务数据的通用基底
    """
    def __init__(self, data_root=None, action_horizon=16, cache_size=16, transform=None, split="train", val_ratio=0.1, seed=42, env_name="B"):
        # 核心改动：如果未指定路径，默认指向被过滤出的单任务文件夹
        if data_root is None:
            data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"filtered_split{env_name.upper()}")
            
        super().__init__(
            data_root=data_root,
            action_horizon=action_horizon,
            cache_size=cache_size,
            transform=transform,
            split=split,
            val_ratio=val_ratio,
            seed=seed,
            env_name=env_name
        )
        print(f" 🎯 [Filtered Mode Active] 已锁定单任务过滤数据源: {data_root}")


class CalvinFilteredBDataset(CalvinFilteredDataset):
    """专门面向【过滤单任务】环境 B (filtered_splitB) 的数据加载子类"""
    def __init__(self, **kwargs):
        super().__init__(env_name="B", **kwargs)


class CalvinFilteredDDataset(CalvinFilteredDataset):
    """专门面向【过滤单任务】测试环境 D (filtered_splitD) 的数据加载子类，用于闭环仿真或指标测算"""
    def __init__(self, **kwargs):
        super().__init__(env_name="D", **kwargs)


class CalvinFilteredMultiEnvDataset(CalvinMultiEnvDataset):
    """
    针对过滤后多环境联合数据集的包装器
    """
    def __init__(self, env_names=["A", "B", "C"], data_root=None, **kwargs):
        self.env_names = [env.upper() for env in env_names]
        self.datasets = []
        
        print(f"\n================ 开始构建【过滤单任务】多环境联合数据集: {self.env_names} ================")
        for env in self.env_names:
            # 动态映射各自对应的 filtered_splitA, filtered_splitB ...
            env_data_root = None if data_root is None else os.path.join(data_root, f"filtered_split{env}")
            
            sub_dataset = CalvinFilteredDataset(
                data_root=env_data_root,
                env_name=env,
                **kwargs
            )
            self.datasets.append(sub_dataset)
            
        self.lengths = [len(ds) for ds in self.datasets]
        self.total_frames = sum(self.lengths)
        print(f" [Filtered MultiEnv] 联合数据集构建完毕！各环境帧数分布: {dict(zip(self.env_names, self.lengths))} | 总计可用帧数: {self.total_frames}\n")


class CalvinFilteredABCDataset(CalvinFilteredMultiEnvDataset):
    """专门面向 A, B, C 三个环境【过滤单任务数据】联合训练的数据加载子类"""
    def __init__(self, data_root=None, action_horizon=16, cache_size=16, transform=None, split="train", val_ratio=0.1, seed=42):
        super().__init__(
            env_names=["A", "B", "C"],
            data_root=data_root,
            action_horizon=action_horizon,
            cache_size=cache_size,
            transform=transform,
            split=split,
            val_ratio=val_ratio,
            seed=seed
        )

if __name__ == "__main__":
    print("\n====== 开始对【过滤后的单任务数据集】进行多环境交叉测试 ======")
    
    try:
        # 测试 1: 验证单环境过滤后的 B 模块
        print("\n--- 正在测试过滤后的环境 B (CalvinFilteredBDataset) ---")
        filtered_dataset_B = CalvinFilteredBDataset(action_horizon=16, split="train", val_ratio=0.1)
        print(f"✨ 过滤环境 B 成功建立！可用单任务总帧数: {len(filtered_dataset_B)}")
        
        # 测试 2: 验证过滤后的 A, B, C 多环境联合加载器
        print("\n--- 正在测试过滤后的 A, B, C 多环境联合数据集 (CalvinFilteredABCDataset) ---")
        filtered_dataset_ABC = CalvinFilteredABCDataset(action_horizon=16, split="train", val_ratio=0.1)
        print(f"✨ 过滤联合数据集 ABC 成功建立！总计单任务帧数: {len(filtered_dataset_ABC)}")
        
        # 测试 3: 验证过滤后的联合 DataLoader 数据流动性
        filtered_dataloader_ABC = DataLoader(filtered_dataset_ABC, batch_size=4, shuffle=True, num_workers=2)
        for batch in filtered_dataloader_ABC:
            print("\n🔥【💪 测试完美通过】过滤后的单任务联合数据集成功产出标准数据包！")
            print(f" ├─ qpos 形状:         {batch['qpos'].shape}       (预期: [batch, 15])")
            print(f" ├─ actions 形状:      {batch['actions'].shape}    (预期: [batch, horizon, 7])")
            print(f" ├─ action_is_pad 形状:{batch['action_is_pad'].shape} (预期: [batch, horizon])")
            print(f" └─ images['image'] 形状: {batch['images']['image'].shape} (预期: [batch, 3, H, W])")
            break
            
    except Exception as e:
        print(f"\n ❌ 测试失败，错误诊断说明如下:\n")
        import traceback
        traceback.print_exc()
