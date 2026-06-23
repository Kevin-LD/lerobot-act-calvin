import pandas as pd
import numpy as np

file = "data/splitD/data/chunk-000/episode_000000.parquet"

print(f"正在读取文件: {file}\n")
df = pd.read_parquet(file)

print("【所有列名及数据类型】:")

for col in df.columns:
    sample = df[col].iloc[0]

    info_str = ""

    # numpy 数组
    if isinstance(sample, np.ndarray):
        if sample.size == 1:
            info_str = f" | 数值: {sample.item()}"
        else:
            info_str = f" | 形状: {sample.shape}"

    # Python 列表
    elif isinstance(sample, list):
        arr = np.asarray(sample)

        if arr.size == 1:
            info_str = f" | 数值: {arr.item()}"
        else:
            info_str = f" | 长度: {len(sample)}"

    # 标量（int、float、bool、str 等）
    elif np.isscalar(sample):
        info_str = f" | 数值: {sample}"

    # 其他可计算长度的对象
    elif hasattr(sample, "__len__") and not isinstance(sample, (str, bytes)):
        info_str = f" | 长度: {len(sample)}"

    print(f"  - {col} ({df[col].dtype}){info_str}")

print("\n【第一帧数据部分预览】:")
print(df.head(1).to_dict(orient="records")[0].keys())
