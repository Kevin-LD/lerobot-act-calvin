import pandas as pd
import glob

# 找到第一个数据文件
files = sorted(glob.glob("data/splitB/data/chunk-*/*.parquet"))
if not files:
    print(" 未找到任何 parquet 文件，请检查路径！")
    exit()

print(f"正在读取文件: {files[0]}\n")
df = pd.read_parquet(files[0])

print(" 【所有列名及数据类型】:")
for col in df.columns:
    # 如果是数组或者列表，顺便打印一下第一帧的形状或长度
    sample = df[col].iloc[0]
    shape_str = f" | 样例形状/长度: {len(sample)}" if hasattr(sample, '__len__') and not isinstance(sample, (str, bytes)) else ""
    print(f"  - {col} ({df[col].dtype}){shape_str}")

print("\n 【第一帧数据部分预览】:")
print(df.head(1).to_dict(orient='records')[0].keys())
