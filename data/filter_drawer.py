import os
import glob
import pandas as pd
from tqdm import tqdm

def filter_and_save_specific_tasks(base_dir, split_name, target_tasks):
    """
    遍历指定的 Split 文件夹，过滤出满足 target_tasks 列表中任意一个 ID 的帧，
    并保持原始的 chunk-xxx 级联目录结构导出。
    """
    print(f"\n🚀 开始过滤 [{split_name}] 中属于 Open Drawer 任务的数据...")
    print(f"🎯 目标 Task 索引列表: {target_tasks}")
    
    source_split_dir = os.path.join(base_dir, split_name)
    target_split_dir = os.path.join(base_dir, f"filtered_{split_name}")
    
    # 匹配当前 split 下的所有 parquet 轨迹文件
    search_path = os.path.join(source_split_dir, "data", "chunk-*", "*.parquet")
    files = glob.glob(search_path)
    
    if not files:
        print(f"  ℹ️ 错误：未在路径 【{source_split_dir}】 下找到任何数据，请检查拼写或路径状态。")
        return
        
    print(f"  📦 共有 {len(files)} 个 Episode 文件待扫描...")
    saved_count = 0
    
    for file_path in tqdm(files, desc=f"Filtering {split_name}"):
        # 读取当前 Episode 的完整数据
        df = pd.read_parquet(file_path)
        
        # 💡 核心改动：使用 .isin() 过滤出属于你指定的 8 个 open drawer 语言指令的所有帧
        filtered_df = df[df["task_index"].isin(target_tasks)]
        
        # 如果该 Episode 中包含我们想要的开抽屉动作，则予以保留
        if not filtered_df.empty:
            # 动态构建相对路径，确保原汁原味的 chunk-xxx 层级不丢失
            rel_path = os.path.relpath(file_path, source_split_dir)
            output_file_path = os.path.join(target_split_dir, rel_path)
            
            # 创建多级父目录
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            
            # 写入全新的、只含有 open drawer 帧的 Parquet 文件
            filtered_df.to_parquet(output_file_path, index=False)
            saved_count += 1
            
    print(f"  ✅ [{split_name}] 过滤清洗完毕!")
    print(f"  📊 结果：成功将 {saved_count}/{len(files)} 个包含开抽屉动作的文件导出至 {target_split_dir}\n")

if __name__ == "__main__":
    # 根据你的 Autodl 容器习惯，当前脚本在 data/ 目录下，base_dir 即为当前目录
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 填入你肉眼排查出来的 splitA 专属 open drawer 任务 ID 列表
    # OPEN_DRAWER_TASKS_A = [3, 10, 16, 22, 43, 100, 155, 341]

    # filter_and_save_specific_tasks(CURRENT_DIR, "splitA", OPEN_DRAWER_TASKS_A)

    # OPEN_DRAWER_TASKS_B = [1, 13, 39, 42, 51, 63, 76, 100, 115]
    
    # filter_and_save_specific_tasks(CURRENT_DIR, "splitB", OPEN_DRAWER_TASKS_B)

    # OPEN_DRAWER_TASKS_C = [17, 23, 27, 49, 60, 86, 111, 313]
    
    # filter_and_save_specific_tasks(CURRENT_DIR, "splitC", OPEN_DRAWER_TASKS_C)

    OPEN_DRAWER_TASKS_D = [8, 10, 46, 72, 81, 110, 120, 324]
    
    filter_and_save_specific_tasks(CURRENT_DIR, "splitD", OPEN_DRAWER_TASKS_D)
    
    
    print("🎉 数据集指定任务清洗过滤工作全部完成！")
