import os
import json
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Ablation Study Visualization Script for Action Horizon")
    parser.add_argument(
        "--json_paths", 
        type=str, 
        nargs="+",  # 支持传入多个 JSON 文件路径
        default=["runs/ablation_20260621_232849/ablation_summary.json", "runs/ablation_20260622_114559/ablation_summary.json"],
        help="Path to one or multiple ablation summary JSON files"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="figure", 
        help="Directory to save the generated figure"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 动态合流多份 JSON 文件中的实验数据
    combined_data = defaultdict(list)
    
    for path in args.json_paths:
        if not os.path.exists(path):
            print(f"警告: 未找到文件 {path}，跳过该路径。")
            continue
            
        print(f"正在读取消融数据: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data_dict = json.load(f)
            results = data_dict.get("results", [])
            
            for item in results:
                if item["status"] == "SUCCESS" and item["eval_error_D"] is not None:
                    mode = item["mode"]
                    horizon = item["horizon"]
                    error = item["eval_error_D"]
                    # 暂存数据用于后续排序去重
                    combined_data[mode].append((horizon, error))

    if not combined_data:
        print("错误: 未能提取到任何合法的成功实验数据，绘图终止。")
        return

    # 2. 严谨的学术风格构图配置
    plt.rcParams["font.family"] = "DejaVu Sans"  # 确保跨平台通用且支持学术常规排版
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.linewidth"] = 1.2
    
    plt.figure(figsize=(7.5, 5.5), dpi=300)  # 保持 300 DPI 达到高清出版物打印标准
    
    # 定义不同实验模式的视觉映射
    style_mapping = {
        "B": {"color": "#E66101", "marker": "o", "label": "Mode B (Single Env)"},
        "ABC": {"color": "#5E3C99", "marker": "s", "label": "Mode ABC (Multi Env)"},
        "test": {"color": "#2CA02C", "marker": "D", "label": "Sanity Check (Test)"}
    }

    # 3. 逐模态提取、排序并绘制折线
    all_horizons = set()
    for mode, p_data in combined_data.items():
        # 按 horizon 升序排列，确保补测的 H=2 完美切入 H=1 和 H=4 之间
        p_data_sorted = sorted(list(set(p_data)), key=lambda x: x[0])
        
        horizons = [x[0] for x in p_data_sorted]
        errors = [x[1] for x in p_data_sorted]
        all_horizons.update(horizons)
        
        # 获取视觉样式，若无匹配则分发默认样式
        cfg = style_mapping.get(mode, {"color": "#333333", "marker": "v", "label": f"Mode {mode}"})
        
        plt.plot(
            horizons, 
            errors, 
            linewidth=2.2, 
            color=cfg["color"], 
            marker=cfg["marker"], 
            markersize=7, 
            label=cfg["label"],
            markeredgecolor="white",  # 白边标记让重叠点更清晰
            markeredgewidth=1.0
        )

    # 4. 坐标轴及图例
    sorted_all_horizons = sorted(list(all_horizons))
    
    plt.xscale("log", base=2)  # 使用以 2 为底的对数轴，让 1, 2, 4, 8, 16, 32, 64 等距排开
    plt.xticks(sorted_all_horizons, labels=[str(h) for h in sorted_all_horizons])
    plt.ylim(bottom=0.09)

    plt.xlabel("Action Horizon ($H$)", fontsize=12, fontweight="bold", labelpad=8)
    plt.ylabel("Avg Action $L_1$ Error on Environment D", fontsize=12, fontweight="bold", labelpad=8)
    plt.title("Ablation Study: Impact of Action Horizon on Cross-Env Generalization", fontsize=12, fontweight="bold", pad=15)
    
    plt.grid(True, which="both", linestyle="--", alpha=0.5, linewidth=0.8)
    plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#E0E0E0", framealpha=0.9)
    
    # 自动微调边缘防止文本裁剪
    plt.tight_layout()
    
    # 5. 保存图像
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "horizon_ablation_curve.png")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"\n消融折线图已成功绘制并保存至: {save_path}")

if __name__ == "__main__":
    main()
