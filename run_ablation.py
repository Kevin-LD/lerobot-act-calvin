import os
import subprocess
import yaml
import sys
import json
from datetime import datetime

def run_experiment(base_config_path, horizon, ablation_root):
    """
    运行单次实验：修改指定配置文件的 action_horizon，启动训练
    """
    if not os.path.exists(base_config_path):
        print(f"未找到基础配置文件: {base_config_path}，跳过此实验。")
        return "SKIP", None, 0.0

    # 1. 读取原始配置
    with open(base_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env_mode = config["dataset"].get("env_mode", "B")
    
    # 2. 动态修改消融核心参数
    config["dataset"]["action_horizon"] = horizon
    
    # 3. 设定传递给 train.py 的基础目录前缀
    target_prefix = f"run_{env_mode}_H{horizon}"
    config["infrastructure"]["save_dir"] = os.path.join(ablation_root, target_prefix)
    config["wandb"]["name"] = f"act_{env_mode}_horizon_{horizon}"

    # 4. 生成临时配置文件供 train.py 和 eval.py 共享读取
    tmp_config_path = f"configs/tmp_ablation_{env_mode}_H{horizon}.yaml"
    os.makedirs("configs", exist_ok=True)
    with open(tmp_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    print("\n" + "="*60)
    print(f"[任务启动] 模式: {env_mode} | Action Horizon: {horizon}")
    print(f"临时配置路径: {tmp_config_path}")
    print(f"目标目录前缀: {config['infrastructure']['save_dir']}")
    print("="*60 + "\n")

    # 5. 执行训练阶段 (train.py)
    cmd_train = [sys.executable, "train.py", "--config", tmp_config_path]
    start_time = datetime.now()
    train_success = False
    
    try:
        subprocess.run(cmd_train, check=True)
        train_success = True
    except subprocess.CalledProcessError as e:
        print(f"\n训练崩溃！模式 {env_mode} | Horizon {horizon} 失败。错误码: {e.returncode}")
    except Exception as e:
        print(f"\n运行期间遭遇未知错误: {str(e)}")

    # 6. 动态路径寻踪：捕获被 train.py 拼接了时间戳后的真实子目录
    actual_save_dir = None
    if train_success and os.path.exists(ablation_root):
        # 扫描总消融目录下所有以 target_prefix 开头的真实文件夹
        candidates = [d for d in os.listdir(ablation_root) 
                      if d.startswith(target_prefix) and os.path.isdir(os.path.join(ablation_root, d))]
        if candidates:
            # 排序后取最新的一个（确保即使重跑也能拿到正确的目录）
            candidates.sort()
            actual_save_dir = os.path.join(ablation_root, candidates[-1])

    # 7. 自动化评估阶段 (eval.py)
    eval_success = False
    if train_success and actual_save_dir:
        checkpoint_path = os.path.join(actual_save_dir, "best_act_policy.pt")
        if os.path.exists(checkpoint_path):
            print(f"\n成功捕获真实权重路径: {checkpoint_path}")
            print(f"开始触发零样本环境 D 泛化评估...")
            
            cmd_eval = [
                sys.executable, "eval.py", 
                "--checkpoint", checkpoint_path, 
                "--config", tmp_config_path,
                "--mode", "none"
            ]
            try:
                subprocess.run(cmd_eval, check=True)
                eval_success = True
            except subprocess.CalledProcessError as e:
                print(f"评估脚本执行失败，错误码: {e.returncode}")
        else:
            print(f"虽定位到真实目录，但未找到最佳权重文件: {checkpoint_path}")
    else:
        if train_success:
            print(f"训练看似成功，但在 {ablation_root} 下未找到以 {target_prefix} 开头的真实输出目录。")

    # 8. 回读 eval.py 生成的局部报告指标
    error_val = None
    if eval_success and actual_save_dir:
        report_file = os.path.join(actual_save_dir, "eval_D_report_none.txt")
        if os.path.exists(report_file):
            try:
                with open(report_file, "r", encoding="utf-8") as rf:
                    for line in rf:
                        if "Avg Action L1 Error on D:" in line:
                            # 提取出冒号后面的浮点误差值
                            error_val = float(line.split(":")[-1].strip())
            except Exception as e:
                print(f"解析评估报告文件时出错: {str(e)}")

    # 9. 清理工作与状态返回
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    print(f"\n批次结束 | 总计耗时: {duration:.2f} 分钟")
    
    if os.path.exists(tmp_config_path):
        os.remove(tmp_config_path)

    if train_success and eval_success:
        return "SUCCESS", error_val, duration
    elif train_success and not eval_success:
        return "TRAIN_OK_EVAL_FAIL", None, duration
    else:
        return "TRAIN_FAILED", None, duration


def main():
    # ================= 实验矩阵配置中心 =================
    horizons_to_test = [1, 4, 8, 16, 32, 64]
    # horizons_to_test = [1, 4]
    
    experiments = [
        # {"name": "test", "config": "configs/overfit_sanity.yaml"},
        {"name": "B", "config": "configs/train_B.yaml"},
        {"name": "ABC", "config": "configs/train_ABC.yaml"}
    ]
    # ===================================================

    # 生成本次实验队列唯一的总时间戳根目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ablation_root = f"runs/ablation_{timestamp}"
    os.makedirs(ablation_root, exist_ok=True)

    print(f"消融实验启动...")
    print(f"本次消融总根目录: {ablation_root}")
    print(f"测试环境模式: {[exp['name'] for exp in experiments]}")
    print(f"测试控制时界: {horizons_to_test}")
    print(f"总计规划实验: {len(experiments) * len(horizons_to_test)} 场\n")

    summary_results = []

    for exp in experiments:
        for horizon in horizons_to_test:
            run_status, error_val, duration = run_experiment(exp["config"], horizon, ablation_root)
            summary_results.append({
                "mode": exp["name"],
                "horizon": horizon,
                "status": run_status,
                "eval_error_D": error_val,
                "duration_min": round(duration, 2)
            })

    # 终局打印运行状态详表
    print("\n" + "#"*50)
    print("消融实验状态报告")
    print("#"*50)
    for res in summary_results:
        err_str = f"{res['eval_error_D']:.5f}" if res['eval_error_D'] is not None else "N/A"
        print(f" 模式: {res['mode']:<5} | Horizon: {res['horizon']:<2} | 状态: {res['status']:<18} | 误差(D): {err_str}")
    print("#"*50)

    # 1. 导出结构化 JSON 报告
    json_report_path = os.path.join(ablation_root, "ablation_summary.json")
    with open(json_report_path, "w", encoding="utf-8") as jf:
        json.dump({
            "meta": {
                "total_experiments": len(summary_results),
                "timestamp": timestamp,
                "finish_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "results": summary_results
        }, jf, indent=4, ensure_ascii=False)

    # 2. 导出可视化 Markdown 报告表格
    md_report_path = os.path.join(ablation_root, "ablation_summary.md")
    with open(md_report_path, "w", encoding="utf-8") as mf:
        mf.write("# 消融实验汇总报告 (Ablation Study Summary Report)\n\n")
        mf.write(f"- **实验总根目录**: `{ablation_root}`\n")
        mf.write(f"- **总计运行批次**: {len(summary_results)} 场\n")
        mf.write(f"- **完成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        mf.write("## 核心结果矩阵\n\n")
        mf.write("| 实验模式 (Mode) | 动作时界 (Horizon) | 最终状态 (Status) | 环境 D 平均动作误差 (Avg L1 Error) | 耗时 (Minutes) |\n")
        mf.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for res in summary_results:
            err_str = f"**{res['eval_error_D']:.5f}**" if res['eval_error_D'] is not None else "`N/A`"
            mf.write(f"| {res['mode']} | {res['horizon']} | `{res['status']}` | {err_str} | {res['duration_min']} |\n")
            
    print(f" 完整消融实验简报已生成至:")
    print(f"   [JSON 格式]: {json_report_path}")
    print(f"   [MD 表格]: {md_report_path}\n")
    # ========================================================

if __name__ == "__main__":
    main()
