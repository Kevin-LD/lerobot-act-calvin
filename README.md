# ACT策略跨环境泛化研究（CALVIN Benchmark）

## 项目简介

本项目基于 CALVIN 数据集研究 Action Chunking with Transformers（ACT）策略的跨环境泛化能力（Cross-Environment Generalization）。

我们重点关注机器人策略在视觉分布发生变化时的零样本迁移（Zero-Shot Transfer）能力，即策略仅在部分环境中训练，然后直接部署到从未见过的新环境中进行测试。实验采用离线动作预测误差（Action L1 Error）和在线任务成功率（Success Rate）两种指标进行评估，并进一步分析多环境训练和 Action Chunking 机制对泛化性能的影响。

## 环境配置

使用以下命令安装项目依赖：

```bash
pip install -r requirements.txt
```

## 数据准备

本项目使用两个数据集。

### 完整 CALVIN 数据集

下载地址：

https://huggingface.co/datasets/xiaoma26/calvin-lerobot/tree/main

### Open Drawer 子数据集

下载地址：

https://huggingface.co/datasets/Kevin-LD/CALVIN_open_drawer/tree/main

下载完成后，请根据配置文件中的数据路径要求放置数据集 (默认在根目录下的 data/)。如有需要，可自行修改配置文件中的数据集路径。

## 模型训练

使用以下命令训练 ACT 模型：

```bash
python train.py
```

更多训练参数可通过以下命令查看：

```bash
python train.py -h
```

## 模型评估

### Action L1 Error 评估

用于评估模型在 Environment D 上的动作预测误差：

```bash
python action_loss_eval.py --checkpoint path/to/checkpoint.pt
```

### Success Rate 评估

用于评估模型在 CALVIN 仿真环境中的任务成功率：

```bash
python success_rate_eval.py --checkpoint path/to/checkpoint.pt
```

更多评估参数可通过以下命令查看：

```bash
python action_loss_eval.py -h
python success_rate_eval.py -h
```

## 预训练权重与 Rollout 视频

模型权重及 Rollout 可视化视频下载地址：

https://drive.google.com/drive/folders/1UVtHY2a5legxZIZzB76gldc701c19pYH?usp=drive_link


## 相关仓库

### Task 1：3D重建与场景生成

https://github.com/F1shermanCNN/CV-Final-Task1
