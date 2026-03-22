# Unsupervised Multi-View Gaze Estimation

本项目面向**无监督多视角视线估计**，基于 Cross-Encoder 思想进行特征解耦学习，并在工程实现中引入注意力与多尺度融合机制，提升少样本下游视线回归性能。

## 项目概览

- 核心任务：从未标注的多视角眼部/面部序列中学习可迁移的 gaze 表征
- 基础框架：Unsupervised Multi-View Cross-Encoder
- 当前有效配置：**2CBAM-MSFFG-50epoch**
- 评估指标：平均角度误差（deg）

## 方法原理

### 1. 无监督多视角交叉编码器

模型利用多视角同步数据的天然约束进行自监督学习：

- 视角变化时，绝对头部姿态和绝对视线可变化
- 眼部外观与相对头部坐标系下的视线模式可保持一致

通过编码器将输入分解到不同语义子空间（头部、相对视线、外观等），再通过特征交换与重建损失驱动解耦，获得可用于下游回归的表示。

### 2. 2CBAM 注意力增强

在编码主干中引入 2 层 CBAM，对关键通道与空间区域进行自适应加权，提升眼部有效特征表达。

### 3. MSFF 多尺度特征融合

融合 L2/L3/L4 多层特征，增强不同感受野信息的互补性，提升对姿态与细粒度眼部模式的建模能力。

### 4. MSFFG 门控残差

在融合分支加入可学习门控系数：

- `beta = sigmoid(beta_logit)`
- `x <- x + beta * msff_out`

该机制用于平衡注意力分支与融合分支贡献，降低直接叠加引发的特征冲突。

## 实验结果

### 主要结果（deg）

| 配置 | final_test | min_val | min_test |
|---|---:|---:|---:|
| nocbam | 9.0878 +- 0.1350 | 8.4077 +- 0.1238 | 8.9965 +- 0.1535 |
| 1CBAM-50epoch | 8.4113 +- 0.1238 | 7.6034 +- 0.3299 | 8.2405 +- 0.0977 |
| **2CBAM-MSFFG-50epoch** | **8.1671 +- 0.1780** | **7.8981 +- 0.1147** | **8.0192 +- 0.1870** |

相较 1CBAM-50epoch，2CBAM-MSFFG-50epoch 在 `final_test` 上提升约 **2.9%**。
相较原始基线 nocbam，2CBAM-MSFFG-50epoch 在 `final_test` 上提升约 **10.1%**。

## 实验配置与复现

最终实验脚本：

- [exps/eve_train_2CBAM_MSFFG_50epoch.txt](exps/eve_train_2CBAM_MSFFG_50epoch.txt)

一键运行：

```bash
python run_experiments.py -e exps/eve_train_2CBAM_MSFFG_50epoch.txt -g 0
```

该脚本包含三阶段流程：

1. `src/train.py`（50 epoch）
2. `src/extract_features.py`
3. `src/estimate_gaze.py`

并执行 5 组重复实验（0001-0005）。

核心参数：

- `--num-cbam 2`
- `--use-msff true`
- `--use-msff-gate true`
- `--msff-beta-init 0.1`
- `--num-epochs 50`

## 仓库结构

- [src/](src/)：训练、特征提取、下游估计代码
- [exps/](exps/)：实验命令脚本
- [configs/](configs/)：配置文件
- [outputs/](outputs/)：模型与特征输出
- [run_experiments.py](run_experiments.py)：实验调度器

## 参考

- Sun et al., Cross-Encoder for Unsupervised Gaze Representation Learning, ICCV 2021
- Gideon et al., Unsupervised Multi-View Gaze Representation Learning, CVPR Workshop 2022
- Woo et al., CBAM: Convolutional Block Attention Module, ECCV 2018
