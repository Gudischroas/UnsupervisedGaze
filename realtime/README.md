# 实时摄像头视线估计

基于训练好的 CrossEncoder + GazeEstimationNet 模型，通过笔记本摄像头实时检测面部并估计视线方向。

## 依赖

在已有项目环境基础上，额外安装：

```bash
conda activate gaze_rocm
pip install mediapipe
```

## 快速开始

```bash
cd realtime
python realtime_gaze.py
```

按 **q** 键退出。

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--camera` | `0` | 摄像头索引 |
| `--encoder-checkpoint` | `../outputs/.../best` | CrossEncoder Encoder 权重目录 |
| `--gaze-checkpoint` | 自动扫描 | GazeEstimationNet `net.pt` 文件路径 |
| `--arrow-length` | `100` | 视线箭头长度（像素） |
| `--no-mirror` | - | 禁用镜像翻转 |
| `--use-cbam` | `1` | 是否使用 CBAM 注意力 (0/1) |
| `--show-patches` | - | 在画面左上角显示眼部裁剪 patch |

## 箭头颜色

- **绿色**：被试者左眼视线方向
- **蓝色**：被试者右眼视线方向
- **红色（粗）**：双眼融合视线方向（鼻尖起点）

## 推理管线

```
摄像头帧 → MediaPipe Face Mesh 人脸检测
         → 裁剪左/右眼 patch (128×128)
         → 灰度化 → 直方图均衡化 → [0,1] 归一化 → 3通道复制
         → CrossEncoder Encoder → gaze特征(12D) + head特征(12D)
         → GazeEstimationNet MLP → pitch/yaw → 3D 视线向量
         → 画面箭头可视化
```

## 注意事项

- 本目录独立于 `src/`，不修改原项目任何文件
- 默认使用 `pt-sga-pair-lr-v222-d111-eve-0001` 的 `best` checkpoint
- `--use-cbam` 参数需与训练时的设置一致（默认为 1）
- 如果摄像头打不开，尝试修改 `--camera` 参数
