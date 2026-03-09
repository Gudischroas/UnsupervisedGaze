#!/usr/bin/env python3
"""
实时摄像头视线估计脚本

使用训练好的 CrossEncoder (Encoder) 和 GazeEstimationNet (MLP)，
通过 MediaPipe Face Mesh 检测面部并裁剪眼部区域，
实时估计视线方向并在画面上绘制箭头。

用法:
    cd realtime
    python realtime_gaze.py
    python realtime_gaze.py --camera 0 --arrow-length 120 --no-mirror
"""

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

# ============================================================
# 路径桥接：将 src/ 加入 sys.path，以复用项目已有模块
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
UTILS_DIR = os.path.join(SRC_DIR, 'utils')

# 项目内部模块（如 config_default.py）使用 sys.path.append("../utils")，
# 所以需要把 src/ 和 src/utils/ 都加入 path
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, UTILS_DIR)

# DefaultConfig 单例在首次实例化时会读取 sys.argv[0] 并验证其为合法文件路径，
# 同时其内部 __get_python_file_contents() 使用 os.path.relpath 计算源文件路径。
# 需要确保 sys.argv[0] 指向真实文件，并且 cwd 在 src/ 下以匹配原项目的运行约定。
_original_cwd = os.getcwd()
os.chdir(SRC_DIR)
sys.argv[0] = os.path.abspath(os.path.join(SCRIPT_DIR, 'realtime_gaze.py'))

import torch
import torch.nn.functional as F
import mediapipe as mp

# ------ 项目模块导入 ------
from core.config_default import DefaultConfig
from models.cross_encoder_nets import BaselineEncoder, BaselineGenerator
from models.gaze_estimator import GazeEstimationNet
from utils.angles import pitch_yaw_to_vector, vector_to_pitch_yaw

# 恢复工作目录
os.chdir(_original_cwd)

# ============================================================
# MediaPipe 眼部关键点索引
# ============================================================
# 左眼轮廓（被试者左眼，画面右侧 —— 镜像前）
LEFT_EYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                173, 157, 158, 159, 160, 161, 246]
# 右眼轮廓（被试者右眼，画面左侧 —— 镜像前）
RIGHT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263,
                 466, 388, 387, 386, 385, 384, 398]
# 鼻尖（用于融合箭头起点）
NOSE_TIP_IDX = 1


def parse_args():
    parser = argparse.ArgumentParser(description='实时摄像头视线估计')
    parser.add_argument('--camera', type=int, default=0,
                        help='摄像头索引 (默认: 0)')
    parser.add_argument('--encoder-checkpoint', type=str,
                        default=os.path.join(PROJECT_ROOT,
                                             'outputs', 'checkpoints',
                                             'pt-sga-pair-lr-v222-d111-eve-0001',
                                             'checkpoints', 'best'),
                        help='CrossEncoder Encoder checkpoint 目录路径')
    parser.add_argument('--gaze-checkpoint', type=str, default='',
                        help='GazeEstimationNet net.pt 文件路径 (留空则自动扫描)')
    parser.add_argument('--arrow-length', type=int, default=100,
                        help='视线箭头长度 (像素, 默认: 100)')
    parser.add_argument('--no-mirror', action='store_true',
                        help='禁用水平镜像翻转')
    parser.add_argument('--show-patches', action='store_true',
                        help='DEBUG: 在画面角落显示裁剪的眼部 patch')
    return parser.parse_args()


# ============================================================
# 配置加载
# ============================================================
def load_training_config(checkpoint_dir):
    """从训练好的模型目录加载配置文件"""
    config_dir = os.path.dirname(os.path.dirname(checkpoint_dir))
    config_json = os.path.join(config_dir, 'configs', 'combined.json')
    
    if not os.path.isfile(config_json):
        print(f'[警告] 未找到训练配置文件: {config_json}')
        print('[警告] 将使用默认配置，可能导致维度不匹配')
        return {}
    
    with open(config_json, 'r') as f:
        training_config = json.load(f)
    
    print(f'[✓] 已加载训练配置: {config_json}')
    print(f'    feature_sizes: {training_config.get("feature_sizes")}')
    print(f'    use_cbam: {training_config.get("use_cbam")}')
    return training_config


# ============================================================
# 模型加载
# ============================================================
def load_encoder(checkpoint_dir, training_config):
    """加载训练好的 Encoder 权重"""
    config = DefaultConfig()
    
    # 应用训练时的配置
    if 'feature_sizes' in training_config:
        from utils.data_types import TypedOrderedDict
        fs = TypedOrderedDict(str, int)
        for k, v in training_config['feature_sizes'].items():
            fs[k] = v
        config.override('feature_sizes', fs)
    
    if 'use_cbam' in training_config:
        config.override('use_cbam', training_config['use_cbam'])

    encoder = BaselineEncoder()

    encoder_path = os.path.join(checkpoint_dir, 'encoder.pt')
    if not os.path.isfile(encoder_path):
        raise FileNotFoundError(f'Encoder 权重文件不存在: {encoder_path}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    state_dict_raw = torch.load(encoder_path, map_location=device)

    # checkpoint 保存时 key 带有 "encoder." 前缀，需要去掉
    state_dict = {}
    for k, v in state_dict_raw.items():
        new_key = k[k.index('.') + 1:] if '.' in k else k
        state_dict[new_key] = v

    encoder.load_state_dict(state_dict)
    encoder.to(device)
    encoder.eval()
    print(f'[✓] Encoder 已加载: {encoder_path}')
    return encoder, device


def find_gaze_checkpoint(project_root, encoder_id='0001'):
    """自动扫描并返回最新的 GazeEstimation net.pt 路径"""
    base_dir = os.path.join(project_root, 'outputs', 'checkpoints',
                            'eg-sga-pair-lr-v222-d111-eve-default')
    # 搜索对应 encoder_id 的第一个 fold (fold 0)
    pattern = os.path.join(base_dir,
                           f'eg-sga-pair-lr-v222-d111-eve-{encoder_id}-0',
                           'checkpoints', '*.pt', 'net.pt')
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        # 尝试所有 fold
        pattern = os.path.join(base_dir,
                               f'eg-sga-pair-lr-v222-d111-eve-{encoder_id}-*',
                               'checkpoints', '*.pt', 'net.pt')
        candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f'未找到 GazeEstimation checkpoint。\n'
            f'搜索路径: {pattern}\n'
            f'请通过 --gaze-checkpoint 手动指定 net.pt 路径。')
    # 返回最新（step 最大）的 checkpoint
    return candidates[-1]


def load_gaze_net(checkpoint_path, training_config):
    """加载训练好的 GazeEstimationNet 权重"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 计算输入特征维度：根据 eval_features 确定使用哪些特征
    eval_features = training_config.get('eval_features', ['gaze', 'head'])
    feature_sizes = training_config.get('feature_sizes', {'app': 32, 'gaze': 12, 'head': 12})
    
    # eval_features 中的 'head' 可能映射到 feature_sizes 中的 'sub' (旧命名)
    num_features = 0
    for feat_name in eval_features:
        if feat_name in feature_sizes:
            num_features += feature_sizes[feat_name]
        elif feat_name == 'head' and 'sub' in feature_sizes:
            # 兼容旧命名：head → sub
            num_features += feature_sizes['sub']
        else:
            print(f'[警告] eval_feature "{feat_name}" 未在 feature_sizes 中找到')
    
    print(f'[✓] GazeEstimationNet 输入维度: {num_features} (来自 {eval_features})')
    gaze_net = GazeEstimationNet(num_features)
    state_dict_raw = torch.load(checkpoint_path, map_location=device)

    # checkpoint 保存时 key 带有 "net." 前缀，需要去掉
    state_dict = {}
    for k, v in state_dict_raw.items():
        if k.startswith('net.'):
            new_key = k[4:]   # 去掉 "net."
        elif '.' in k:
            new_key = k[k.index('.') + 1:]
        else:
            new_key = k
        state_dict[new_key] = v

    gaze_net.load_state_dict(state_dict)
    gaze_net.to(device)
    gaze_net.eval()
    print(f'[✓] GazeEstimationNet 已加载: {checkpoint_path}')
    return gaze_net


# ============================================================
# 图像预处理（复现训练管线）
# ============================================================
def preprocess_eye_patch(patch_bgr):
    """
    将裁剪的 BGR 眼部 patch 处理为模型输入 tensor。
    流程与训练完全一致：
      resize 128x128 → 灰度 → 直方图均衡化 → [0,1] 归一化 → 3通道复制
    """
    # Resize to 128x128
    patch = cv2.resize(patch_bgr, (128, 128))
    # 灰度化
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    # 直方图均衡化（对应训练中的 PIL ImageOps.equalize）
    gray = cv2.equalizeHist(gray)
    # 归一化到 [0, 1]
    gray_float = gray.astype(np.float32) / 255.0
    # 转为 tensor: (1, 1, 128, 128)
    tensor = torch.from_numpy(gray_float).unsqueeze(0).unsqueeze(0)
    # 扩展到 3 通道 (ResNet-18 需要 3 通道输入)
    tensor = tensor.repeat(1, 3, 1, 1)
    return tensor


# ============================================================
# MediaPipe 眼部裁剪
# ============================================================
def get_eye_bbox(landmarks, eye_indices, frame_h, frame_w, expand_ratio=1.5):
    """
    根据 MediaPipe 关键点计算眼部 bounding box。
    返回 (x1, y1, x2, y2) 的正方形区域。
    """
    xs = [landmarks[i].x * frame_w for i in eye_indices]
    ys = [landmarks[i].y * frame_h for i in eye_indices]

    cx = int(np.mean(xs))
    cy = int(np.mean(ys))

    # 以眼部宽高的较大值为基准，扩展为正方形
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    size = int(max(w, h) * expand_ratio)
    half = size // 2

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(frame_w, cx + half)
    y2 = min(frame_h, cy + half)

    return x1, y1, x2, y2, cx, cy


def crop_eye_patch(frame, landmarks, eye_indices, frame_h, frame_w):
    """裁剪眼部区域，返回 (patch_bgr, center_x, center_y) 或 None"""
    x1, y1, x2, y2, cx, cy = get_eye_bbox(landmarks, eye_indices, frame_h, frame_w)

    # 确保区域有效
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None

    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return None

    return patch, cx, cy


# ============================================================
# 推理
# ============================================================
@torch.no_grad()
def estimate_gaze_direction(encoder, gaze_net, eye_tensor, device, eval_features, feature_sizes):
    """
    从预处理后的眼部 tensor 推理视线方向。
    返回 3D 单位向量 (numpy array, shape=(3,))。
    """
    eye_tensor = eye_tensor.to(device)

    # Encoder 提取特征
    features, confidences = encoder(eye_tensor)
    
    # 拼接 eval_features 指定的特征
    feat_list = []
    for feat_name in eval_features:
        if feat_name in features:
            feat_list.append(features[feat_name])
        elif feat_name == 'head' and 'sub' in features:
            # 兼容旧命名：head → sub
            feat_list.append(features['sub'])
        else:
            raise KeyError(f'特征 "{feat_name}" 未在 Encoder 输出中找到')
    
    combined = torch.cat(feat_list, dim=-1)

    # GazeEstimationNet 期望输入 shape (B, V, N)
    combined = combined.unsqueeze(1)   # (1, 1, 24)
    pred = gaze_net(combined)          # (1, 1, 3) — 3D 单位向量

    return pred.squeeze().cpu().numpy()


# ============================================================
# 可视化
# ============================================================
def draw_gaze_arrow(frame, center, gaze_3d, color, length=100, thickness=2):
    """
    在画面上绘制视线方向箭头。
    gaze_3d: (x, y, z) 3D 单位向量
    映射到 2D: dx = x, dy = -y (图像 y 轴向下)
    """
    dx = gaze_3d[0]
    dy = -gaze_3d[1]  # 翻转 y 轴

    # 归一化 2D 方向
    norm = np.sqrt(dx ** 2 + dy ** 2)
    if norm < 1e-6:
        return

    dx /= norm
    dy /= norm

    start = (int(center[0]), int(center[1]))
    end = (int(center[0] + dx * length), int(center[1] + dy * length))
    cv2.arrowedLine(frame, start, end, color, thickness, tipLength=0.3)


def draw_eye_bbox(frame, landmarks, eye_indices, frame_h, frame_w, color):
    """绘制眼部检测区域的矩形框"""
    x1, y1, x2, y2, _, _ = get_eye_bbox(landmarks, eye_indices, frame_h, frame_w)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)


# ============================================================
# 主循环
# ============================================================
def main():
    args = parse_args()

    # ----------------------------------------------------------
    # 1. 加载训练配置
    # ----------------------------------------------------------
    print('正在加载训练配置...')
    training_config = load_training_config(args.encoder_checkpoint)
    
    # 提取关键配置项
    eval_features = training_config.get('eval_features', ['gaze', 'head'])
    feature_sizes = training_config.get('feature_sizes', {'app': 32, 'gaze': 12, 'head': 12})

    # ----------------------------------------------------------
    # 2. 加载模型
    # ----------------------------------------------------------
    print('正在加载模型...')
    encoder, device = load_encoder(args.encoder_checkpoint, training_config)

    if args.gaze_checkpoint:
        gaze_ckpt = args.gaze_checkpoint
    else:
        gaze_ckpt = find_gaze_checkpoint(PROJECT_ROOT)
    gaze_net = load_gaze_net(gaze_ckpt, training_config)

    # ----------------------------------------------------------
    # 3. 初始化 MediaPipe Face Mesh
    # ----------------------------------------------------------
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # ----------------------------------------------------------
    # 4. 打开摄像头
    # ----------------------------------------------------------
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f'[✗] 无法打开摄像头 {args.camera}')
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f'[✓] 摄像头 {args.camera} 已打开  |  按 q 退出')

    mirror = not args.no_mirror
    fps_deque = []
    ARROW_LEN = args.arrow_length

    # 颜色定义 (BGR)
    COLOR_LEFT_EYE = (0, 255, 0)      # 绿色 — 被试者左眼
    COLOR_RIGHT_EYE = (255, 0, 0)     # 蓝色 — 被试者右眼
    COLOR_FUSED = (0, 0, 255)         # 红色 — 融合

    try:
        while True:
            t_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # 镜像翻转（更自然的交互体验）
            if mirror:
                frame = cv2.flip(frame, 1)

            frame_h, frame_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --------------------------------------------------
            # MediaPipe 人脸检测
            # --------------------------------------------------
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                # 绘制眼部检测区域
                draw_eye_bbox(frame, landmarks, LEFT_EYE_IDX, frame_h, frame_w, COLOR_LEFT_EYE)
                draw_eye_bbox(frame, landmarks, RIGHT_EYE_IDX, frame_h, frame_w, COLOR_RIGHT_EYE)

                gaze_vectors = []

                # ------ 左眼 ------
                left_result = crop_eye_patch(frame, landmarks, LEFT_EYE_IDX, frame_h, frame_w)
                if left_result is not None:
                    left_patch, left_cx, left_cy = left_result
                    left_tensor = preprocess_eye_patch(left_patch)
                    left_gaze = estimate_gaze_direction(encoder, gaze_net, left_tensor, device,
                                                        eval_features, feature_sizes)
                    draw_gaze_arrow(frame, (left_cx, left_cy), left_gaze,
                                    COLOR_LEFT_EYE, ARROW_LEN, 2)
                    gaze_vectors.append(left_gaze)

                    # DEBUG: 显示预处理后的 patch
                    if args.show_patches:
                        patch_show = cv2.resize(left_patch, (64, 64))
                        frame[10:74, 10:74] = patch_show

                # ------ 右眼 ------
                right_result = crop_eye_patch(frame, landmarks, RIGHT_EYE_IDX, frame_h, frame_w)
                if right_result is not None:
                    right_patch, right_cx, right_cy = right_result
                    right_tensor = preprocess_eye_patch(right_patch)
                    right_gaze = estimate_gaze_direction(encoder, gaze_net, right_tensor, device,
                                                         eval_features, feature_sizes)
                    draw_gaze_arrow(frame, (right_cx, right_cy), right_gaze,
                                    COLOR_RIGHT_EYE, ARROW_LEN, 2)
                    gaze_vectors.append(right_gaze)

                    # DEBUG: 显示预处理后的 patch
                    if args.show_patches:
                        patch_show = cv2.resize(right_patch, (64, 64))
                        frame[10:74, 84:148] = patch_show

                # ------ 融合箭头（红色，鼻尖起点）------
                if len(gaze_vectors) > 0:
                    fused_gaze = np.mean(gaze_vectors, axis=0)
                    fused_gaze /= (np.linalg.norm(fused_gaze) + 1e-8)
                    nose_x = int(landmarks[NOSE_TIP_IDX].x * frame_w)
                    nose_y = int(landmarks[NOSE_TIP_IDX].y * frame_h)
                    draw_gaze_arrow(frame, (nose_x, nose_y), fused_gaze,
                                    COLOR_FUSED, int(ARROW_LEN * 1.3), 3)

            else:
                cv2.putText(frame, 'No face detected', (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # --------------------------------------------------
            # FPS 显示
            # --------------------------------------------------
            t_end = time.time()
            fps = 1.0 / max(t_end - t_start, 1e-6)
            fps_deque.append(fps)
            if len(fps_deque) > 30:
                fps_deque.pop(0)
            avg_fps = np.mean(fps_deque)
            cv2.putText(frame, f'FPS: {avg_fps:.1f}', (frame_w - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 图例
            cv2.putText(frame, 'L', (20, frame_h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LEFT_EYE, 2)
            cv2.putText(frame, 'R', (20, frame_h - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RIGHT_EYE, 2)
            cv2.putText(frame, 'Fused', (20, frame_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_FUSED, 2)

            cv2.imshow('Realtime Gaze Estimation', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()
        print('[✓] 已退出')


if __name__ == '__main__':
    main()
