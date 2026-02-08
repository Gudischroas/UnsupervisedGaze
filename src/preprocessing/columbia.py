"""Columbia Gaze Dataset Preprocessing Script

处理 Columbia_Gaze_Data_Set 数据集
数据集结构：
  - 56个受试者（0001-0056）
  - 每个受试者有多张静态图像
  - 文件名格式: {subject}_{distance}_{pitch}_{vertical}_{horizontal}.jpg
    例如: 0001_2m_0P_0V_0H.jpg
  - 头部姿态角度：
    - Pitch(俯仰角): -30, -15, 0, 15, 30 度
    - Vertical(垂直角): -10, 0, 10 度  
    - Horizontal(水平角): -15, -10, -5, 0, 5, 10, 15 度
"""

import logging
import os
from typing import List, Dict, Tuple
import re
import sys
import pathlib
from frozendict import frozendict
import bz2
import pickle
import _pickle as cPickle
from multiprocessing import Pool

import cv2 as cv
import h5py
import numpy as np

file_dir_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(file_dir_path, ".."))
import core.training as training
from utils.data_types import MultiDict
from utils.angles import pitch_yaw_to_vector

logger = logging.getLogger(__name__)
config, device = training.script_init_common()


class ColumbiaConfig():
    """Columbia 数据集配置"""
    face_size = [256, 256]  # width, height
    eyes_size = [128, 128]  # width, height - not applicable for Columbia (full face images)
    camera_frame_type = 'face'  # full | face (Columbia is full face)
    
    # 头部姿态角度范围
    pitches = [-30, -15, 0, 15, 30]  # Pitch (俯仰角)
    verticals = [-10, 0, 10]  # Vertical (垂直角)
    horizontals = [-15, -10, -5, 0, 5, 10, 15]  # Horizontal (水平角)
    distances = ['2m']  # Only 2m distance in Columbia dataset

columbia_config = ColumbiaConfig()


def parse_filename(filename: str) -> Dict:
    """
    从文件名解析出Columbia数据集的信息
    
    示例: 0001_2m_0P_0V_0H.jpg
    返回: {
        'subject': '0001',
        'distance': '2m',
        'pitch': 0,
        'vertical': 0,
        'horizontal': 0
    }
    """
    pattern = r'(\d{4})_(\d+m)_(-?\d+)P_(-?\d+)V_(-?\d+)H\.jpg'
    match = re.match(pattern, filename)
    
    if not match:
        return None
    
    return {
        'subject': match.group(1),
        'distance': match.group(2),
        'pitch': int(match.group(3)),
        'vertical': int(match.group(4)),
        'horizontal': int(match.group(5))
    }


class ColumbiaDataset():
    """Columbia Gaze Dataset 处理类"""
    
    def __init__(self, dataset_path: str,
                 participants_to_use: List[str] = None):
        """
        初始化 Columbia 数据集
        
        Args:
            dataset_path: 数据集路径
            participants_to_use: 要使用的受试者列表，如 ['0001', '0002', ...]
        """
        self.path = dataset_path
        
        # 如果没有指定受试者，使用所有受试者
        if participants_to_use is None:
            participants_to_use = ['%04d' % i for i in range(1, 57)]  # 0001-0056
        
        self.participants_to_use = participants_to_use
        
        # 验证数据集路径
        assert os.path.isdir(self.path), f"Dataset path does not exist: {self.path}"
        
        logger.info('Initialized Columbia dataset class for: %s' % self.path)
        logger.info('Using %d participants' % len(self.participants_to_use))
    
    def extract_face_patch(self, image: np.ndarray, target_size: List[int] = None) -> np.ndarray:
        """
        提取面部区域（这里直接用整个图像或简单裁剪）
        对于Columbia数据集，图像已经是面部居中的，可以直接使用或简单调整大小
        
        Args:
            image: 输入图像 (H, W, C)
            target_size: 目标大小 [width, height]
        
        Returns:
            面部补丁 (C, H, W)
        """
        if target_size is not None:
            image = cv.resize(image, tuple(target_size))
        
        # 转换为 (C, H, W) 和归一化
        image = np.transpose(image, [2, 0, 1])
        image = image.astype(np.float32)
        image = image / 255.0  # 归一化到 [0, 1]
        
        return image
    
    def pitch_yaw_to_gaze_vector(self, pitch: float, yaw: float) -> np.ndarray:
        """
        将俯仰角(pitch)和偏航角(yaw)转换为目光方向向量
        
        Args:
            pitch: 俯仰角（度）
            yaw: 偏航角（度）
        
        Returns:
            目光方向向量 (3,)
        """
        pitch_rad = np.radians(pitch)
        yaw_rad = np.radians(yaw)
        
        # 标准的球面坐标到笛卡尔坐标转换
        x = np.sin(yaw_rad) * np.cos(pitch_rad)
        y = -np.sin(pitch_rad)
        z = np.cos(yaw_rad) * np.cos(pitch_rad)
        
        return np.array([x, y, z], dtype=np.float32)
    
    def process_single_image(self, image_path: str, metadata: Dict) -> Dict:
        """
        处理单个图像
        
        Args:
            image_path: 图像文件路径
            metadata: 从文件名解析的元数据
        
        Returns:
            处理后的数据条目
        """
        entry = {}
        
        # 读取图像
        image = cv.imread(image_path)
        if image is None:
            logger.warning(f"Failed to read image: {image_path}")
            return None
        
        # 转换 BGR -> RGB
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        # 提取面部补丁
        entry['frame'] = self.extract_face_patch(image, columbia_config.face_size)
        
        # 从头部姿态计算目光方向
        # Columbia 数据集中的角度就代表目光的俯仰和偏航角度
        pitch = metadata['pitch']
        horizontal = metadata['horizontal']  # 水平角作为偏航角
        
        gaze_vector = self.pitch_yaw_to_gaze_vector(pitch, horizontal)
        entry['gaze_dir'] = gaze_vector
        
        # 存储元数据
        entry['subject'] = metadata['subject']
        entry['pitch'] = pitch
        entry['vertical'] = metadata['vertical']
        entry['horizontal'] = horizontal
        entry['distance'] = metadata['distance']
        
        return entry
    
    def preprocess(self, output_dir: str):
        """
        预处理整个数据集
        
        Args:
            output_dir: 输出目录
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        patches = MultiDict(['sub', 'head', 'gaze', 'app'])
        
        image_count = 0
        skipped_count = 0
        
        # 遍历所有受试者
        for participant_id in sorted(self.participants_to_use):
            participant_dir = os.path.join(self.path, participant_id)
            
            if not os.path.isdir(participant_dir):
                logger.warning(f"Participant directory not found: {participant_dir}")
                continue
            
            # 获取该受试者的所有图像
            image_files = sorted([
                f for f in os.listdir(participant_dir)
                if f.endswith('.jpg') and f.startswith(participant_id)
            ])
            
            logger.info(f"Processing participant {participant_id}: {len(image_files)} images")
            
            for image_file in image_files:
                # 解析文件名
                metadata = parse_filename(image_file)
                if metadata is None:
                    logger.warning(f"Failed to parse filename: {image_file}")
                    skipped_count += 1
                    continue
                
                # 处理图像
                image_path = os.path.join(participant_dir, image_file)
                entry = self.process_single_image(image_path, metadata)
                
                if entry is None:
                    skipped_count += 1
                    continue
                
                # 创建输出路径
                # 结构: {participant}/{image_name_without_ext}.pbz2
                output_subdir = os.path.join(output_dir, participant_id)
                os.makedirs(output_subdir, exist_ok=True)
                
                image_name_without_ext = image_file.replace('.jpg', '')
                output_path = os.path.join(output_subdir, image_name_without_ext + '.pbz2')
                
                # 保存为压缩pickle
                with bz2.BZ2File(output_path, 'w') as f:
                    cPickle.dump(entry, f)
                
                # 更新索引
                sub = frozendict({'participant': participant_id})
                tags = {
                    'sub': sub,
                    'head': 'camera',
                    'app': 'face',
                    'gaze': image_name_without_ext
                }
                
                output_path_rel = os.path.join(participant_id, image_name_without_ext + '.pbz2')
                patches[tags] = output_path_rel
                
                image_count += 1
                
                if image_count % 100 == 0:
                    logger.info(f"Processed {image_count} images")
        
        # 保存索引
        index_path = os.path.join(output_dir, 'index.pbz2')
        with bz2.BZ2File(index_path, 'w') as f:
            cPickle.dump(patches, f)
        
        logger.info(f"Preprocessing complete!")
        logger.info(f"  Total images processed: {image_count}")
        logger.info(f"  Skipped: {skipped_count}")
        logger.info(f"  Output index: {index_path}")


def split_dataset(total_subjects: int = 56, 
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15):
    """
    将受试者分为训练/验证/测试集
    
    Args:
        total_subjects: 总受试者数
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    
    Returns:
        {'train': [...], 'val': [...], 'test': [...]}
    """
    import random
    
    all_subjects = ['%04d' % i for i in range(1, total_subjects + 1)]
    random.shuffle(all_subjects)
    
    n_train = int(len(all_subjects) * train_ratio)
    n_val = int(len(all_subjects) * val_ratio)
    
    train = all_subjects[:n_train]
    val = all_subjects[n_train:n_train + n_val]
    test = all_subjects[n_train + n_val:]
    
    return {
        'train': train,
        'val': val,
        'test': test
    }


if __name__ == '__main__':
    """
    使用方式:
    conda activate gaze_rocm
    cd /home/gudi/project/UnsupervisedGaze/cvpr
    python src/preprocessing/columbia.py
    """
    
    columbia_input_path = os.path.expanduser('~/project/data/Columbia_Gaze_Data_Set')
    columbia_output_path = os.path.expanduser('~/project/data/columbia_preprocessed')
    
    # 检查输入路径
    if not os.path.isdir(columbia_input_path):
        print(f"Error: Dataset path not found: {columbia_input_path}")
        sys.exit(1)
    
    logger.info(f"Columbia dataset input path: {columbia_input_path}")
    logger.info(f"Columbia dataset output path: {columbia_output_path}")
    
    # 分割数据集
    splits = split_dataset()
    
    logger.info(f"Train: {len(splits['train'])} subjects")
    logger.info(f"Val: {len(splits['val'])} subjects")
    logger.info(f"Test: {len(splits['test'])} subjects")
    
    # 处理每个分割
    for split_name, participant_list in splits.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {split_name} split ({len(participant_list)} subjects)")
        logger.info(f"{'='*60}")
        
        output_dir = os.path.join(columbia_output_path, split_name)
        
        dataset = ColumbiaDataset(
            dataset_path=columbia_input_path,
            participants_to_use=participant_list
        )
        
        dataset.preprocess(output_dir)
    
    logger.info("\n" + "="*60)
    logger.info("All preprocessing complete!")
    logger.info("="*60)
