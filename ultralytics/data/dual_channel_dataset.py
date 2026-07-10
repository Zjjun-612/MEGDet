import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER
import torch
from copy import deepcopy

class DualChannelDataset(YOLODataset):
    """支持RGB-IR双通道输入的YOLO数据集"""
    
    def __init__(self, *args, **kwargs):
        # 在调用父类初始化之前保存img_path
        self.ir_path = kwargs.get('img_path', '')  # 现在主路径是IR图像路径
        super().__init__(*args, **kwargs)
        
        # 获取RGB图像路径
        # 将 /data/DroneVehicle/merge/train/ir/images 转换为 /data/DroneVehicle/merge/train/rgb/images
        self.rgb_path = str(Path(self.ir_path).parent.parent / 'rgb' / 'images')
        LOGGER.info(f'IR images path: {self.ir_path}')
        LOGGER.info(f'RGB images path: {self.rgb_path}')
        
        # 验证RGB图像目录存在
        if not os.path.exists(self.rgb_path):
            raise FileNotFoundError(f'RGB images directory not found: {self.rgb_path}')
        
        # 预先扫描RGB图像，建立存在性缓存以避免重复检查
        self.rgb_exists_cache = {}
        LOGGER.info('Building RGB images cache...')
        for ir_file in self.im_files:
            rgb_file = self.get_rgb_path(ir_file)
            self.rgb_exists_cache[rgb_file] = os.path.exists(rgb_file)
        
        rgb_count = sum(self.rgb_exists_cache.values())
        total_count = len(self.rgb_exists_cache)
        LOGGER.info(f'RGB images cache built: {rgb_count}/{total_count} RGB images found')
    
    def get_rgb_path(self, ir_path):
        """根据IR图像路径获取对应的RGB图像路径"""
        # 将ir路径转换为rgb路径
        return str(ir_path).replace('/ir/', '/rgb/')
    
    def get_ir_path(self, rgb_path):
        """根据RGB图像路径获取对应的IR图像路径"""
        # 将rgb路径转换为ir路径
        return str(rgb_path).replace('/rgb/', '/ir/')
    
    def load_image(self, i):
        """加载IR和RGB图像并融合为6通道"""
        # 获取IR图像路径（现在是主路径）
        ir_path = self.im_files[i]
        
        # 获取对应的RGB图像路径
        rgb_path = self.get_rgb_path(ir_path)
        
        # 读取IR图像
        ir_img = cv2.imread(ir_path)
        if ir_img is None:
            # 如果IR图像不存在，尝试从RGB路径读取作为IR
            ir_img = cv2.imread(rgb_path)
            if ir_img is None:
                raise FileNotFoundError(f'Neither IR nor RGB image found: {ir_path}, {rgb_path}')
        
        # 读取RGB图像 - 使用缓存检查文件是否存在以避免警告
        if self.rgb_exists_cache.get(rgb_path, False):
            rgb_img = cv2.imread(rgb_path)
            # 如果文件存在但读取失败，使用IR图像
            if rgb_img is None:
                rgb_img = ir_img.copy()
        else:
            # 如果RGB图像不存在，直接使用IR图像作为RGB通道
            rgb_img = ir_img.copy()
        
        # 确保IR图像尺寸与RGB图像一致
        if ir_img.shape[:2] != rgb_img.shape[:2]:
            ir_img = cv2.resize(ir_img, (rgb_img.shape[1], rgb_img.shape[0]))
        
        # 融合为6通道图像：前3通道RGB，后3通道IR
        img = np.concatenate([rgb_img, ir_img], axis=2)  # (H, W, 6)
        
        h0, w0 = img.shape[:2]  # 原始高宽
        r = self.imgsz / max(h0, w0)  # 缩放比例
        if r != 1:  # 如果需要缩放
            interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        
        # 维护与父类相同的缓存/缓冲逻辑，保证 mosaic 等增强所需的 buffer 不为空
        if self.augment:
            self.ims[i], self.im_hw0[i], self.im_hw[i] = img, (h0, w0), img.shape[:2]
            self.buffer.append(i)
            if 1 < len(self.buffer) >= self.max_buffer_length:
                j = self.buffer.pop(0)
                if self.cache != "ram":
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    
    def get_image_and_label(self, index):
        """获取图像和标签信息，重写以使用6通道图像"""
        # 复制标签信息
        label = deepcopy(self.labels[index])
        label.pop("shape", None)  # shape is for rect, remove it
        
        # 用6通道图像替换原始图像
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        
        return self.update_labels_info(label)
