import torch
import cv2
import numpy as np
from PIL import Image

def visualize_opacity_flow(flow, save_path = None):
    H, W, _ = flow.shape
    
    # 计算光流的大小和方向
    magnitude, angle = cv2.cartToPolar(flow[..., 0].cpu().numpy(), flow[..., 1].cpu().numpy())
    
    if save_path:
        np.savetxt(save_path, angle, fmt = '%.6f')

    # 归一化角度到 [0, 1] 以映射到 HSV 色调
    hsv = np.zeros((H, W, 3), dtype=np.float32)
    # hsv[..., 0] = (angle * 180 / np.pi) % 180  # 角度转换到 OpenCV 的 HSV 范围（0-180）
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 1  # 饱和度设为最大值
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)  # 归一化光流大小

    # 转换为 BGR 格式用于保存
    bgr = cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    return bgr  # 返回图像数组