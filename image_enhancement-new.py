# -*- coding: utf-8 -*-
"""
实验二：图像增强
使用 OpenCV / scikit-image / NumPy / Matplotlib 完成：
  1. 图像读取与色彩空间转换（BGR -> RGB / GRAY）
  2. 为图像添加椒盐噪声与高斯噪声
  3. 用均值滤波、中值滤波、高斯滤波对两种噪声图去噪并对比

运行方式：
    python image_enhancement.py
输入：input/p1.png
输出：output/ 目录下的若干对比图
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # 不弹出 GUI 窗口，直接保存图片
from matplotlib import pyplot as plt
from skimage.util import random_noise

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

INPUT_PATH = os.path.join("input", "p1.png")
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)


# 2.2 读取原始图像并进行色彩空间转换
img = cv2.imread(INPUT_PATH)
assert img is not None, f"图像读取失败，请检查路径: {INPUT_PATH}"
h, w = img.shape[:2]
print(f"图像尺寸 (H x W x C) = {h} x {w} x {img.shape[2]}")

# OpenCV 读取顺序为 B, G, R。取 (100, 100) 处像素
y, x = 100, 100
b, g, r = img[y, x]
print(f"像素点 ({y}, {x}) 的 BGR = ({b}, {g}, {r})，对应 RGB = ({r}, {g}, {b})")

# BGR -> RGB
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# BGR -> GRAY
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(6, 6))
plt.imshow(rgb_img)
plt.title("Original Image (RGB)")
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_original_rgb.png"), dpi=200)
plt.close()

plt.figure(figsize=(6, 6))
plt.imshow(gray_img, cmap="gray")
plt.title("Gray Image")
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_gray.png"), dpi=200)
plt.close()


# 2.3 添加噪声（椒盐噪声 / 高斯噪声）
# random_noise 返回 [0, 1] 的 float 图像
sp_noise = random_noise(rgb_img, mode="s&p", amount=0.4)            # 椒盐噪声：40% 像素被替换为黑白点
gus_noise = random_noise(rgb_img, mode="gaussian", mean=0.2, var=0.03)  # 高斯噪声：每像素叠加高斯分布偏差

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(rgb_img); axes[0].set_title("Original"); axes[0].axis("off")
axes[1].imshow(sp_noise); axes[1].set_title("Salt & Pepper Noise (amount=0.4)"); axes[1].axis("off")
axes[2].imshow(gus_noise); axes[2].set_title("Gaussian Noise (mean=0.2, var=0.03)"); axes[2].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_noise_comparison.png"), dpi=200)
plt.close()

# 2.4 图像滤波：均值 / 中值 / 高斯，分别处理两种噪声
# OpenCV 滤波函数要求 uint8 类型，先把 [0,1] float 转回 [0,255] uint8
sp_u8 = (np.clip(sp_noise, 0, 1) * 255).astype(np.uint8)
gus_u8 = (np.clip(gus_noise, 0, 1) * 255).astype(np.uint8)

K = 5  # 滤波核大小（奇数）

# 均值滤波
mean_sp = cv2.blur(sp_u8, (K, K))
mean_gus = cv2.blur(gus_u8, (K, K))

# 中值滤波
median_sp = cv2.medianBlur(sp_u8, K)
median_gus = cv2.medianBlur(gus_u8, K)

# 高斯滤波
gauss_sp = cv2.GaussianBlur(sp_u8, (K, K), 0)
gauss_gus = cv2.GaussianBlur(gus_u8, (K, K), 0)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].imshow(mean_sp);    axes[0, 0].set_title("S&P Noise + Mean Filter");    axes[0, 0].axis("off")
axes[0, 1].imshow(median_sp);  axes[0, 1].set_title("S&P Noise + Median Filter");  axes[0, 1].axis("off")
axes[0, 2].imshow(gauss_sp);   axes[0, 2].set_title("S&P Noise + Gaussian Filter"); axes[0, 2].axis("off")
axes[1, 0].imshow(mean_gus);   axes[1, 0].set_title("Gaussian Noise + Mean Filter");   axes[1, 0].axis("off")
axes[1, 1].imshow(median_gus); axes[1, 1].set_title("Gaussian Noise + Median Filter"); axes[1, 1].axis("off")
axes[1, 2].imshow(gauss_gus);  axes[1, 2].set_title("Gaussian Noise + Gaussian Filter");axes[1, 2].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "04_filter_comparison.png"), dpi=200)
plt.close()

print("所有结果图已保存到 output/ 目录")
