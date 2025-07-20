import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def main():
    # 示例图片路径
    # noisy_image_path = 'ex3.png'  # 输入的噪声图像路径
    noisy_image_path = 'lable3.png'  # 输入的噪声图像路径
    save_path = 'lable3_saved.png'  # 重新保存的路径

    # 打开图片并转换为灰度图
    noisy_image = Image.open(noisy_image_path).convert('L')

    # 重新保存图片
    noisy_image.save(save_path)
    print(f"Image saved to {save_path}")


if __name__ == '__main__':
    main()