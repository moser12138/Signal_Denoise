import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 设置日志记录
log_dir = "../output"
os.makedirs(log_dir, exist_ok=True)
from models.tradition_models import NAFNet, DRUNet, MPRNet, DANet
from models.own_model import SignalDenoiseNet

# 加载保存的模型
def load_model(model_path, device='cuda'):

    # model = NAFNet()
    # model = DRUNet()
    # model = MPRNet()
    # model = DANet()
    model = SignalDenoiseNet()


    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# 可视化并保存降噪后的图像
def visualize_and_save(noisy_img, denoised_img, save_dir, idx):
    # 确保 noisy_img 和 denoised_img 是 numpy 数组
    if isinstance(noisy_img, torch.Tensor):
        noisy_img = noisy_img.squeeze().cpu().numpy()
    if isinstance(denoised_img, torch.Tensor):
        denoised_img = denoised_img.squeeze().cpu().numpy()

    # 归一化到 [0, 255] 范围
    noisy_img = (noisy_img - noisy_img.min()) / (noisy_img.max() - noisy_img.min() + 1e-8) * 255
    denoised_img = (denoised_img - denoised_img.min()) / (denoised_img.max() - denoised_img.min() + 1e-8) * 255

    # 转换为 uint8
    noisy_img = np.uint8(noisy_img)
    denoised_img = np.uint8(denoised_img)

    # 创建保存文件夹
    os.makedirs(save_dir, exist_ok=True)

    # 保存图片
    Image.fromarray(noisy_img).convert("L").save(os.path.join(save_dir, f"noisy_{idx}.png"))
    Image.fromarray(denoised_img).convert("L").save(os.path.join(save_dir, f"denoised_{idx}.png"))

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(noisy_img, cmap='gray')
    axes[0].set_title('Noisy')
    axes[0].axis('off')

    axes[1].imshow(denoised_img, cmap='gray')
    axes[1].set_title('Denoised')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


# 测试并保存降噪结果
def denoise_image(model, noisy_img, device='cuda', save_dir="output/denoised"):
    noisy_img = noisy_img.unsqueeze(0).to(device)  # 添加 batch 维度

    with torch.no_grad():
        denoised_img = model(noisy_img)

    denoised_img = denoised_img.squeeze().cpu().numpy()  # 去掉 batch 维度和通道维度
    visualize_and_save(noisy_img.squeeze(), denoised_img, save_dir, idx=53)

# 主程序
def main():
    # model_path = os.path.join(log_dir, 'best_2_1.lst.pth')
    # model_path = os.path.join(log_dir, 'best_2_2.lst.pth')
    # model_path = os.path.join(log_dir, 'best_2_3.lst.pth')
    # model_path = os.path.join(log_dir, 'best_2_4.lst.pth')
    model_path = os.path.join(log_dir, 'best_2_5.lst.pth')

    if not os.path.exists(model_path):
        print("No trained model found.")
        return

    # 加载模型
    model = load_model(model_path, device='cuda')

    # 示例图片路径
    # noisy_image_path = './img/11.png'  # 输入的噪声图像路径
    # noisy_image_path = './img/12.png'  # 输入的噪声图像路径
    # noisy_image_path = './img/13.png'  # 输入的噪声图像路径

    # noisy_image_path = './img/21.png'  # 输入的噪声图像路径
    # noisy_image_path = './img/22.png'  # 输入的噪声图像路径
    # noisy_image_path = './img/23.png'  # 输入的噪声图像路径

    # noisy_image_path = './img/31.png'  # 输入的噪声图像路径
    # noisy_image_path = './img/32.png'  # 输入的噪声图像路径
    # noisy_image_path = './img/33.png'  # 输入的噪声图像路径

    # noisy_image_path = './img/41.png'  # 输入的噪声图像路径
    # noisy_image_path = './img/42.png'  # 输入的噪声图像路径
    # noisy_image_path = './img/43.png'  # 输入的噪声图像路径

    # noisy_image_path = './img/51.png'  # 输入的噪声图像路径
    # noisy_image_path = './img/52.png'  # 输入的噪声图像路径
    noisy_image_path = './img/53.png'  # 输入的噪声图像路径
    noisy_image = Image.open(noisy_image_path).convert('L')

    # 转换为张量
    transform = transforms.Compose([transforms.ToTensor()])
    noisy_image_tensor = transform(noisy_image)

    # 对图片进行降噪并保存结果
    denoise_image(model, noisy_image_tensor, save_dir="output/denoised_example")

if __name__ == '__main__':
    main()
