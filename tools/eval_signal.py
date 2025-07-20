import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import logging
from PIL import Image
import numpy as np
from models.tradition_models import NAFNet, DRUNet, MPRNet, DANet
from models.own_model import SignalDenoiseNet

# 设置日志记录
log_dir = "../output"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'test.log'), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 读取 lst 文件的数据集定义
class SpectrogramDataset(Dataset):
    def __init__(self, lst_file, transform=None):
        self.transform = transform
        self.data_pairs = []
        with open(lst_file, 'r') as f:
            for line in f:
                noisy_path, clean_path = line.strip().split()
                self.data_pairs.append((noisy_path, clean_path))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.data_pairs[idx]
        noisy_image = Image.open(noisy_path).convert('L')
        clean_image = Image.open(clean_path).convert('L')

        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)

        return noisy_image, clean_image


# 计算SSIM相似度
def compute_ssim(img1, img2):
    img1 = img1.squeeze().cpu().numpy()
    img2 = img2.squeeze().cpu().numpy()
    return ssim(img1, img2, data_range=img2.max() - img2.min())


# 测试函数
def test_model(model, dataloader, device='cuda'):
    model.to(device)
    model.eval()
    total_ssim = 0
    count = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Testing")
        for noisy_img, clean_img in progress_bar:
            noisy_img, clean_img = noisy_img.to(device), clean_img.to(device)
            outputs = model(noisy_img)

            batch_ssim = sum(compute_ssim(outputs[i], clean_img[i]) for i in range(outputs.shape[0])) / outputs.shape[0]
            total_ssim += batch_ssim
            count += 1
            progress_bar.set_postfix(ssim=batch_ssim)

    avg_ssim = total_ssim / count
    return avg_ssim


# 主测试程序
def main(num):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_lst_dir = '../data/lst_files/test/'  # 测试 lst 文件所在目录
    test_lst_files = [os.path.join(test_lst_dir, f) for f in os.listdir(test_lst_dir) if
                      f.startswith("snr_") and f.endswith(f"_{num}.lst")]

    if not test_lst_files:
        logging.warning(f"No test lst files found for NUM {num}")
        print(f"No test lst files found for NUM {num}")
        return

    # model = NAFNet()
    # model = DRUNet()
    # model = MPRNet()
    # model = DANet()
    model = SignalDenoiseNet()



    avg_ssim_scores = []

    for test_lst_file in test_lst_files:
        parts = test_lst_file.split('_')
        model_path = os.path.join(log_dir, f"best_{parts[-3]}_{parts[-1]}.pth")
        if not os.path.exists(model_path):
            logging.warning("No trained model found.")
            print("No trained model found.")
            return
        model.load_state_dict(torch.load(model_path, map_location="cuda"))


        test_dataset = SpectrogramDataset(test_lst_file, transform=transform)
        test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

        ssim_score = test_model(model, test_dataloader)
        avg_ssim_scores.append(ssim_score)
        logging.info(f"best_{parts[-3]}_{parts[-1]}.pth")
        logging.info(f"Tested on {test_lst_file}, SSIM: {ssim_score:.4f}")
        print(f"Tested on {test_lst_file}, SSIM: {ssim_score:.4f}")

    overall_avg_ssim = np.mean(avg_ssim_scores)
    logging.info(f"SNR {num} - Average SSIM: {overall_avg_ssim:.4f}")
    print(f"SNR {num} - Average SSIM: {overall_avg_ssim:.4f}")


if __name__ == '__main__':
    # main(1)
    main(2)

