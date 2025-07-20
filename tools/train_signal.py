import sys
import os

# 获取项目根目录
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import logging
from models.tradition_models import NAFNet, DRUNet, MPRNet, DANet
from models.own_model import SignalDenoiseNet

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（假设项目根目录是当前目录的上一级）
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到 sys.path
sys.path.append(project_root)


# 设置日志记录
log_dir = "../output"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'training.log'), level=logging.INFO,
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

# 自动加载指定 SNR 的 lst 文件
def get_lst_files_by_num(lst_dir, num):
    lst_files = []
    for file in os.listdir(lst_dir):
        if file.startswith("snr_") and file.endswith(f"_{num}.lst"):
            lst_files.append(os.path.join(lst_dir, file))
    return lst_files

# 组合损失函数定义
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha

    def forward(self, output, target):
        mse = self.mse_loss(output, target)
        ssim = self.ssim_loss(output, target)
        return self.alpha * mse + (1 - self.alpha) * (1 - ssim)

    def ssim_loss(self, img1, img2):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu1 = img1.mean()
        mu2 = img2.mean()
        sigma1 = img1.var()
        sigma2 = img2.var()
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
        return ssim

# 训练函数
def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, train_lst_file, num_epochs=20, eval_interval=5, device='cuda'):
    model.to(device)
    best_accuracy = 0.0
    parts = train_lst_file.split('_')
    best_model_path = os.path.join(log_dir, f"best_{parts[-3]}_{parts[-1]}.pth")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for noisy_img, clean_img in progress_bar:
            noisy_img, clean_img = noisy_img.to(device), clean_img.to(device)
            optimizer.zero_grad()
            outputs = model(noisy_img)
            loss = criterion(outputs, clean_img)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_dataloader)
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # 每 eval_interval 轮计算一次准确度，并保存最佳模型
        if (epoch + 1) % eval_interval == 0:
            model.eval()
            total_loss = 0
            with torch.no_grad():
                for noisy_img, clean_img in test_dataloader:
                    noisy_img, clean_img = noisy_img.to(device), clean_img.to(device)
                    outputs = model(noisy_img)
                    loss = criterion(outputs, clean_img)
                    total_loss += loss.item()
            avg_val_loss = total_loss / len(test_dataloader)
            accuracy = 1 - avg_val_loss
            logging.info(f"Validation Accuracy: {accuracy:.4f}")
            print(f"Validation Accuracy: {accuracy:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"New best model saved with accuracy: {best_accuracy:.4f}")
                print(f"New best model saved with accuracy: {best_accuracy:.4f}")

# 主程序
def main(num):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_lst_dir = '../data/lst_files/train/'  # 训练 lst 文件所在目录
    test_lst_dir = '../data/lst_files/test/'  # 测试 lst 文件所在目录
    train_lst_files = get_lst_files_by_num(train_lst_dir, num)
    test_lst_files = get_lst_files_by_num(test_lst_dir, num)

    if not train_lst_files:
        logging.warning(f"No train lst files found for NUM {num}")
        print(f"No train lst files found for NUM {num}")
        return

    if not test_lst_files:
        logging.warning(f"No test lst files found for NUM {num}")
        print(f"No test lst files found for NUM {num}")
        return

    for train_lst_file in train_lst_files:
        logging.info(f"Training with dataset: {train_lst_file}")
        print(f"Training with dataset: {train_lst_file}")
        train_dataset = SpectrogramDataset(train_lst_file, transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

        test_lst_file = test_lst_files[0]  # 选取一个测试集 lst 进行评估
        test_dataset = SpectrogramDataset(test_lst_file, transform=transform)
        test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

        # model = NAFNet()
        # model = DRUNet()
        # model = MPRNet()
        # model = DANet()
        model = SignalDenoiseNet()


        # criterion = CombinedLoss(alpha=0.8)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_model(model, train_dataloader, test_dataloader, criterion, optimizer, train_lst_file, num_epochs=60, eval_interval=4)


if __name__ == '__main__':
    main(2)