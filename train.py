# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import CaptchaCNN
from dataset import CaptchaDataset
from utils import label_to_onehot
import matplotlib.pyplot as plt

# 超參數
batch_size = 8  # 一次訓練 8 張圖片
epochs = 10      # 訓練 10 次
learning_rate = 0.001

# 資料轉換
transform = transforms.Compose([
    transforms.Resize((160, 80)),  # 調整圖片大小
    transforms.ToTensor(),         # 轉為張量
    transforms.Normalize((0.5,), (0.5,))  # 正規化
])

# 加載訓練資料集
train_dataset = CaptchaDataset('data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 初始化模型、損失函數和優化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CaptchaCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 訓練函數
def train():
    model.train()
    train_loss = []
    train_acc = []
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels_onehot = torch.stack([label_to_onehot(label) for label in labels]).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_onehot)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # 計算準確率
            predicted = outputs.view(-1, 4, 36).argmax(dim=2)
            for i in range(len(labels)):
                pred_label = predicted[i].tolist()
                true_label = [int(c) if c.isdigit() else ord(c) - ord('a') + 10 for c in labels[i]]
                if pred_label == true_label:
                    correct += 1
                total += 1

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'captcha_model.pth')

    # 繪製學習曲線
    plt.plot(train_loss, label='Training Loss')
    plt.plot(train_acc, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Learning Curve')
    plt.show()

if __name__ == '__main__':
    train()