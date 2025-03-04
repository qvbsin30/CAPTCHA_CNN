# test.py
import torch
from torch.utils.data import DataLoader
from model import CaptchaCNN
from dataset import CaptchaDataset
from utils import onehot_to_label
from torchvision import transforms

# 資料轉換
transform = transforms.Compose([
    transforms.Resize((160, 80)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加載測試資料集
test_dataset = CaptchaDataset('data/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 載入模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CaptchaCNN().to(device)
model.load_state_dict(torch.load('captcha_model.pth'))
model.eval()

# 測試函數
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            predicted = outputs.view(-1, 4, 36).argmax(dim=2)
            for i in range(len(labels)):
                pred_label = ''.join([chr(ord('0') + idx) if idx < 10 else chr(ord('a') + idx - 10) for idx in predicted[i]])
                if pred_label == labels[i]:
                    correct += 1
                total += 1
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    test()