# utils.py
import torch

def label_to_onehot(label, num_classes=36):
    onehot = torch.zeros(4, num_classes)
    for i, char in enumerate(label):
        if char.isdigit():
            idx = int(char)
        else:
            idx = ord(char) - ord('a') + 10  # a-z 對應 10-35
        onehot[i, idx] = 1
    return onehot.view(-1)  # 展平成一維 (4 * 36)

def onehot_to_label(onehot):
    onehot = onehot.view(4, 36)
    label = ''
    for i in range(4):
        idx = torch.argmax(onehot[i]).item()
        if idx < 10:
            label += str(idx)
        else:
            label += chr(idx - 10 + ord('a'))
    return label