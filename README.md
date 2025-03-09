以下是根據您提供的 Python 檔案內容，為您的驗證碼識別專案撰寫的完整 README 文件。這個 README 包含專案概覽、功能、安裝方法、使用說明、專案結構、模型架構、貢獻指南和授權資訊，旨在幫助使用者快速理解並使用您的專案。

---

# Captcha Recognition with CNN

## 概覽
這個專案是一個基於卷積神經網絡（CNN）的驗證碼識別系統。它的主要目標是生成包含 4 位驗證碼的圖片（由小寫英文和數字組成），並訓練一個 CNN 模型來識別這些圖片中的文本。驗證碼圖片包含背景噪點、隨機干擾線和圖片扭曲等特徵，以模擬真實環境並增加識別難度。

---

## 功能
- **驗證碼生成**：自動生成 4 位驗證碼圖片，並保存到 `train` 和 `test` 資料夾中。
- **資料集處理**：自定義的 `CaptchaDataset` 類，用於加載和預處理驗證碼圖片及其標籤。
- **模型訓練**：使用 CNN 模型進行訓練，支持批量處理並可視化學習曲線。
- **模型測試**：在測試集上評估模型的識別準確率。
- **工具函數**：提供標籤與 one-hot 編碼之間的轉換功能。

---

## 安裝
按照以下步驟設置專案環境：

1. **複製專案**：
   ```bash
   git clone https://github.com/your-username/captcha-recognition.git
   ```
   （請將 `your-username` 替換為您的 GitHub 用戶名）

2. **進入專案目錄**：
   ```bash
   cd captcha-recognition
   ```

3. **安裝依賴**：
   確保已安裝 Python 3.x，然後運行：
   ```bash
   pip install torch torchvision pillow matplotlib
   ```
   這些依賴包括：
   - `torch` 和 `torchvision`：用於構建和訓練 CNN 模型。
   - `pillow`：用於圖片處理。
   - `matplotlib`：用於繪製學習曲線。

---

## 使用方法
按照以下步驟運行專案：

1. **生成驗證碼圖片**：
   運行以下指令生成訓練和測試用的驗證碼圖片：
   ```bash
   python generate_captcha.py
   ```
   - 這將在 `train` 資料夾中生成 10,000 張圖片，在 `test` 資料夾中生成 1,000 張圖片。
   - 圖片尺寸為 180x100 像素，包含噪點、干擾線和扭曲效果。

2. **訓練模型**：
   運行訓練腳本：
   ```bash
   python train.py
   ```
   - 訓練參數：批量大小 8，訓練 10 個 epoch，學習率 0.001。
   - 訓練完成後，模型將保存為 `captcha_model.pth`，並顯示損失和準確率的學習曲線。

3. **測試模型**：
   運行測試腳本：
   ```bash
   python test.py
   ```
   - 這將在測試集上評估模型的準確率並輸出結果。

---

## 專案結構
```
captcha-recognition/
│
├── generate_captcha.py   # 生成驗證碼圖片的腳本
├── model.py              # CNN 模型定義
├── utils.py              # 工具函數（標籤轉換）
├── dataset.py            # 自定義資料集類別
├── train.py              # 訓練腳本
├── test.py               # 測試腳本
├── train/                # 訓練圖片資料夾
├── test/                 # 測試圖片資料夾
└── captcha_model.pth     # 訓練好的模型（訓練後生成）
```

---

## 模型架構
模型是一個自定義的 CNN，結構如下：
- **輸入**：160x80 像素的灰階圖（1 通道）。
- **卷積層**：
  - Conv1：1 -> 32 通道，3x3 卷積核，padding=1。
  - Conv2：32 -> 64 通道，3x3 卷積核，padding=1。
  - Conv3：64 -> 128 通道，3x3 卷積核，padding=1，含 BatchNorm。
  - Conv4：128 -> 256 通道，3x3 卷積核，padding=1。
  - Conv5：256 -> 512 通道，3x3 卷積核，padding=1，含 BatchNorm。
- **池化層**：5 個 MaxPool2d (2x2)，逐步將圖片尺寸從 160x80 降至 5x2。
- **全連接層**：512 * 5 * 2 -> 4 * 36（4 位驗證碼，每位 36 類：0-9 和 a-z）。
- **激活函數**：ReLU。

---

## 貢獻
歡迎對此專案進行貢獻！請按照以下步驟操作：
1. Fork 這個專案。
2. 創建一個新分支：
   ```bash
   git checkout -b feature-branch
   ```
3. 提交您的更改並推送 Pull Request。

---



---

## 注意事項
- **字體文件**：運行 `generate_captcha.py` 前，確保系統中已安裝 `arial.ttf` 字體，或修改代碼以使用其他字體。
- **硬體支持**：訓練和測試時，若有 GPU 可用將自動使用，否則使用 CPU。
- **圖片尺寸**：模型輸入尺寸為 160x80 像素，灰階圖，生成圖片時會調整為此大小。

---
