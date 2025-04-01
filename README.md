# Captcha Recognition with CNN

## Overview
This project is a captcha recognition system based on Convolutional Neural Networks (CNN). Its primary goal is to generate images containing 4-character captchas (comprising lowercase letters and numbers) and to train a CNN model to recognize the text in these images. The captcha images include features such as background noise, random interference lines, and image distortions to simulate real-world conditions and increase recognition difficulty.
![4wc6](https://github.com/user-attachments/assets/d8edac55-734e-486c-a9bf-77294f3c69df)

---

## Features
- **Captcha Generation**: Automatically generates 4-character captcha images and saves them into the `train` and `test` folders.
- **Dataset Processing**: A custom `CaptchaDataset` class is provided to load and preprocess the captcha images along with their labels.
- **Model Training**: Utilizes a CNN model for training, supports batch processing, and visualizes the learning curve.
- **Model Testing**: Evaluates the model's recognition accuracy on the test set.
- **Utility Functions**: Offers functions to convert between labels and one-hot encoding.

---

## Installation
Follow the steps below to set up the project environment:

**Install Dependencies**:  
Ensure that Python 3.x is installed, and then run:
```bash
pip install torch torchvision pillow matplotlib
```
These dependencies include:
- `torch` and `torchvision`: Used to build and train the CNN model.
- `pillow`: For image processing.
- `matplotlib`: For plotting the learning curves.

---

## Usage
Follow these steps to run the project:

1. **Generate Captcha Images**:  
   Execute the following command to generate captcha images for training and testing:
   ```bash
   python generate_captcha.py
   ```
   - This command creates 10,000 images in the `train` folder and 1,000 images in the `test` folder.
   - The images are 180x100 pixels, featuring noise, interference lines, and distortion effects.

2. **Train the Model**:  
   Run the training script:
   ```bash
   python train.py
   ```
   - Training parameters: batch size of 8, 10 epochs, learning rate of 0.001.
   - Upon completion, the model is saved as `captcha_model.pth`, and the learning curves for loss and accuracy are displayed.

3. **Test the Model**:  
   Run the testing script:
   ```bash
   python test.py
   ```
   - This evaluates the model's accuracy on the test set and outputs the results.

---

## Model Architecture
The model is a custom CNN with the following structure:
- **Input**: A 160x80 pixel grayscale image (1 channel).
- **Convolutional Layers**:
  - **Conv1**: 1 → 32 channels, 3×3 kernel, padding=1.
  - **Conv2**: 32 → 64 channels, 3×3 kernel, padding=1.
  - **Conv3**: 64 → 128 channels, 3×3 kernel, padding=1, includes BatchNorm.
  - **Conv4**: 128 → 256 channels, 3×3 kernel, padding=1.
  - **Conv5**: 256 → 512 channels, 3×3 kernel, padding=1, includes BatchNorm.
- **Pooling Layers**: 5 MaxPool2d layers (2×2) progressively reduce the image dimensions from 160×80 to 5×2.
- **Fully Connected Layer**: Transforms from 512 × 5 × 2 to 4 × 36 (4-character captcha, with each character classified among 36 classes: digits 0-9 and letters a-z).
- **Activation Function**: ReLU.

---

## Learning Curve and Test Accuracy
![learning_curve](https://github.com/user-attachments/assets/71b63250-5188-45d1-b670-3a8f2f4fcb27)

![test_accuracy](https://github.com/user-attachments/assets/217f246b-5237-46ac-9b73-01930e79cc0e)

