from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import string
import os

# 生成4位驗證碼文本（小寫英文+數字）
def generate_captcha_text(length=4):
    characters = string.ascii_lowercase + string.digits  # a-z 和 0-9
    return ''.join(random.choice(characters) for _ in range(length))

# 生成單張驗證碼圖片
def generate_captcha_image(captcha_text, width=180, height=100):
    # 創建空白圖片，背景為白色
    image = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    # 添加背景噪點
    for _ in range(random.randint(100, 200)):
        xy = (random.randrange(0, width), random.randrange(0, height))
        draw.point(xy, fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    
    # 添加隨機干擾線
    for _ in range(random.randint(3, 5)):
        start = (random.randint(0, width), random.randint(0, height))
        end = (random.randint(0, width), random.randint(0, height))
        draw.line([start, end], fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), width=1)
    
    # 繪製驗證碼文字
    font = ImageFont.truetype("arial.ttf", size=36)  # 需確保字體文件存在
    for i, char in enumerate(captcha_text):
        x = 10 + i * 40  # 每個字符間隔40像素
        y = random.randint(10, 30)  # 隨機高度增加干擾
        draw.text((x, y), char, font=font, fill=(0, 0, 0))  # 黑色文字
    
    # 扭曲圖片
    image = image.transform((width, height), Image.AFFINE, (1, -0.3, 0, -0.1, 1, 0), resample=Image.BILINEAR)
    image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)  # 增強邊緣效果
    
    return image

# 生成並保存驗證碼圖片到指定資料夾
def generate_and_save_captcha(folder, count):
    if not os.path.exists(folder):
        os.makedirs(folder)  # 創建資料夾
    generated = set()  # 用集合避免重複
    while len(generated) < count:
        captcha_text = generate_captcha_text()
        if captcha_text not in generated:
            image = generate_captcha_image(captcha_text)
            image.save(os.path.join(folder, f"{captcha_text}.png"))
            generated.add(captcha_text)

# 生成train和test資料夾中的圖片
generate_and_save_captcha('train', 10000)
generate_and_save_captcha('test', 1000)

# 檢查生成結果
train_count = len(os.listdir('train'))
test_count = len(os.listdir('test'))
print(f"Train資料夾圖片數量: {train_count}")
print(f"Test資料夾圖片數量: {test_count}")