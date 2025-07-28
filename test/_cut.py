import cv2
import numpy as np
import os

# 1. 在这里自定义你的参数 ↓↓↓
img_path = '/Users/ymhave/Downloads/le.jpg'  # 你的字帖图片路径
out_dir = '/Users/ymhave/Downloads/02'       # 分割后图片存放的文件夹
row_num = 11                                      # 字帖的总行数
col_num = 8                                       # 字帖的总列数
resize_size = 400                                 # 目标图片宽高（单位px）

def split_and_resize(img_path, out_dir, row_num, col_num, size=400):
    img = cv2.imread(img_path)
    if img is None:
        print("图片读取失败，请检查路径和文件名！")
        return

    h, w = img.shape[:2]

    cell_h = h // row_num
    cell_w = w // col_num

    os.makedirs(out_dir, exist_ok=True)
    count = 1

    for i in range(row_num):
        for j in range(col_num):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            char_img = img[y1:y2, x1:x2]
            # 调整为指定尺寸
            char_img_resized = cv2.resize(char_img, (size, size), interpolation=cv2.INTER_AREA)
            out_path = os.path.join(out_dir, f'{i+1:02d}_{j+1:02d}.png')
            cv2.imwrite(out_path, char_img_resized)
            count += 1
    print(f"已分割并保存 {count-1} 个字图片到 {out_dir}")

# 2. 直接运行，不需要命令行参数
split_and_resize(img_path, out_dir, row_num, col_num, size=resize_size)
