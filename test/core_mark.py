import cv2
import numpy as np
import os
import math
import json

# 批量识别出汉字的重心并计算出平均差异值的模块
user_folder = '/Users/ymhave/Downloads/Work'
expert_folder = '/Users/ymhave/Downloads/Expert'

user_files = [f for f in os.listdir(user_folder) if f.endswith('.png') or f.endswith('.jpg')]
expert_files = [f for f in os.listdir(expert_folder) if f.endswith('.png') or f.endswith('.jpg')]
common_files = sorted(set(user_files) & set(expert_files))

def get_centroid(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取文件: {image_path}")
        return None
    h, w = img.shape[:2]
    scale = min(400 / w, 400 / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.ones((400, 400), dtype=np.uint8) * 255
    x_offset = (400 - nw) // 2
    y_offset = (400 - nh) // 2
    canvas[y_offset:y_offset+nh, x_offset:x_offset+nw] = resized
    img_400 = canvas
    _, binary = cv2.threshold(img_400, 127, 255, cv2.THRESH_BINARY_INV)
    M = cv2.moments(binary)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        return (cx, cy)
    else:
        return (0, 0)

print(f"共有 {len(common_files)} 对图片.")

results = {}
for fn in common_files:
    user_path = os.path.join(user_folder, fn)
    expert_path = os.path.join(expert_folder, fn)

    user_centroid = get_centroid(user_path)
    expert_centroid = get_centroid(expert_path)

    if user_centroid is None or expert_centroid is None:
        continue

    dist = math.sqrt((user_centroid[0] - expert_centroid[0])**2 + (user_centroid[1] - expert_centroid[1])**2)

    results[fn] = {
        "expert_centroid": {"x": expert_centroid[0], "y": expert_centroid[1]},
        "user_centroid": {"x": user_centroid[0], "y": user_centroid[1]},
        "centroid_distance": dist
    }

# 计算平均重心差异
distances = [v["centroid_distance"] for v in results.values() if v["centroid_distance"] is not None]
if distances:
    average_distance = sum(distances) / len(distances)
else:
    average_distance = float('nan')

results["average_centroid_distance"] = average_distance

# 构造 JSON 文件名，文件夹名间用下划线连接
user_name = os.path.basename(os.path.normpath(user_folder))
expert_name = os.path.basename(os.path.normpath(expert_folder))
json_filename = f"{expert_name}_{user_name}.json"
json_path = os.path.join(os.getcwd(), json_filename)

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"重心差异及平均值结果已保存至：{json_path}")
print(f"平均重心差异距离: {average_distance:.4f}")
