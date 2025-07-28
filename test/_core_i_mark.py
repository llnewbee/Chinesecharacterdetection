import cv2
import numpy as np
import os
import math

# 个人作品文件夹
user_folder = '/Users/ymhave/Downloads/Work'
# 专家作品文件夹
expert_folder = '/Users/ymhave/Downloads/Expert'

user_files = [f for f in os.listdir(user_folder) if f.endswith('.png') or f.endswith('.jpg')]
expert_files = [f for f in os.listdir(expert_folder) if f.endswith('.png') or f.endswith('.jpg')]

# 取交集，即同名文件名单
common_files = sorted(set(user_files) & set(expert_files))

def get_centroid(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取文件: {image_path}")
        return None
    # 等比例缩放+补白为400x400
    h, w = img.shape[:2]
    scale = min(400 / w, 400 / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.ones((400, 400), dtype=np.uint8) * 255
    x_offset = (400 - nw) // 2
    y_offset = (400 - nh) // 2
    canvas[y_offset:y_offset+nh, x_offset:x_offset+nw] = resized
    img_400 = canvas

    # 二值化
    _, binary = cv2.threshold(img_400, 127, 255, cv2.THRESH_BINARY_INV)

    # 计算重心
    M = cv2.moments(binary)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        return (cx, cy), img_400
    else:
        return (0, 0), img_400

print(f"共有 {len(common_files)} 对图片.")

for fn in common_files:
    user_path = os.path.join(user_folder, fn)
    expert_path = os.path.join(expert_folder, fn)

    user_centroid, user_img = get_centroid(user_path)
    expert_centroid, expert_img = get_centroid(expert_path)

    if user_centroid is None or expert_centroid is None:
        continue

    # 计算重心差异欧式距离
    dist = math.sqrt((user_centroid[0] - expert_centroid[0]) ** 2 + (user_centroid[1] - expert_centroid[1]) ** 2)

    # 把两张图放一起显示，左专家右用户，画重心点
    height, width = 400, 400
    combined_img = np.ones((height, width * 2, 3), dtype=np.uint8) * 255

    expert_bgr = cv2.cvtColor(expert_img, cv2.COLOR_GRAY2BGR)
    user_bgr = cv2.cvtColor(user_img, cv2.COLOR_GRAY2BGR)

    # 画重心点，专家红色，用户绿色
    cv2.circle(expert_bgr, (int(expert_centroid[0]), int(expert_centroid[1])), 6, (0,0,255), -1)
    cv2.circle(user_bgr, (int(user_centroid[0]), int(user_centroid[1])), 6, (0,255,0), -1)

    combined_img[:, :width, :] = expert_bgr
    combined_img[:, width:, :] = user_bgr
    os.makedirs("output", exist_ok=True)
    cv2.imwrite(os.path.join("output", f"{fn}_compare.png"), combined_img)

    cv2.imshow(f"{fn} 重心差异 {dist:.2f}", combined_img)
    print(f"{fn}: 专家重心 {expert_centroid}, 用户重心 {user_centroid}, 差异距离 {dist:.2f}")

    key = cv2.waitKey(0)
    cv2.destroyAllWindows()