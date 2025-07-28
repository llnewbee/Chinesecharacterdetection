import os
import cv2
import json

#批量识别角点的模块
size=400
def get_image_files(folder):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]

def resize_image(img, size=(size, size)):
    return cv2.resize(img, size)

def detect_corners(img, max_corners=100):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=0.01, minDistance=10)
    if corners is not None:
        corners = corners.reshape(-1, 2)
        return [(int(x), int(y)) for x, y in corners]
    else:
        return []

def process_folder(folder):
    image_files = get_image_files(folder)
    result = {}
    for img_path in sorted(image_files):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_resized = resize_image(img)
        corners = detect_corners(img_resized)

        # 将 key 设为文件名（不含扩展名）
        key = os.path.basename(img_path)

        result[key] = corners
    return result

def visualize_corners(image_path, corners, save_path=None, show=False):
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return
    img = cv2.resize(img, (size, size))  # 保证坐标一致
    for x, y in corners:
        cv2.circle(img, (x, y), radius=2, color=(0, 255,0), thickness=-1)
    if save_path:
        cv2.imwrite(save_path, img)
        print(f"图像已保存至: {save_path}")
    if show:
        cv2.imshow("Corners", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ===== 使用区域 =====
folder_path = '/Users/ymhave/Downloads/Work'
folder_name = os.path.basename(folder_path.rstrip("/\\"))
corners_dict = process_folder(folder_path)
print(corners_dict)
print()
# ask = input('是否查看图片(y/n)')
# if ask=='y':
#     for key, value in corners_dict.items():
#         visualize_corners(folder_path+'/'+key, value, save_path=key+'_corners.png', show=True)

with open(folder_name+'.json', "w", encoding="utf-8") as f:
    json.dump(corners_dict, f, ensure_ascii=False, indent=4)
