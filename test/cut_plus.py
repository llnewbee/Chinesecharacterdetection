import os
import cv2
import numpy as np
from PIL import Image
#新版 作品图像透视变化和裁剪模块

# ===================== 配置 =====================
IMG_PATH = '/Users/ymhave/Downloads/file/zi.jpg'
SAVE_DIR = '/Users/ymhave/Downloads/Expert/'
N_ROWS, N_COLS = 11, 8
OUT_SIZE = (400, 400)
PADDING_RATIO, PADDING_PX_MIN = 0.03, 6
DEBUG_SAVE = True
os.makedirs(SAVE_DIR, exist_ok=True)

# ============== 工具函数 ==============
def order_quad(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1); d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]; bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def score_rect(rect_w, rect_h, img_w, img_h, ncols, nrows):
    # 面积比（越大越好，但不应接近整张纸）
    area_ratio = (rect_w * rect_h) / (img_w * img_h + 1e-6)
    area_score = np.clip((area_ratio - 0.10) / 0.70, 0.0, 1.0)  # 10%~80%最佳
    # 长宽比接近 N_COLS/N_ROWS
    target = ncols / max(1e-6, nrows)
    ratio = rect_w / max(1e-6, rect_h)
    ratio_score = np.exp(-((ratio - target) ** 2) / (2 * (0.15 * target) ** 2))  # ±15%容忍
    return 0.6 * area_score + 0.4 * ratio_score

def is_near_blank(img_bgr, thr_std=2.0):
    # 透视后若几乎纯色，std 很小
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return g.std() < thr_std

# ============== 读图与预处理 ==============
orig = cv2.imread(IMG_PATH)
if orig is None: raise FileNotFoundError(IMG_PATH)
h0, w0 = orig.shape[:2]
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

# Blackhat 强化黑线（网格/外框/字迹）
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
# 二值化 + 边缘
_, th = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
edges = cv2.Canny(th, 50, 150)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)), iterations=2)

if DEBUG_SAVE:
    cv2.imwrite(os.path.join(SAVE_DIR, '_dbg_blackhat.png'), blackhat)
    cv2.imwrite(os.path.join(SAVE_DIR, '_dbg_th.png'), th)
    cv2.imwrite(os.path.join(SAVE_DIR, '_dbg_edges.png'), edges)

# ============== 寻找最佳四边形外框 ==============
cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
best_quad, best_score = None, -1.0
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) != 4: continue
    rect = cv2.boundingRect(approx)
    x, y, w, h = rect
    # 过滤极小/极扁
    if w < 0.25*w0 or h < 0.25*h0: continue
    sc = score_rect(w, h, w0, h0, N_COLS, N_ROWS)
    if sc > best_score:
        best_score = sc
        best_quad = approx.reshape(-1,2)

if best_quad is None:
    raise RuntimeError('未找到字帖外框，请检查图片或放宽阈值。')

quad = order_quad(best_quad)

dbg_quad = orig.copy()
cv2.polylines(dbg_quad, [quad.astype(np.int32)], True, (0,255,0), 3)
if DEBUG_SAVE:
    cv2.imwrite(os.path.join(SAVE_DIR, '_dbg_quad.png'), dbg_quad)

# ============== 透视矫正（整体拉直） ==============
cell_w, cell_h = 100, 100  # 仅作网格基准
dst = np.array([[0,0],
                [N_COLS*cell_w-1,            0],
                [N_COLS*cell_w-1, N_ROWS*cell_h-1],
                [0,             N_ROWS*cell_h-1]], dtype=np.float32)

M = cv2.getPerspectiveTransform(quad, dst)
warped = cv2.warpPerspective(orig, M, (N_COLS*cell_w, N_ROWS*cell_h))

# 如果矫正结果近似纯色，说明外框取错或顺序错误，回退：改用整图均分（至少可用）
if is_near_blank(warped):
    # 试图交换右上/左下（极少数顺序问题）
    quad_swapped = np.array([quad[0], quad[3], quad[2], quad[1]], dtype=np.float32)
    M2 = cv2.getPerspectiveTransform(quad_swapped, dst)
    warped2 = cv2.warpPerspective(orig, M2, (N_COLS*cell_w, N_ROWS*cell_h))
    if not is_near_blank(warped2):
        warped = warped2
    else:
        # 仍然纯色，最后兜底：用整图等分（保证不至于空）
        warped = cv2.resize(orig, (N_COLS*cell_w, N_ROWS*cell_h), interpolation=cv2.INTER_LINEAR)

if DEBUG_SAVE:
    cv2.imwrite(os.path.join(SAVE_DIR, '_debug_warped.jpg'), warped)

# ============== 画调试网格（红横蓝竖） ==============
if DEBUG_SAVE:
    dbg = warped.copy()
    for i in range(N_ROWS + 1):
        y = int(i * cell_h)
        cv2.line(dbg, (0, y), (dbg.shape[1]-1, y), (0,0,255), 2)
    for j in range(N_COLS + 1):
        x = int(j * cell_w)
        cv2.line(dbg, (x, 0), (x, dbg.shape[0]-1), (255,0,0), 2)
    cv2.imwrite(os.path.join(SAVE_DIR, '_grid_debug.png'), dbg)

# ============== 均分切割并保存 ==============
count = 0
for i in range(N_ROWS):
    for j in range(N_COLS):
        x1, x2 = int(j*cell_w), int((j+1)*cell_w)
        y1, y2 = int(i*cell_h), int((i+1)*cell_h)

        pad_x = max(PADDING_PX_MIN, int((x2-x1)*PADDING_RATIO))
        pad_y = max(PADDING_PX_MIN, int((y2-y1)*PADDING_RATIO))
        xx1, xx2 = x1 + pad_x, x2 - pad_x
        yy1, yy2 = y1 + pad_y, y2 - pad_y
        if xx2 <= xx1 or yy2 <= yy1:  # 极端情况下跳过
            continue

        crop = warped[yy1:yy2, xx1:xx2]
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        crop_resized = crop_pil.resize(OUT_SIZE, Image.Resampling.LANCZOS)
        count += 1
        crop_resized.save(os.path.join(SAVE_DIR, f'{i+1:02d}_{j+1:02d}.png'))

print(f'完成：保存 {count} 张单字图片 → {SAVE_DIR}')
