import os
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import math
import json
import csv
from typing import Dict, List, Tuple, Optional
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.spatial import procrustes


            #                      _oo8oo_
            #                     o8888888o
            #                     88" . "88
            #                     (| -_- |)
            #                     0\  =  /0
            #                   ___/'==='\___
            #                 .' \\|     |// '.
            #                / \\|||  :  |||// \
            #               / _||||| -:- |||||_ \
            #              |   | \\\  -  /// |   |
            #              | \_|  ''\---/''  |_/ |
            #              \  .-\__  '-'  __/-.  /
            #            ___'. .'  /--.--\  '. .'___
            #         ."" '<  '.___\_<|>_/___.'  >' "".
            #        | | :  `- \`.:`\ _ /`:.`/ -`  : | |
            #        \  \ `-.   \_ __\ /__ _/   .-` /  /
            #    =====`-.____`.___ \_____/ ___.`____.-`=====
            #                      `=---=`


            #              佛祖保佑         永无bug

def cut(IMG_PATH,SAVE_DIR,N_ROWS,N_COLS,OUT_SIZE=(400, 400),PADDING_RATIO=0.03, PADDING_PX_MIN=6,): # cut_plus.py 的函数
    # IMG_PATH = '～/wo.jpg' 原始字帖地址
    # SAVE_DIR = '～/Work/' 保存地址
    # N_ROWS, N_COLS = 11, 8 增加精确度：行/列
    # OUT_SIZE = (400, 400) 需求图片大小
    # PADDING_RATIO, PADDING_PX_MIN = 0.03, 6  填充比例, 最小填充像素
    DEBUG_SAVE = False
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

    print (f'完成：保存 {count} 张单字图片 → {SAVE_DIR}')
#输入参数，会在对应目录生成对应张图片喵

def core(user_folder,expert_folder,save_dir):
    # user_folder = '～/Work' 用户文件夹地址
    # expert_folder = '～/Expert' 专家文件夹地址
    # save_dir 保存目录

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
    json_filename = 'coreall.json'   #f"{expert_name}_{user_name}.json"
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, json_filename)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"重心差异及平均值结果已保存至：{json_path}")
    print(f"平均重心差异距离: {average_distance:.4f}")
#输入参数会生成coreall.json其中对应键'average_centroid_distance'为平均重心距离

def cner(folder_path,save_dir,size=400):
    # folder_path = '～/Expert'
    # save_dir 保存目录
    folder_name = os.path.basename(folder_path.rstrip("/\\"))
    # size=400
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
    corners_dict = process_folder(folder_path)
    print(corners_dict)
    # ask = input('是否查看图片(y/n)')
    # if ask=='y':
    #     for key, value in corners_dict.items():
    #         visualize_corners(folder_path+'/'+key, value, save_path=key+'_corners.png', show=True)
    # save_dir = "output"   # 保存目录
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在

    json_path = os.path.join(save_dir, folder_name + '.json')

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(corners_dict, f, ensure_ascii=False, indent=4)
#输入参数会生成文件夹名.json，是作品每个图片的角点集

def cal(expert_json,user_json,output_dir):
    # expert_json = "/Users/ymhave/Documents/归档/数学建模/01.json"  专家数据
    #     user_json   = "/Users/ymhave/Documents/归档/数学建模/02.json" 

    #     output_dir  = "/Users/ymhave/Documents/归档/数学建模/out"
    # ===================== 配置区 =====================
    FASTDTW_RADIUS = 3           # fastdtw 搜索半径（2~3 更稳但更慢）
    DRAW_DEBUG = False            # 是否导出调试图（两条对齐曲线叠画）
    DEBUG_DIRNAME = "_debug_curves"
    # =================================================

    # ---------------- 工具函数 ----------------
    def load_points_json(path: str) -> Dict[str, List[List[float]]]:
        """读取 JSON，统一为 dict[str, list[[x,y],...]]。"""
        with open(path, "r", encoding="utf-8") as f:
            _score = json.load(f)
        norm = {}
        for k, v in _score.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (list, tuple)):
                norm[k] = [[float(p[0]), float(p[1])] for p in v]
            else:
                norm[k] = []
        return norm

    def normalize_bbox(points: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """将点集按其外接矩形映射到 [0,1]^2，降低尺度/平移差异对 DTW 的影响。"""
        if points.size == 0:
            return points
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        w = max(x_max - x_min, eps)
        h = max(y_max - y_min, eps)
        out = points.copy()
        out[:, 0] = (out[:, 0] - x_min) / w
        out[:, 1] = (out[:, 1] - y_min) / h
        return out

    def dtw_align(expert_pts: np.ndarray, user_pts: np.ndarray, radius: int = FASTDTW_RADIUS):
        """返回 DTW 对齐后的两组点、DTW 距离与匹配路径。"""
        distance, path = fastdtw(expert_pts, user_pts, radius=radius, dist=euclidean)
        A_aligned = np.asarray([expert_pts[i] for i, _ in path], dtype=float)
        B_aligned = np.asarray([user_pts[j]   for _, j in path], dtype=float)
        return A_aligned, B_aligned, float(distance), path

    def draw_debug_curve(pair_name: str, A: np.ndarray, B: np.ndarray, save_dir: str, size: int = 512):
        """
        导出对齐后的两条折线（绿: 专家，红: 用户）。
        A, B 为 DTW 对齐后的同长度序列。函数内部再做 bbox 归一化方便视觉比较。
        """
        os.makedirs(save_dir, exist_ok=True)
        canvas = np.ones((size, size, 3), dtype=np.uint8) * 255

        def to_px(P):
            P = normalize_bbox(P)         # 这里二次做归一化用于可视化
            P = np.clip(P, 0, 1)
            Q = np.zeros_like(P)
            Q[:, 0] = (P[:, 0] * (size - 40) + 20)
            Q[:, 1] = (P[:, 1] * (size - 40) + 20)
            return Q.astype(int)

        A_px, B_px = to_px(A), to_px(B)

        # 画折线
        for i in range(1, len(A_px)):
            cv2.line(canvas, tuple(A_px[i-1]), tuple(A_px[i]), (0, 180, 0), 2)
        for i in range(1, len(B_px)):
            cv2.line(canvas, tuple(B_px[i-1]), tuple(B_px[i]), (0, 0, 200), 2)

        # 画端点
        for p in (A_px[0], A_px[-1]):
            cv2.circle(canvas, tuple(p), 4, (0, 120, 0), -1)
        for p in (B_px[0], B_px[-1]):
            cv2.circle(canvas, tuple(p), 4, (0, 0, 180), -1)

        out_path = os.path.join(save_dir, f"{pair_name}.png")
        cv2.imwrite(out_path, canvas)

    def weighted_average(values: List[float], weights: Optional[List[float]] = None) -> float:
        """计算等权或加权平均。"""
        vals = [v for v in values if not math.isnan(v)]
        if len(vals) == 0:
            return float("nan")
        if not weights:
            return float(np.mean(vals))
        wts = [w for v, w in zip(values, weights) if not math.isnan(v)]
        vs  = [v for v in values if not math.isnan(v)]
        sw = float(np.sum(wts))
        if sw <= 0:
            return float(np.mean(vs))
        return float(np.sum(np.array(vs) * np.array(wts)) / sw)

    # ---------------- 主流程 ----------------
    def batch_compare(
        expert_json_path: str,
        user_json_path: str,
        out_csv_path: str,
        normalize_before_dtw: bool = True,
        draw_debug: bool = DRAW_DEBUG,
        # weighting 选项：'none'（等权）、'matches'（按DTW匹配对数）、'expert_len'、'user_len'
        weighting: str = "matches",
        metrics_json_path: Optional[str] = None
    ):
        assert weighting in ("none", "matches", "expert_len", "user_len"), \
            "weighting 必须是 'none' | 'matches' | 'expert_len' | 'user_len'"

        expert = load_points_json(expert_json_path)
        user   = load_points_json(user_json_path)

        # 用键交集对齐（即使两侧顺序一致，依旧建议以键为准更稳）
        keys = sorted(set(expert.keys()) & set(user.keys()))
        missing_in_user   = sorted(set(expert.keys()) - set(user.keys()))
        missing_in_expert = sorted(set(user.keys())   - set(expert.keys()))

        if missing_in_user:
            print(f"[警告] 仅在专家JSON中出现、用户缺失：{missing_in_user[:10]}{' ...' if len(missing_in_user)>10 else ''}")
        if missing_in_expert:
            print(f"[警告] 仅在用户JSON中出现、专家缺失：{missing_in_expert[:10]}{' ...' if len(missing_in_expert)>10 else ''}")

        rows = []
        procrustes_list: List[float] = []
        dtw_per_point_list: List[float] = []
        weights_list: List[float] = []  # 与 weighting 策略一致的权重

        # 调试图目录
        dbg_dir = os.path.join(os.path.dirname(out_csv_path), DEBUG_DIRNAME) if draw_debug else None
        if draw_debug:
            os.makedirs(dbg_dir, exist_ok=True)

        for k in keys:
            A = np.asarray(expert[k], dtype=float)
            B = np.asarray(user[k], dtype=float)

            # 基本校验
            if A.ndim != 2 or B.ndim != 2 or A.shape[1] != 2 or B.shape[1] != 2 or len(A) == 0 or len(B) == 0:
                rows.append([k, "nan", "nan", len(A), len(B), 0])
                procrustes_list.append(float("nan"))
                dtw_per_point_list.append(float("nan"))
                weights_list.append(0.0)
                continue

            # DTW 前可做归一化（建议 True，提升稳健性）
            A_in = normalize_bbox(A) if normalize_before_dtw else A
            B_in = normalize_bbox(B) if normalize_before_dtw else B

            # DTW 对齐
            A_aligned, B_aligned, dtw_dist, path = dtw_align(A_in, B_in, radius=FASTDTW_RADIUS)
            m = len(path)  # 匹配对数（用于 weighting='matches'）

            if len(A_aligned) < 2 or len(B_aligned) < 2:
                rows.append([k, "nan", "nan", len(A), len(B), m])
                procrustes_list.append(float("nan"))
                dtw_per_point_list.append(float("nan"))
                weights_list.append(float(m if weighting == "matches" else (
                    len(A) if weighting == "expert_len" else (len(B) if weighting == "user_len" else 1.0)
                )))
                continue

            # Procrustes 形状差异（越小越相似）
            _, _, disparity = procrustes(A_aligned, B_aligned)
            # DTW 每点距离（路径总距离 / 匹配点数）
            dtw_per_point = dtw_dist / max(m, 1)

            rows.append([k, f"{disparity:.6f}", f"{dtw_per_point:.6f}", len(A), len(B), m])

            procrustes_list.append(float(disparity))
            dtw_per_point_list.append(float(dtw_per_point))

            # 根据策略确定样本权重
            if weighting == "matches":
                weights_list.append(float(m))
            elif weighting == "expert_len":
                weights_list.append(float(len(A)))
            elif weighting == "user_len":
                weights_list.append(float(len(B)))
            else:  # 'none'
                weights_list.append(1.0)

            # 调试图（对齐后的两条曲线）
            if draw_debug:
                draw_debug_curve(os.path.splitext(k)[0], A_aligned, B_aligned, dbg_dir)

        # 写 CSV（逐图指标）
        os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
        with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "procrustes_score", "dtw_per_point", "len_expert", "len_user", "matches"])
            writer.writerows(rows)

        # ===== 计算平均差异值（等权 & 加权）=====
        avg_procrustes_eq = weighted_average(procrustes_list, None)              # 等权
        avg_dtw_eq        = weighted_average(dtw_per_point_list, None)

        if weighting == "none":
            avg_procrustes_w = avg_procrustes_eq
            avg_dtw_w        = avg_dtw_eq
        else:
            avg_procrustes_w = weighted_average(procrustes_list, weights_list)   # 加权
            avg_dtw_w        = weighted_average(dtw_per_point_list, weights_list)

        print(f"[完成] 样本数: {len(keys)}")
        print(f" - 平均 Procrustes（等权） : {avg_procrustes_eq:.6f}")
        print(f" - 平均 DTW/point（等权）  : {avg_dtw_eq:.6f}")
        print(f" - 平均 Procrustes（加权={weighting}) : {avg_procrustes_w:.6f}")
        print(f" - 平均 DTW/point（加权={weighting})  : {avg_dtw_w:.6f}")
        print(f"结果明细 CSV 已保存至：{out_csv_path}")

        # 写 metrics.json 摘要
        if metrics_json_path is None:
            metrics_json_path = os.path.join(os.path.dirname(out_csv_path) or ".", "metrics.json")
        try:
            with open(metrics_json_path, "w", encoding="utf-8") as fm:
                json.dump({
                    "num_images": len(keys),
                    "weighting": weighting,
                    "avg_procrustes_equal": avg_procrustes_eq,
                    "avg_dtw_per_point_equal": avg_dtw_eq,
                    "avg_procrustes_weighted": avg_procrustes_w,
                    "avg_dtw_per_point_weighted": avg_dtw_w
                }, fm, ensure_ascii=False, indent=2)
            print(f"指标摘要写入：{metrics_json_path}")
        except Exception as e:
            print(f"[提示] 写入指标摘要失败：{e}")

        # 返回一个字典，便于上层复用
        return {
            "num_images": len(keys),
            "weighting": weighting,
            "avg_procrustes_equal": avg_procrustes_eq,
            "avg_dtw_per_point_equal": avg_dtw_eq,
            "avg_procrustes_weighted": avg_procrustes_w,
            "avg_dtw_per_point_weighted": avg_dtw_w
        }

    # ---------------- 脚本入口示例 ----------------
    if __name__ == "__main__":
        # 替换为你的路径
        os.makedirs(output_dir, exist_ok=True)
        out_csv     = os.path.join(output_dir, "dtw_procrustes_scores.csv")
        metrics_json= os.path.join(output_dir, "metrics.json")

        # weighting 选项：'none' 等权；'matches' 按 DTW 匹配对数；'expert_len' 按专家点数；'user_len' 按用户点数
        result = batch_compare(
            expert_json_path=expert_json,
            user_json_path=user_json,
            out_csv_path=out_csv,
            normalize_before_dtw=True,
            draw_debug=False,
            weighting="matches",       # <<< 这里切换加权策略
            metrics_json_path=metrics_json
        )
#输入参数会生成metrics.json，我们选择"avg_procrustes_weighted"作为《差异值》

def code(core_json_path,corncer_json_path):
    # core_json_path = '/Users/ymhave/Documents/归档/数学建模/Expert_Work.json'  重心差异json地址
    # corncer_json_path = "/Users/ymhave/Documents/归档/数学建模/metrics.json" 角点差异json地址

    with open(core_json_path, "r", encoding="utf-8") as f:
        _score_core = json.load(f)  # 将 JSON 文件内容转换为 Python 字典或列表等对象

    with open(corncer_json_path, "r", encoding="utf-8") as f:
        _score_cor = json.load(f)  # 将 JSON 文件内容转换为 Python 字典或列表等对象
        
    # 现在 _score 是个字典（或列表），例如访问字典特定键：
    core=_score_core["average_centroid_distance"]/100
    if core>1:
        core=1
    cor=_score_cor["avg_procrustes_weighted"]

    de_average=core*0.5+cor*0.5
    score=(1-de_average)*100
    return(f'{score}%')
#输入参数会生成最终评分

def file(folder_path,file_dir,file_name):
    # folder_path = "～/file" 读取图片的文件夹路径
    # file_dir = "～/1" 文件储存地址
    # file_name = 'test' 输出的json文件名

    # 支持的图片扩展名（小写）
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}

    files = os.listdir(folder_path)
    files.sort()  # 按文件名排序

    d={}

    for filename in files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in image_extensions:
            full_path = os.path.join(folder_path, filename)
            if os.path.isfile(full_path):
                d[filename]=full_path
                print(full_path)

    file_path=file_dir+'/'+file_name+'.json'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # 使用with语句打开文件，并写入json
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(d, json_file, ensure_ascii=False, indent=4)
    return d

def read(json_path: str) -> dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        _score = json.load(f)
    return _score
# ------------数据输入处----------
or_work_path = '～/test/work' #原始待评价作品目录
or_expert_path = '～/test/expert' #原始专家作品目录
path = '～/test/down' #工作目录(保存文件)
N_ROWS = 11   #规定字帖行数
N_COLS = 8   #规定字帖列数
#------------施工区请勿打扰喵----------
work_list=file(or_work_path,path+'/work',file_name='work')
expert_list=file(or_expert_path,path+'/expert',file_name='expert')
#专家作品处理
for key_expert in expert_list:
        value_expert = expert_list[key_expert]
        print(key_expert, value_expert)
        cut(value_expert,path+'/expert/cuting/'+key_expert+'dir',N_ROWS,N_COLS)
        cner(path+'/expert/cuting/'+key_expert+'dir',path+'/expert/json/'+key_expert+'dir')
score={}
_score=[]
#选手作品处理
for key_work in work_list:
    value_work = work_list[key_work]
    print(key_work, value_work)
    cut(value_work,path+'/work/cuting/'+key_work+'dir',N_ROWS,N_COLS)
    for key_expert in expert_list:
        value_expert = expert_list[key_expert]
        # 计算core
        work_cuting=path+'/work/cuting/'+key_work+'dir'
        expert_cuting=path+'/expert/cuting/'+key_expert+'dir'
        json_dir=path+'/json/'+key_expert+'_'+key_work #json文件保存目录
        core(work_cuting,expert_cuting,json_dir)
        cner(path+'/work/cuting/'+key_work+'dir',path+'/work/json/'+key_work+'dir')
        cal(path+'/expert/json/'+key_expert+'dir/'+key_expert+'dir.json',path+'/work/json/'+key_work+'dir/'+key_work+'dir.json',json_dir)
        score[key_work+'_'+key_expert]=code(json_dir+'/coreall.json',json_dir+'/metrics.json')
        _score.append(score[key_work+'_'+key_expert])
    percent = 0.3  # 保留前30%
    # 计算要保留的个数
    n = math.ceil(len(_score) * percent)
    # 获取前 n 个最大值
    top_str = sorted(_score, reverse=True)[:n]
    top_n = [float(x.replace('%', '')) for x in top_str]
    av_score = sum(top_n) / len(top_n)
    score[key_work]=av_score
    

output_path = path+'/score.json'

# 确保目录存在
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 写入 JSON 文件
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(score, f, ensure_ascii=False, indent=4)

print("导出成功：", output_path)



filtered = {k: v for k, v in score.items() if isinstance(v, float)}

sorted_items = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
top_percent = 0.3
top_dict = math.ceil(len(sorted_items) * top_percent)
final = dict(sorted_items[:top_dict])
output_path = path+'/final.json'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(final, f, ensure_ascii=False, indent=4)
save_path = path+'/final.csv'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['key', 'value'])  # 表头
    for k, v in final.items():
        writer.writerow([k, v])

print("文件已保存至：", save_path)
# 生成箱形图
def plot(data):
    values = list(data.values())
    df = pd.DataFrame(values, columns=["score"])

    # 计算统计量
    median = df["score"].median()
    mean = df["score"].mean()
    sorted_scores = sorted(values, reverse=True)
    top_n = math.ceil(len(values) * 0.3)
    top_threshold = sorted_scores[top_n - 1]  # 前30%的最小值

    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6))
    box = ax.boxplot(df["score"], vert=True, patch_artist=True, showmeans=True)

    # 添加标注线
    ax.axhline(median, color='orange', linestyle='--', label=f'Median: {median:.2f}')
    ax.axhline(mean, color='blue', linestyle='--', label=f'Average: {mean:.2f}')
    ax.axhline(top_threshold, color='green', linestyle='--', label=f'Threshold of the top 30%: {top_threshold:.2f}')

    # 添加标题和图例
    ax.set_title("Box plot with score values (including mean, median, and top 30%)")
    ax.set_ylabel("score")
    ax.legend(loc='upper right')

    # 保存图像到指定目录
    save_path = path+'/plot.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    print(f"图像已保存到：{save_path}")
plot(final)