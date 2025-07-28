import os
import json
import csv
import math
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.spatial import procrustes

# ===================== 配置区 =====================
FASTDTW_RADIUS = 3           # fastdtw 搜索半径（2~3 更稳但更慢）
DRAW_DEBUG = True            # 是否导出调试图（两条对齐曲线叠画）
DEBUG_DIRNAME = "_debug_curves"
# =================================================

# ---------------- 工具函数 ----------------
def load_points_json(path: str) -> Dict[str, List[List[float]]]:
    """读取 JSON，统一为 dict[str, list[[x,y],...]]。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    norm = {}
    for k, v in data.items():
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

def draw_debug_curve(pair_name: str, A: np.ndarray, B: np.ndarray, save_dir: str, size: int = 512, 
                     expert_img_dir: str = None, user_img_dir: str = None, alpha: float = 0.35):
    """
    画专家与用户对齐曲线，并将原图（专家、用户，均400x400）灰度处理后半透明叠加。
    图片文件名与pair_name一致（自动匹配扩展名）。
    """
    os.makedirs(save_dir, exist_ok=True)
    canvas = np.ones((size, size, 3), dtype=np.uint8) * 255

    # 原图读取和半透明灰度融合
    base = canvas.copy()
    overlay_list = []
    for folder in [expert_img_dir, user_img_dir]:
        if folder:
            imgpath = None
            for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                try_path = os.path.join(folder, f"{pair_name}{ext}")
                if os.path.exists(try_path):
                    imgpath = try_path
                    break
            if imgpath:
                img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
                    overlay_list.append(img_rgb)
                else:
                    print(f"未能读取图片: {imgpath}")

    if overlay_list:
        overlay_sum = np.zeros_like(base, dtype=np.float32)
        for img in overlay_list:
            overlay_sum += img.astype(np.float32) / len(overlay_list)
        overlay_sum = overlay_sum.astype(np.uint8)
        cv2.addWeighted(overlay_sum, alpha, base, 1 - alpha, 0, base)

    def to_px(P):
        P = normalize_bbox(P)
        P = np.clip(P, 0, 1)
        Q = np.zeros_like(P)
        Q[:, 0] = (P[:, 0] * (size - 40) + 20)
        Q[:, 1] = (P[:, 1] * (size - 40) + 20)
        return Q.astype(int)

    A_px, B_px = to_px(A), to_px(B)

    for i in range(1, len(A_px)):
        cv2.line(base, tuple(A_px[i-1]), tuple(A_px[i]), (0, 180, 0), 2)
    for i in range(1, len(B_px)):
        cv2.line(base, tuple(B_px[i-1]), tuple(B_px[i]), (0, 0, 200), 2)
    for p in (A_px[0], A_px[-1]):
        cv2.circle(base, tuple(p), 4, (0, 120, 0), -1)
    for p in (B_px[0], B_px[-1]):
        cv2.circle(base, tuple(p), 4, (0, 0, 180), -1)

    out_path = os.path.join(save_dir, f"{pair_name}.png")
    cv2.imwrite(out_path, base)

def weighted_average(values: List[float], weights: Optional[List[float]] = None) -> float:
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
    weighting: str = "matches",
    metrics_json_path: Optional[str] = None,
    expert_img_dir: Optional[str] = None,
    user_img_dir: Optional[str] = None,
):
    assert weighting in ("none", "matches", "expert_len", "user_len"), \
        "weighting 必须是 'none' | 'matches' | 'expert_len' | 'user_len'"

    expert = load_points_json(expert_json_path)
    user   = load_points_json(user_json_path)
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
    weights_list: List[float] = []

    dbg_dir = os.path.join(os.path.dirname(out_csv_path), DEBUG_DIRNAME) if draw_debug else None
    if draw_debug:
        os.makedirs(dbg_dir, exist_ok=True)

    for k in keys:
        A = np.asarray(expert[k], dtype=float)
        B = np.asarray(user[k], dtype=float)
        if A.ndim != 2 or B.ndim != 2 or A.shape[1] != 2 or B.shape[1] != 2 or len(A) == 0 or len(B) == 0:
            rows.append([k, "nan", "nan", len(A), len(B), 0])
            procrustes_list.append(float("nan"))
            dtw_per_point_list.append(float("nan"))
            weights_list.append(0.0)
            continue

        A_in = normalize_bbox(A) if normalize_before_dtw else A
        B_in = normalize_bbox(B) if normalize_before_dtw else B
        A_aligned, B_aligned, dtw_dist, path = dtw_align(A_in, B_in, radius=FASTDTW_RADIUS)
        m = len(path)

        if len(A_aligned) < 2 or len(B_aligned) < 2:
            rows.append([k, "nan", "nan", len(A), len(B), m])
            procrustes_list.append(float("nan"))
            dtw_per_point_list.append(float("nan"))
            weights_list.append(float(m if weighting == "matches" else (
                len(A) if weighting == "expert_len" else (len(B) if weighting == "user_len" else 1.0)
            )))
            continue

        _, _, disparity = procrustes(A_aligned, B_aligned)
        dtw_per_point = dtw_dist / max(m, 1)
        rows.append([k, f"{disparity:.6f}", f"{dtw_per_point:.6f}", len(A), len(B), m])
        procrustes_list.append(float(disparity))
        dtw_per_point_list.append(float(dtw_per_point))

        if weighting == "matches":
            weights_list.append(float(m))
        elif weighting == "expert_len":
            weights_list.append(float(len(A)))
        elif weighting == "user_len":
            weights_list.append(float(len(B)))
        else:
            weights_list.append(1.0)

        if draw_debug:
            draw_debug_curve(
                os.path.splitext(k)[0], A_aligned, B_aligned, dbg_dir,
                size=512,
                expert_img_dir=expert_img_dir,
                user_img_dir=user_img_dir,
                alpha=0.35
            )

    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "procrustes_score", "dtw_per_point", "len_expert", "len_user", "matches"])
        writer.writerows(rows)

    avg_procrustes_eq = weighted_average(procrustes_list, None)
    avg_dtw_eq        = weighted_average(dtw_per_point_list, None)

    if weighting == "none":
        avg_procrustes_w = avg_procrustes_eq
        avg_dtw_w        = avg_dtw_eq
    else:
        avg_procrustes_w = weighted_average(procrustes_list, weights_list)
        avg_dtw_w        = weighted_average(dtw_per_point_list, weights_list)

    print(f"[完成] 样本数: {len(keys)}")
    print(f" - 平均 Procrustes（等权） : {avg_procrustes_eq:.6f}")
    print(f" - 平均 DTW/point（等权）  : {avg_dtw_eq:.6f}")
    print(f" - 平均 Procrustes（加权={weighting}) : {avg_procrustes_w:.6f}")
    print(f" - 平均 DTW/point（加权={weighting})  : {avg_dtw_w:.6f}")
    print(f"结果明细 CSV 已保存至：{out_csv_path}")

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
    expert_json = "/Users/ymhave/Documents/归档/数学建模/Expert.json"
    user_json   = "/Users/ymhave/Documents/归档/数学建模/Work.json"

    output_dir  = "/Users/ymhave/Documents/归档/数学建模/e"
    os.makedirs(output_dir, exist_ok=True)
    out_csv     = os.path.join(output_dir, "dtw_procrustes_scores.csv")
    metrics_json= os.path.join(output_dir, "metrics.json")

    # 请设置专家图片文件夹和用户图片文件夹
    expert_img_dir = "/Users/ymhave/Downloads/Expert"
    user_img_dir   = "/Users/ymhave/Downloads/Work"

    result = batch_compare(
        expert_json_path=expert_json,
        user_json_path=user_json,
        out_csv_path=out_csv,
        normalize_before_dtw=True,
        draw_debug=True,
        weighting="matches",
        metrics_json_path=metrics_json,
        expert_img_dir=expert_img_dir,
        user_img_dir=user_img_dir
    )
