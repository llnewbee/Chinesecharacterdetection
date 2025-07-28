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
DRAW_DEBUG = False            # 是否导出调试图（两条对齐曲线叠画）
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
    expert_json = "/Users/ymhave/Documents/归档/数学建模/01.json"
    user_json   = "/Users/ymhave/Documents/归档/数学建模/02.json"

    output_dir  = "/Users/ymhave/Documents/归档/数学建模/out"
    os.makedirs(output_dir, exist_ok=True)
    out_csv     = os.path.join(output_dir, "dtw_procrustes_scores.csv")
    metrics_json= os.path.join(output_dir, "metrics.json")

    # weighting 选项：'none' 等权；'matches' 按 DTW 匹配对数；'expert_len' 按专家点数；'user_len' 按用户点数
    result = batch_compare(
        expert_json_path=expert_json,
        user_json_path=user_json,
        out_csv_path=out_csv,
        normalize_before_dtw=True,
        draw_debug=True,
        weighting="matches",       # <<< 这里切换加权策略
        metrics_json_path=metrics_json
    )
