# -*- coding: utf-8 -*-
"""
plate_dip.py —— 路线 A：传统数字图像处理（DIP）车牌识别流水线
==============================================================

完整实现五个阶段，与项目计划书一一对应：

    ① 车牌定位   HSV 颜色阈值 + Sobel/Canny 边缘 + 形态学 + 多维度评分
    ② ROI 预处理 灰度化 → CLAHE 增强 → 中值/高斯滤波
    ③ 倾斜校正   三级回退：Hough 直线 → 连通域 minAreaRect → 放弃校正
    ④ 字符分割   垂直投影 + 波谷检测 + 宽度先验的合并/强切
    ⑤ 字符识别   模板匹配（归一化相关系数 NCC）+ 车牌格式规则校验

设计取舍
--------
* 颜色定位在灰度化之前完成——灰度化会丢失颜色信息，而颜色是中国车牌最强的先验。
* 定位阶段把颜色、边缘、几何、位置四类特征加权打分，而不是依赖单一特征，
  单一特征在复杂光照下都会失效。
* 字符分割是整个传统路线的瓶颈：中文省份汉字（如"川""蒙"）笔画离散，
  垂直投影会把一个字符切成多段。这里用"宽度先验"做合并，是传统路线的
  核心工程技巧，也是这条路线误差累积的根源。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# 保证同目录下的模块可以被导入（无论从哪个工作目录启动脚本）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synthetic_data as sd  # noqa: E402

# ==========================================================================
# 常量
# ==========================================================================

CHAR_W, CHAR_H = 32, 64          # 字符归一化尺寸
PLATE_ASPECT = 440.0 / 140.0     # 国标车牌长宽比 ≈ 3.14

# 蓝色车牌的 HSV 阈值（OpenCV 的 H 取 0~179）
HSV_BLUE_LOWER = np.array([95, 80, 60], dtype=np.uint8)
HSV_BLUE_UPPER = np.array([140, 255, 255], dtype=np.uint8)


# ==========================================================================
# ① 车牌定位
# ==========================================================================

@dataclass
class PlateCandidate:
    """一个车牌候选框及其打分明细。"""
    bbox: Tuple[int, int, int, int]      # (x, y, w, h)
    score: float = 0.0
    detail: Dict[str, float] = field(default_factory=dict)
    contour: Optional[np.ndarray] = None  # 该候选对应的掩膜轮廓，供透视校正使用


class PlateLocator:
    """
    车牌定位器：多特征融合 + 加权评分排序。

    评分公式（权重与项目计划书一致）：
        score = 0.20 × 颜色密度 + 0.25 × 边缘密度 + 0.10 × S 通道边缘密度
              + 0.15 × 长宽比得分 + 0.15 × 对比度得分 + 0.15 × 位置得分
    """

    def __init__(self) -> None:
        # 小核：桥接车牌内白色描边造成的环状断裂
        self._close_small = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        # 横向大核：把被字符挖空的区域连成整块
        self._close_wide = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))

    # ---- 特征 1：HSV 颜色阈值分割 ----
    def _color_segment(self, image: np.ndarray) -> np.ndarray:
        """
        生成车牌掩膜。

        这里做两级闭运算而不是一级，原因是车牌内部有白色字符和白色描边，
        直接用原始掩膜做连通域分析会把一块车牌拆成"外圈细环 + 内部孔洞块"
        多个区域，导致后续选出错的候选框。
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_BLUE_LOWER, HSV_BLUE_UPPER)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_small)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_wide)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        return mask

    def _raw_color_mask(self, image: np.ndarray) -> np.ndarray:
        """未经形态学处理的原始蓝色掩膜，用于统计真实颜色占比。"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, HSV_BLUE_LOWER, HSV_BLUE_UPPER)

    # ---- 特征 2：Sobel / Canny 边缘检测 ----
    def _edge_detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny：双阈值，抓强边缘
        canny = cv2.Canny(gray, 50, 150)

        # Sobel 垂直方向：车牌字符是密集的竖直纹理，在水平方向梯度最强
        sobel_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        sobel_x = cv2.convertScaleAbs(sobel_x)
        _, sobel_bin = cv2.threshold(sobel_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        edges = cv2.bitwise_or(canny, sobel_bin)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, self._close_wide)

        # S 通道上的边缘：捕捉蓝色饱和区域内部的字符纹理
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        s_edge = cv2.Canny(hsv[:, :, 1], 50, 150)
        s_edge = cv2.morphologyEx(s_edge, cv2.MORPH_CLOSE, self._close_wide)
        return edges, s_edge

    # ---- 特征融合 + 轮廓分析 ----
    def _morphology_analyze(
        self,
        image: np.ndarray,
        color_mask: np.ndarray,
        edge_mask: np.ndarray,
    ) -> List[dict]:
        """
        从车牌掩膜中提取候选框。

        用外轮廓（RETR_EXTERNAL）而不是连通域统计：轮廓会忽略内部孔洞，
        一块车牌就是一个候选，不会被白色字符挖出的孔洞拆散。
        """
        h_img, w_img = image.shape[:2]
        img_area = float(h_img * w_img)
        raw_color = self._raw_color_mask(image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[dict] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 40 or h < 14:
                continue
            box_area = float(w * h)
            if box_area < img_area * 0.004:
                continue
            ratio = w / float(h)
            if ratio < 1.8 or ratio > 7.0:
                continue

            # 轮廓面积 / 外接矩形面积：过滤掉 L 形等不规则区域
            fill_ratio = cv2.contourArea(cnt) / box_area
            if fill_ratio < 0.45:
                continue

            roi_raw = raw_color[y:y + h, x:x + w]
            roi_edge = edge_mask[y:y + h, x:x + w]
            roi_gray = cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
            s_roi = hsv[y:y + h, x:x + w, 1]

            candidates.append({
                "bbox": (int(x), int(y), int(w), int(h)),
                "color_density": float(np.count_nonzero(roi_raw)) / box_area,
                "edge_density": float(np.count_nonzero(roi_edge)) / box_area,
                "s_edge_density": float(np.count_nonzero(s_roi > 100)) / box_area,
                "aspect_ratio": ratio,
                "contrast": float(roi_gray.std()) / 128.0,
                "area_ratio": box_area / img_area,
                "cy_norm": (y + h / 2.0) / h_img,   # 垂直相对位置
                "contour": cnt,
            })
        return candidates

    # ---- 多维度评分排序 ----
    @staticmethod
    def _aspect_score(ratio: float) -> float:
        """长宽比越接近国标 3.14 得分越高（高斯型衰减）。"""
        return float(np.exp(-((ratio - PLATE_ASPECT) ** 2) / (2 * 1.4 ** 2)))

    @staticmethod
    def _position_score(cy_norm: float) -> float:
        """车牌通常位于画面中下部，越靠下得分越高。"""
        return float(np.clip(cy_norm, 0.0, 1.0))

    @staticmethod
    def _area_score(area_ratio: float) -> float:
        """车牌占画面比例越大越可能是真车牌（排除背景中的小蓝块）。"""
        return float(min(area_ratio / 0.10, 1.0))

    def _score_and_sort(self, candidates: List[dict]) -> List[PlateCandidate]:
        """
        多维度加权评分。

        权重设计说明：长宽比与面积是**形状先验**，最能区分"整块车牌"
        与"背景里的蓝色小块"，因此占主导；颜色/边缘密度作为纹理证据，
        位置作为弱先验。这与计划书里"几何筛选是关键步骤"的判断一致。
        """
        scored: List[PlateCandidate] = []
        for cand in candidates:
            aspect = self._aspect_score(cand["aspect_ratio"])
            area = self._area_score(cand["area_ratio"])
            s = (
                0.30 * aspect
                + 0.25 * area
                + 0.20 * min(cand["color_density"], 1.0)
                + 0.15 * min(cand["edge_density"] * 2.0, 1.0)
                + 0.10 * self._position_score(cand["cy_norm"])
            )
            d = {k: (round(float(v), 4) if isinstance(v, (int, float, np.floating))
                     else v) for k, v in cand.items() if k != "contour"}
            d["aspect_score"] = round(aspect, 4)
            d["area_score"] = round(area, 4)
            scored.append(PlateCandidate(cand["bbox"], round(float(s), 4), d,
                                         contour=cand["contour"]))
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored

    # ---- 对外接口 ----
    def locate(self, image: np.ndarray, topk: int = 1) -> List[PlateCandidate]:
        color_mask = self._color_segment(image)
        edge_mask, _ = self._edge_detect(image)
        cands = self._morphology_analyze(image, color_mask, edge_mask)
        return self._score_and_sort(cands)[:topk]


# ==========================================================================
# ② ROI 预处理 + ③ 倾斜校正
# ==========================================================================

class PlatePreprocessor:
    """
    车牌 ROI 预处理与几何校正。

    校正策略：**四角点检测 + 透视校正**。
    先由车牌掩膜轮廓求最小外接旋转矩形（minAreaRect）得到四个角点，
    再做透视变换，把倾斜、带透视的车牌一次性摆正成标准矩形。

    相比"先按外接矩形裁剪、再估算角度旋转"的做法，透视校正的优点是
    一步同时解决**倾斜**和**透视形变**两个问题，而且校正后车牌被紧贴
    地裁成矩形，不再有斜向的白色边框残留在图里干扰字符分割——这正是
    传统路线里"边框残留导致投影法分割失败"的常见根源。
    """

    CANON = (440, 140)          # 校正后的标准车牌尺寸（国标长宽比 3.14）

    def __init__(self, clahe_clip: float = 2.0, clahe_grid: int = 8) -> None:
        self._clahe = cv2.createCLAHE(clipLimit=clahe_clip,
                                      tileGridSize=(clahe_grid, clahe_grid))

    # ---- ② 预处理 ----
    def process(
        self,
        image: np.ndarray,
        contour: Optional[np.ndarray] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        返回 (二值图, 校正后的灰度图, 残余校正角)。

        二值图中字符为白（前景）、背景为黑。
        传入 contour 时走透视校正；只给 bbox 时退化为直接裁剪。
        """
        if contour is not None:
            plate = self._perspective_correct(image, contour, self.CANON)
        else:
            plate = self._extract_roi(image, bbox)

        gray = self._to_gray(plate)
        gray = self._enhance(gray)                 # CLAHE
        gray = self._denoise(gray)                 # 中值 + 高斯
        gray, angle = self._correct_residual_skew(gray)   # 三级回退（残余微调）
        binary = self._binarize(gray)              # OTSU
        binary = self._trim_border(binary)         # 去掉残存边框
        return binary, gray, angle

    # ---- 四角点透视校正 ----
    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """把四个角点排成 (左上, 右上, 右下, 左下) 的顺序。"""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        rect[0] = pts[np.argmin(s)]        # 左上：x+y 最小
        rect[2] = pts[np.argmax(s)]        # 右下：x+y 最大
        rect[1] = pts[np.argmin(d)]        # 右上：y-x 最小
        rect[3] = pts[np.argmax(d)]        # 左下：y-x 最大
        return rect

    def _perspective_correct(
        self,
        image: np.ndarray,
        contour: np.ndarray,
        size: Tuple[int, int],
    ) -> np.ndarray:
        h, w = image.shape[:2]
        cnt = contour.copy()
        cnt[:, 0, 0] = np.clip(cnt[:, 0, 0], 0, w - 1)
        cnt[:, 0, 1] = np.clip(cnt[:, 0, 1], 0, h - 1)

        box = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float32)
        src = self._order_points(box)

        W, H = size
        dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image, M, (W, H),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)

    # ---- 基础图像处理 ----
    @staticmethod
    def _extract_roi(image: np.ndarray,
                     bbox: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
        if bbox is None:
            return image.copy()
        x, y, w, h = bbox
        x0, y0 = max(0, x), max(0, y)
        x1 = min(image.shape[1], max(x0 + 1, x + w))
        y1 = min(image.shape[0], max(y0 + 1, y + h))
        return image[y0:y1, x0:x1].copy()

    @staticmethod
    def _to_gray(roi: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    def _enhance(self, gray: np.ndarray) -> np.ndarray:
        """CLAHE 限制对比度自适应直方图均衡化——避免全局均衡化放大噪声。"""
        return self._clahe.apply(gray)

    @staticmethod
    def _denoise(gray: np.ndarray) -> np.ndarray:
        """中值滤波去椒盐噪声并保护边缘，再做高斯平滑。"""
        median = cv2.medianBlur(gray, 3)
        return cv2.GaussianBlur(median, (5, 5), 1.0)

    @staticmethod
    def _trim_border(binary: np.ndarray) -> np.ndarray:
        """裁掉二值图四边的残存白边（车牌外框/铆钉）。"""
        h, w = binary.shape[:2]
        mx, my = int(w * 0.04), int(h * 0.10)
        if w - 2 * mx < 10 or h - 2 * my < 5:
            return binary
        return binary[my:h - my, mx:w - mx].copy()

    # ---- ③ 残余倾斜校正（三级回退） ----
    def _correct_residual_skew(self, gray: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        透视校正后车牌已基本摆正；这里只做小幅残余校正。

        三级回退：Hough 直线 → 连通域 minAreaRect → 放弃校正。
        只在检测到的角度落在 (0.5°, 10°) 之间时才应用，避免过度校正。
        """
        h, w = gray.shape[:2]
        edges = cv2.Canny(gray, 50, 150)

        angle = self._detect_angle_hough(edges, w)
        if angle is None:
            angle = self._detect_angle_contour(gray)
        if angle is None or not (0.5 <= abs(angle) <= 10.0):
            return gray, 0.0

        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        return rotated, float(angle)

    @staticmethod
    def _detect_angle_hough(edges: np.ndarray, w: int) -> Optional[float]:
        """检测近似水平的直线，按长度加权求平均倾角。"""
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                                minLineLength=max(20, w // 2), maxLineGap=8)
        if lines is None:
            return None
        # 兼容不同 OpenCV 版本：(N,1,4) 与 (N,4) 两种返回形状
        lines = np.asarray(lines).reshape(-1, 4)
        angles, weights = [], []
        for x1, y1, x2, y2 in lines:
            if x2 == x1:
                continue
            ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(ang) <= 10:                      # 残余倾斜只可能是小角度
                angles.append(ang)
                weights.append(np.hypot(x2 - x1, y2 - y1))
        if len(angles) < 2:
            return None
        return float(np.average(angles, weights=weights))

    @staticmethod
    def _detect_angle_contour(gray: np.ndarray) -> Optional[float]:
        """对二值化后的最大连通域求最小外接矩形，取其倾角。"""
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)))
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        big = max(contours, key=cv2.contourArea)
        if cv2.contourArea(big) < 200:
            return None
        (_, _), (_, _), ang = cv2.minAreaRect(big)
        while ang > 45:                 # 统一到 [-45, 45]
            ang -= 90
        while ang < -45:
            ang += 90
        return float(ang) if abs(ang) <= 10 else None

    @staticmethod
    def _binarize(gray: np.ndarray) -> np.ndarray:
        """OTSU 全局阈值；字符为亮（白字），故用 THRESH_BINARY。"""
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return binary


# ==========================================================================
# ④ 字符分割
# ==========================================================================

class CharSegmenter:
    """
    基于垂直投影的字符分割。

    流程图（与计划书一致）：
        去边框 → 垂直投影 → 波谷检测 → 宽度先验合并/强切 → 长度校验

    "宽度先验"是应对中文省份汉字笔画离散的关键：
    像"川"这样的字会被投影法切成 3 段，必须依据标准字符宽度把它们合并回去。
    """

    def __init__(self, expected_chars: int = 7) -> None:
        self.expected_chars = expected_chars

    # ---- 去边框 ----
    @staticmethod
    def _remove_border(binary: np.ndarray) -> np.ndarray:
        """
        轻量裁边，去掉车牌外框描边与铆钉，避免投影峰值出现在两端。

        主要的边框裁剪已在预处理的透视校正阶段完成，这里只做兜底，
        避免过度裁剪吃掉省份汉字的笔画。
        """
        h, w = binary.shape[:2]
        mx, my = int(w * 0.015), int(h * 0.03)
        if w - 2 * mx < 10 or h - 2 * my < 5:
            return binary
        return binary[my:h - my, mx:w - mx].copy()

    # ---- 垂直投影求连通段 ----
    @staticmethod
    def _projection_runs(binary: np.ndarray,
                         thr_ratio: float = 0.10) -> List[Tuple[int, int]]:
        proj = binary.sum(axis=0).astype(np.float32) / 255.0
        if proj.max() <= 0:
            return []
        active = proj > proj.max() * thr_ratio

        runs: List[Tuple[int, int]] = []
        start = None
        for i, a in enumerate(active):
            if a and start is None:
                start = i
            elif not a and start is not None:
                runs.append((start, i))
                start = None
        if start is not None:
            runs.append((start, len(active)))
        return runs

    # ---- 把投影得到的小段合并成恰好 N 个字符 ----
    def fit_runs(self, runs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        把垂直投影产生的若干小段整理成恰好 ``expected_chars`` 个字符区间。

        主策略是**间隙排序法**：字符与字符之间的间隙，必然大于同一字符内部
        笔画之间的间隙（例如"川""蒙"的笔画虽然离散，但彼此靠得很近）。
        因此只需在全部间隙中挑出最大的 N-1 个作为字符边界即可。

        这条思路比"按固定宽度比例合并"稳健得多：中文字符宽、拉丁字符窄，
        用同一个宽度阈值会把窄的拉丁字符误并进相邻字符，导致后续所有字符
        整体错位——这正是传统路线里最难缠的一类分割失败。
        """
        if not runs:
            return []

        # 主策略：段数多于字符数时，用最大的 N-1 个间隙切开
        if len(runs) > self.expected_chars:
            gaps = [(runs[i + 1][0] - runs[i][1], i) for i in range(len(runs) - 1)]
            n_cut = self.expected_chars - 1
            cut_at = sorted(i for _, i in sorted(gaps, key=lambda g: -g[0])[:n_cut])

            merged: List[Tuple[int, int]] = []
            start = runs[0][0]
            for i in cut_at:
                merged.append((start, runs[i][1]))
                start = runs[i + 1][0]
            merged.append((start, runs[-1][1]))
            runs = merged

        # 兜底：段数不足时，按宽度比例强切最宽的一段
        while 0 < len(runs) < self.expected_chars:
            widths = [e - s for s, e in runs]
            i = int(np.argmax(widths))
            s, e = runs[i]
            span = runs[-1][1] - runs[0][0]
            exp_w = span / float(self.expected_chars)
            if e - s < exp_w * 0.9:
                break                            # 已经不够切了，放弃
            mid = (s + e) // 2
            runs[i] = (s, mid)
            runs.insert(i + 1, (mid, e))

        return runs

    # ---- 对外接口 ----
    def segment(self, binary: np.ndarray) -> List[np.ndarray]:
        clean = self._remove_border(binary)
        if clean.size == 0:
            return []
        runs = self._projection_runs(clean)
        runs = self.fit_runs(runs)

        chars: List[np.ndarray] = []
        for s, e in runs:
            char = clean[:, s:e]
            if char.size == 0:
                continue
            # 垂直方向按墨迹裁剪后再归一化：否则"1"和"津"这类字符的
            # 有效高度不同，统一拉伸到同一尺寸后仍无法与模板对齐。
            rows = np.where(char.sum(axis=1) > 0)[0]
            if len(rows) == 0:
                continue
            char = char[rows.min():rows.max() + 1, :]
            if not self._validate_char(char):
                continue
            chars.append(cv2.resize(char, (CHAR_W, CHAR_H),
                                    interpolation=cv2.INTER_AREA))
        return chars

    @staticmethod
    def _validate_char(char: np.ndarray) -> bool:
        """宽高比检查：过窄（噪声）或过宽（粘连）的段丢弃。"""
        h, w = char.shape[:2]
        if h <= 0 or w <= 0:
            return False
        ratio = w / float(h)
        return 0.05 <= ratio <= 1.6


# ==========================================================================
# ⑤ 字符识别：模板匹配
# ==========================================================================

class TemplateLibrary:
    """
    标准字符模板库：31 个省份汉字 + 24 个字母（去除 I/O）+ 10 个数字。

    注意：这里用同一套字体渲染模板，是为了让 Demo 能自洽地跑通。
    真实项目中模板应来自真实车牌截图（不同字体、不同磨损程度），
    否则模板匹配的准确率会被高估——这正是传统路线的局限之一。
    """

    def __init__(self, font_size: int = 64) -> None:
        self.templates: Dict[str, np.ndarray] = {}
        self._build(font_size)

    def _render(self, ch: str, font_size: int) -> Optional[np.ndarray]:
        from PIL import Image, ImageDraw
        font = sd.load_font(font_size)
        img = Image.new("L", (font_size * 2, font_size * 2), 0)
        draw = ImageDraw.Draw(img)
        box = draw.textbbox((0, 0), ch, font=font)
        w, h = box[2] - box[0], box[3] - box[1]
        if w <= 0 or h <= 0:
            return None
        draw.text(((img.width - w) / 2 - box[0], (img.height - h) / 2 - box[1]),
                  ch, font=font, fill=255)
        arr = np.array(img)
        # 裁到墨迹外接框，再去掉细边
        ys, xs = np.where(arr > 60)
        if len(xs) == 0:
            return None
        arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        _, binar = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return cv2.resize(binar, (CHAR_W, CHAR_H), interpolation=cv2.INTER_AREA)

    def _build(self, font_size: int) -> None:
        chars = list(sd.PROVINCES) + list(sd.LETTERS) + list(sd.DIGITS)
        for ch in chars:
            t = self._render(ch, font_size)
            if t is not None:
                self.templates[ch] = t

    # 位移搜索半径：容忍分割带来的 1~2 像素错位
    SHIFT = 2

    @classmethod
    def _similarity(cls, char_img: np.ndarray, tmpl: np.ndarray) -> float:
        """
        二值前景的 IoU（交并比），并在 ±SHIFT 像素范围内做位移搜索取最优。

        为什么不用 cv2.matchTemplate 的 TM_CCOEFF_NORMED：该系数会先减去
        均值再算相关，对 0/255 的二值字形并不稳定，实测会把大量字符误判
        成同一类；而 IoU 直接度量"前景重合程度"，对二值笔画更可靠。
        """
        a = char_img > 127
        b = tmpl > 127
        if not a.any() or not b.any():
            return 0.0

        best = 0.0
        for dy in range(-cls.SHIFT, cls.SHIFT + 1):
            for dx in range(-cls.SHIFT, cls.SHIFT + 1):
                shifted = np.roll(np.roll(b, dy, axis=0), dx, axis=1)
                inter = np.logical_and(a, shifted).sum()
                union = np.logical_or(a, shifted).sum()
                if union:
                    best = max(best, float(inter) / float(union))
        return best

    def match(self, char_img: np.ndarray,
              topk: int = 3) -> List[Tuple[str, float]]:
        """模板匹配，返回 Top-k 候选 (字符, 相似度)。"""
        scores: List[Tuple[str, float]] = []
        for ch, tmpl in self.templates.items():
            if tmpl.shape != char_img.shape:
                tmpl = cv2.resize(tmpl, (char_img.shape[1], char_img.shape[0]))
            scores.append((ch, self._similarity(char_img, tmpl)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topk]


class FormatValidator:
    """车牌格式规则校验：第 1 位省份汉字，第 2 位字母，后 5 位字母/数字。"""

    PROVINCES = set(sd.PROVINCES)
    LETTERS = set(sd.LETTERS)
    ALNUM = set(sd.ALNUM)

    @classmethod
    def validate(cls, text: str) -> Tuple[bool, List[int]]:
        """返回 (是否合法, 违规位置列表)。"""
        bad: List[int] = []
        if len(text) != 7:
            return False, list(range(len(text)))
        if text[0] not in cls.PROVINCES:
            bad.append(0)
        if text[1] not in cls.LETTERS:
            bad.append(1)
        for i in range(2, 7):
            if text[i] not in cls.ALNUM:
                bad.append(i)
        return len(bad) == 0, bad


# ==========================================================================
# 路线 A 总调度
# ==========================================================================

class DIPPlateRecognizer:
    """传统 DIP 路线：定位 → 预处理 → 校正 → 分割 → 模板匹配识别。"""

    def __init__(self, expected_chars: int = 7) -> None:
        self.locator = PlateLocator()
        self.preprocessor = PlatePreprocessor()
        self.segmenter = CharSegmenter(expected_chars=expected_chars)
        self.library = TemplateLibrary()

    def recognize(self, image: np.ndarray, debug: bool = False) -> dict:
        out: dict = {
            "text": None, "confidence": 0.0, "bbox": None,
            "chars": [], "n_chars": 0, "angle": 0.0,
            "valid": False, "stages": {},
        }

        # ① 定位
        cands = self.locator.locate(image, topk=1)
        if not cands:
            out["stages"]["locate"] = "失败：未找到候选车牌区域"
            return out
        best = cands[0]
        out["bbox"] = best.bbox
        out["stages"]["locate"] = f"通过（评分 {best.score:.3f}）"

        # ②③ 预处理：四角点透视校正 + 增强 + 二值化
        binary, gray, angle = self.preprocessor.process(
            image, contour=best.contour, bbox=best.bbox)
        out["angle"] = round(angle, 2)

        # ④ 分割
        chars = self.segmenter.segment(binary)
        out["n_chars"] = len(chars)
        out["chars"] = chars
        if not chars:
            out["stages"]["segment"] = "失败：未分割出字符"
            return out

        # ⑤ 识别
        text, confs = "", []
        for ch_img in chars:
            top = self.library.match(ch_img, topk=1)
            if top:
                text += top[0][0]
                confs.append(top[0][1])
        out["text"] = text
        out["confidence"] = round(float(np.mean(confs)) if confs else 0.0, 4)
        out["valid"], _ = FormatValidator.validate(text)

        if debug:
            out["binary"] = binary
            out["gray"] = gray
        return out


# ==========================================================================
# 自测
# ==========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="传统 DIP 车牌识别自测")
    parser.add_argument("-n", "--num", type=int, default=30, help="测试样本数")
    parser.add_argument("--seed", type=int, default=200, help="起始随机种子")
    parser.add_argument("--verbose", action="store_true", help="逐条打印结果")
    args = parser.parse_args()

    rec = DIPPlateRecognizer()
    n = args.num
    full_ok = 0          # 整牌全对
    seg_ok = 0           # 字符数正确
    char_total = 0       # 字符总数
    char_hit = 0         # 字符级命中

    print("=" * 78)
    print(f"传统 DIP 路线 —— 端到端自测（{n} 个合成样本，起始种子 {args.seed}）")
    print("=" * 78)

    for i in range(n):
        s = sd.make_sample(seed=args.seed + i)
        r = rec.recognize(s.image)
        pred = r["text"] or ""

        hit = (pred == s.text)
        full_ok += int(hit)
        seg_ok += int(r["n_chars"] == len(s.text))
        char_total += len(s.text)
        char_hit += sum(1 for a, b in zip(pred, s.text) if a == b)

        if args.verbose or not hit:
            print(f"[{'命中' if hit else '未中'}] 真值 {s.text} → 预测 {pred:8s} "
                  f"字符数 {r['n_chars']}  置信度 {r['confidence']:.3f}")

    print("-" * 78)
    print(f"整牌准确率     ：{full_ok}/{n} = {full_ok / n:.1%}")
    print(f"字符级准确率   ：{char_hit}/{char_total} = {char_hit / char_total:.1%}")
    print(f"分割字符数正确率：{seg_ok}/{n} = {seg_ok / n:.1%}")
