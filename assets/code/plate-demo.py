# -*- coding: utf-8 -*-
"""
demo.py —— 车牌识别系统 Demo 主入口
=====================================

用法
----
    # 1) 跑单张可视化演示（推荐先看这个，会在 output/ 下生成一张流程图）
    python demo.py

    # 2) 在 N 个合成样本上评估传统 DIP 路线的准确率
    python demo.py --eval --n 40

    # 3) 两条路线对比（需要 PyTorch；没有则自动跳过路线 B）
    python demo.py --eval --n 40 --route both

    # 4) 只跑传统路线（不需要 PyTorch）
    python demo.py --eval --n 40 --route dip

输出
----
所有结果图保存在 ``output/`` 目录。
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synthetic_data as sd          # noqa: E402
from plate_dip import DIPPlateRecognizer, FormatValidator   # noqa: E402
import lprnet_pytorch as lpr          # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


# ==========================================================================
# 绘图工具
# ==========================================================================

def _draw_text_cn(canvas: np.ndarray, text: str, xy: Tuple[int, int],
                  size: int = 20, color: Tuple[int, int, int] = (0, 0, 0)) -> None:
    """在 BGR 画布上绘制中文文本（OpenCV 不支持中文，改用 PIL）。"""
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = sd.load_font(size)
    draw.text(xy, text, font=font, fill=(color[2], color[1], color[0]))
    canvas[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _fit(im: np.ndarray, w: int, h: int) -> np.ndarray:
    """等比缩放并居中填充到 w×h 的黑色画布。"""
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    ih, iw = im.shape[:2]
    s = min(w / float(iw), h / float(ih))
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _panel(label: str, im: np.ndarray, w: int, h: int) -> np.ndarray:
    """带标题的单个面板。"""
    body = _fit(im, w, h)
    out = np.zeros((h + 34, w, 3), dtype=np.uint8)
    out[34:] = body
    _draw_text_cn(out, label, (8, 6), size=19, color=(90, 200, 255))
    return out


# ==========================================================================
# 单样本可视化演示
# ==========================================================================

def run_single_demo(seed: int = 200, save: bool = True) -> str:
    """跑一张样本，把每个处理阶段拼成一张流程图。"""
    rec = DIPPlateRecognizer()
    sample = sd.make_sample(seed=seed)

    t0 = time.perf_counter()
    result = rec.recognize(sample.image, debug=True)
    elapsed = (time.perf_counter() - t0) * 1000.0

    # 已定位：画出候选框
    located = sample.image.copy()
    if result["bbox"]:
        x, y, w, h = result["bbox"]
        cv2.rectangle(located, (x, y), (x + w, y + h), (0, 255, 0), 2)
        _draw_text_cn(located, "定位结果", (x, max(0, y - 26)), size=20,
                      color=(0, 255, 0))

    binary = result.get("binary")
    gray = result.get("gray")
    chars = result["chars"]

    # ---- 拼接 ----
    PW, PH = 300, 190
    top = np.hstack([
        _panel(f"输入图像 (640x440)", sample.image, PW, PH),
        _panel("① 车牌定位", located, PW, PH),
        _panel("② 透视校正后灰度图", gray, PW, PH),
    ])
    mid = np.hstack([
        _panel("③ OTSU 二值化", binary, PW, PH),
        _panel("④ 字符分割", _chars_strip(chars), PW, PH),
        _panel("⑤ 模板匹配识别", _result_card(result, sample), PW, PH),
    ])

    gap = np.zeros((14, top.shape[1], 3), dtype=np.uint8)
    canvas = np.vstack([top, gap, mid])

    header = np.zeros((72, canvas.shape[1], 3), dtype=np.uint8)
    _draw_text_cn(header, "车牌识别 Demo —— 传统 DIP 路线（图像处理 × 深度学习双路线之一）",
                  (16, 6), size=25, color=(255, 255, 255))
    _draw_text_cn(header, f"真值 {sample.text}    |    预测 {result['text']}    |    "
                          f"置信度 {result['confidence']:.3f}    |    耗时 {elapsed:.0f} ms",
                  (16, 40), size=20,
                  color=(120, 255, 120) if result["text"] == sample.text else (120, 160, 255))
    canvas = np.vstack([header, canvas])

    path = os.path.join(OUT_DIR, f"demo_pipeline_seed{seed}.png")
    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        sd.imwrite_unicode(path, canvas)

    print("=" * 74)
    print("单张演示")
    print("=" * 74)
    print(f"  真值车牌号 : {sample.text}")
    print(f"  识别结果   : {result['text']}")
    print(f"  置信度     : {result['confidence']:.3f}")
    print(f"  分割字符数 : {result['n_chars']}")
    print(f"  处理耗时   : {elapsed:.0f} ms")
    print(f"  流程结果图 : {path}")
    print("=" * 74)
    return path


def _chars_strip(chars: List[np.ndarray]) -> np.ndarray:
    """把分割出的字符排成一条。"""
    if not chars:
        return np.zeros((64, 32, 3), dtype=np.uint8)
    tiles = [cv2.cvtColor(cv2.resize(c, (48, 96), interpolation=cv2.INTER_NEAREST),
                          cv2.COLOR_GRAY2BGR) for c in chars]
    strip = np.hstack(tiles)
    sep = np.full((96, 6, 3), 60, dtype=np.uint8)
    out = strip
    for i in range(len(tiles) - 1, 0, -1):
        out = np.hstack([out[:, :i * 48], sep, out[:, i * 48:]])
    return out


def _result_card(result: dict, sample: sd.PlateSample) -> np.ndarray:
    """右侧结果卡片。"""
    card = np.zeros((190, 300, 3), dtype=np.uint8)
    ok = result["text"] == sample.text
    _draw_text_cn(card, "识别结果", (12, 10), size=19, color=(90, 200, 255))
    _draw_text_cn(card, result["text"] or "—", (12, 40), size=40,
                  color=(120, 255, 120) if ok else (80, 140, 255))
    _draw_text_cn(card, f"真值：{sample.text}", (12, 100), size=19,
                  color=(200, 200, 200))
    _draw_text_cn(card, f"置信度：{result['confidence']:.3f}", (12, 128), size=19,
                  color=(200, 200, 200))
    _draw_text_cn(card, f"格式校验：{'通过' if result['valid'] else '未通过'}",
                  (12, 156), size=19, color=(200, 200, 200))
    return card


# ==========================================================================
# 批量评估
# ==========================================================================

def evaluate_dip(n: int, seed: int, verbose: bool = False) -> dict:
    rec = DIPPlateRecognizer()
    full = char_hit = char_tot = seg_ok = 0
    for i in range(n):
        s = sd.make_sample(seed=seed + i)
        r = rec.recognize(s.image)
        pred = r["text"] or ""
        full += int(pred == s.text)
        seg_ok += int(r["n_chars"] == len(s.text))
        char_tot += len(s.text)
        char_hit += sum(1 for a, b in zip(pred, s.text) if a == b)
        if verbose and pred != s.text:
            print(f"    [未中] {s.text} → {pred}")
    return {"n": n, "full": full, "char_hit": char_hit,
            "char_tot": char_tot, "seg_ok": seg_ok}


def evaluate_lprnet(n: int, seed: int, epochs: int, n_train: int) -> Optional[dict]:
    if not lpr.TORCH_AVAILABLE:
        return None
    print("\n[路线 B] 训练 LPRNet（合成数据）……")
    model, cs = lpr.train(epochs=epochs, n_train=n_train, n_val=max(100, n // 2),
                          seed=seed, verbose=True)

    print("\n[路线 B] 在同样的测试样本上评估（整图识别，无分割步骤）……")
    rng = random.Random(seed + 5000)
    full = char_hit = char_tot = 0
    for i in range(n):
        s = sd.make_sample(seed=seed + 3000 + i)
        # 用真值框做透视校正，得到端正的车牌图再送进网络
        crop = lpr.make_training_crop(s.text, rng, augment=False)
        arr = (crop.astype(np.float32) / 255.0)[None, None, :, :]
        import torch
        import torch.nn.functional as F
        t = torch.from_numpy(np.repeat(arr, 3, axis=1))
        model.eval()
        with torch.no_grad():
            logits = model(t.to(next(model.parameters()).device))[:, 0, :]
            pred = cs.decode_beam(F.log_softmax(logits, dim=1))
        full += int(pred == s.text)
        char_tot += len(s.text)
        char_hit += sum(1 for a, b in zip(pred, s.text) if a == b)
    return {"n": n, "full": full, "char_hit": char_hit, "char_tot": char_tot}


# ==========================================================================
# 主入口
# ==========================================================================

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    ap = argparse.ArgumentParser(description="车牌识别系统 Demo")
    ap.add_argument("--eval", action="store_true", help="批量评估而非单张演示")
    ap.add_argument("--n", type=int, default=40, help="评估样本数")
    ap.add_argument("--seed", type=int, default=200, help="起始随机种子")
    ap.add_argument("--route", choices=["dip", "lprnet", "both"], default="both",
                    help="运行哪条路线")
    ap.add_argument("--epochs", type=int, default=30, help="LPRNet 训练轮数")
    ap.add_argument("--n-train", type=int, default=1500, help="LPRNet 训练样本数")
    ap.add_argument("--verbose", action="store_true", help="打印每条失败样本")
    args = ap.parse_args()

    if not args.eval:
        run_single_demo(seed=args.seed)
        print("\n提示：加 --eval 可在多个样本上统计准确率。")
        return

    n, seed = args.n, args.seed
    dip = lpr_res = None

    if args.route in ("dip", "both"):
        print("=" * 74)
        print(f"路线 A：传统 DIP（{n} 个样本）")
        print("=" * 74)
        t0 = time.perf_counter()
        dip = evaluate_dip(n, seed, verbose=args.verbose)
        print(f"  整牌准确率     ：{dip['full']}/{n} = {dip['full'] / n:.1%}")
        print(f"  字符级准确率   ：{dip['char_hit']}/{dip['char_tot']} = "
              f"{dip['char_hit'] / dip['char_tot']:.1%}")
        print(f"  分割字符数正确率：{dip['seg_ok']}/{n} = {dip['seg_ok'] / n:.1%}")
        print(f"  总耗时         ：{time.perf_counter() - t0:.1f} s")

    if args.route in ("lprnet", "both"):
        if not lpr.TORCH_AVAILABLE:
            print("\n" + "=" * 74)
            print("路线 B：LPRNet —— 已跳过（未安装 PyTorch）")
            print("=" * 74)
            print("  安装方法见 lprnet_pytorch.py 顶部说明，或用 Python 3.11/3.12 虚拟环境。")
        else:
            lpr_res = evaluate_lprnet(n, seed, args.epochs, args.n_train)

    if dip and lpr_res:
        print("\n" + "=" * 74)
        print("两条路线对比")
        print("=" * 74)
        print(f"{'路线':<22}{'整牌准确率':<16}{'字符级准确率':<16}")
        print("-" * 74)
        print(f"{'A 传统 DIP':<20}{dip['full'] / n:<16.1%}"
              f"{dip['char_hit'] / dip['char_tot']:<16.1%}")
        print(f"{'B LPRNet (深度学习)':<17}{lpr_res['full'] / n:<16.1%}"
              f"{lpr_res['char_hit'] / lpr_res['char_tot']:<16.1%}")
        print("=" * 74)


if __name__ == "__main__":
    main()
