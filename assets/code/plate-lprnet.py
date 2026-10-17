# -*- coding: utf-8 -*-
"""
lprnet_pytorch.py —— 路线 B：LPRNet 端到端深度学习车牌识别
============================================================

LPRNet（License Plate Recognition Network）是一种轻量级**端到端**车牌识别模型：
它跳过了"字符分割"这一步，直接把整张车牌图映射成字符序列。这与传统 DIP
路线的根本差别在于：

    DIP   ：定位 → 预处理 → 校正 → **分割** → 识别   （逐步累积误差）
    LPRNet：定位 → 预处理 → 校正 → **整图序列识别**   （端到端，无分割误差）

本文件实现 PPT 中描述的网络结构：

    Conv ×4 (64 → 128 → 256 → 512)
      → BiLSTM (hidden = 256)
      → FC (512 → 66 类)
      → CTC 解码

其中 66 类 = 31 个省份汉字 + 24 个字母（去除 I/O）+ 10 个数字 + 1 个 CTC blank。

依赖
----
需要 PyTorch：``pip install torch torchvision``

注意：PyTorch 目前尚未发布 Python 3.14 的官方 wheel。若你的解释器是 3.14，
请用 Python 3.11 / 3.12 创建虚拟环境后再安装：

    py -3.12 -m venv .venv
    .venv\\Scripts\\activate
    pip install torch torchvision
    python lprnet_pytorch.py --train

未安装 PyTorch 时，本模块导入不会失败——``TORCH_AVAILABLE`` 会是 False，
demo.py 会自动跳过路线 B 并给出提示。
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synthetic_data as sd  # noqa: E402

# --------------------------------------------------------------------------
# 可选依赖：PyTorch
# --------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:                                   # pragma: no cover
    torch = None                                      # type: ignore
    nn = None                                         # type: ignore
    F = None                                          # type: ignore
    TORCH_AVAILABLE = False


# --------------------------------------------------------------------------
# 字符集与 CTC 编解码
# --------------------------------------------------------------------------

INPUT_H, INPUT_W = 24, 94          # LPRNet 标准输入尺寸（高 × 宽）


class Charset:
    """
    车牌字符集：31 省份汉字 + 24 字母（无 I/O）+ 10 数字，外加 1 个 CTC blank。

    索引约定：0 ~ N-1 为真实字符，N 为 blank。
    """

    def __init__(self) -> None:
        self.chars = list(sd.PROVINCES) + list(sd.LETTERS) + list(sd.DIGITS)
        self.blank = len(self.chars)          # blank 索引 = 65
        self.num_classes = len(self.chars) + 1
        self._idx = {c: i for i, c in enumerate(self.chars)}

    def __len__(self) -> int:
        return self.num_classes

    def encode(self, text: str) -> List[int]:
        """车牌号 → 索引序列。"""
        out = []
        for ch in text:
            if ch not in self._idx:
                raise ValueError(f"字符 {ch!r} 不在字符集中")
            out.append(self._idx[ch])
        return out

    def decode_greedy(self, log_probs: "torch.Tensor") -> str:
        """
        CTC 贪心解码：取每帧最大概率的类别，再去重、去 blank。

        参数 log_probs 形状为 (T, C)。
        """
        idx = log_probs.argmax(dim=1).tolist()
        out, prev = [], None
        for i in idx:
            if i != prev and i != self.blank:
                out.append(self.chars[i])
            prev = i
        return "".join(out)

    def decode_beam(self, log_probs: "torch.Tensor", beam: int = 4) -> str:
        """
        CTC 前缀束搜索解码（beam search）。

        入参必须是**已经取过 log_softmax** 的对数概率，形状 (T, C)。
        相比贪心解码，束搜索会保留多条候选路径，在字符粘连或模糊处更稳健。
        """
        lp = log_probs
        T = lp.shape[0]
        beams: List[Tuple[List[int], float, Optional[int]]] = [([], 0.0, None)]
        for t in range(T):
            cand: List[Tuple[List[int], float, Optional[int]]] = []
            for seq, score, last in beams:
                for c in range(self.num_classes):
                    p = float(lp[t, c])
                    if c == self.blank:
                        cand.append((seq, score + p, None))
                    elif c == last:
                        cand.append((seq, score + p, c))       # 重复字符合并
                    else:
                        cand.append((seq + [c], score + p, c))
            # 合并相同前缀，保留概率最高的 beam 条
            best: dict = {}
            for seq, score, last in cand:
                key = (tuple(seq), last)
                if key not in best or score > best[key]:
                    best[key] = score
            beams = [(list(k[0]), v, k[1])
                     for k, v in sorted(best.items(), key=lambda kv: -kv[1])[:beam]]
        seq = beams[0][0]
        return "".join(self.chars[i] for i in seq)


# --------------------------------------------------------------------------
# 网络结构
# --------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class ConvBNReLU(nn.Module):
        """卷积 + 批归一化 + ReLU 的标准组合。"""

        def __init__(self, cin: int, cout: int, k: int = 3,
                     stride: int = 1, pad: int = 1) -> None:
            super().__init__()
            self.conv = nn.Conv2d(cin, cout, k, stride, pad, bias=False)
            self.bn = nn.BatchNorm2d(cout)
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

    class LPRNet(nn.Module):
        """
        LPRNet：卷积主干 → 双向 LSTM → 全连接分类头 → CTC。

        输入  (B, 3, 24, 94)
        输出  (T, B, num_classes)，T = 宽度方向的下采样长度，供 CTC Loss 使用。
        """

        def __init__(self, num_classes: int = 66,
                     lstm_hidden: int = 256,
                     dropout: float = 0.5) -> None:
            super().__init__()
            # 主干：4 个卷积阶段，通道数 64 → 128 → 256 → 512
            self.backbone = nn.Sequential(
                ConvBNReLU(3, 64),
                nn.MaxPool2d(2, 2),          # 24×94 → 12×47
                ConvBNReLU(64, 128),
                nn.MaxPool2d(2, 2),          # 12×47 → 6×23
                ConvBNReLU(128, 256),
                ConvBNReLU(256, 512),
            )
            # 序列建模：双向 LSTM 读取宽度方向的字符序列
            self.lstm = nn.LSTM(512, lstm_hidden, num_layers=1,
                                bidirectional=True, batch_first=True)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(lstm_hidden * 2, num_classes)

        def forward(self, x):
            f = self.backbone(x)                 # (B, 512, 6, 23)
            f = f.mean(dim=2)                    # 高度方向池化 → (B, 512, 23)
            f = f.permute(0, 2, 1)               # (B, 23, 512)  作为时间序列
            f, _ = self.lstm(f)                  # (B, 23, 512)
            f = self.dropout(f)
            f = self.fc(f)                       # (B, 23, num_classes)
            return f.permute(1, 0, 2)            # (T=23, B, C)，CTC 要求的时间在前


# --------------------------------------------------------------------------
# 训练数据准备
# --------------------------------------------------------------------------

def make_training_crop(text: str, rng, augment: bool = True) -> np.ndarray:
    """
    生成一张用于训练的**已裁好的**车牌图（24×94 灰度）。

    训练阶段直接使用裁好的车牌，把"定位/校正"当成上游已解决的模块——
    这与真实工程中的做法一致：识别模型只负责"整图 → 字符序列"。
    """
    panel = sd.render_plate_panel(text)               # 440×140 BGR

    if augment:
        h, w = panel.shape[:2]
        # 透视扰动
        j = rng.uniform(0.0, 0.05) * w
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([
            [rng.uniform(0, j), rng.uniform(0, j)],
            [w - rng.uniform(0, j), rng.uniform(0, j)],
            [w - rng.uniform(0, j), h - rng.uniform(0, j)],
            [rng.uniform(0, j), h - rng.uniform(0, j)],
        ])
        panel = cv2.warpPerspective(
            panel, cv2.getPerspectiveTransform(src, dst), (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # 亮度抖动
        alpha = rng.uniform(0.7, 1.3)
        beta = rng.uniform(-35, 35)
        panel = cv2.convertScaleAbs(panel, alpha=alpha, beta=beta)

        # 轻微平移 / 裁剪，模拟边界截断
        if rng.random() < 0.5:
            k = rng.randint(0, 3)
            panel = panel[:, k:w - (2 - k)] if w - (2 - k) > k else panel

        # 模糊
        if rng.random() < 0.5:
            k = rng.choice([3, 5])
            panel = cv2.GaussianBlur(panel, (k, k), 0)

    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)


def build_dataset(n: int, seed: int = 0):
    """构造 n 个 (3×24×94 归一化张量, 标签文本) 样本。"""
    import random as _random
    rng = _random.Random(seed)
    xs, ys = [], []
    for i in range(n):
        text = sd.random_plate_text(rng)
        crop = make_training_crop(text, rng, augment=True)
        # 归一化到 [0,1]，并复制成 3 通道（LPRNet 使用 3 通道输入）
        arr = (crop.astype(np.float32) / 255.0)[None, :, :]
        xs.append(np.repeat(arr, 3, axis=0))
        ys.append(text)
    return np.stack(xs), ys


# --------------------------------------------------------------------------
# 训练
# --------------------------------------------------------------------------

def train(
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    n_train: int = 1500,
    n_val: int = 200,
    seed: int = 0,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Tuple["LPRNet", "Charset"]:
    """
    在合成数据上训练 LPRNet。

    超参数与项目 PPT 中一致：Adam(lr=1e-3)，CTC Loss(blank=65)，
    梯度裁剪 max_norm=10。
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "未检测到 PyTorch。请先安装：pip install torch torchvision\n"
            "（注意 PyTorch 暂不支持 Python 3.14，建议用 Python 3.11/3.12 虚拟环境）"
        )

    cs = Charset()
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = LPRNet(num_classes=cs.num_classes).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ctc = nn.CTCLoss(blank=cs.blank, zero_infinity=True)

    if verbose:
        print(f"[训练] 设备={dev}  训练样本={n_train}  验证样本={n_val}  "
              f"类别数={cs.num_classes}  blank={cs.blank}")

    x_np, y_txt = build_dataset(n_train, seed=seed)
    x = torch.from_numpy(x_np)

    n = len(y_txt)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total, steps = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = x[idx].to(dev)

            # 按 batch 内长度拼接标签（CTC 要求的一维格式）
            lens = [len(cs.encode(y_txt[j])) for j in idx]
            flat = torch.cat([torch.tensor(cs.encode(y_txt[j]), dtype=torch.long)
                              for j in idx]).to(dev)

            logits = model(xb)                       # (T, B, C)
            log_probs = F.log_softmax(logits, dim=2)
            T = log_probs.shape[0]
            in_lens = torch.full((len(idx),), T, dtype=torch.long).to(dev)

            loss = ctc(log_probs, flat, in_lens,
                       torch.tensor(lens, dtype=torch.long).to(dev))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)   # 梯度裁剪
            opt.step()
            total += float(loss.item())
            steps += 1

        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs):
            acc = evaluate(model, cs, n=n_val, seed=seed + 999, device=dev,
                           verbose=False)
            print(f"  epoch {ep:3d}/{epochs}  loss={total / max(steps,1):.4f}  "
                  f"验证整牌准确率={acc:.1%}")

    return model, cs


@torch.no_grad() if TORCH_AVAILABLE else (lambda f: f)
def evaluate(model, cs: Charset, n: int = 200, seed: int = 999,
             device=None, verbose: bool = True) -> float:
    """在合成验证集上评估整牌准确率。"""
    import random as _random
    rng = _random.Random(seed)
    model.eval()
    dev = next(model.parameters()).device
    ok = 0
    for _ in range(n):
        text = sd.random_plate_text(rng)
        crop = make_training_crop(text, rng, augment=False)
        arr = (crop.astype(np.float32) / 255.0)[None, None, :, :]
        t = torch.from_numpy(np.repeat(arr, 3, axis=1)).to(dev)
        logits = model(t)                     # (T,1,C)
        pred = cs.decode_greedy(logits[:, 0, :])
        ok += int(pred == text)
    acc = ok / float(n)
    if verbose:
        print(f"验证整牌准确率：{ok}/{n} = {acc:.1%}")
    return acc


@torch.no_grad() if TORCH_AVAILABLE else (lambda f: f)
def infer(model, cs: Charset, plate_bgr: np.ndarray,
          use_beam: bool = True, device=None) -> str:
    """对一张已裁好的车牌图（BGR）做识别。"""
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
    arr = (gray.astype(np.float32) / 255.0)[None, None, :, :]
    t = torch.from_numpy(np.repeat(arr, 3, axis=1))
    model.eval()
    dev = next(model.parameters()).device
    logits = model(t.to(dev))[:, 0, :]
    lp = F.log_softmax(logits, dim=1)
    return cs.decode_beam(lp) if use_beam else cs.decode_greedy(lp)


def export_onnx(model, path: str = "lprnet.onnx") -> None:
    """导出 ONNX，便于脱离 PyTorch 做轻量部署。"""
    if not TORCH_AVAILABLE:
        raise RuntimeError("需要安装 PyTorch 才能导出 ONNX")
    model.eval()
    dummy = torch.randn(1, 3, INPUT_H, INPUT_W)
    torch.onnx.export(model, dummy, path,
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"},
                                    "logits": {1: "batch"}},
                      opset_version=11)
    print(f"已导出 ONNX 模型：{path}")


# --------------------------------------------------------------------------
# 命令行入口
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="LPRNet 训练 / 评估")
    ap.add_argument("--train", action="store_true", help="训练模型")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--save", type=str, default="lprnet_plate.pt")
    ap.add_argument("--onnx", type=str, default="", help="导出 ONNX 的路径")
    args = ap.parse_args()

    if not TORCH_AVAILABLE:
        print("=" * 74)
        print("未检测到 PyTorch，路线 B（LPRNet）暂时无法运行。")
        print()
        print("原因：当前解释器是 Python %d.%d。" % sys.version_info[:2])
        print("      PyTorch 官方尚未发布 Python 3.14 的 wheel。")
        print()
        print("解决办法（任选其一）：")
        print("  1) 用 Python 3.11/3.12 建虚拟环境后安装：")
        print("       py -3.12 -m venv .venv")
        print("       .venv\\Scripts\\activate")
        print("       pip install torch torchvision")
        print("       python lprnet_pytorch.py --train")
        print("  2) 暂时只跑路线 A（传统 DIP），它不需要 PyTorch：")
        print("       python demo.py --route dip")
        print("=" * 74)
        sys.exit(0)

    if args.train:
        model, cs = train(epochs=args.epochs, n_train=args.n_train,
                          n_val=args.n_val)
        torch.save({"state_dict": model.state_dict(),
                    "charset": cs.chars}, args.save)
        print(f"模型已保存：{args.save}")
        if args.onnx:
            export_onnx(model, args.onnx)
    else:
        print("用法：python lprnet_pytorch.py --train")
