#!/usr/bin/env python3
"""审查用图像增强工具（几何不变，不改标签）。

视图分为两类：
1. 亮度/对比度增强：Lab CLAHE、强对比、灰阶 CLAHE、HSV-V
2. 结构/残差高亮：top-hat、black-hat、形态学叠加、边缘

逻辑对齐 Tools/ImageProcessTool，并补充颜色空间与形态学视图。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class EnhancePreset:
    name: str
    clahe_clip: float
    clahe_tile: int
    bilateral_d: int
    bilateral_sigma_color: float
    bilateral_sigma_space: float


PRESETS: dict[str, EnhancePreset] = {
    "enhanced": EnhancePreset(
        name="enhanced",
        clahe_clip=2.0,
        clahe_tile=8,
        bilateral_d=9,
        bilateral_sigma_color=75.0,
        bilateral_sigma_space=75.0,
    ),
    "strong": EnhancePreset(
        name="strong",
        clahe_clip=4.0,
        clahe_tile=8,
        bilateral_d=5,
        bilateral_sigma_color=50.0,
        bilateral_sigma_space=50.0,
    ),
}


# view_key -> (dir_suffix, short_label, help_text)
# dir is images_<key> except raw which may be images_raw
VIEW_CATALOG: dict[str, tuple[str, str, str]] = {
    "raw": ("raw", "原图", "原始 RGB"),
    "enhanced": ("enhanced", "Lab增强", "白平衡+Lab-L CLAHE clip=2"),
    "strong": ("strong", "强对比", "白平衡+Lab-L CLAHE clip=4"),
    "lab_l": ("lab_l", "Lab-L", "Lab 亮度通道 CLAHE 伪彩"),
    "hsv_v": ("hsv_v", "HSV-V", "HSV 明度通道 CLAHE"),
    "hsv_s": ("hsv_s", "HSV-S", "HSV 饱和度（色差/污渍敏感）"),
    "gray": ("gray", "灰阶", "灰度 + CLAHE"),
    "tophat": ("tophat", "亮残差", "形态学 top-hat 亮缺陷候选"),
    "blackhat": ("blackhat", "暗残差", "形态学 black-hat 暗缺陷候选"),
    "morph": ("morph", "残差叠", "亮红+暗青 叠在灰阶上"),
    "edges": ("edges", "边缘", "Sobel 边缘幅值"),
    "gamma": ("gamma", "Gamma", "暗部提亮 gamma=0.5"),
}

# Default order for review UI (raw first)
DEFAULT_VIEW_KEYS = [
    "raw",
    "enhanced",
    "strong",
    "lab_l",
    "hsv_v",
    "hsv_s",
    "gray",
    "tophat",
    "blackhat",
    "morph",
    "edges",
    "gamma",
]


def imread_unicode(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".png"
    if ext not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
        ext = ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def apply_white_balance(image: np.ndarray) -> np.ndarray:
    xphoto = getattr(cv2, "xphoto", None)
    if xphoto is not None and hasattr(xphoto, "createSimpleWhiteBalance"):
        return xphoto.createSimpleWhiteBalance().balanceWhite(image)

    result = image.astype(np.float32)
    channel_means = result.reshape(-1, 3).mean(axis=0)
    gray_mean = float(channel_means.mean())
    scale = gray_mean / np.maximum(channel_means, 1.0)
    result *= scale
    return np.clip(result, 0, 255).astype(np.uint8)


def to_bgr(gray_or_color: np.ndarray) -> np.ndarray:
    if gray_or_color.ndim == 2:
        return cv2.cvtColor(gray_or_color, cv2.COLOR_GRAY2BGR)
    return gray_or_color


def clahe_gray(gray: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def enhance_image(image: np.ndarray, preset: EnhancePreset | str = "enhanced") -> np.ndarray:
    if isinstance(preset, str):
        preset = PRESETS[preset]
    white_balanced = apply_white_balance(image)
    lab = cv2.cvtColor(white_balanced, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)

    if preset.bilateral_d > 0:
        lightness = cv2.bilateralFilter(
            lightness,
            preset.bilateral_d,
            preset.bilateral_sigma_color,
            preset.bilateral_sigma_space,
        )

    lightness = clahe_gray(lightness, preset.clahe_clip, preset.clahe_tile)
    enhanced_lab = cv2.merge([lightness, channel_a, channel_b])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def view_lab_l(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(apply_white_balance(image), cv2.COLOR_BGR2LAB)
    l_ch = clahe_gray(lab[:, :, 0], clip=3.0, tile=8)
    # TURBO makes local contrast easier to see for humans
    color = cv2.applyColorMap(l_ch, cv2.COLORMAP_TURBO)
    return color


def view_hsv_v(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(apply_white_balance(image), cv2.COLOR_BGR2HSV)
    v = clahe_gray(hsv[:, :, 2], clip=3.0, tile=8)
    return to_bgr(v)


def view_hsv_s(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(apply_white_balance(image), cv2.COLOR_BGR2HSV)
    s = clahe_gray(hsv[:, :, 1], clip=2.0, tile=8)
    # high saturation regions often related to stain/oxidation tint
    color = cv2.applyColorMap(s, cv2.COLORMAP_MAGMA)
    return color


def view_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(apply_white_balance(image), cv2.COLOR_BGR2GRAY)
    return to_bgr(clahe_gray(gray, clip=3.0, tile=8))


def _morph_kernel(size: int = 21) -> np.ndarray:
    k = size if size % 2 == 1 else size + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def view_tophat(image: np.ndarray, kernel_size: int = 21) -> np.ndarray:
    gray = cv2.cvtColor(apply_white_balance(image), cv2.COLOR_BGR2GRAY)
    kernel = _morph_kernel(kernel_size)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    tophat = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    tophat = clahe_gray(tophat, clip=2.0, tile=8)
    return cv2.applyColorMap(tophat, cv2.COLORMAP_HOT)


def view_blackhat(image: np.ndarray, kernel_size: int = 21) -> np.ndarray:
    gray = cv2.cvtColor(apply_white_balance(image), cv2.COLOR_BGR2GRAY)
    kernel = _morph_kernel(kernel_size)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blackhat = clahe_gray(blackhat, clip=2.0, tile=8)
    return cv2.applyColorMap(blackhat, cv2.COLORMAP_OCEAN)


def view_morph_overlay(image: np.ndarray, kernel_size: int = 21) -> np.ndarray:
    """Bright residual=red, dark residual=cyan on CLAHE gray base."""
    gray = cv2.cvtColor(apply_white_balance(image), cv2.COLOR_BGR2GRAY)
    base = to_bgr(clahe_gray(gray, clip=2.0, tile=8))
    kernel = _morph_kernel(kernel_size)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    # adaptive-ish thresholds
    thr_t = max(8, int(np.percentile(tophat, 97)))
    thr_b = max(8, int(np.percentile(blackhat, 97)))
    bright = tophat >= thr_t
    dark = blackhat >= thr_b
    out = base.copy()
    out[bright] = (0, 0, 255)  # BGR red
    out[dark] = (255, 255, 0)  # cyan
    return out


def view_edges(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(apply_white_balance(image), cv2.COLOR_BGR2GRAY)
    gray = clahe_gray(gray, clip=2.0, tile=8)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(mag, cv2.COLORMAP_INFERNO)


def view_gamma(image: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    """gamma < 1 brightens dark regions."""
    inv = 1.0 / max(gamma, 1e-6)
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(apply_white_balance(image), table)


VIEW_BUILDERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "enhanced": lambda im: enhance_image(im, "enhanced"),
    "strong": lambda im: enhance_image(im, "strong"),
    "lab_l": view_lab_l,
    "hsv_v": view_hsv_v,
    "hsv_s": view_hsv_s,
    "gray": view_gray,
    "tophat": view_tophat,
    "blackhat": view_blackhat,
    "morph": view_morph_overlay,
    "edges": view_edges,
    "gamma": view_gamma,
}


def build_views(
    image: np.ndarray,
    view_keys: Iterable[str] | None = None,
) -> dict[str, np.ndarray]:
    keys = list(view_keys) if view_keys is not None else list(DEFAULT_VIEW_KEYS)
    out: dict[str, np.ndarray] = {}
    for key in keys:
        if key == "raw":
            out[key] = image
            continue
        builder = VIEW_BUILDERS.get(key)
        if builder is None:
            raise KeyError(f"Unknown view key: {key}")
        out[key] = builder(image)
    return out


def mean_brightness(image: np.ndarray) -> float:
    if image.ndim == 2:
        return float(image.mean())
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def estimate_mean_brightness(path: Path) -> float | None:
    """Brightness estimate via reduced grayscale when available."""
    reduced = imread_unicode(path, flags=getattr(cv2, "IMREAD_REDUCED_GRAYSCALE_8", 64))
    if reduced is None:
        image = imread_unicode(path)
        if image is None:
            return None
        return mean_brightness(image)
    return float(reduced.mean())


def brightness_bucket(mean_luma: float) -> str:
    if mean_luma < 40:
        return "dark"
    if mean_luma < 100:
        return "mid"
    return "bright"


def conf_bucket(min_low_conf: float) -> str:
    if min_low_conf < 0.25:
        return "very_low"
    return "low"


def area_bucket(mean_area: float) -> str:
    if mean_area < 0.001:
        return "tiny"
    if mean_area < 0.01:
        return "small"
    return "normal"


@dataclass
class BoxRec:
    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    confidence: float | None

    @property
    def area(self) -> float:
        return self.w * self.h


def read_boxes(label_path: Path) -> list[BoxRec]:
    if not label_path.is_file():
        return []
    boxes: list[BoxRec] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) not in (5, 6):
            continue
        conf = float(parts[5]) if len(parts) == 6 else None
        boxes.append(
            BoxRec(
                class_id=int(parts[0]),
                cx=float(parts[1]),
                cy=float(parts[2]),
                w=float(parts[3]),
                h=float(parts[4]),
                confidence=conf,
            )
        )
    return boxes


def low_conf_boxes(boxes: Iterable[BoxRec], threshold: float = 0.40) -> list[BoxRec]:
    return [b for b in boxes if b.confidence is not None and b.confidence < threshold]


def process_one_image(
    src: Path,
    out_root: Path,
    view_keys: Iterable[str] | None = None,
    stem: str | None = None,
    suffix: str | None = None,
    copy_raw_bytes: bool = True,
) -> dict:
    """Write images_<view>/ for requested views. Returns brightness stats."""
    image = imread_unicode(src)
    if image is None:
        raise RuntimeError(f"Failed to read image: {src}")

    stem = stem or src.stem
    suffix = suffix or src.suffix
    keys = list(view_keys) if view_keys is not None else list(DEFAULT_VIEW_KEYS)
    views = build_views(image, keys)

    for key, img in views.items():
        dir_name = f"images_{VIEW_CATALOG.get(key, (key, key, ''))[0]}"
        dst = out_root / dir_name / f"{stem}{suffix}"
        if key == "raw" and copy_raw_bytes:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                dst.write_bytes(src.read_bytes())
            continue
        if not imwrite_unicode(dst, img):
            raise RuntimeError(f"Failed to write view {key}: {dst}")

    luma = mean_brightness(image)
    return {
        "mean_brightness": round(luma, 2),
        "brightness_bucket": brightness_bucket(luma),
        "views": keys,
    }


# Back-compat wrapper used by older prepare script
def process_one_image_legacy(
    src: Path,
    raw_dst: Path,
    enhanced_dst: Path,
    strong_dst: Path,
    copy_raw: bool = True,
) -> dict:
    image = imread_unicode(src)
    if image is None:
        raise RuntimeError(f"Failed to read image: {src}")
    stats = {
        "mean_brightness": round(mean_brightness(image), 2),
        "brightness_bucket": brightness_bucket(mean_brightness(image)),
    }
    if copy_raw:
        raw_dst.parent.mkdir(parents=True, exist_ok=True)
        if not raw_dst.exists():
            raw_dst.write_bytes(src.read_bytes())
    else:
        imwrite_unicode(raw_dst, image)
    imwrite_unicode(enhanced_dst, enhance_image(image, "enhanced"))
    imwrite_unicode(strong_dst, enhance_image(image, "strong"))
    return stats


def stratified_sample_indices(
    strata: list[str],
    n: int,
    seed: int,
) -> list[int]:
    rng = np.random.default_rng(seed)
    by_stratum: dict[str, list[int]] = {}
    for idx, key in enumerate(strata):
        by_stratum.setdefault(key, []).append(idx)
    for key in by_stratum:
        rng.shuffle(by_stratum[key])

    keys = sorted(by_stratum.keys())
    selected: list[int] = []
    pos = {k: 0 for k in keys}
    while len(selected) < n and any(pos[k] < len(by_stratum[k]) for k in keys):
        for k in keys:
            if pos[k] < len(by_stratum[k]):
                selected.append(by_stratum[k][pos[k]])
                pos[k] += 1
                if len(selected) >= n:
                    break
    return selected


def discover_view_dirs(dataset: Path) -> list[str]:
    """Return view keys that have images_<key> directories with files."""
    found: list[str] = []
    for key, (suffix, _label, _help) in VIEW_CATALOG.items():
        d = dataset / f"images_{suffix}"
        if d.is_dir() and any(d.iterdir()):
            found.append(key)
    # also accept any images_* not in catalog
    for d in sorted(dataset.glob("images_*")):
        if not d.is_dir():
            continue
        key = d.name[len("images_") :]
        if key not in found and any(d.iterdir()):
            found.append(key)
    # keep catalog order first
    ordered = [k for k in DEFAULT_VIEW_KEYS if k in found]
    for k in found:
        if k not in ordered:
            ordered.append(k)
    return ordered


# ---------------------------------------------------------------------------
# Box-level ROI helpers (for hard-to-see tiny / low-conf boxes)
# ---------------------------------------------------------------------------

ROI_PANEL_KEYS = [
    "raw",
    "lab",
    "gray",
    "gamma",
    "tophat",
    "blackhat",
    "edges",
    "local_eq",
]


def yolo_to_xyxy(
    box: BoxRec,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    x1 = int(round((box.cx - box.w / 2) * img_w))
    y1 = int(round((box.cy - box.h / 2) * img_h))
    x2 = int(round((box.cx + box.w / 2) * img_w))
    y2 = int(round((box.cy + box.h / 2) * img_h))
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(x1 + 1, min(img_w, x2))
    y2 = max(y1 + 1, min(img_h, y2))
    return x1, y1, x2, y2


def expand_xyxy(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    img_w: int,
    img_h: int,
    pad_ratio: float = 1.5,
    min_size: int = 64,
) -> tuple[int, int, int, int]:
    """Expand box with padding; ensure minimum crop size for tiny boxes."""
    bw = x2 - x1
    bh = y2 - y1
    pad_x = max(int(bw * pad_ratio), (min_size - bw) // 2 if bw < min_size else 0)
    pad_y = max(int(bh * pad_ratio), (min_size - bh) // 2 if bh < min_size else 0)
    # also expand very thin boxes
    if bw < min_size:
        pad_x = max(pad_x, (min_size - bw + 1) // 2)
    if bh < min_size:
        pad_y = max(pad_y, (min_size - bh + 1) // 2)
    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y)
    nx2 = min(img_w, x2 + pad_x)
    ny2 = min(img_h, y2 + pad_y)
    return nx1, ny1, nx2, ny2


def view_local_eq(image: np.ndarray) -> np.ndarray:
    """Strong local contrast on ROI: CLAHE on L with high clip."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a, b = cv2.split(lab)
    l_ch = clahe_gray(l_ch, clip=6.0, tile=4)
    return cv2.cvtColor(cv2.merge([l_ch, a, b]), cv2.COLOR_LAB2BGR)


def build_roi_panels(roi_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Apply review transforms on a small ROI (already cropped)."""
    return {
        "raw": roi_bgr.copy(),
        "lab": enhance_image(roi_bgr, "strong"),
        "gray": view_gray(roi_bgr),
        "gamma": view_gamma(roi_bgr, gamma=0.45),
        "tophat": view_tophat(roi_bgr, kernel_size=max(9, (min(roi_bgr.shape[:2]) // 8) | 1)),
        "blackhat": view_blackhat(roi_bgr, kernel_size=max(9, (min(roi_bgr.shape[:2]) // 8) | 1)),
        "edges": view_edges(roi_bgr),
        "local_eq": view_local_eq(roi_bgr),
    }


def structure_score(roi_bgr: np.ndarray) -> dict[str, float]:
    """Heuristic scores: low structure often means empty / noise box."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    # center crop ~50% to reduce pad context
    h, w = gray.shape
    y1, y2 = h // 4, h - h // 4
    x1, x2 = w // 4, w - w // 4
    core = gray[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else gray
    gx = cv2.Sobel(core, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(core, cv2.CV_32F, 0, 1, ksize=3)
    edge = float(np.mean(np.sqrt(gx * gx + gy * gy)))
    std = float(core.std())
    # local range
    blur = cv2.GaussianBlur(core, (0, 0), 1.2)
    high = float(np.mean(np.abs(core.astype(np.float32) - blur.astype(np.float32))))
    return {
        "edge_mean": round(edge, 3),
        "std": round(std, 3),
        "highfreq": round(high, 3),
        # composite 0~1-ish after soft norm (not calibrated)
        "structure": round(min(1.0, (edge / 40.0 + std / 40.0 + high / 20.0) / 3.0), 4),
    }


def draw_box_on_roi(
    roi_bgr: np.ndarray,
    box_xyxy_in_full: tuple[int, int, int, int],
    crop_xyxy: tuple[int, int, int, int],
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    out = roi_bgr.copy()
    cx1, cy1, cx2, cy2 = crop_xyxy
    bx1, by1, bx2, by2 = box_xyxy_in_full
    x1 = bx1 - cx1
    y1 = by1 - cy1
    x2 = bx2 - cx1
    y2 = by2 - cy1
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


def make_montage(
    panels: dict[str, np.ndarray],
    panel_order: list[str] | None = None,
    cell_size: int = 320,
    labels: dict[str, str] | None = None,
    cols: int = 4,
) -> np.ndarray:
    order = panel_order or list(panels.keys())
    labels = labels or {}
    cells: list[np.ndarray] = []
    for key in order:
        img = panels[key]
        img = to_bgr(img)
        cell = cv2.resize(img, (cell_size, cell_size), interpolation=cv2.INTER_NEAREST)
        # label bar
        cv2.rectangle(cell, (0, 0), (cell_size, 28), (0, 0, 0), -1)
        text = labels.get(key, key)
        cv2.putText(
            cell,
            text,
            (6, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cells.append(cell)

    rows = (len(cells) + cols - 1) // cols
    while len(cells) < rows * cols:
        cells.append(np.zeros((cell_size, cell_size, 3), dtype=np.uint8))
    grid_rows = []
    for r in range(rows):
        row = np.hstack(cells[r * cols : (r + 1) * cols])
        grid_rows.append(row)
    return np.vstack(grid_rows)


def crop_box_roi(
    image: np.ndarray,
    box: BoxRec,
    pad_ratio: float = 1.5,
    min_size: int = 96,
) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Return (roi, box_xyxy, crop_xyxy)."""
    h, w = image.shape[:2]
    box_xyxy = yolo_to_xyxy(box, w, h)
    crop_xyxy = expand_xyxy(*box_xyxy, w, h, pad_ratio=pad_ratio, min_size=min_size)
    x1, y1, x2, y2 = crop_xyxy
    roi = image[y1:y2, x1:x2].copy()
    return roi, box_xyxy, crop_xyxy
