#!/usr/bin/env python3
"""
电解镍等级判定规则（从原项目 start.py 提取并封装）。

原项目逻辑：
- 检测模型：YOLOv5（ryw.pt），输入尺寸 1024×888，输出缺陷框坐标。
- 分类模型：ResNet（ryw.pth），2 类（bad=0 / good=1），但当前规则主要依赖缺陷比例。
- 输出 type：
    0 无板
    1 3号板
    2 烫洗板
    3 合格整版
    4 合格600
    5 合格不足600
    6 2号板

本模块不加载深度学习模型，仅对给定的检测框列表执行原规则，
便于在 WebAPI 中替换 start.py 内联代码、做回归测试或生成伪标签。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# 电解镍 YOLO 标签 → A/B/C/气孔 分组（与原项目 start.py 一致）
# A: 水印/污渍      B: 黄渍/气孔/结粒/疙瘩/绿色结晶/缺角/严重水渍
# C: 发乌/烧板/酸污染   Airhole: 气孔
A_LABELS = {0, 1}
B_LABELS = {2, 4, 5, 6, 10, 12, 13}
C_LABELS = {7, 8, 11}
AIRHOLE_LABELS = {4}
NODULE_LABEL = 3


@dataclass
class DianjieGradeResult:
    type_id: int
    type_name: str
    has_nodule: bool
    nodule_status: int  # 0 合格无结瘤 / 1 合格有结瘤 / 2 烫洗板 / 3 其他
    ratios: dict[str, float]
    defects: list[dict[str, Any]]


class DianjieGradeRule:
    """电解镍等级判定规则封装。"""

    TYPE_NAMES = {
        0: "无板",
        1: "3号板",
        2: "烫洗板",
        3: "合格整版",
        4: "合格600",
        5: "合格不足600",
        6: "2号板",
    }

    def __init__(
        self,
        nodule_area_threshold: float = 13000.0,
        nodule_aspect_threshold: float = 3.0,
        good_a_ratio: float = 0.2,
        good_b_ratio: float = 0.1,
        good_airhole_ratio: float = 0.01,
        good_c_ratio: float = 0.03,
        wash_b_ratio: float = 0.05,
        wash_c_ratio: float = 0.05,
        c_severe_ratio: float = 0.5,
        b_severe_ratio: float = 0.5,
    ):
        self.nodule_area_threshold = nodule_area_threshold
        self.nodule_aspect_threshold = nodule_aspect_threshold
        self.good_a_ratio = good_a_ratio
        self.good_b_ratio = good_b_ratio
        self.good_airhole_ratio = good_airhole_ratio
        self.good_c_ratio = good_c_ratio
        self.wash_b_ratio = wash_b_ratio
        self.wash_c_ratio = wash_c_ratio
        self.c_severe_ratio = c_severe_ratio
        self.b_severe_ratio = b_severe_ratio

    @staticmethod
    def _to_box(det: dict[str, Any]) -> tuple[float, float, float, float]:
        """从中心宽高转左上右下。"""
        cx, cy, w, h = det["cx"], det["cy"], det["w"], det["h"]
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    @staticmethod
    def _read_group(output: list[dict[str, Any]], labels: set[int]) -> list[list[int]]:
        """按标签组提取框坐标，并缩放到 4069×3554 坐标系（与原项目一致）。"""
        plist = []
        for d in output:
            if int(d["label"]) in labels:
                x1, y1, x2, y2 = DianjieGradeRule._to_box(d)
                plist.append(
                    [
                        int(x1 / 4),
                        int(y1 / 4),
                        int(x2 / 4),
                        int(y2 / 4),
                    ]
                )
        return plist

    @staticmethod
    def _scalculate(array: list[list[int]]) -> int:
        """缺陷像素面积（去重）。"""
        if not array:
            return 0
        # 用近似积分去重，与原项目一致
        canvas = np.zeros((4069, 3554), dtype=np.uint8)
        for x1, y1, x2, y2 in array:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(4069, x2), min(3554, y2)
            if x2 > x1 and y2 > y1:
                canvas[x1:x2, y1:y2] = 1
        return int(canvas.sum())

    def classify(
        self,
        output: list[dict[str, Any]],
        plate_area: float,
        predicted_idx: int | None = None,
    ) -> DianjieGradeResult:
        """
        根据检测框执行等级判定。

        Args:
            output: 检测框列表，每个元素为 dict(label, cx, cy, w, h)。
            plate_area: 镍板面积（1024×888 坐标系下的 width*height）。
            predicted_idx: 可选，2 类分类模型输出（bad=0/good=1），原规则仅在"有结瘤"分支使用。
        """
        flag_nodule = False
        filtered: list[dict[str, Any]] = []

        for det in output:
            label = int(det["label"])
            if label == NODULE_LABEL:
                _, _, w, h = det["cx"], det["cy"], det["w"], det["h"]
                ratio = max(h / w, w / h) if min(w, h) > 0 else 1.0
                area = w * h
                if area < self.nodule_area_threshold and ratio < self.nodule_aspect_threshold:
                    continue
                flag_nodule = True
            filtered.append(det)

        a_area = self._scalculate(self._read_group(filtered, A_LABELS))
        b_area = self._scalculate(self._read_group(filtered, B_LABELS))
        c_area = self._scalculate(self._read_group(filtered, C_LABELS))
        airhole_area = self._scalculate(self._read_group(filtered, AIRHOLE_LABELS))

        a_ratio = a_area / plate_area if plate_area > 0 else 0.0
        b_ratio = b_area / plate_area if plate_area > 0 else 0.0
        c_ratio = c_area / plate_area if plate_area > 0 else 0.0
        airhole_ratio = airhole_area / plate_area if plate_area > 0 else 0.0

        ratios = {
            "A_ratio": a_ratio,
            "B_ratio": b_ratio,
            "C_ratio": c_ratio,
            "Airhole_ratio": airhole_ratio,
        }

        type_id = 4  # 默认合格600
        nodule_status = 3

        # 合格整版
        if (
            not flag_nodule
            and a_ratio < self.good_a_ratio
            and b_ratio < self.good_b_ratio
            and airhole_ratio < self.good_airhole_ratio
            and c_ratio < self.good_c_ratio
        ):
            type_id = 3
            nodule_status = 0
        # 有结瘤且分类为 good -> 3号板
        elif flag_nodule and predicted_idx == 1:
            type_id = 1
            nodule_status = 1
        # 烫洗板
        elif (
            not flag_nodule
            and a_ratio >= self.good_a_ratio
            and b_ratio < self.wash_b_ratio
            and c_ratio < self.wash_c_ratio
        ):
            type_id = 2
            nodule_status = 2
        # C 类 > 50% -> 3号板
        elif c_ratio > self.c_severe_ratio:
            type_id = 1
            nodule_status = 3
        # B 类 > 50% 且 C 类 < 50% -> 2号板
        elif b_ratio > self.b_severe_ratio and c_ratio < self.c_severe_ratio:
            type_id = 6
            nodule_status = 3

        return DianjieGradeResult(
            type_id=type_id,
            type_name=self.TYPE_NAMES.get(type_id, "未知"),
            has_nodule=flag_nodule,
            nodule_status=nodule_status,
            ratios=ratios,
            defects=filtered,
        )


def dianjie_grade_from_xyxy(
    xyxy: np.ndarray,
    plate_area: float,
    predicted_idx: int | None = None,
    rule: DianjieGradeRule | None = None,
) -> DianjieGradeResult:
    """
    从 YOLOv5 xyxy 输出（numpy 数组，每行 [x1,y1,x2,y2,conf,cls]）直接判定等级。
    """
    output = []
    for det in xyxy:
        x1, y1, x2, y2, _, cls = det
        output.append(
            {
                "label": int(cls),
                "cx": float((x1 + x2) / 2),
                "cy": float((y1 + y2) / 2),
                "w": float(x2 - x1),
                "h": float(y2 - y1),
            }
        )
    rule = rule or DianjieGradeRule()
    return rule.classify(output, plate_area, predicted_idx)


if __name__ == "__main__":
    # 简单回归测试
    rule = DianjieGradeRule()
    # 模拟合格整版：无结瘤，A/B/C/气孔都很小
    result = rule.classify([], plate_area=100000)
    print("空板（默认）:", result)

    # 模拟 3号板：C 类占比大
    output = [{"label": 7, "cx": 512, "cy": 444, "w": 400, "h": 400}]
    result = rule.classify(output, plate_area=100000)
    print("C类大缺陷:", result)
