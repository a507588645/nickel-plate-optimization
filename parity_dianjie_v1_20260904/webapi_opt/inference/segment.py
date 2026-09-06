"""镍条分割与镍板判断业务逻辑。"""
from __future__ import annotations

import logging

from app_settings import get_app_settings
from inference import models as model_state
from segment.segpredict import infer


logger = logging.getLogger("jinchuan.inference")
BUSINESS_RULES = get_app_settings().business_rules


def nietiao_segment(image_path: str) -> dict:
    """原 /nietiaoSeg 业务逻辑。"""
    x_range = BUSINESS_RULES.nietiao_seg.x_range
    y_range = BUSINESS_RULES.nietiao_seg.y_range
    flag = True

    try:
        width, height, angle, center_x, center_y, box = infer(
            model=model_state.model,
            stride=model_state.stride,
            pt=model_state.pt,
            names=model_state.names,
            source=image_path,
        )
        if width == -1:
            return {"code": 0, "msg": "未检测到镍条", "data": {}}

        x1, y1 = box[0]
        x2, y2 = box[1]
        x3, y3 = box[2]
        x4, y4 = box[3]
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        x3 = int(x3)
        y3 = int(y3)
        x4 = int(x4)
        y4 = int(y4)

        for point in box:
            x, y = point
            if not (x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1]):
                flag = False
                break

        if flag:
            if width > BUSINESS_RULES.nietiao_seg.min_size or height > BUSINESS_RULES.nietiao_seg.min_size:
                result = {
                    "width": width,
                    "height": height,
                    "angle": angle,
                    "center_x": center_x,
                    "center_y": center_y,
                    "point1": [x1, y1],
                    "point2": [x2, y2],
                    "point3": [x3, y3],
                    "point4": [x4, y4],
                }
                return {"code": 1, "msg": "success", "data": result}
            else:
                return {"code": 0, "msg": "镍条不完整", "data": {}}
        else:
            return {"code": 0, "msg": "镍条位于抓取区域外", "data": {}}

    except Exception:
        logger.exception("nietiao_seg_failed")
        return {"code": 0, "msg": "nietiao segmentation failed", "data": {}}


def is_nieban(image_path: str) -> dict:
    """原 /isNieban 业务逻辑。"""
    try:
        width, height, angle, center_x, center_y, rect = infer(
            model=model_state.model,
            stride=model_state.stride,
            pt=model_state.pt,
            names=model_state.names,
            source=image_path,
        )
        if width == -1:
            return {"code": 1, "msg": "未检测到镍板", "data": {"type": 1}}
        return {"code": 1, "msg": "success", "data": {"type": 0}}
    except Exception:
        logger.exception("is_nieban_failed")
        return {"code": 0, "msg": "is nieban failed", "data": {}}
