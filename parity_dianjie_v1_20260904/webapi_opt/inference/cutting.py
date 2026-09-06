"""化线 / 切割业务逻辑。"""
from __future__ import annotations

import logging

from app_paths import debug_file_path
from app_settings import get_app_settings
from preprocess import (
    bioplist,
    cutmsg,
    findcent,
    jianbian,
    readarray,
    readarray_C,
    read_nocorner,
    readimg,
)


logger = logging.getLogger("jinchuan.inference")
APP_SETTINGS = get_app_settings()


def huaxian(image1: str, image2: str, json_file1: str, json_file2: str) -> dict:
    """原 /cutting 业务逻辑。"""
    try:
        img1 = readimg(image1)
        img2 = readimg(image2)

        result1 = findcent(img1)
        img1_x1 = int(result1[3][0] - result1[1] / 2)
        img1_y1 = int(result1[3][1] - result1[2] / 2)
        img1_x2 = int(result1[3][0] + result1[1] / 2)
        img1_y2 = int(result1[3][1] + result1[2] / 2)
        img1_edge = [img1_x1, img1_y1, img1_x2, img1_y2]

        result2 = findcent(img2)
        img2_x1 = int(result2[3][0] - result2[1] / 2)
        img2_y1 = int(result2[3][1] - result2[2] / 2)
        img2_x2 = int(result2[3][0] + result2[1] / 2)
        img2_y2 = int(result2[3][1] + result2[2] / 2)
        img2_edge = [img2_x1, img2_y1, img2_x2, img2_y2]

        # 以下步骤为预处理算法读取板面中心点坐标 center
        center1 = result1[3]
        center2 = result2[3]
        center1 = (512, 444)
        center2 = (512, 444)

        # 求取镍板宽度
        width1 = result1[1]
        width2 = result2[1]
        height1 = result1[2]
        height2 = result2[2]

        # 返回除水渍、污渍、结瘤以外的缺陷信息和结瘤信息 plist, nodule_info, ndtype
        nieban1_info = readarray(json_file1)
        nieban2_info = readarray(json_file2)
        nieban1_info_C = readarray_C(json_file1)
        nieban2_info_C = readarray_C(json_file2)
        nieban1_info_nocorner = read_nocorner(json_file1)
        nieban2_info_nocorner = read_nocorner(json_file2)

        defects1 = nieban1_info[0]
        defects2 = nieban2_info[0]
        defects1_C = nieban1_info_C
        defects2_C = nieban2_info_C
        nodule_info1 = nieban1_info[1]
        nodule_info2 = nieban2_info[1]

        nodule_edge1 = jianbian(nodule_info1, img1_edge)
        nodule_edge2 = jianbian(nodule_info2, img2_edge)
        nodule_edge = nodule_edge1 + nodule_edge2

        djtype = 5
        djtype1 = nieban1_info[2]
        djtype2 = nieban2_info[2]

        if djtype1 == 0 and djtype2 == 0:
            djtype = 0
        if djtype1 != 3 and djtype2 != 3:
            if djtype1 == 2 or djtype2 == 2:
                djtype = 2

        defects = bioplist(center1, center2, width1, width2, defects1, defects2)
        defects_C = bioplist(center1, center2, width1, width2, defects1_C, defects2_C)
        defects_nocorner = bioplist(
            center1, center2, width1, width2, nieban1_info_nocorner, nieban2_info_nocorner
        )
        fmsg = cutmsg(center1, defects, defects_C, nodule_edge, defects_nocorner, djtype)

        if APP_SETTINGS.runtime.debug_artifacts_enabled:
            with open(debug_file_path("bioplist.txt"), "w", encoding="utf-8") as f:
                f.write("plist1=" + str(defects1))
                f.write("\r\n")
                f.write("plist2=" + str(defects2))

        return {"code": 1, "msg": "success", "data": fmsg}

    except Exception:
        logger.exception("cutting_failed")
        return {"code": 0, "msg": "cutting failed", "data": {}}
