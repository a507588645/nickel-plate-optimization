# ryw.pt 检查 union 原标签准确性报告（2026-07-23）

> 用途：L1 汇总集划分前的标签质量摸底。**注意：ryw.pt 不能当裁判**——它的伪标签线刚被弃用，
> 本报告量化的是「原标签 ↔ ryw.pt」的**分歧**，分歧本身不等于原标签错误，最终裁决靠人工复核。

- 数据：`union_original_labels_v1` 有标签图 1455 张（dianjienie 1335 按序映射 + labels_original 120）。
- 推理：ryw.pt，imgsz=1024，conf 下限 0.05，NMS IoU=0.45，GPU。`predictions.csv` 全量保存。
- 主口径：同类 id 匹配，conf≥0.25，IoU≥0.3。

## 1. 各类召回分层（原标签框被 ryw.pt 同类预测覆盖比例）

| 层 | 类 | recall（c0.25/i0.3） | 解读 |
| --- | --- | --- | --- |
| 盲区（≈0） | **yellowstain 0/143**、fawu 0/6、NiOH 0/12、heibian 0/12、mijixiaokeng 0/33 | 教师完全不认（电解镍教师 vs 电积镍域差 + 稀少类退化）；这些类的原标签只能靠人工 |
| 低（<0.25） | overfire 0.038、greenstain 0.090、airhole 0.131、bubble 0.134、watermark 0.170、suanwuran 0.216 | 严重分歧，复核重点 |
| 中（0.46~0.60） | quejiao 0.537、waterstain 0.536、wuzi 0.503、nodule 0.488、bump 0.464 | 半一致：框画法风格差异（大区域 vs 小碎块）为主因之一 |

- IoU 0.1 放宽后提升有限（如 waterstain 0.536→0.557、airhole 0.131→0.152）→ **低召回不是口径太严，是真分歧**。
- conf 0.15 与 0.25 几乎无差 → 不是置信度截断问题。

## 2. 分歧方向

1. **教师漏检原标签**：混淆对中「无预测」占主导（waterstain 4173/9150 框无预测、watermark 2143、airhole 1606、bubble 856、overfire 499）。
2. **教师多报**：conf≥0.25 且与原框 IoU<0.1 的额外预测 waterstain 7876、wuzi 4579、bump 750、watermark 215——原标注对密集缺陷只标了显著部分（欠标）或教师过火，两者皆有。
3. **类间串扰**：watermark↔waterstain（131+103）、wuzi↔watermark（89）、airhole→bubble（61）。

## 3. 结论（对 L1 的含义）

1. **原标签仍是主真值**：ryw.pt 与原标注是双向系统性分歧，不能用 ryw.pt 批量否决原标签（循环论证，AGENTS.md §8）。
2. **原标签也不能全信**：欠标（密集缺陷只标部分）很可能存在，worst_images.csv TOP50 是首批人工复核对象。
3. **教师盲区类（yellowstain/fawu/NiOH/heibian/mijixiaokeng/overfire/greenstain）的检查只能纯人工**，ryw.pt 对这些类无检查能力。
4. 后续补充标注的正确姿势：ryw.pt 的额外预测只作**候选**，人工确认后再补入，不自动并入。

## 4. 产物

- `predictions.csv`：1455 张全部预测（conf≥0.05）。
- `per_class_recall.csv`：conf×IoU 全网格召回。
- `confusion_pairs.csv`：原类→预测类（含无预测）配对计数。
- `worst_images.csv`：分歧最大 TOP50（score=漏检原框+0.5×额外框）。
- `summary.json`：主口径汇总。

## 5. 复现

```text
E:\PythonProject\ImetricCodes\JinchuanYolov5sTrain\venv\Scripts\python.exe check_union_labels_with_ryw.py
# 仅分析（不重推理）：加 --analyze-only
```
