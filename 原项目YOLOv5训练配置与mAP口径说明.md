# 原项目 YOLOv5 训练配置与 mAP 口径说明

| 项 | 内容 |
| --- | --- |
| 日期 | 2026-07-19 |
| 状态 | 已落盘结论（只读审计，未改训练代码） |
| 依据 | `JinchuanYolov5sTrain`、`ryw.pt` 权重元数据、`val.py`、Ultralytics 默认配置 |

---

## 1. 核心结论（先看这个）

1. **现场 `ryw.pt` 的 hyp 基本是 YOLOv5 官方默认 `hyp.scratch-low`，不是深度搜参结果。**
2. **项目改的是入口默认值（数据 / cfg / weights / epochs / patience）和模型 yaml 的 `nc`，不是私有神级超参。**
3. **项目检测验证口径被改松：`val.py` 使用 `linspace(0.1, 0.55, 10)`，却仍打印成 mAP50 / mAP50-95。**
4. 因此：
   - 报表 **mAP50** ≈ 实际 **mAP@0.1**
   - 报表 **mAP50-95** ≈ 实际 **mAP@0.1:0.55**
   - **不是** 标准 COCO/YOLO 的 mAP@0.5 与 mAP@0.5:0.95
5. 这能解释：同一任务上「官方 YOLOv5 / YOLO11 标准验证」数字往往明显低于项目 `train.py` 日志。
6. 同仓库分割验证仍是标准：`segment/val.py` 使用 `linspace(0.5, 0.95, 10)`。

---

## 2. 现场权重 `ryw.pt` 的真实训练配置

来源：权重内嵌 `opt` / `hyp`  
路径：

- `E:\PythonProject\ImetricCodes\JinchuanPythonWebapi\ryw.pt`
- `E:\PythonProject\Tools\ModelAutoLabelTool\weights\yolov5\ryw.pt`

日期字段：`2026-03-27`

| 项 | 值 | 是否默认/说明 |
| --- | --- | --- |
| 初始化权重 | `yolov5s.pt` | 官方 COCO 预训练 |
| 结构配置 | `models/yolov5m.yaml` | **文件名 m，实际 depth=0.33 / width=0.50，规模≈YOLOv5s** |
| 参数量 | ≈7.07M | 不是标准 YOLOv5m（≈21M） |
| 数据 | `data/nieban1100_xiangmu.yaml` | 电解镍项目集，18 类 |
| epochs | 300 | 项目入口默认 |
| batch | **32** | 比当前 `train.py` 默认 16 更大 |
| imgsz | 640 | 默认 |
| optimizer | SGD | 默认 |
| hyp | `hyp.scratch-low` 全套 | **官方 low-aug 模板** |
| patience | 100 | 默认 |
| seed | 0 | 默认 |
| cos_lr | False | 线性衰减 |
| multi_scale | False | 默认关 |
| image_weights | False | 默认关 |
| label_smoothing | 0.0 | 默认关 |
| 损失 | 标准 CIoU box loss | `loss.py` 中 NWD 为注释，**未启用** |

### hyp.scratch-low 关键数值

| 类别 | 参数 | 值 |
| --- | --- | ---: |
| 学习率 | lr0 | 0.01 |
| | lrf | 0.01 |
| 优化器 | momentum | 0.937 |
| | weight_decay | 0.0005 |
| warmup | warmup_epochs | 3.0 |
| | warmup_momentum | 0.8 |
| | warmup_bias_lr | 0.1 |
| 损失 | box | 0.05 |
| | cls | 0.5 |
| | obj | 1.0 |
| 匹配 | iou_t | 0.20 |
| | anchor_t | 4.0 |
| | fl_gamma | 0.0 |
| 增强 | hsv_h/s/v | 0.015 / 0.7 / 0.4 |
| | translate | 0.1 |
| | scale | 0.5 |
| | fliplr | 0.5 |
| | degrees/shear/perspective/flipud | 0 |
| | mosaic | 1.0 |
| | mixup | 0.0 |
| | copy_paste | 0.0 |

结论：**超参本身几乎是官方默认；工程入口与数据被项目化。**

---

## 3. 项目 `train.py` 默认入口

路径：`E:\PythonProject\ImetricCodes\JinchuanYolov5sTrain\train.py`

| 参数 | 默认 |
| --- | --- |
| --weights | `yolov5s.pt` |
| --cfg | `models/yolov5m.yaml` |
| --data | `data/nieban1100_xiangmu.yaml`（电解）；电积注释为 `DianJiNieBan.yaml` |
| --hyp | `data/hyps/hyp.scratch-low.yaml` |
| --epochs | 300 |
| --batch-size | 16 |
| --imgsz | 640 |
| --optimizer | SGD |
| --patience | 100 |
| --seed | 0 |

当前训练工程工作区的 `models/yolov5m.yaml` 顶部为电积 5 类注释且：

```yaml
nc: 5
depth_multiple: 0.33
width_multiple: 0.50
```

即：**仍是 s 规模系数**，不是标准 m。

重要边界：该训练工程工作区存在用户未提交改动；`nc: 5` 相对 Git 已提交版本的 `nc: 18` 是本地修改，`models/yolov5m_dianjinie.yaml` 也是未跟踪文件。因此它们只能说明当前拟实验配置，**不能**作为 `ryw.pt` 的历史训练依据。`ryw.pt` 的内嵌模型 YAML 已直接确认 `nc: 18`、`depth_multiple: 0.33`、`width_multiple: 0.50`。

另有未跟踪的 `models/yolov5m_dianjinie.yaml` 使用真 m 规模（0.67/0.75）；用于新实验前必须纳入版本控制，并且不要与 `ryw.pt` 混用。

---

## 4. 与 YOLO11m / Ultralytics 默认的差距

| 层级 | 原项目 / ryw | YOLO11 Ultralytics 默认（本机 8.4.90） |
| --- | --- | --- |
| 框架 | 经典 YOLOv5 | Ultralytics YOLO11 |
| hyp 来源 | 官方 scratch-low | 官方 v8/11 默认 |
| 是否深度调参 | 基本否 | 基本否（外层预算可改） |
| 模型规模 | ≈s（ryw≈7M） | yolo11m 为真 m |
| loss | box/obj/cls | box/cls/**dfl** |
| box 权重 | 0.05 | 7.5（**不同 loss 体系，不可直接抄数字**） |
| optimizer | SGD | auto（常落到 AdamW） |
| close_mosaic | 无此概念 | 默认最后 10 epoch 关 mosaic |
| nbs | 无 | 64 |
| 常见预算 | 300 ep / 640 | 用户脚本常 100 ep / 576 等 |

用户侧 `models/train_yolo.py` 通常只覆盖 epochs/imgsz/batch/seed/workers/patience，其余吃 Ultralytics 默认。

已有 YOLO11m 运行示例：

- 本地：`runs/detect/runs_yolo11/exp_dianjinie_yolo11m/args.yaml` → optimizer=auto, box=7.5, close_mosaic=10
- Kaggle 对齐尝试：`models/kaggle_outputs/yolo11m-resume-source/.../yolo11m/args.yaml` → optimizer=SGD, close_mosaic=0，但仍是 box=7.5/dfl=1.5

---

## 5. mAP 口径错误（最重要）

### 5.1 代码事实

`JinchuanYolov5sTrain/val.py`：

```python
iouv = torch.linspace(0.1, 0.55, 10, device=device)  # 注释仍写 mAP@0.5:0.95
```

官方 / Ultralytics：

```python
iouv = torch.linspace(0.5, 0.95, 10)
```

同仓库 `segment/val.py` 仍是标准 `0.5~0.95`。

该改动在仓库初始提交即存在；训练过程每轮 val 都走此逻辑。

### 5.2 报表含义对照

| 训练日志打印名 | 代码实际 | 标准官方含义 |
| --- | --- | --- |
| mAP50 | **mAP@0.1**（第一档 IoU=0.1） | mAP@**0.5** |
| mAP50-95 | **mAP@0.1:0.55**（0.10→0.55 十档平均） | mAP@**0.5:0.95** |

命名纠正：

- 可称第一列为 **mAP@0.1**（口语 mAP10）
- 第二列应称 **mAP@0.1:0.55**
- **不是** mAP@0.1:0.95，也不是标准 mAP50-95

### 5.3 为何会“官方 YOLOv5 完全比不上项目 train.py”

优先解释：

1. **尺子不同**：项目更松，标准更严。
2. **模型规模常对不齐**：ryw≈s；官方复现常误用真 yolov5m。
3. **数据/划分不同**：电解 18 类旧集 vs 当前 Dianjinie0211 5 类 / dianjienie 17 类。
4. **训练预算不同**：300ep/batch32 vs 更短训。
5. **入口不同**：经典 YOLOv5 train.py vs Ultralytics 封装。

hyp 本身不是主因。

### 5.4 为何有人可能用 0.1（推断，文档未写死）

可能动机（**待确认，非已证业务决策**）：

- 小目标多，严格 0.5 很难
- 标注框不贴边
- 业务更关心检出而非框精
- 曲线更好看

无论动机如何，**对外对比与模型选择必须用标准 0.5 / 0.5:0.95**。  
若业务要宽松检出，应另列 `mAP@0.1`，禁止继续冒充 mAP50。

---

## 6. 历史结果数字（注意口径）

### 6.1 项目侧旧实验日志（走宽松 val 的可能性高）

`out_xiangmu/baseline.txt` 等（约 363 张 val）：

| 实验 | 报表 mAP50 | 报表 mAP50-95 |
| --- | ---: | ---: |
| baseline | 0.569 | 0.500 |
| +NWD | 0.572 | 0.504 |
| BiFPN+NWD | 0.551 | 0.472 |

论文侧 `结果/yolov5m.txt`：0.457 / 0.383（另一套 12 类/数据版本）

### 6.2 近期 Dianjinie0211 项目代码重训参考（实际为 s 规模）

`runs/train/exp_dianjinie_yolov5m` 的首次运行确实指定了 `--weights yolov5m.pt`，但同时指定了 `--cfg models/yolov5m.yaml`。本项目 `train.py` 使用 `Model(cfg or ckpt['model'].yaml, ...)`，因此非空的 `cfg` 优先于官方权重内嵌 YAML。训练日志记录为 `Transferred 57/349 items from yolov5m.pt`，并打印 `7,033,114 parameters`；续训产物 `exp_dianjinie_yolov5m2/weights/best.pt` 内嵌 YAML 也是 `depth_multiple: 0.33`、`width_multiple: 0.50`、5 类。

结论：该新权重使用了**官方 YOLOv5m 预训练权重的部分可匹配参数**，但最终训练架构是本地同名 YAML 的 **s 规模**，不是未改结构的官方 YOLOv5m；实验名中的 `yolov5m` 具有误导性。其 `results.csv` 历史最佳为：

- 宽松 mAP@0.1 = 0.32188（epoch 114）
- 宽松 mAP@0.1:0.55 = 0.25949（epoch 114）

手册门槛暂记为 mAP50>0.3219、mAP50-95>0.2595。  
**在统一为标准 IoU 前，这些门槛不能当作与 YOLO11 官方 val 直接可比的目标，也不能标为官方 YOLOv5m 的结果。**

### 6.3 结构改造实验结论（项目集）

BiFPN / NWD / ASPP 等多为实验；项目集上 baseline 已很强，NWD 略升或持平，BiFPN 可能掉点。  
**上线 `ryw.pt` 仍是标准检测结构 + 标准 CIoU，未启用 NWD。**

---

## 7. 不要混淆的权重

| 权重 | 规模 | 数据 | 用途 |
| --- | --- | --- | --- |
| **ryw.pt（现场）** | ≈s，18 类 | nieban1100_xiangmu | 电解镍生产检测 |
| `exp_dianjinie_yolov5m2/weights/best.pt` | ≈s，5 类 | Dianjinie0211 | 新训练；由 `yolov5m.pt` 部分迁移初始化，实际使用本地 s 规模 YAML |
| baseline.pt | 真 m，12 类 | nieban1000_article | 论文/基线实验 |
| JinchuanYolov5sTrain/best.pt | ASPPv3 改造，12 类 | 同上 | 结构改造实验 |
| 预训练 yolov5m.pt | 真 m COCO | — | 初始化用，不是业务 best |

---

## 8. 正确对比清单

### 必须统一

1. 同一 `data.yaml` 与划分
2. 同一 `imgsz`
3. 同一验证入口与 **IoU 向量**
4. 同一模型规模（s 就比 s，m 就比 m）
5. 标准指标名：`mAP@0.5`、`mAP@0.5:0.95`

### 建议先做的最小验证

把 `val.py` 临时改回：

```python
iouv = torch.linspace(0.5, 0.95, 10, device=device)
```

对同一 `best.pt` / `ryw.pt` 重跑 val，记录：

| 权重 | 宽松 mAP@0.1 | 宽松 mAP@0.1:0.55 | 标准 mAP@0.5 | 标准 mAP@0.5:0.95 |
| --- | ---: | ---: | ---: | ---: |
| ... |  |  |  |  |

差值即“虚高幅度”。

### 若要尽量复现 ryw 训练日程

```bash
python train.py \
  --weights yolov5s.pt \
  --cfg models/yolov5m.yaml \
  --data <同一份yaml> \
  --hyp data/hyps/hyp.scratch-low.yaml \
  --epochs 300 \
  --batch-size 32 \
  --imgsz 640 \
  --optimizer SGD \
  --patience 100 \
  --seed 0
```

并确认 yaml 为 `depth_multiple: 0.33`、`width_multiple: 0.50`。

### 对齐到 YOLO11 时

可对齐：epochs/imgsz/patience/seed、SGD、lr0/lrf/momentum/weight_decay、mosaic/mixup、close_mosaic=0。  
**不要**把 YOLO11 的 `box` 从 7.5 改成 0.05 当作“复现 ryw”——loss 定义不同。

---

## 9. 对当前优化工作的含义

1. 商业约束若保持 `ryw.pt` 不变：优化优先在阈值、规则、决策层；引用旧 mAP 时必须注明口径。
2. 若重训 YOLO11/26 并与项目 YOLOv5 比：先修/绕过宽松 val，否则比较无效。
3. 操作手册中基于 `exp_dianjinie_yolov5m2` 的门槛，应标为「项目 s 规模、宽松 val 历史参考」，待标准 val 后重定。
4. 未完成：尚未在本会话对 `ryw.pt` / 现有 best 用标准 IoU 重跑实测对照表。

---

## 10. 相关路径速查

| 用途 | 路径 |
| --- | --- |
| 训练工程 | `E:\PythonProject\ImetricCodes\JinchuanYolov5sTrain` |
| 训练入口 | `...\train.py` |
| 验证口径 | `...\val.py`（问题行） |
| 标准对照 | `...\segment\val.py` |
| hyp | `...\data\hyps\hyp.scratch-low.yaml` |
| 现场权重 | `...\JinchuanPythonWebapi\ryw.pt` |
| 近期重训 | `...\runs\train\exp_dianjinie_yolov5m2\` |
| 项目结果文本 | `...\out_xiangmu\`、`...\结果\` |
| 本说明 | `E:\PythonProject\优化\原项目YOLOv5训练配置与mAP口径说明.md` |
