# D4 电积镍 detector business-output 对比报告

- 运行开始：`2026-08-28T02:42:22`。
- 墙钟耗时：`167.98` 秒。
- 评估图像：`216` 张；图像选择 stride=`1`，`stride=1` 表示全量去重集。
- 产物目录：`E:\PythonProject\优化\d4_dianji_business_compare_20260828_024222`。
- 本阶段仅分析，不切换 `models.yaml`，不调用生产 classifier；生产 grade `type` 按要求排除，只比较 defects 列表。

## 方法
- 当前模型：`weights/slf.pt`，YOLOv5 adapter，5 类，类序 `0=waterstain, 1=waterblock, 2=colorstain, 3=nodule, 4=bubble`，阈值 `0.25`。
- 候选模型：`weights/dianjienie_yolo11m_ryw18_640.onnx`，Ultralytics adapter，RYW18 18 类，`imgsz=640`，阈值 `0.25`。
- 两个 adapter 均接收同一张原始 BGR ndarray；没有把业务层的 `(1024, 888)` 副本传入 detector。框坐标按 adapter 契约解释为原图像素坐标。
- 每张图记录 detector business defect-list 的框数和所有框面积之和。面积比为 `sum(box_area_px) / image_area_px`，重叠框不做 union。
- 同名对照只使用已确认映射：`waterstain`（SLF 0 / RYW18 1）、`nodule`（3 / 3）、`bubble`（4 / 5）。`waterblock`、`colorstain` 不参与伪映射，候选其他 15 类单列为 candidate-only。
- 权重 SHA-256：`slf.pt=d15831042281747965d0a4f4eb2dd85c95b121e0112fbd90ea2e200cd8dbeb40`；`dianjienie_yolo11m_ryw18_640.onnx=901ccf6006f44689483aaf6f0d1d1b8c18132a85c7d881be7e62fcdccc255b21`。

### 验证集布局
- 数据集根目录：`E:\PythonProject\datasets\Dianjinie0211`。
- 图像枚举规则：只使用常见图像后缀，按文件名排序；`.npy` 不作为检测输入。
- `images/val`：216 个图像文件，目录总文件 432；扩展名统计：`.npy`=216, `.png`=216；标签目录 `labels/val`：216 个 txt 标签，含 `classes.txt`。
- `valid/images`：216 个图像文件，目录总文件 432；扩展名统计：`.npy`=216, `.png`=216；标签目录 `valid/labels`：216 个 txt 标签，含 `classes.txt`。
- 布局核对 `images/val` vs `valid/images`：共同文件 216，文件名集合相同=True，内容相同=True。
- 去重后实际评估 216 张独立 PNG/图像；同名标签缺失 0 张。标签只做布局记录，不参与本次 detector-output 比较。
- 发现但忽略的非图像文件：`.npy`=432。

## 总体结果

| 模型 | 平均 defect count | 检出图像数 | 图像级 detection rate |
|---|---:|---:|---:|
| `slf.pt` | 9.2083 | 215/216 | 99.54% |
| `yolo11m_640.onnx` | 6.9769 | 214/216 | 99.07% |
| candidate - slf | -2.2315 | - | -0.4630% |

### 差异分布

以下 signed delta 定义为 `candidate - slf`；面积 delta 的单位为 ratio。

| 指标 | median | mean | p90 | max |
|---|---:|---:|---:|---:|
| count delta | -2.0000 | -2.2315 | 4.0000 | 20.0000 |
| area-ratio delta | -0.008222 | 0.044209 | 0.362393 | 0.984740 |
| absolute count delta | 3.0000 | 4.4074 | 11.0000 | 21.0000 |
| absolute area-ratio delta | 0.037500 | 0.107225 | 0.396520 | 0.984740 |

### 已确认同名类别总数

| 类别 | SLF class id | RYW18 class id | SLF 总框数 | candidate 总框数 | candidate - SLF |
|---|---:|---:|---:|---:|---:|
| `waterstain` | 0 | 1 | 835 | 749 | -86 |
| `nodule` | 3 | 3 | 0 | 0 | +0 |
| `bubble` | 4 | 5 | 558 | 19 | -539 |

### SLF 类别总数

| SLF class id | 类别 | 总框数 | 生产输出 label |
|---:|---|---:|---:|
| 0 | `waterstain` | 835 | 1 |
| 1 | `waterblock` | 570 | 0 |
| 2 | `colorstain` | 26 | 2 |
| 3 | `nodule` | 0 | 3 |
| 4 | `bubble` | 558 | 5 |

### Candidate-only 类别总数

candidate-only 定义为 RYW18 class id 不在 `{1, 3, 5}` 的检测框。

| RYW18 class id | 类别 | 总框数 |
|---:|---|---:|
| 0 | `wuzi` | 294 |
| 2 | `yellowstain` | 0 |
| 4 | `airhole` | 239 |
| 6 | `bump` | 44 |
| 7 | `fawu` | 3 |
| 8 | `overfire` | 9 |
| 9 | `NiOH` | 0 |
| 10 | `greenstain` | 3 |
| 11 | `suanwuran` | 0 |
| 12 | `quejiao` | 0 |
| 13 | `watermark` | 146 |
| 14 | `youpao` | 0 |
| 15 | `heibian` | 1 |
| 16 | `mijixiaokeng` | 0 |
| 17 | `fanxitong` | 0 |

## Top 20 divergence images

排序分数严格为 `abs(count_delta) + abs(area_ratio_delta)`，其中 delta 均为 `candidate - slf`。审查图像见 [`review.html`](review.html)。

| rank | image | SLF count | candidate count | count delta | SLF area | candidate area | area delta | score | rendered |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `20230515153500820_0_2.png` | 34 | 13 | -21 | 11.06% | 2.79% | -8.27% | 21.082712 | `review_images/01_20230515153500820_0_2.png` |
| 2 | `20230519171742091_0_1.png` | 25 | 45 | +20 | 10.65% | 5.91% | -4.75% | 20.047452 | `review_images/02_20230519171742091_0_1.png` |
| 3 | `20230514161156186_0_2.png` | 26 | 6 | -20 | 5.68% | 2.05% | -3.63% | 20.036297 | `review_images/03_20230514161156186_0_2.png` |
| 4 | `113554top_0.png` | 21 | 2 | -19 | 1.75% | 52.54% | 50.78% | 19.507850 | `review_images/04_113554top_0.png` |
| 5 | `20230514160147721_0_2.png` | 23 | 5 | -18 | 9.04% | 0.55% | -8.49% | 18.084900 | `review_images/05_20230514160147721_0_2.png` |
| 6 | `20230519163356955_0_2.png` | 26 | 10 | -16 | 20.67% | 15.99% | -4.68% | 16.046754 | `review_images/06_20230519163356955_0_2.png` |
| 7 | `20230515163512376_0_2.png` | 2 | 17 | +15 | 0.53% | 5.87% | 5.34% | 15.053371 | `review_images/07_20230515163512376_0_2.png` |
| 8 | `113353top_0.png` | 20 | 6 | -14 | 3.66% | 2.65% | -1.02% | 14.010153 | `review_images/08_113353top_0.png` |
| 9 | `20230519154708145_0_2.png` | 18 | 5 | -13 | 7.72% | 5.95% | -1.77% | 13.017714 | `review_images/09_20230519154708145_0_2.png` |
| 10 | `20230515153624955_0_2.png` | 22 | 9 | -13 | 2.71% | 1.36% | -1.34% | 13.013443 | `review_images/10_20230515153624955_0_2.png` |
| 11 | `20230515163228460_0_2.png` | 2 | 14 | +12 | 0.87% | 6.90% | 6.03% | 12.060250 | `review_images/11_20230515163228460_0_2.png` |
| 12 | `20230519153621078_0_2.png` | 15 | 27 | +12 | 11.45% | 5.68% | -5.77% | 12.057676 | `review_images/12_20230519153621078_0_2.png` |
| 13 | `20230519154902446_0_2.png` | 21 | 9 | -12 | 9.77% | 12.03% | 2.26% | 12.022597 | `review_images/13_20230519154902446_0_2.png` |
| 14 | `20230519171823582_0_1.png` | 6 | 18 | +12 | 3.02% | 3.67% | 0.65% | 12.006483 | `review_images/14_20230519171823582_0_1.png` |
| 15 | `20230515153259296_0_2.png` | 15 | 4 | -11 | 15.84% | 0.19% | -15.65% | 11.156466 | `review_images/15_20230515153259296_0_2.png` |
| 16 | `20230519155231834_0_2.png` | 19 | 8 | -11 | 11.59% | 25.65% | 14.06% | 11.140590 | `review_images/16_20230519155231834_0_2.png` |
| 17 | `20230519160130067_0_2.png` | 16 | 5 | -11 | 16.25% | 6.83% | -9.42% | 11.094198 | `review_images/17_20230519160130067_0_2.png` |
| 18 | `20230514160756977_0_2.png` | 20 | 9 | -11 | 6.23% | 0.44% | -5.79% | 11.057912 | `review_images/18_20230514160756977_0_2.png` |
| 19 | `20230519164636511_0_2.png` | 25 | 36 | +11 | 12.73% | 17.85% | 5.12% | 11.051169 | `review_images/19_20230519164636511_0_2.png` |
| 20 | `20230519162601238_0_2.png` | 18 | 7 | -11 | 18.81% | 23.09% | 4.28% | 11.042810 | `review_images/20_20230519162601238_0_2.png` |

## 产物

- [`per_image.csv`](per_image.csv)：每张图的框数、面积比、delta 和指定类别计数。
- [`per_box.csv`](per_box.csv)：两模型所有原始框的 model/image/class id/confidence/xywh 及审计字段。
- [`review.html`](review.html)：无外部脚本或 CSS 依赖的 side-by-side 审查页，图像使用同目录相对路径。
- 已渲染审查比较：`20` 张；HTML 中每张比较对应一个 `<img>` 引用。

## 限制与闸门

- 本次没有 ground-truth comparison；标签仅用于核对目录布局，没有计算 precision、recall 或 mAP。
- `waterblock` 与 `colorstain` 没有确认的 RYW18 映射；它们保持为 SLF-only，不做猜测或合并。
- 全部检测使用单一 confidence threshold `0.25`，没有类别专用阈值或后处理调参扫描。
- 两个 detector 均为 CPU inference；耗时和数值表现不代表 GPU 部署性能。
- pass/fail gate 不由本报告自动判定，而是由用户对 `review.html` 中 largest disagreements 的人工 adjudication 决定。
