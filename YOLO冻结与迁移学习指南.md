# YOLO 冻结、解冻与迁移学习指南

## 1. 本文解决的问题

本文用于区分以下四种容易混淆的操作：

1. 从 COCO 预训练权重开始训练镍板缺陷模型。
2. 冻结部分 Backbone，让检测头先适应新类别。
3. 从已经训练好的 YOLO11m 或 YOLO26m继续学习新数据。
4. 恢复一次意外中断的训练。

## 2. Backbone、Neck 和 Head

YOLO 检测模型可以粗略分为三部分：

```text
输入图像
  ↓
Backbone：提取边缘、纹理、角点和局部形状
  ↓
Neck：融合浅层细节和深层语义
  ↓
Head：输出类别、位置和置信度
```

COCO 预训练权重主要提供通用视觉特征。镍板训练需要让模型进一步学习金属表面纹理、极小缺陷和三光通道之间的关系。

冻结 Backbone 的作用：

- 减少初始阶段可训练参数，降低小数据快速过拟合的风险。
- 暂时保留预训练的通用边缘和纹理特征。
- 让 Neck 和 Head 先适应新的缺陷类别。

冻结不能直接解决：

- 缺陷在缩放后只剩一两个像素。
- 标签漏标或类别错误。
- 三光通道没有对齐。
- 稀有类别没有足够样本。

小目标问题仍应优先使用高分辨率、切片和准确标签解决。

## 3. `freeze` 参数是什么意思

Ultralytics 中：

```python
freeze=10
```

表示冻结模型前 10 个模块，而不是抽象意义上固定冻结“10个卷积层”。YOLO11m 和 YOLO26m 的模块结构不同，因此相同数字不保证冻结完全相同的网络区域。

推荐把以下值作为实验变量：

```text
freeze=0    不冻结
freeze=10   冻结前10个模块，推荐迁移学习起点
freeze=15   更强冻结，仅作为对照
```

不建议直接把旧 PDF 中的 `freeze=20` 当作统一答案。冻结范围过大时，模型可能无法适应镍板纹理和新三光输入。

## 4. 四种训练场景

### 4.1 场景 A：从 COCO 权重训练旧数据

适用条件：第一次训练 YOLO11m 或 YOLO26m。

阶段 1，短暂冻结：

```python
from ultralytics import YOLO

model = YOLO("yolo11m.pt")
model.train(
    data=r"E:\PythonProject\datasets\dataset_v1\data.yaml",
    epochs=20,
    imgsz=960,
    batch=4,
    freeze=10,
    optimizer="SGD",
    lr0=0.001,
    seed=42,
    project=r"E:\PythonProject\models\runs\transfer",
    name="yolo11m_stage1_freeze10",
)
```

阶段 2，全部解冻：

```python
from ultralytics import YOLO

model = YOLO(
    r"E:\PythonProject\models\runs\transfer\"
    r"yolo11m_stage1_freeze10\weights\best.pt"
)
model.train(
    data=r"E:\PythonProject\datasets\dataset_v1\data.yaml",
    epochs=150,
    imgsz=960,
    batch=4,
    freeze=0,
    optimizer="SGD",
    lr0=0.0005,
    seed=42,
    project=r"E:\PythonProject\models\runs\transfer",
    name="yolo11m_stage2_unfreeze",
)
```

YOLO26m只需把模型文件和实验名称换成 `yolo26m.pt`，其他公平比较参数保持一致。

### 4.2 场景 B：旧数据模型继续学习新三光数据

适用条件：已经有用旧数据训练完成的同架构 `best.pt`。

```python
model = YOLO(r"E:\path\to\old_data_yolo11m_best.pt")
```

推荐先使用旧数据和新数据混合集，而不是只训练新数据，防止模型遗忘旧场景。

阶段 1：冻结前 10 个模块，训练 5 至 10 轮。

阶段 2：全部解冻，使用较低学习率训练 50 至 100 轮。

由于三光通道具有新的物理含义，冻结阶段必须较短，最终必须全部解冻。

### 4.3 场景 C：从中断点恢复原训练

只有以下条件基本不变时才使用：

```python
model = YOLO(r"E:\path\to\last.pt")
model.train(resume=True)
```

适用条件：

- 同一模型。
- 同一数据集和划分。
- 同一训练目录。
- 只是断电、超时或进程中断。

如果更换为旧新混合数据集，不应使用 `resume=True`，而应加载 `best.pt` 后启动一次新的 `train()`。

### 4.4 场景 D：YOLOv5m 迁移到 YOLO11m 或 YOLO26m

以下做法不可行：

```python
YOLO("old_yolov5m_best.pt").train_as_yolo11()
```

YOLOv5m、YOLO11m 和 YOLO26m 的结构与检测头不同，旧 YOLOv5m 权重不能直接变成新架构的初始化权重。

正确方式：

1. YOLO11m 从 `yolo11m.pt` 开始。
2. YOLO26m 从 `yolo26m.pt` 开始。
3. 旧 YOLOv5m用于生成伪标签、提供历史基线或模型一致性投票。

## 5. 推荐冻结实验矩阵

在同一数据、同一划分和同一随机种子下比较：

| 实验 | 阶段1 | 阶段2 | 目的 |
| --- | --- | --- | --- |
| A | `freeze=0` | 无 | 完全微调基线 |
| B | `freeze=10, 20 epochs` | `freeze=0` | 推荐迁移方案 |
| C | `freeze=15, 20 epochs` | `freeze=0` | 验证更强冻结是否有利 |

判断顺序：

1. mAP50-95。
2. 小目标和关键类别 AP。
3. Recall。
4. 三个随机种子的平均值和标准差。

冻结策略的提升如果只在一个随机种子出现，不能视为稳定结论。

## 6. 新三光数据的特殊设置

新三光输入固定为：

```text
R = 左光
G = 右光
B = 中间光
```

建议：

- 冻结阶段不超过 10 至 20 轮。
- 最终阶段必须 `freeze=0`。
- 关闭通道交换和随机灰度。
- 强 HSV 色相、饱和度增强默认关闭或显著减小。
- 旧单光数据复制为三通道时，记录 `source_type=legacy_single`。
- 新三光数据记录 `source_type=trilinear_three_light`。

## 7. 常见错误

- 把 `freeze=20` 当作所有模型通用的 Backbone 范围。
- 全程冻结 Backbone，导致模型无法学习新三光纹理。
- 更换数据集后仍使用 `resume=True`。
- 用 YOLOv5m 权重直接初始化 YOLO11m。
- 只看总 mAP，不看稀有类别和新旧数据域指标。
- 同时改变冻结、输入尺寸、优化器和增强，导致无法判断提升来源。

## 8. 本项目推荐起点

```text
阶段1：freeze=10，10至20轮，imgsz=960
阶段2：freeze=0，100至150轮，降低学习率
对照组：从第一轮开始 freeze=0
```

如果显存不足，优先降低 batch，而不是把 `imgsz` 降回 640。极小缺陷任务中，输入分辨率通常比冻结层数更关键。
