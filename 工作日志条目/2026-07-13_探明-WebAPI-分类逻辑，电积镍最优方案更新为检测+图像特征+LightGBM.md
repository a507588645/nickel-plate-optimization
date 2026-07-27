## 2026-07-13 | 探明 WebAPI 分类逻辑，电积镍最优方案更新为检测+图像特征+LightGBM

| 字段 | 内容 |
| --- | --- |
| 执行者 | Grok / Claude |
| 任务目标 | 解决用户反馈「没有好的输出」：明确电解镍保留多类硬编码规则、电积镍需要稳定 4 类；在原项目逻辑与现有 folder 数据集基础上找到当前最优可部署方案 |
| 状态 | 已完成（方案选定与 API 验证）；folder ResNet/CNN 因 CPU 算力不足未收敛；样本扩充后建议复验 |
| 关联用户请求 | 电解镍原项目多类硬编码输出符合企业要求但数据集是二分类；电积镍实际应用效果不好需优化成四分类；现在没有好的输出 |

### 改动文件

- `dianjie_grade_rule.py`：新建脚本，将 `start.py` 中电解镍多类判定规则（type=0~6）提取为可复用模块。
- `grade_classifier_unified_api_v2.py`：新建统一 API，电解镍走规则模块、电积镍走「检测特征 + ResNet18 图像特征 + LightGBM」。
- `train_resnet_folder_classifier.py`：新建脚本，尝试在 folder 4 类数据上微调 ResNet18（CPU 训练过慢，未达可用指标）。
- `train_cnn_folder_classifier.py`：新建脚本，尝试轻量 CNN 基线（CPU 训练过慢，未达可用指标）。
- `compare_dianji_classifiers.py`：新建脚本，对比 LightGBM 与 folder ResNet 在电积镍验证集上的表现。
- `train_grade_classifier.py`：将 ResNet18 加载方式从 `torch.hub.load` 改为 `torchvision.models.resnet18`，使其能使用本地缓存权重。
- `grade_classifier_api.py` / `grade_classifier_unified_api_v2.py`：修复 DataFrame 碎片化警告。

### 产物路径

| 路径 | 说明 |
| --- | --- |
| `classify_from_detection_v1\models\dianji_lgbm_imgfeat.pkl` | **推荐：电积镍 4 类，检测特征 + ResNet18 图像特征 + LightGBM** |
| `classify_from_detection_v1\models\training_summary_imgfeat.json` | 图像特征模型指标：CV acc 0.794 / F1-macro 0.677；Val acc 0.736 / F1-macro 0.514 |
| `classify_from_detection_v1\models\dianji_lgbm_weighted.pkl` | 备选：纯检测特征 + 类别权重，Val acc 0.643 / F1-macro 0.521 |
| `classify_from_detection_v1\models\resnet_dianji_fast\best_model.pth` | folder ResNet18 128px 早停模型（epoch 1，Val acc 0.449 / F1-macro 0.437），未达可用 |
| `classify_from_detection_v1\models\cnn_dianji_weighted_os\best_model.pth` | folder CNN 早停模型（epoch 1，效果极差），未达可用 |
| `classify_from_detection_v1\models\dianji_comparison_fast.json` | LightGBM vs folder ResNet 对比表 |
| `classify_from_detection_v1\models\dianji_feature_importance_imgfeat.csv` | 图像特征模型特征重要性 |

### 关键命令或入口

```text
# 电积镍当前推荐推理（统一 API v2）
cd E:\PythonProject\优化
.venv_classify\Scripts\python grade_classifier_unified_api_v2.py

# 在 WebAPI 优化分支 start.py 中替换分类分支
from grade_classifier_unified_api_v2 import UnifiedGradeAPI
api = UnifiedGradeAPI()
# 电解镍接口返回 type=0~6
result = api.predict_dianjie(image_path)
# 电积镍接口返回 type=1/2/3/4（bad/9996/9950/3#）
result = api.predict_dianji(image_path)

# 重新训练图像特征模型
cd E:\PythonProject\优化
.venv_classify\Scripts\python train_grade_classifier.py --tasks dianji --add-image-features --output-suffix _imgfeat --seed 42 --device cpu

# 重新训练 folder ResNet（建议 GPU）
.venv_classify\Scripts\python train_resnet_folder_classifier.py --data-root E:/PythonProject/datasets/classify_4types_dianji --class-names 3# 9950 9996 bad --output-dir ... --device cuda
```

### 结论

- **WebAPI 现状**：
  - 电解镍 `DianjieDetection.post` 实际输出 7 类（`type=0~6`），基于检测缺陷面积比例硬编码，**企业认可**；分类模型 `resnet` 仅 2 类（good/bad），主要做辅助。
  - 电积镍 `DianjiDetection.post` 注释里是 4 类（3#/9950/9996/bad），但实际代码把 4 类 `slf_resnet` 注释掉，走了 2 类 `resnet`，所以输出逻辑残缺，**与用户反馈一致**。
- **电解镍方案**：保持原硬编码规则，已提取到 `dianjie_grade_rule.py`，可直接在 WebAPI 中替换内联代码，不改变判定语义。
- **电积镍方案对比**（固定 seed=42）：

| 方案 | Val Acc | Val F1-macro | 备注 |
| --- | ---: | ---: | --- |
| LightGBM 检测特征 + class_weight | 0.643 | 0.521 | 旧推荐 |
| **LightGBM 检测 + ResNet18 图像特征** | **0.736** | **0.514** | **新推荐**，准确率提升明显 |
| folder ResNet18 128px（epoch 1） | 0.449 | 0.437 | CPU 训练过慢，未收敛 |
| folder CNN 128px（epoch 1） | 0.030 | 0.015 | 几乎不可用 |

- 图像特征模型在 5 折 CV 上表现更稳健（acc 0.794 / F1-macro 0.677），验证集 F1-macro 与 class_weight 方案持平但准确率更高，**建议作为当前部署方案**。
- 电积镍根本瓶颈仍是样本：3# 仅 34 张、bad 仅 28 张（train+valid），稀有类召回/精度受限。

### 未完成项

- [ ] folder ResNet18/CNN 在 GPU 上完整收敛后再与 LightGBM 做最终对比
- [ ] 在 WebAPI `优化分支` 中替换 `start.py` 的 `DianjieDetection` / `DianjiDetection`
- [ ] 补充电积镍 3#、bad 类样本并重新训练最终模型
- [ ] 多随机种子复验图像特征模型稳定性

### 下一步

1. 用户在 WebAPI `优化分支` 导入 `grade_classifier_unified_api_v2.py`，按示例替换两个接口。
2. 收集并标注更多 3# 与 bad 样本，解决稀有类瓶颈。
3. 有 GPU 环境后，用 `train_resnet_folder_classifier.py` 重训 folder ResNet 做最终对比。

### 风险与待确认

- 风险：电积镍 3#/bad 样本过少，当前最优模型对这两类的置信度和泛化仍有限，现场可能误分。
- 风险：统一 API v2 每次推理需加载 ResNet18 提取图像特征，CPU 上额外耗时约 100~300ms/张；如部署 latency 敏感，可回退到纯检测特征 LightGBM。
- 待确认：WebAPI `优化分支` 当前代码是否与 `grade_classifier_unified_api_v2.py` 的输入输出字段兼容（字段已尽量对齐原 `start.py`）。

| 字段 | 内容 |
| --- | --- |
| 执行者 | Grok / Claude |
| 任务目标 | 不改动原分类/检测数据集，用软连接构建新数据集；以原项目 YOLOv5 检测输出为特征训练 LightGBM，解决电积镍只输出两类的问题 |
| 状态 | 已完成（核心模型）；ResNet18 图像特征实验因网络下载失败未验证 |
| 关联用户请求 | 根据 JinchuanPythonWebapi 分类逻辑新建分类模型；电积镍必须稳定输出 4 类；用软连接不改动原数据；AGENTS.md/CLAUDE.md 增加每次任务记录流程 |

### 改动文件

- `AGENTS.md`：新增第 5 节「工作日志与会话交接」，规定任务完成后必须写入 `工作日志.md`，后续 Agent 开始前必须读取「当前状态摘要」和最近 3～5 条条目。
- `CLAUDE.md`：「开始任务」与「结束任务」增加强制读取/写入 `工作日志.md` 的条款。
- `build_grade_dataset.py`：新建脚本，用软连接/路径清单组织 `classify_4types_dianji` 与 `classify_2types`，调用原项目检测模型提取缺陷框特征，不改动原数据。
- `train_grade_classifier.py`：新建脚本，训练 LightGBM 等级分类器；支持类别权重、随机过采样、SMOTE、可选 ResNet18 图像特征。
- `infer_grade_classifier.py`：新建脚本，从检测框或图片路径推理等级。
- `grade_classifier_api.py`：新建 API 封装，可直接替换 `start.py` 中的硬编码分类逻辑。
- `README.md`：补充分类器产物与使用入口（已在前序步骤更新）。

### 产物路径

| 路径 | 说明 |
| --- | --- |
| `E:\PythonProject\优化\.venv_classify` | 隔离虚拟环境（lightgbm / scikit-learn / torch / opencv / torchvision） |
| `E:\PythonProject\优化\classify_from_detection_v1` | 软连接数据集 + 特征/模型根目录 |
| `classify_from_detection_v1\dianji\features.csv` | 电积镍 986 样本 × 40 检测特征 |
| `classify_from_detection_v1\dianjie\features.csv` | 电解镍 1415 样本 × 144 检测特征 |
| `classify_from_detection_v1\models\dianji_lgbm_weighted.pkl` | **推荐：电积镍 4 类，类别权重，val F1-macro 最高 0.521** |
| `classify_from_detection_v1\models\dianji_lgbm_oversample.pkl` | 电积镍 4 类，随机过采样，val accuracy 最高 0.661 |
| `classify_from_detection_v1\models\dianjie_lgbm.pkl` | 电解镍 2 类基线，val accuracy 0.829，F1-macro 0.756 |
| `classify_from_detection_v1\models\training_summary*.json` | 各模型评估指标 |
| `classify_from_detection_v1\models\dianji_feature_importance*.csv` | 特征重要性 |

### 关键命令或入口

```text
# 构建软连接数据集并提取特征（已执行）
cd E:\PythonProject\优化
.venv_classify\Scripts\python build_grade_dataset.py --tasks dianji dianjie

# 训练电积镍 4 类分类器（推荐：类别权重）
.venv_classify\Scripts\python train_grade_classifier.py --tasks dianji --class-weight --output-suffix _weighted --seed 42

# 训练电积镍 4 类分类器（随机过采样）
.venv_classify\Scripts\python train_grade_classifier.py --tasks dianji --oversample random --output-suffix _oversample --seed 42

# 单张推理
.venv_classify\Scripts\python infer_grade_classifier.py --task dianji --image path/to/image.jpg --model classify_from_detection_v1/models/dianji_lgbm_weighted.pkl

# 接入 WebAPI（在优化分支 start.py 中替换硬编码逻辑）
from grade_classifier_api import GradeClassifierAPI
grade_api = GradeClassifierAPI(model_root=r"E:/PythonProject/优化/classify_from_detection_v1/models")
result = grade_api.predict("dianji", image_path)
```

### 结论

- 电积镍分类器已能稳定输出 **4 类（0=3#、1=9950、2=9996、3=bad）**，不再硬编码为两类。
- 电积镍 4 类样本极不均衡：3# 34 张、bad 28 张，9950/9996 各约 450 张；稀有类召回仍是瓶颈。
- 方案对比（固定 seed=42，5 折分层 CV）：

| 方案 | Val Acc | Val F1-macro | 备注 |
| --- | ---: | ---: | --- |
| 基线 | 0.643 | 0.463 | 无均衡处理 |
| +class_weight | 0.643 | **0.521** | **推荐**，宏观 F1 最高 |
| +random oversample | 0.661 | 0.507 | 准确率最高，但 F1-macro 略低 |
| +class_weight + oversample | 0.626 | 0.482 | 两者叠加未提升 |

- 电解镍 2 类基线表现良好（val acc 0.829，F1-macro 0.756），可直接接入；其原硬编码逻辑在业务现场已被认为可用。
- ResNet18 整板图像特征实验因 `torch.hub` 下载权重被远程断开而失败，未验证是否能进一步改善 3#/bad 类。
- 所有训练均未改动原始分类数据集 `classify_4types_dianji` 与 `classify_2types`，仅通过软连接引用原图。

### 未完成项

- [ ] ResNet18 图像特征消融实验（网络可用后重跑）
- [ ] 在 WebAPI `优化分支` 中替换硬编码分类逻辑并端到端验证
- [ ] 补充电积镍 3#、bad 类样本，再训最终模型
- [ ] 多随机种子（≥3）复验，确认提升稳定性

### 下一步

1. 用户在 WebAPI `优化分支` 的 `start.py` 中导入 `GradeClassifierAPI`，按示例替换现有分类调用。
2. 网络恢复后重跑 `--add-image-features` 实验，评估 3#/bad 类提升幅度。
3. 收集并标注更多 3# 与 bad 样本，解决根本瓶颈。

### 风险与待确认

- 风险：电积镍 3#、bad 样本过少（各约 30 张），当前模型对这两类的泛化能力弱，现场误分风险高。
- 待确认：原项目 `JinchuanPythonWebapi` 当前运行分支及 `start.py` 中分类逻辑的具体位置；需在 `优化分支` 上修改。

| 字段 | 内容 |
| --- | --- |
| 执行者 | Grok |
| 任务目标 | 落盘工具包路径；强调数据集改动规范；核对用户已完成的重标注与审查功能 |
| 状态 | 已完成（记录与核实）；人工审查应用 v2 仍待做 |
| 关联用户请求 | 记录 Tools；数据集改动勿动原件；记得已完成重标注与审查功能 |

### 改动文件

- `工作日志.md`：本条 + 摘要
- `README.md`：增加 Tools 与数据改动原则；伪标签完成状态
- `CLAUDE.md`：关键事实补充 Tools 与只增不改
- `AGENTS.md`：适用范围点名 Tools；数据安全与用户约定对齐（已有版本化规则，补充 Tools 入口）

### 产物路径（核实，非本次新生成）

#### 工具包

| 路径 | 说明 |
| --- | --- |
| `E:\PythonProject\Tools` | 工具包根目录 |
| `E:\PythonProject\Tools\ModelAutoLabelTool` | 批量自动标注（YOLOv5 / Ultralytics / RF-DETR 适配器） |
| `E:\PythonProject\Tools\ModelAutoLabelTool\weights\yolov5\ryw.pt` | 电解镍教师权重（伪标签 v1 使用） |
| `E:\PythonProject\Tools\ModelAutoLabelTool\weights\yolov5\slf.pt` | 电积镍相关 YOLOv5 权重 |
| `E:\PythonProject\Tools\ImageProcessTool` | 图像预处理（白平衡/CLAHE 等，几何不变可复制原标签） |

操作手册亦引用：`ModelAutoLabelTool` 用于无人工伪标签闭环。

#### 数据集改动原则（用户约定 + 项目规则）

1. **尽量不改原有数据集**（原图、`labels/` 人工标签、`data.yaml` 主配置）。
2. 需要新标注时：**新建版本化标签目录**（如 `labels_pseudo_v1/`、将来的 `labels_pseudo_v1_v2/`），禁止覆盖 `labels/`。
3. 需要新划分/新视图时：优先 **软连接** 或路径清单指向原图，在新版本根目录组装；失败可退化为路径记录（见 `build_grade_dataset.py`）。
4. 清洗、重划分必须输出到**新版本目录** + 映射/清单，不能原地改唯一副本（`AGENTS.md` 数据安全）。

#### 已完成：伪标签重标注 v1（本机核实）

| 数据集 | 原 `labels/` | `labels_pseudo_v1` | `labels_pseudo_v1_raw` | manifests |
| --- | --- | --- | --- | --- |
| `Dianjinie0211` | 993 文件（保留） | 1052 | 1052 | 4 文件 |
| `dianjienie` | 1349 文件（保留） | 1350 | 1350 | 4 文件 |

- 教师：`ryw.pt`，统一 18 类伪标签体系（与当前检测 YAML 5/17 类不同，训练前须映射/筛选）。
- 文档：`伪标签重标注报告_v1.md`、`伪标签审查流程记录_v1.md`。
- **原始 `labels/` 未改动**（与报告一致）。

#### 已完成：审查功能（工具与 UI，本机核实）

| 产物 | 路径 |
| --- | --- |
| 综合审查页（推荐） | `E:\PythonProject\优化\visualizations\pseudo_v1_comprehensive_review.html` |
| 旧版审查页 | `...\pseudo_v1_review.html` |
| 分状态样例图 | `visualizations/pseudo_v1_{accepted,uncertain,rejected}_only/` |
| 生成脚本 | `generate_comprehensive_review.py`、`generate_status_visualizations.py`、`generate_pseudo_label_review.py` |
| 应用审查结果 | `apply_review_results.py` → 应写入 `labels_pseudo_v1_v2/`，**不改 v1** |

- 流程记录写明：模拟审查已跑通 apply；**真实全量人工审查后的 `labels_pseudo_v1_v2` 在本机尚不存在**（两数据集 `Test-Path` 均为 False）。
- 故状态表述：**重标注完成 + 审查功能已建好**；**人审结果落地 v2 仍为下一步**。

### 关键命令或入口

```text
# 自动标注工具
cd E:\PythonProject\Tools\ModelAutoLabelTool

# 图像预处理
cd E:\PythonProject\Tools\ImageProcessTool

# 打开综合审查（或先起本地服务）
# E:\PythonProject\优化\visualizations\pseudo_v1_comprehensive_review.html
python -m http.server 8767 --directory E:\PythonProject\优化\visualizations

# 应用审查导出（导出 jsonl 之后）
python E:\PythonProject\优化\apply_review_results.py --review-result <导出的 jsonl>
```

### 结论

- `E:\PythonProject\Tools` 为项目工具包（自动标注 + 图像处理），与 `优化` 文档目录、`JinchuanPythonWebapi` 代码仓分离。
- 用户要求与 `AGENTS.md` 一致：**数据集只增版本、软链/新标签，不覆盖原件**。
- 用户记忆正确：v1 重标注与审查 UI/应用脚本已完成；尚未看到审查后的 v2 标签目录。

### 未完成项

- [ ] 人工审查进度与是否已有导出的 `pseudo_v1_review_result.jsonl`（本条未在磁盘搜到 v2）
- [ ] 生成并校验 `labels_pseudo_v1_v2`
- [ ] 训练前处理 18 类伪标签 vs 现行 5/17 类 YAML 的映射
- [ ] `dianjienie` 伪标签 train/val 重复 stem 清洗（审查流程记录已提示）

### 下一步

1. 人审 uncertain 优先 → 导出结果 → apply 出 v2
2. 任何新数据工作继续只用新目录/软链
3. WebAPI 优化仍在 `优化分支`

### 风险与待确认

- 风险：误把 `labels_pseudo_v1` 当最终人工标；或训练直接指 18 类伪标签却用 5/17 类 YAML
- 待确认：人审是否已在其他机器导出过结果但未拷回本机

---
