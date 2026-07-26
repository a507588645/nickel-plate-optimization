# dianjienie RYW18 锚点训练交接

## 当前可训练数据

- 锚点：`E:\PythonProject\datasets\dianjienie_ryw18_anchor_v1`。
- 标签来源：`dianjienie` 历史人工标签；源 id `0..16` 不改，仅类名对齐 RYW18，另保留 id `17 fanxitong`。
- 图片：1325 张；框：18232 个；独立冻结测试集：198 张；开发集：1127 张，固定为 3 个分组多标签 Fold。
- 原 `dianjienie` 的 `train/valid` 划分已不再用于任何新实验。
- `ryw.pt` 只作为后续候选教师，不能用作 YOLO11m/YOLO26m 的初始化权重。

## L1 实际路线：复用旧图像，不复用旧标签和旧划分

`lly3323w/dianjienie` 在 Kaggle 中的职责只是**只读图像仓库**。运行器只从其中取已核验的 1325 张源图；它不会读取该 Dataset 内旧的 17 类标签，也不会使用其旧 `train/valid` 划分。

| 层 | 内容 | 在 L1 中的用途 |
| --- | --- | --- |
| 旧 Kaggle 图片集 | 1350 张历史图片及旧目录结构 | 只提供其中已核验的 1325 张图片字节 |
| `dianjienie_ryw18_anchor_v1` | 18 类标签独立副本、去重记录、198 张冻结测试集、3 折分组划分 | 当前唯一训练与验证定义 |
| Kaggle 配置 Dataset | 锚点标签、`source_manifest`、固定 Fold、训练脚本 | 将旧图片与新锚点安全组装到 `/kaggle/working` |

因此，L1 的第一轮并不是“拿旧 `dianjienie` 直接训练”，而是：`旧图片 + anchor_v1 的新标签副本 + anchor_v1 的新分组划分`。旧 Dataset 中多出的 25 张图、旧标签和旧划分均不进入本轮训练。

### 不做人工逐框标注时的自动路线

1. **挂载校验**：先运行 `--prepare-only`，确认实际恢复 `1325` 图、`1325` 标签以及 Fold 1 的 `751/376/198` 划分；此步不消耗 GPU。
2. **统一基线**：GPU 配额足够后，以完全相同的参数训练 YOLO11m 和 YOLO26m 各 3 个 Fold。先跑两者 Fold 1 验证环境，再补齐其余 4 个任务；`ryw.pt` 仅作旧教师/对照，不作跨架构初始化权重。
3. **模型选择**：汇总 3 折开发集指标，选择候选架构和阈值；198 张冻结测试集在此阶段不参与训练、早停、调参或选模型。
4. **OOF 自动复核**：对每张开发集图片仅使用“未见过该图”的 Fold 模型预测，再与另一架构的 OOF 预测及 `ryw.pt` 候选比较。原历史框默认保留；模型漏检不能自动删原框；类别冲突只记录为冲突；空预测只记为 `uncertain_empty`，不自动当成无缺陷。
5. **无人工共识接纳**：仅当至少两个独立模型给出同类框且 IoU `>= 0.55` 时，才可作为新增/改类候选。所有候选先保留第六列置信度，按 `accepted >= 0.70`、`uncertain 0.40-0.70`、`rejected 0.15-0.40` 留痕；单模型候选不进训练标签。
6. **新版本而非覆盖**：候选、自动决策和置信度标签都写入 `E:\PythonProject\优化` 的版本化目录。只有完成共识后，才由它生成一个新的五列训练视图；`anchor_v1`、原 `dianjienie` 标签和冻结测试标签始终保留不改。新视图必须重新跑相同 Fold，旧指标仅作历史参考。
7. **最终确认**：只在架构、阈值和训练视图都冻结后，用未改动的 198 张测试集做一次最终评估；随后再按固定 epoch/交叉验证结论训练最终部署权重并导出 ONNX。

`youpao` 与 `fanxitong` 当前没有监督实例，自动共识不能凭空补出可靠标签；`heibian`、`mijixiaokeng`、`NiOH`、`fawu` 等极稀少类也只能标记为低证据，不能据此宣称已解决。`Dianjinie0211` 是另一套 5 类任务，仍需单独训练和报告，不与当前 18 类锚点混训。

## Kaggle 私有 Dataset 训练（当前主路径）

本机不具备可用训练环境时，使用私有 Kaggle Dataset + Kaggle GPU Notebook。不要把图片上传到 GitHub；本锚点图片约 21.25 GiB，GitHub 不适合作为训练数据载体。Google Drive 不需要使用：当前 Kaggle 账号已有私有图片集 `lly3323w/dianjienie`，可直接复用。

已完成只读核验：既有私有图片集含 1350 张 PNG；新 RYW18 锚点使用的 1325 张源图全部存在，路径和文件大小逐项一致。私有配置 Dataset `lly3323w/dianjienie-ryw18-anchor-v1-config` 已处于 `ready`，只含标签、固定划分和脚本，不含图片。Kaggle 会自动展开其配置归档；必须通过顶层启动器恢复配置，不能直接运行归档内的训练运行器。

### 1. 创建私有 Dataset

本机已有 Kaggle API 凭据。配置 Dataset 已创建；以下命令仅用于需要重建本地发布包时，不要对已存在的同名 Dataset 重复执行 `--publish`：

```powershell
cd E:\PythonProject\优化
python .\build_kaggle_dianjienie_ryw18_config_dataset.py --publish
```

该脚本会再次核验既有图片集，读取当前用户名，并生成或创建 `dianjienie-ryw18-anchor-v1-config` 配置 Dataset。不要上传原 `dianjienie` 目录、GitHub 仓库或任何旧随机划分。

### 2. 先做无 GPU 挂载校验

在 Kaggle 新建私有 Python Notebook，同时附加两个私有 Dataset：已有图片集 `dianjienie` 和 `dianjienie-ryw18-anchor-v1-config`。当前账户页面提示“接近耗尽 GPU 配额”，因此先不要启动训练或选择 GPU；先运行下列无 GPU 校验：

```python
!python /kaggle/input/dianjienie-ryw18-anchor-v1-config/kaggle_dianjienie_ryw18_archive_bootstrap.py --model yolo11m.pt --fold 1 --prepare-only
```

预期显示 `images=1325`、`labels=1325`，并成功恢复固定锚点。该命令不安装依赖、不检查 GPU、不训练。

### 3. 配额恢复后启动首轮 Fold 1

在 Notebook 设置中选择 GPU，并开启 Internet（首次需要下载固定依赖与公开预训练权重）。确认 Kaggle 显示的可用时长足以容纳单次完整训练后，单独运行以下两次，不要并行运行：

```python
!python /kaggle/input/dianjienie-ryw18-anchor-v1-config/kaggle_dianjienie_ryw18_archive_bootstrap.py --model yolo11m.pt --fold 1
```

```python
!python /kaggle/input/dianjienie-ryw18-anchor-v1-config/kaggle_dianjienie_ryw18_archive_bootstrap.py --model yolo26m.pt --fold 1
```

运行器会从已有图片集中只链接锚点的 1325 张图，映射到新版 18 类标签与固定划分；固定安装项目已有 Kaggle 基线使用的 `torch==2.6.0`、`torchvision==0.21.0`、CUDA 12.4 wheel、`numpy==1.26.4`、`ultralytics==8.4.66`，然后检查 GPU、18 类顺序与固定划分。若显存不足，两种模型都使用相同的更小 `--batch` 重跑，例如 `--batch 4`；不可只改其中一个模型。

每次成功训练会输出：

- run 目录：`/kaggle/working/runs/dianjienie_medium_<model>_640_kaggle_seed0_cm0_anchorv1_fold01/`。
- 可下载压缩包：`/kaggle/working/dianjienie_medium_<model>_640_kaggle_seed0_cm0_anchorv1_fold01_results.zip`。

保存 Notebook Version 后下载每个结果压缩包，并将完整 run 归档到 `E:\PythonProject\models\runs` 对应目录。先核对两次 Fold 1 都有 `weights/best.pt`、`args.yaml`、`results.csv`、`requested_config.json`、`environment.json`、`dataset_config/` 和曲线，再开始 Fold 2/3。

## 先做校验

在有训练环境的机器上，从 `E:\PythonProject\优化` 运行：

```powershell
python .\train_dianjienie_ryw18_anchor.py --model yolo11m.pt --fold 1 --dry-run
python .\train_dianjienie_ryw18_anchor.py --model yolo26m.pt --fold 1 --dry-run
```

两条命令应分别报告 `train=751`、`val=376`、`test_frozen=198`。训练环境需要已安装支持所选模型的 `ultralytics`、PyTorch 和可用 GPU；本机当前未安装 `ultralytics`，因此这里只完成了数据和配置校验，未启动训练。

锚点可复制到其他 GPU 机器后直接传入锚点根目录；训练脚本会根据平铺的 `images/` 文件名重建运行时 `train.txt`、`val.txt`、`test.txt`，不会使用锚点内保存的 Windows 绝对路径。Kaggle 必须使用本文件前述的 `kaggle_dianjienie_ryw18_archive_bootstrap.py`，不要把旧 `dianjienie` 图片集直接传给底层训练器，因为它不含修正后的 18 类锚点标签和固定划分。

## 最小启动

先只跑 Fold 1 的两个模型，确认 GPU 环境、模型权重和输出目录都正常。它们是可比较的首轮基线，但不是最终的三折结论：

```powershell
python .\train_dianjienie_ryw18_anchor.py --model yolo11m.pt --fold 1
python .\train_dianjienie_ryw18_anchor.py --model yolo26m.pt --fold 1
```

两个 run 完成后保留其完整目录，再继续下面的完整三折方案。

## 统一基线

两种模型和三个 Fold 必须使用完全相同的训练参数：`epochs=300`、`imgsz=640`、`batch=8`、`optimizer=SGD`、`patience=100`、`seed=0`、`close_mosaic=0`。该组合对齐本项目登记的 YOLO11m/YOLO26m 标准比较组；类别、数据划分和输出目录则已替换为本锚点。

正式模型选择时补全六个任务；Fold 1 的两条若已完成，不要覆盖，直接补 Fold 2 和 Fold 3：

```powershell
python .\train_dianjienie_ryw18_anchor.py --model yolo11m.pt --fold 1
python .\train_dianjienie_ryw18_anchor.py --model yolo11m.pt --fold 2
python .\train_dianjienie_ryw18_anchor.py --model yolo11m.pt --fold 3
python .\train_dianjienie_ryw18_anchor.py --model yolo26m.pt --fold 1
python .\train_dianjienie_ryw18_anchor.py --model yolo26m.pt --fold 2
python .\train_dianjienie_ryw18_anchor.py --model yolo26m.pt --fold 3
```

本地默认产物位于 `E:\PythonProject\models\runs\dianjienie_medium_yolo11m_640_lab_seed0_cm0_anchorv1_fold01` 等目录；Kaggle 将 `lab` 替换为 `kaggle`。脚本会自动写入 `requested_config.json`、`environment.json`、`dataset_config/` 和符合 run 规范的 `README.md`；每个 run 还应保留 Ultralytics 生成的 `weights/best.pt`、`weights/last.pt`、`args.yaml`、`results.csv`、曲线和混淆矩阵。

## 使用边界

- `splits/test.txt` 只在选择完模型和阈值之后做一次最终确认；不能参与早停、选模型或调阈值。
- `youpao`、`fanxitong` 为零监督类，训练输出虽保留 18 个类别，但这两类不可评、不可宣称可识别。
- `heibian`、`mijixiaokeng` 的独立 group 少于 3，不能要求每个开发 Fold 都有可靠逐类指标；`NiOH`、`fawu` 也属于极稀少类别。
- 若显存不足，可同时下调两种模型的 `--batch`，但不得只改其中一个模型；需要改变输入尺寸或增强时，应作为新的对比实验，不能覆盖本基线。
- 更换数据或划分后禁止使用 `resume=True`；本入口已显式固定为 `resume=False`。

## 训练后自动路线

六个 run 完成后，下一步由自动流程执行：对锚点做 OOF 推理，使每张图只由未见过它的 Fold 模型预测；再与另一架构及 `ryw.pt` 做同类、IoU 阈值一致性匹配。只把至少两模型一致的新增/改类候选写入版本化 `labels_with_conf/`，并另导出五列共识训练版本；原始历史标签和本锚点均不覆盖。单模型预测和空检图不自动当成负样本。
