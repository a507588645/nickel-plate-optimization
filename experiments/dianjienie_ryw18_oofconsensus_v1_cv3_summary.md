# dianjienie RYW18 OOF 共识候选 v1 三折汇总

## 1. 范围与状态

| 项目 | 内容 |
| --- | --- |
| 数据版本 | `dianjienie_ryw18_oofconsensus_v1_dev_only_r1` |
| 数据范围 | 1,127 张开发图、15,671 个五列候选标签；18 类 RYW18 |
| 折划分 | Fold 1=`751/376`，Fold 2=`752/375`，Fold 3=`751/376` |
| 冻结集边界 | `release-validation` 不读、不链接、不评估 |
| 训练 | 两个模型各 3 个 Kaggle GPU run，官方预训练、300 epochs、640、batch 8、SGD、seed 0、`resume=false` |
| 统一复评 | 本机 `E:/PythonProject/models/.venv`：Python 3.10.11、Ultralytics 8.4.66、Torch 2.6.0+cu124；`imgsz=640`、batch 4、FP32、`conf=0.001`、`iou=0.7`、`max_det=300` |
| 当前结论状态 | `auto_candidate_selected`：自动化训练、归档、三折统一复评及历史锚点自动非回退闸门完成；无需填写人工评测，YOLO11m 进入无人工候选训练 |

候选五列标签包含 15,426 个历史框与 245 个 OOF accepted 新框。其自身开发集指标只能用于同一候选视图内的暂定模型对照，不能证明候选标签优于历史标签锚点；因此另以候选权重在未改动历史锚点标签上的同 Fold 复评作为无人工安全闸门，详见 `../l1_auto_gate_oofconsensusv1_20260809/report.md`。

## 2. 统一开发集复评结果

| 模型 | Fold | 验证图 | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11m | 1 | 376 | 0.49024 | 0.30787 | 0.31292 | 0.16513 |
| YOLO11m | 2 | 375 | 0.42342 | 0.23208 | 0.26962 | 0.17194 |
| YOLO11m | 3 | 376 | 0.49413 | 0.39202 | 0.34760 | 0.21910 |
| YOLO26m | 1 | 376 | 0.50327 | 0.32701 | 0.30722 | 0.15927 |
| YOLO26m | 2 | 375 | 0.45340 | 0.32600 | 0.27356 | 0.16023 |
| YOLO26m | 3 | 376 | 0.54795 | 0.27801 | 0.35582 | 0.22548 |

| 模型 | Precision 均值±样本标准差 | Recall 均值±样本标准差 | mAP50 均值±样本标准差 | mAP50-95 均值±样本标准差 |
| --- | ---: | ---: | ---: | ---: |
| YOLO11m | 0.46927±0.03975 | 0.31066±0.08001 | 0.31005±0.03907 | 0.18539±0.02939 |
| YOLO26m | 0.50154±0.04730 | 0.31034±0.02800 | 0.31220±0.04135 | 0.18166±0.03795 |

YOLO11m 的平均 mAP50-95 高 `0.00373`，YOLO26m 的平均 mAP50 高 `0.00215`。这两个差值均远小于折间样本标准差，且两者只有一个随机种子，因此不构成架构优劣或最终权重选择证据。

## 3. 归档与复评证据

| 模型 / Fold | 训练 run | 复评证据 |
| --- | --- | --- |
| YOLO11m / 1 | `E:/PythonProject/models/runs/dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_fold01` | `E:/PythonProject/优化/l1_reeval_yolo11m_oofconsensusv1_fold01/metrics_per_class_reeval.json` |
| YOLO11m / 2 | `E:/PythonProject/models/runs/dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_fold02` | `E:/PythonProject/优化/l1_reeval_yolo11m_oofconsensusv1_fold02/metrics_per_class_reeval.json` |
| YOLO11m / 3 | `E:/PythonProject/models/runs/dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_fold03` | `E:/PythonProject/优化/l1_reeval_yolo11m_oofconsensusv1_fold03/metrics_per_class_reeval.json` |
| YOLO26m / 1 | `E:/PythonProject/models/runs/dianjienie_medium_yolo26m_640_kaggle_seed0_cm0_oofconsensusv1_fold01` | `E:/PythonProject/优化/l1_reeval_yolo26m_oofconsensusv1_fold01/metrics_per_class_reeval.json` |
| YOLO26m / 2 | `E:/PythonProject/models/runs/dianjienie_medium_yolo26m_640_kaggle_seed0_cm0_oofconsensusv1_fold02` | `E:/PythonProject/优化/l1_reeval_yolo26m_oofconsensusv1_fold02/metrics_per_class_reeval.json` |
| YOLO26m / 3 | `E:/PythonProject/models/runs/dianjienie_medium_yolo26m_640_kaggle_seed0_cm0_oofconsensusv1_fold03` | `E:/PythonProject/优化/l1_reeval_yolo26m_oofconsensusv1_fold03/metrics_per_class_reeval.json` |

Fold 2 和 Fold 3 的全部归档均通过 ZIP CRC、必需产物、`best.pt`、`last.pt`、`requested_config.json` 与固定开发集清单核验。Fold 3 ZIP SHA-256：YOLO11m=`199c86ef7572e76e7409b566cbb3deec186cbad0212fd4ac0672164bebba5c58`，YOLO26m=`3fbeadb462ecfd0c7955de9ae0603c37bfb953eec9d3dfad6e31fdbe97a0d8d4`。

## 4. 已完成与未完成闸门

- 已完成：OOF 候选版两个模型的 6 个 Fold 训练、归档、固定开发集复评、三折数值汇总，以及将候选权重按未改动历史锚点标签进行的 6 次同 Fold 自动复评。
- 已完成：自动非回退闸门。规则与结果见 `../l1_auto_gate_oofconsensusv1_20260809/report.md`：两个模型均通过；YOLO11m 在候选标签自评和历史锚点复评下的绝对平均 mAP50-95 均略高，因此作为无人工候选主模型。
- 不做：`l1_oof_evaluation_contract_v1` 的人工决定不再是 L1 的阻塞条件；它保留为可选审查素材，不修改候选标签。
- 未完成：YOLO11m OOF 候选全开发集训练、一次历史 release-validation 自动确认、ONNX 导出和一致性检查。
- 当前决策：启动无人工候选训练；在其自动发布验证完成前，不启动高分辨率训练或部署。
