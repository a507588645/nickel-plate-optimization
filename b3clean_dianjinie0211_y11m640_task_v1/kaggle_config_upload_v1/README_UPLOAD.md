# B3-clean Dianjinie0211 Config 数据集（v3）

由 `build_b3clean_config_dataset.py` 生成；图片复用私有 Dataset `lly3323w/dianjinie`，本 config 只含清单、全量 SHA-256 校验表与训练器。与 v2 的差异：划分改为 b3clean_split_v1（按采集键排除 grade-test 泄漏 159 图，train 653 / val 178）。上传：进程内 API `dataset_create_version`；kernel 见 `b3clean-dianjinie0211-y11m640-seed0.py`。
