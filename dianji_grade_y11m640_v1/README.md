# dianji_grade_y11m640_v1 产物说明

- 状态：**unconfirmed_pending_enterprise**，企业确认前不得用于生产。
- 数据：归档等级视图 1072 张，train/valid/test=`750/161/161`，seed=42。
- 特征：冻结 YOLO11m@640 新提取 43 列 BASE + 既有 CSV 按 SHA-256 复用 20 列 `tx__`，共 63 列。
- 检测权重 SHA-256：`901ccf6006f44689483aaf6f0d1d1b8c18132a85c7d881be7e62fcdccc255b21`
- 训练目标：`v3_texture_only` 等价配置，LightGBM best_iteration=`186`。
- 主要文件：
  - `per_box_y11m640.csv`：9672 个逐框检测审计行
  - `features_y11m640_base43.csv`：43 列新检测特征
  - `features_y11m640_v1.csv`：训练用 63 列特征
  - `metrics_v3_texture_only.json`：目标配置 valid/test 逐类指标与混淆矩阵
  - `model_comparison.csv`：既有训练入口生成的对照结果
  - `reject_curve.csv`：valid-only 阈值和 test 一次性报告
  - `lgbm_model.txt`：LightGBM Booster
  - `deploy_contract.json`：部署输入、映射、阈值和权重契约
  - `training_predictions.csv`：训练时三 split 预测快照
  - `roundtrip_summary.json` / `roundtrip_comparison.csv`：fresh-process 3 图回建验证
  - `report.md`：中文指标报告
- 本次目标指标：valid acc/macroF1=`0.7453/0.7058`；test acc/macroF1=`0.7453/0.6797`。
- 拒判契约候选：目标 coverage=`0.70`，threshold=`0.7362`（valid coverage=`0.7019`；test coverage=`0.7391`）。
- 历史参考（非本次结果）：历史 `ryw.pt@1536` `v3_texture_only` test acc/macroF1=`0.7764/0.6631`。

