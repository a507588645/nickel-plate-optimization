# L1 E3 P2 检测头本机零 GPU 冒烟

- 状态：`dry_run_passed`
- 运行时间：`2026-08-23T05:19:13.352129+00:00`
- 范围：仅合成 CPU 张量；未读取数据集、未读取 `release-validation`、未启动训练或 Kaggle。
- 运行时：Python `3.10.11`；Torch `2.6.0+cu124`；Ultralytics `8.4.66`。
- 强制设备：`cpu`，进程 `CUDA_VISIBLE_DEVICES=-1`。
- YAML：`E:\PythonProject\优化\l1_e3_p2_smoke_20260823\yolo11m_p2.yaml`，SHA-256=`1738eb00730cd0403f291a954b2e0f19a77d9fb3ad83dc1bce353743e3acf5ac`。

## 拓扑

- 基于运行时自带 `yolo11.yaml` 的 YOLO11m 主干，新增 P3→P2 上采样、原生 P2 拼接和 P2→P3 回流；检测层为 P2/P3/P4/P5 四尺度。
- 使用模块仅为 `Conv`、`C3k2`、`SPPF`、`C2PSA`、`nn.Upsample`、`Concat`、`Detect`；未修改或 vendor patch Ultralytics，未使用 SPD-Conv/NWD。

## 结果

- 输入 YAML 与运行时路径检查通过；未构建模型、未请求 FLOPs、未运行前向/反向。
- 后续须不带 `--dry-run` 执行实际 CPU 冒烟后，才能判定 E3 是否可进入 Kaggle Fold 1 训练准备。

## 限制

- 本验证未使用真实图像或标签，不评估 mAP、类别 AP/Recall、训练速度或显存；这些均需在 E3 的单独授权 Kaggle Fold 1 中按固定开发边界验证。
- 结果属于 `auto_only` 工程可行性证据，不得作为人工真值或最终模型选择结论。
