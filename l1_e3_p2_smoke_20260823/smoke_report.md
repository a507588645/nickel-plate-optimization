# L1 E3 P2 检测头本机零 GPU 冒烟

- 状态：`passed`
- 运行时间：`2026-08-23T04:53:13.600724+00:00`
- 范围：仅合成 CPU 张量；未读取数据集、未读取 `release-validation`、未启动训练或 Kaggle。
- 运行时：Python `3.10.11`；Torch `2.6.0+cu124`；Ultralytics `8.4.66`。
- 强制设备：`cpu`，进程 `CUDA_VISIBLE_DEVICES=-1`。
- YAML：`E:\PythonProject\优化\l1_e3_p2_smoke_20260823\yolo11m_p2.yaml`，SHA-256=`1738eb00730cd0403f291a954b2e0f19a77d9fb3ad83dc1bce353743e3acf5ac`。

## 拓扑

- 基于运行时自带 `yolo11.yaml` 的 YOLO11m 主干，新增 P3→P2 上采样、原生 P2 拼接和 P2→P3 回流；检测层为 P2/P3/P4/P5 四尺度。
- 使用模块仅为 `Conv`、`C3k2`、`SPPF`、`C2PSA`、`nn.Upsample`、`Concat`、`Detect`；未修改或 vendor patch Ultralytics，未使用 SPD-Conv/NWD。

## 结果

- YAML 解析与模型构建：通过；检测步长 `[4.0, 8.0, 16.0, 32.0]`。
- 合成前向/反向：`3` 个 microbatch 全部通过。
- P2 参数量：`21,811,608`；标准 YOLO11m：`20,066,886`；增量：`1,744,722`。
- P2 FLOPs@640：`119.3830912` GFLOPs；标准 YOLO11m：`68.2617344` GFLOPs；增量：`51.121357` GFLOPs。
- 冒烟闸门：`pass_local_smoke`。这只证明固定运行时下的解析、构建和 CPU 梯度链路可用，不代表训练收益或可进入 Fold 2/3。

## 限制

- 本验证未使用真实图像或标签，不评估 mAP、类别 AP/Recall、训练速度或显存；这些均需在 E3 的单独授权 Kaggle Fold 1 中按固定开发边界验证。
- 结果属于 `auto_only` 工程可行性证据，不得作为人工真值或最终模型选择结论。
