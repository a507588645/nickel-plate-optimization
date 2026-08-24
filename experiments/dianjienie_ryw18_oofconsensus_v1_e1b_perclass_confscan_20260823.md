# 模型优化实验记录

## 1. 基本信息

| 项目 | 内容 |
| --- | --- |
| 实验编号 | `dianjienie_ryw18_oofconsensus_v1_e1b_perclass_confscan_20260823` |
| 所属主线 | L1 小目标研究 / E1b 逐类别置信度阈值扫描 |
| 日期 | 2026-08-23 |
| 执行人 | Codex |
| 实验状态 | `complete`；按计划的固定原始锚点严格门槛为 `not_passed_strict_fixed_anchor`。 |
| 证据等级 | `provisional`；输入、边界与复现完整，但阈值选择和评测共用同一 `auto_only` 开发 OOF。 |
| 父计划 / 决策闸道 | `L1_小目标优化分阶段计划.md` §5 E1b |
| 研究假设 | 将微小缺陷类别的运行置信度阈值与其他类别解耦，可提高焦点类召回且不显著损失总体 mAP50-95。 |
| 唯一主要变量 | 每次仅将一个焦点类阈值设为 `0.05` 至 `0.50`，其他 17 类固定为 `0.25`。 |
| 训练 / 推理 | 未训练、未加载权重、未做新推理；仅重评已有 OOF 预测转储。 |

## 2. 数据与边界

| 项目 | 内容 |
| --- | --- |
| 评测成员 | 固定开发 OOF Fold 1/2/3，共 1,127 图、15,671 个目标。 |
| 来源预测 | Phase 0 YOLO11m 原始转储 149,826 框；E1 YOLO26m 转储 100,292 框；E1 WBF IoU=0.55 转储 169,187 框。 |
| 焦点类 | `wuzi`、`waterstain`、`airhole`、`bubble`、`greenstain`、`watermark`。 |
| 阈值 | `0.05` 至 `0.50`，步长 `0.05`；基准运行阈值为 `0.25`。 |
| 冻结测试集 | 未读取、未挂载、未链接、未评测；`release-validation` 为 `null`。 |
| 标签边界 | `auto_only` 开发标签；不得用本结果宣称人工真值、泛化收益或部署阈值。 |

## 3. 固定锚点与双重口径

E1 固定原始锚点使用预测文件原有的 `0.001` 置信度下限，已逐字节来源转储重现：YOLO26m mAP50-95=`0.148939031666318`，WBF IoU=0.55 mAP50-95=`0.16751980771806158`。

运行阈值的公平局部对照是“所有类别为 `0.25`，仅覆盖一个焦点类”，因为两侧均经过相同的运行截断。该对照与固定原始 mAP 锚点不是同一口径，必须同时报告：

| 变体 | 全类 `0.25` 运行对照 mAP50-95 | 相对固定原始锚点 |
| --- | ---: | ---: |
| YOLO26m | 0.112763 | -0.036176 |
| WBF IoU=0.55 | 0.135488 | -0.032032 |

因此，不允许把相对 `0.25` 运行对照的局部召回收益误写成相对 E1 原始 mAP 锚点的收益。

## 4. 闸门结果

运行对照门槛为：焦点类 Recall 相对同类 `0.25` 对照提高至少 `+0.05`，且 pooled mAP50-95 相对该运行对照不低于 `-0.005`。共有 22 个单类覆盖通过该局部门槛，例如：

| 变体 | 类别 | 阈值 | Recall 增量 vs `0.25` | pooled mAP50-95 增量 vs `0.25` |
| --- | --- | ---: | ---: | ---: |
| WBF IoU=0.55 | waterstain | 0.05 | +0.126103 | +0.001353 |
| WBF IoU=0.55 | airhole | 0.05 | +0.115159 | +0.000458 |
| YOLO26m | watermark | 0.05 | +0.111444 | +0.001501 |

最终计划门槛仍以计划中写明的固定 E1 原始 mAP50-95 锚点判断。22 个局部候选相对其各自固定原始锚点均下降约 `0.031` 至 `0.036`，没有一个满足“下降不超过 `0.005`”。

**最终决定：`not_passed_strict_fixed_anchor`。** 不生成联合六类阈值配置，不与 E1 WBF 叠加，不改动部署阈值或既有 640 `auto_only` 候选。

## 5. 复现与证据

| 内容 | 路径 |
| --- | --- |
| 入口 | `E:/PythonProject/优化/run_l1_oof_perclass_conf_scan.py` |
| 最终输出 | `E:/PythonProject/优化/l1_oof_perclass_conf_scan_20260823_r3/` |
| 机器可读闸门 | `E:/PythonProject/优化/l1_oof_perclass_conf_scan_20260823_r3/gate_decision.json` |
| 人类可读报告 | `E:/PythonProject/优化/l1_oof_perclass_conf_scan_20260823_r3/report.md` |
| 预测输入审计 | `E:/PythonProject/优化/l1_oof_perclass_conf_scan_20260823_r3/preflight.json` |

```text
E:\PythonProject\models\.venv\Scripts\python.exe run_l1_oof_perclass_conf_scan.py --dry-run
E:\PythonProject\models\.venv\Scripts\python.exe run_l1_oof_perclass_conf_scan.py --output-root .\l1_oof_perclass_conf_scan_20260823_r3
```

## 6. 限制与下一步

- 同一开发 OOF 同时用于扫描、选择和计分，存在选择偏差。
- 预测已是模型 NMS 后输出；本研究无法恢复已被模型 NMS 删除的候选框。
- 本实验只测单类别覆盖，未测六类联合阈值，且严格门槛已否决继续扩展。
- 后续 L1 GPU 研究仅可在 E3 P2 或 E4 Copy-Paste 的单独用户授权下启动；两者均不能据此实验直接提交。
