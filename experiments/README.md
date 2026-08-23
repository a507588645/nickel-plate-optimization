# 实验目录说明

本目录保存正式实验卡和跨运行汇总，不保存大模型权重、原始数据或唯一训练证据。

## 1. 四层证据结构

| 层级 | 位置 | 作用 |
| --- | --- | --- |
| 当前计划 | `当前工作计划.md` | 决定当前优先级、资源预算、闸道和停止条件 |
| 事件日志 | `工作日志.md`、`工作日志条目/` | 记录执行事实、阻塞、交接和历史更正 |
| 实验卡 | `experiments/<experiment_id>.md` | 记录一次可复现实验的假设、唯一主变量、配置、结果和决策 |
| 原始证据 | `E:\PythonProject\models\runs\<run_id>` | 保存权重、配置、环境、训练日志、曲线和标准复评 |
| 比较结论 | `E:\PythonProject\models\results\` 或本目录的汇总 Markdown | 只引用已核验的原始证据，说明比较口径和限制 |

## 2. 实验卡最小要求

每个正式实验必须写清：

- 实验编号、所属主线和当前状态：`planned`、`running`、`complete`、`blocked`、`invalid` 或 `abandoned`。
- 研究假设、只改变的主要变量、基线和进入条件。
- 数据版本、split manifest/hash、类别顺序、测试集访问边界。
- 运行环境、模型、权重、训练参数、预计与实际资源消耗。
- 完整产物路径、统一复评入口、总体和逐类指标。
- 决策门结果：继续、停止、转备用路线或仅作参考。
- 未解决风险和下一步，不把单次结果写成稳定结论。

## 3. 命名与比较规则

推荐实验编号：

```text
<line>_<dataset>_<model>_<imgsz>_<variable>_seed<seed>[_foldNN]
```

不同数据版本、标签版本、验证清单、输入尺寸或主要变量必须使用不同实验编号。跨版本指标不得直接排名；OOF 候选五列视图不能标记为人工真值。

## 4. 证据等级

| 等级 | 含义 |
| --- | --- |
| `valid` | 配置、产物、统一复评和比较边界均完整，可进入当前决策 |
| `provisional` | 结果可用但缺少多 seed、三折或外部验证，只能作暂定结论 |
| `inconclusive` | OOM、超时、未完成或资源配置改变，不能支持正负结论 |
| `invalid` | 数据泄漏、配置不一致、权重损坏或验证口径错误，不得比较 |

## 5. 汇总报告规则

跨实验汇总应只回答三个问题：

1. 同一数据和验证口径下，哪个变量是否改善了目标指标。
2. 改善是否覆盖关键类别，资源成本是否可接受。
3. 结论是否足够进入下一闸道、交付或部署。

历史实验不删除；若结论被更正，追加更正记录并保留原实验卡。

## 6. 当前专项状态

- `dianjienie_ryw18_oofconsensus_v1_yolo11m_tile1024o25_phase1_wbfedge_fold01_kaggle.md`：已完成且证据完整的单 Fold `auto_only` 停止闸门。后处理虽清除了大量低置信/边缘切片框，但所有切片 profile 仍显著低于固定整图基线；Phase 2/3 不启动。
- `dianjienie_ryw18_oofconsensus_v1_yolo11m_e2_mosaicpair_fold01_kaggle.md`：`complete`、`provisional`。两个完成 ZIP 的最小必需证据已逐成员 CRC 提取并通过成对合同校验；Mosaic=0 候选 mAP50-95 比 Mosaic=1 对照低 `0.006518`，且 `wuzi/airhole/bubble` Recall 下降，结论为 `not_passed_do_not_expand`，不启动 Fold 2/3。
