# Dianjinie0211 只读审计摘要（B3 前置）

- 生成时间：2026-09-04T23:38:59
- 数据集：`E:\PythonProject\datasets\Dianjinie0211`

| 项 | train | val |
| --- | ---: | ---: |
| png 图片 | 836 | 216 |
| 标签 txt | 774 | 217 |
| 缺标签图片 | 62 | 0 |
| 孤立标签 | 0 | 1 |
| 空标签 | 0 | 1 |
| 框总数 | 17655 | 4337 |

- 文件类型：train {'.npy': 836, '.png': 836}，val {'.npy': 216, '.png': 216}
- 类别分布：train {'waterstain': 7532, 'waterblock': 8219, 'colorstain': 566, 'nodule': 5, 'bubble': 1333}，val {'waterstain': 1873, 'waterblock': 1215, 'colorstain': 309, 'bubble': 940}
- train/val 主名冲突：0
- train/val 前缀码分组重叠：0
- 与等级视图主名重叠：986
- 标签格式错误：train 0，val 5

## 警告
- train 有 62 张 png 无标签文件，需人工确认是负样本还是漏标
