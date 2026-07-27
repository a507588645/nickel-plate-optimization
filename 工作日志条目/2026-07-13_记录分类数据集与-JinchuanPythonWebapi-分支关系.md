## 2026-07-13 | 记录分类数据集与 JinchuanPythonWebapi 分支关系

| 字段 | 内容 |
| --- | --- |
| 执行者 | Grok |
| 任务目标 | 落盘用户提供的分类数据路径与项目代码分支约定，供后续 Agent 续作 |
| 状态 | 已完成 |
| 关联用户请求 | 记录 classify 数据集路径；`308` 为主分支，`优化分支` 为未来改进且未 push |

### 改动文件

- `工作日志.md`：本条 + 更新当前状态摘要
- `CLAUDE.md`：补充分类数据集路径、WebAPI 仓库与分支约定
- `README.md`：补充相关路径索引（便于人工查找）

### 产物路径

无新数据/训练产物；以下为用户声明并经本机路径/Git 核实的关键位置：

| 角色 | 路径 / 名称 | 核实 |
| --- | --- | --- |
| 电积镍分类数据集（原 4 类） | `E:\PythonProject\datasets\classify_4types_dianji` | 存在；`train`/`valid`；train 子类约 `3#`(24)、`9950`(372)、`9996`(342)、`bad`(21) |
| 电解镍分类数据集（2 类） | `E:\PythonProject\datasets\classify_2types` | 存在；`train`/`valid`；train 子类约 `bad`(254)、`good`(729) |
| 项目代码仓库 | `E:\PythonProject\ImetricCodes\JinchuanPythonWebapi` | 存在；remote `origin` → cnb.cool imetric/jinchuan |
| 主分支（现场/稳定） | `308` | 跟踪 `origin/308`；最新 `f2fe325` |
| 优化开发分支 | `优化分支` | **仅本地**，无 upstream；相对 `308` 约 **超前 7 个提交**；最新 `ac4b44d`（FastAPI 迁移与权重加载等） |
| 记录时工作区分支 | `308` | `git branch --show-current` 为 `308`（未 checkout 在 `优化分支`） |

`优化分支` 相对 `308` 的本地提交（新→旧，摘录）：

1. `ac4b44d` 迁移到原生 FastAPI 并修复 PyTorch 权重加载  
2. `8c26523` 优化代码增加模型支持  
3. `267c355` 新增 Ultralytics 检测器适配器占位  
4. `bb47bfc` 清理未用代码与非版本文件  
5. `7b60f3a` 权重集中到 `weights/` 等整理  
6. `36b3a65` 可插拔适配器 `adapters/` + `models.yaml`  
7. `aa6aeb1` 电解镍检测接入 RF-DETR 适配器（WIP）

### 关键命令或入口

```text
# 仓库
cd E:\PythonProject\ImetricCodes\JinchuanPythonWebapi

# 主分支（已跟踪 origin）
git checkout 308

# 本地优化分支（尚未 push，无 upstream）
git checkout 优化分支

# 本目录内已有引用（classify_from_detection_v1）
# --dianji-root  默认 E:/PythonProject/datasets/classify_4types_dianji
# --dianjie-root 默认 E:/PythonProject/datasets/classify_2types
```

### 结论

- **电积镍等级/分类原数据**用 `classify_4types_dianji`（4 类目录结构）；**电解镍**用 `classify_2types`（good/bad）。与检测集 `dianjienie` / `Dianjinie0211` 不同，不可混用路径。
- **代码主分支是 `308`**（应对齐线上/现场基线）；**未来改进在本地 `优化分支`**，用户明确尚未 push，改代码时默认应在该分支进行。
- 本机记录时刻当前 checkout 在 `308`，续作优化前需先 `git checkout 优化分支`，避免改动落在主分支。
- `E:\PythonProject\优化\classify_from_detection_v1` 已默认指向上述两个 classify 路径（见其 README）。

### 未完成项

- [ ] `优化分支` 尚未 push；是否推远程、远程名由用户决定
- [ ] 优化分支上的 RF-DETR 等适配器仍有 WIP 痕迹，完整度待在该分支上核对
- [ ] 分类集与 `classify_from_detection_v1` 产物、LightGBM 权重的版本对应关系未在本条展开

### 下一步

1. 任何改 `JinchuanPythonWebapi` 的任务：先确认分支是否为 `优化分支`
2. 分类/等级相关任务：区分检测数据集 vs 上述 classify 数据集
3. 用户若 push `优化分支`，在本日志追加一条并写 remote 分支名

### 风险与待确认

- 风险：未 push 的本地分支仅在本机；换机器或误删则丢失优化提交
- 风险：当前若停在 `308` 上直接改文件，会污染主分支工作树
- 待确认：`优化分支` 最终是否以该中文名推送，或改名为英文分支再推

---
