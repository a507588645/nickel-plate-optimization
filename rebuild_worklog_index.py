#!/usr/bin/env python3
"""重建工作日志条目索引表（条目文件夹版）。

扫描 ``工作日志条目/`` 下的全部条目文件（每个文件首行为
``## YYYY-MM-DD[ HH:MM] | 标题``），在主文件 ``工作日志.md`` 中重建
「条目索引」表：``| 日期 | [标题](条目文件链接) | 主线 |``，
并同步表头「全 N 条」总数。索引按 日期+时间 倒序（最新在上；
无时间的条目排在当日最后，再按标题排序）。

规则（与 AGENTS.md §5 一致）：

- 只重写索引表区块与总数行，不改动任何条目文件；
- 主线标签为人工归号：先从现有索引表按标题继承（兼容旧的行号版表格），
  其次查 ``TAG_OVERRIDES`` 种子表，最后回落「其他」并在输出中列出，
  便于人工补标；
- 写后自动复检：每行链接指向的文件必须存在，且文件首行条目头与
  索引行的日期、标题一致；错位即非零退出。

用法::

    python rebuild_worklog_index.py            # 重建并写回
    python rebuild_worklog_index.py --dry-run  # 只审计，不写文件
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ENTRY_RE = re.compile(r"^## (\d{4}-\d{2}-\d{2})(?:\s+(\d{2}):(\d{2}))?\s*\|\s*(.+?)\s*$")
INDEX_COUNT_RE = re.compile(r"^## 条目索引（全 \d+ 条；最新在上）\s*$")
LINK_RE = re.compile(r"\[(.+?)\]\((.+?)\)")
TABLE_HEADER = "| 日期 | 标题 | 主线 |"
TABLE_SEP = "| --- | --- | --- |"
VALID_TAGS = {"L1", "L2", "L3", "对比", "数据集", "审查", "配置", "其他"}

# 人工归号种子：条目从未上过索引时生效；正常重建时标签从现有索引表继承。
TAG_OVERRIDES = {
    "提交 L1 代码文档与实验记录": "配置",
    "跨电脑工作区与数据集同步基础": "配置",
    "未来计划与工作记录机制只读评估": "配置",
    "补齐旧方案五文件本地归档": "配置",
    "建立旧方案与现行优化桥接索引": "配置",
    "纠正三光与等级模型实施顺序": "配置",
    "旧方案文档与当前优化进度只读对照审计": "配置",
    "修正工作日志显式主线标签覆盖规则": "配置",
    "用户要求保存结果并暂停，后续继续": "配置",
    "清理会话临时文件": "配置",
    "清理所有临时/测试产物": "配置",
    "L1中心总体计划与三线新线分层": "L1",
    "修正 L1 路线将 OOF 提升为数据质量主线": "L1",
    "L1 主控计划只读审阅与闸道改进建议": "L1",
    "项目计划重构方案：单交付主线与冻结规则": "配置",
    "澄清低质量数据默认保留完整双模型 OOF 路线": "L1",
    "L1 OOF评测契约冻结并启动YOLO11m Fold2": "L1",
    "L1 OOF Fold2 双模型并行启动": "L1",
    "L1 YOLO11m OOF Fold2归档并流水启动Fold3": "L1",
    "L1 YOLO26m OOF Fold2归档并因配额暂缓Fold3": "L1",
    "L1 YOLO26m OOF Fold3已提交并与YOLO11m并行": "L1",
    "L1 OOF双模型Fold3远端完成待归档": "L1",
    "L1 OOF候选Fold2-3归档复评与三折汇总": "L1",
    "L1 OOF无人工自动闸门通过与最终训练包准备": "L1",
    "L1 OOF最终YOLO11m训练已提交Kaggle": "L1",
    "L1 OOF最终YOLO26m并行训练已提交Kaggle": "L1",
    "L1 OOF最终候选归档、发布确认与ONNX交付": "L1",
    "L1-hr 新版标签 1280 Fold 1 已提交 Kaggle": "L1",
    "L1-hr 1280 Fold 1 归档并判定不扩展": "L1",
    "L1-切片 Fold1 目录初始化错误修复并重提 Kaggle 版本 2": "L1",
    "L1-切片 Fold1 缺失Ultralytics修复并重提 Kaggle 版本 3": "L1",
    "L1-切片 version 2 索引标签修正与状态复核": "L1",
    "L1-切片训练推理 Fold1 已提交 Kaggle": "L1",
    "L1-切片 version 3 中断证据回收与独立评测准备": "L1",
    "L1-切片恢复检查点评测已提交 Kaggle": "L1",
    "L1-切片独立评测完成并停止扩展": "L1",
    "L1 小目标 Phase0 完成与 Phase1 评测包准备": "L1",
    "L1 小目标 Phase1 后处理评测已提交": "L1",
    "L1 小目标 Phase1 归档验证并停止切片方向": "L1",
    "L1 E2 成对 Mosaic 消融任务包与校验器准备": "L1",
    "L1 E2 成对 Mosaic 任务已提交 Kaggle": "L1",
    "L1 E2 成对 Mosaic 验收完成并停止扩展": "L1",
    "L1 计划追加 E1b 逐类置信度阈值扫描": "L1",
    "L1 E1b、E3、E4 零 GPU并行执行完成": "L1",
    "L1 E3/E4 Fold 1 并行 GPU 提交": "L1",
    "L1 E3/E4 version 1 错误修复并重提 version 2": "L1",
    "L1 E3/E4 version 2 共同崩溃诊断并重提 version 3": "L1",
    "L1 E3/E4 version 3 绘图回调诊断并重提 version 4": "L1",
    "L1 E3/E4 version 4 回收评测并停止扩展": "L1",
    "L1 E5 数据中心计划只读审计": "L1",
    "L1 YOLO11m 冻结层训练可行性只读核验": "L1",
    "L1 YOLO11m Freeze10 成对 Fold 1 训练已提交 Kaggle": "L1",
    "L1 YOLO11m Freeze10 成对训练 v1 故障修复并重提 v2": "L1",
    "L1 YOLO11m Freeze10 成对 Fold 1 归档复评与闸门未通过": "L1",
    "L1-研究生毕业与论文旁路路线图制定": "L1",
    "L1-创建独立研究生毕业工作区": "L1",
    "L1-独立论文仓库 Git 与治理架构初始化": "L1",
    "L1 YOLO26m Fold 1 正式归档并完成固定开发集复评": "L1",
    "L1 YOLO11m Fold 2 已提交 Kaggle": "L1",
    "L1 YOLO11m Fold 2 正式归档并完成固定开发集复评": "L1",
    "L1 YOLO26m Fold 2 已提交 Kaggle": "L1",
    "L1 YOLO26m Fold 2 正式归档并完成固定开发集复评": "L1",
    "L1 两架构 Fold 3 并行提交，YOLO26m Fold 2 归档中": "L1",
    "L1 两架构 Fold 3 远端完成并开始并行归档": "L1",
    "L1 两架构 Fold 3 正式归档并完成三折开发集汇总": "L1",
    "L1 OOF 双模型一致性自动候选完成": "L1",
    "L1 OOF候选开发集视图与Kaggle预检已提交": "L1",
    "L1 OOF预检v1参数修复并重提v2": "L1",
    "L1 OOF候选双模型Fold1预检通过并行提交": "L1",
    "L1 OOF候选双模型Fold1归档复评并触发Kaggle配额阻塞": "L1",
    "L1 OOF候选CPU训练可行性复核": "L1",
    "L1 当前进度与可信度只读复核": "L1",
    "L1 锚点框尺寸只读诊断验证小目标假设": "L1",
    "L1 锚点框尺寸诊断复核发现边界问题": "L1",
    "L1 锚点框尺寸dev-only诊断更正版报告": "L1",
    "L1 后续改为Kaggle-only高分辨率验证计划": "L1",
    "建立 Agent 工作日志与会话交接流程": "配置",
    "记录分类数据集与 JinchuanPythonWebapi 分支关系": "配置",
    "整理文档地图（README 索引化）": "配置",
    "探明 WebAPI 分类逻辑，电积镍最优方案更新为检测+图像特征+LightGBM": "L2",
    "基于 308 分支规则构建电解镍多类 folder 数据集，发现仅覆盖 type 3/4": "L3",
    "统一整理电解镍/电积镍等级分类数据集到一个根目录": "数据集",
    "分析电积镍特征分布，生成候选 4 级严重程度数据集": "L2",
    "撰写分类模型优化工作总结文档": "L2",
    "网络调研电积镍等级标准并制定分类解决方案": "L2",
    "构建企业 0309 电积镍独立数据集及旧新合并视图": "数据集",
    "使用 ryw.pt 重标企业 0309 数据并重新生成辅助分类": "数据集",
    "修正企业目录等级语义并生成电积镍四分类 v3": "L2",
    "L2 LightGBM小样本过拟合只读诊断": "L2",
    "更正 L2 原始等级数据集与合并训练视图关系": "L2",
    "固定数据直接企业落地的 L2 LightGBM 方案": "L2",
    "整理电积镍数据唯一入口并冻结中间产物用途": "数据集",
    "删除全部特征提取和等级划分派生数据集": "数据集",
    "删除两检测集伪标签与 manifests 派生目录": "数据集",
    "删除伪标签汇总与类别映射派生文件": "数据集",
    "删除 visualizations 并构建企业硬链接数据集": "数据集",
    "企业数据集与原始数据集完全查重": "数据集",
    "汇总两镍种数据并统一使用 ryw.pt 生成扁平 YOLO 伪标签": "数据集",
    "为两套扁平伪标签生成可视化审查页": "审查",
}


def read_lines(path: Path) -> tuple[list[str], str]:
    """按字节读取并拆分行，保留原行尾风格（LF 或 CRLF）。"""
    raw = path.read_bytes().decode("utf-8")
    ending = "\r\n" if raw.count("\r\n") * 2 > raw.count("\n") else "\n"
    return raw.replace("\r\n", "\n").split("\n"), ending


def scan_entries_dir(entries_dir: Path) -> list[dict]:
    if not entries_dir.is_dir():
        sys.exit(f"[error] 条目文件夹不存在：{entries_dir}")
    entries = []
    for path in sorted(entries_dir.iterdir()):
        if path.suffix.lower() != ".md":
            if path.is_file():
                print(f"[warn] 跳过非 .md 文件：{path.name}")
            continue
        head = path.read_bytes().decode("utf-8").split("\n", 5)[:5]
        m = next((ENTRY_RE.match(ln) for ln in head if ENTRY_RE.match(ln)), None)
        if not m:
            print(f"[warn] 前 5 行未找到条目头，已跳过：{path.name}")
            continue
        entries.append({
            "date": m.group(1),
            "time": f"{m.group(2)}:{m.group(3)}" if m.group(2) else "",
            "title": m.group(4),
            "filename": path.name,
        })
    if not entries:
        sys.exit(f"[error] {entries_dir} 中未扫到任何条目文件")
    dup = [t for t in {e["title"] for e in entries} if sum(e["title"] == t for e in entries) > 1]
    if dup:
        print(f"[warn] 存在重复标题 {len(dup)} 个，索引链接按文件区分但标签继承可能串号：{dup[:3]}")
    return entries


def find_table(lines: list[str]) -> tuple[int, int]:
    """定位索引表区块 [start, end)：表头行 + 分隔行 + 全部数据行（兼容旧行号版）。"""
    start = next((i for i, ln in enumerate(lines)
                  if ln.strip().startswith(("| 日期 |", "| 行号 |"))), None)
    if start is None:
        sys.exit("[error] 未找到索引表表头行（`| 日期 |` 或旧版 `| 行号 |`）")
    end = start + 2
    while end < len(lines) and lines[end].lstrip().startswith("|"):
        end += 1
    return start, end


def parse_index_tags(lines: list[str], start: int, end: int) -> dict[str, str]:
    """从现有索引数据行解析 标题 -> 主线标签（兼容链接版与旧行号版）。"""
    tags: dict[str, str] = {}
    for ln in lines[start + 2 : end]:
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) == 5:  # 新版：| 日期 | [标题](链接) | 标签 |
            m = LINK_RE.search(parts[2])
            if m:
                tags[m.group(1)] = parts[3]
        elif len(parts) == 7 and parts[1].isdigit():  # 旧版：| 行号 | 日期 | 标题 | 标签 | 位置 |
            tags[parts[3]] = parts[4]
    return tags


def sort_key(entry: dict):
    # 最新在上（倒序）；无时间条目用 00:00 使其沉到当日最后；再按标题稳定次序。
    return (entry["date"], entry["time"] if entry["time"] else "00:00", entry["title"])


def main() -> int:
    ap = argparse.ArgumentParser(description="重建工作日志条目索引表（条目文件夹版）")
    ap.add_argument("--main", default="工作日志.md", help="主文件路径（默认：工作日志.md）")
    ap.add_argument("--entries-dir", default="工作日志条目", help="条目文件夹")
    ap.add_argument("--dry-run", action="store_true", help="只审计并打印结果，不写文件")
    args = ap.parse_args()

    main_path = Path(args.main)
    entries_dir = Path(args.entries_dir)
    lines, ending = read_lines(main_path)

    tbl_start, tbl_end = find_table(lines)
    old_row_count = tbl_end - tbl_start - 2
    tags = parse_index_tags(lines, tbl_start, tbl_end)

    entries = scan_entries_dir(entries_dir)
    rows, new_titles = [], []
    for e in sorted(entries, key=sort_key, reverse=True):
        inherited_tag = tags.get(e["title"])
        # An old automatic "其他" must not defeat a later explicit mainline tag.
        tag = TAG_OVERRIDES.get(e["title"]) if inherited_tag == "其他" else inherited_tag
        tag = tag or TAG_OVERRIDES.get(e["title"])
        if tag is None:
            tag = "其他"
            new_titles.append(e["title"])
        rows.append({**e, "tag": tag})
    stale = sorted(set(tags) - {e["title"] for e in entries})

    block = [TABLE_HEADER, TABLE_SEP] + [
        f"| {r['date']} | [{r['title']}]({entries_dir.as_posix()}/{r['filename']}) | {r['tag']} |"
        for r in rows
    ]
    new_lines = lines[:tbl_start] + block + lines[tbl_end:]
    new_lines = [
        INDEX_COUNT_RE.sub(f"## 条目索引（全 {len(rows)} 条；最新在上）", ln) for ln in new_lines
    ]
    while new_lines and not new_lines[-1].strip():
        new_lines.pop()
    new_lines.append("")

    print(f"条目文件：{len(entries)} 个；索引数据行：旧 {old_row_count} -> 新 {len(rows)}")
    if new_titles:
        print(f"新收录条目（标签为继承/种子/其他，请人工复核）: {len(new_titles)} 条")
        for t in new_titles:
            print(f"  + {t}")
    if stale:
        print(f"[warn] 旧索引中存在、条目文件夹已扫不到的标题 {len(stale)} 条：")
        for t in stale:
            print(f"  ? {t}")

    if args.dry_run:
        print("[dry-run] 未写文件。")
        return 0

    main_path.write_bytes(ending.join(new_lines).encode("utf-8"))

    # 写后复检：重读主文件索引，逐行核对链接文件存在且首行条目头一致。
    check_lines, _ = read_lines(main_path)
    cs, ce = find_table(check_lines)
    bad = 0
    for ln in check_lines[cs + 2 : ce]:
        parts = [p.strip() for p in ln.split("|")]
        m = LINK_RE.search(parts[2]) if len(parts) == 5 else None
        target = entries_dir / (m.group(2).rsplit("/", 1)[-1]) if m else None
        head = target.read_bytes().decode("utf-8").split("\n", 1)[0] if target and target.is_file() else ""
        hm = ENTRY_RE.match(head)
        if not (m and target.is_file() and hm and hm.group(1) == parts[1] and hm.group(4) == m.group(1)):
            bad += 1
            print(f"  ! 索引行异常：{ln[:80]}")
    if bad:
        print(f"[error] 复检失败，{bad} 行索引与条目文件错位。")
        return 2
    if ce - cs - 2 != len(entries):
        print(f"[error] 复检失败，索引行数 {ce - cs - 2} 与条目文件数 {len(entries)} 不一致。")
        return 2
    print(f"[ok] 已写回 {main_path}，复检通过：{len(rows)} 行索引与条目文件一一对应。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
