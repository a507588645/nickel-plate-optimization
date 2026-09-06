# 金川镍板检测 WebAPI 六大算法技术解析

> 项目：`E:/PythonProject/ImetricCodes/JinchuanPythonWebapi`（YOLOv5 + ResNet + Flask WebAPI，端口 5005）
> 用途：研究生接手前的代码研读底稿。所有结论均来自源码逐行阅读，缺陷类别名以权重文件（`ryw.pt` / `slf.pt`）内嵌 names 为准验证过。
>
> **分支与核实说明（2026-08-30 补充）**：本文描述的是 **`308` 分支**代码（核实基准 HEAD = `5cb215a`）；后续算法优化与融合落在 **`优化分支`**，引用本文行号前必须先确认目标分支。2026-08-30 已经 Kimi 对照 308 分支源码与权重文件逐条核实：8 条红旗、决策树阈值、类别名、标定常数全部成立，行号几乎零漂移；本次收编时已修正两处实质错误（类别 16 拼写 `mijixiaokeng`、§3.3 >800 过滤的坐标尺度）。仍存的轻微不严谨处（不影响主结论）：§0「比例恰好都是 4」中 3554/888≈4.0023 是近似（888×4=3552≠3554）；§七.11 debug 脏文件清单另遗漏 `rightbug.txt`、`bottomcutbug.txt`、`bioplist.txt`；§0 加密密钥派生实为「机器ID 先经固定盐+SHA1 取半+Base62，再 SHA256」。

## 0. 全局架构与模型清单

Flask + flask_restful 服务（`start.py`），启动时加载 5 个模型（`start.py:588-592`）：

| 加载函数 | 权重 | 用途 | 加载方式 |
|---|---|---|---|
| `load_yolov5()` (start.py:43) | `ryw.pt` | 电解镍缺陷检测，**18 类** | 优先读 `.enc` 加密文件，否则 `torch.hub.load('.', 'custom', source='local')` 本地加载，外包 `AutoShape` |
| `load_slf_yolov5()` (start.py:56) | `slf.pt` | 电积镍缺陷检测，**5 类** | 同上 |
| `load_resnet()` (start.py:67) | `ryw.pth` | 电解镍整板分类，fc 输出 **2 类**（bad=0/good=1） | `torch.load` 反序列化完整 pickled 模型（非 state_dict），类为 `resnet.ResNet18` |
| `load_slf_resnet()` (start.py:82) | `slf.pth` | 电积镍整板分类，fc 输出 **4 类**（3#/9950/9996/bad） | 同上 |
| `load_yolov5seg()` (start.py:100) | `best.pt` | 镍条实例分割（2 类 background/nietiao） | `segment/segpredict.loadmodel()` → `DetectMultiBackend` |

7 个路由（`start.py:577-583`）：

| 路由 | 类 | 对应算法 |
|---|---|---|
| `/dianjiEclass` | `DianjieDetection` | 电解镍分类 + 缺陷检测 |
| `/dianjiclass` | `DianjiDetection` | 电积镍分类 + 缺陷检测 |
| `/nietiaoSeg` | `NietiaoSeg` | 镍条识别 |
| `/cutting` | `Huaxian` | 划线算法 |
| `/preprocess` | `Preprocess` | 预处理（旋转校正，配套） |
| `/abdetect` | `AbDetect` | 拍照质检（暗/亮/模糊，配套） |
| `/isNieban` | `IsNieban` | 有无板判断（复用镍条分割模型） |

**YOLOv5 推理通用流程**：`AutoShape`（`models/common.py:623`）默认 NMS 参数 conf=0.25、iou=0.45、agnostic=False、max_det=1000，代码中均未覆盖；输入原始路径（4096×3554），`results.xyxy[0]` 坐标是原图尺度。

**贯穿全场的两个手法**：
1. 原图 4096×3554 **÷4** 到 1024×888 工作坐标系做中间计算，输出前再 **×4** 回原图坐标（两者比例恰好都是 4，代码中 `banw/tow` 与 `banh/toh` 混用却不出错）。
2. 缺陷面积用布尔掩码置位求**并集**（重叠区域只算一次），见 `scalculate`。

模型加密机制（`model_decryptor.py`）：文件头魔数 `MODEL_ENC_V1`，AES-CBC，密钥 = SHA256(机器ID)，机器ID 取主板序列号或注册表 MachineGuid；模型参数另做 0x55 XOR 混淆。当前目录无 `.enc` 文件，实际走明文回退分支。

---

# 一、电解镍分类 + 缺陷检测（`/dianjiEclass`）

入口：`DianjieDetection.post()`（`start.py:118-234`）。

## 1.1 功能定位

输入一张电解镍板图像路径，输出整板等级 `type`、结瘤标志 `nodule`、缺陷框列表 `defects`。评级 = ResNet 整板分类 + YOLOv5 缺陷框面积占比统计的组合决策树。

## 1.2 缺陷 label 0-17 真实类别（从 `ryw.pt` 权重二进制提取，权威）

```
0 wuzi 污渍        1 waterstain 水渍      2 yellowstain 黄渍
3 nodule 边部结瘤   4 airhole 气孔        5 bubble 结粒
6 bump 疙瘩         7 fawu 发乌           8 overfire 烧板
9 NiOH 氢氧化镍     10 greenstain 绿色结晶 11 suanwuran 酸污染
12 quejiao 缺角     13 watermark 严重水渍 14 youpao 油泡
15 heibian 黑边     16 mijixiaokeng 密集小坑 17 fanxitong 反析铜
```

⚠️ 仓库里的 `data/nieban420.yaml` 和 `data/nieban420_xiangmu.yaml` 都是训练早期旧配置，与生产权重**不一致**，不能作为类别依据。

## 1.3 逐步计算逻辑

1. **ResNet 分类**（start.py:127-136）：图像 → ToTensor → Resize(224,224) → ImageNet mean/std 归一化 → `resnet(image)` → argmax。语义：bad=0 / good=1。
2. **YOLOv5 检测**（start.py:143）：`yolo(image_path)` → `results.xyxy[0]`，每框转 `{label, cx, cy, w, h}`（原图坐标）。
3. **边部结瘤过滤**（start.py:158-165）：对 label==3 的框，面积 = w×h，长宽比 = 长边/短边。**面积 < 13000 px² 且长宽比 < 3.0 → 忽略**（continue）；否则 `flag_nodule=True`。
   - 几何含义：真实边部结瘤是沿板边延伸的长条状或大体积凸起；面积小且形状近方圆的框视为噪声/镍耳误检。13000 是原图尺度（约 114×114 px 见方）。
4. `flag_wuzi_watersatin`（start.py:152,168-170）：检测到 label∈{2,4..13} 时置 False——**计算后从未使用，死代码**（对应被注释的旧分支 start.py:147-148、211-212）。
5. **镍板区域求取**：`image1 = readimg()` 缩到 1024×888 → `findcent(image1)` → `nieban_area = width×height`（1024×888 尺度）。
6. **四类缺陷面积统计**：`read_A/read_B/read_C/read_airhole` 筛选 label、坐标 **÷4** 到 1024×888 尺度，交 `scalculate` 求并集面积：
   - `read_A`（preprocess.py:326）：label∈{0,1} = 污渍、水渍 → A 类（可烫洗）
   - `read_B`（preprocess.py:343）：label∈{2,4,5,6,10,12,13} = 黄渍、气孔、结粒、疙瘩、绿色结晶、缺角、严重水渍 → B 类
   - `read_C`（preprocess.py:360）：label∈{7,8,11} = 发乌、烧板、酸污染 → C 类
   - `read_airhole`（preprocess.py:377）：label==4 气孔（B 类子集，单独设更严阈值）
7. **占比**：`X_ratio = X_area / nieban_area`（同一尺度，自洽）。
8. **决策树**（start.py:214-230，按 if 顺序短路）：

```
① 无结瘤 且 A<0.2 且 B<0.1 且 Airhole<0.01 且 C<0.03
        → type=3 合格整版, nodule=0
② 有结瘤 且 ResNet 判 good(1)
        → type=1 3号板, nodule=1            ← 唯一用到 ResNet 结果的活分支
③ 无结瘤 且 A≥0.2 且 B<0.05 且 C<0.05
        → type=2 烫洗板, nodule=2           ← 只有水渍/污渍超标才烫洗
④ C>0.5  → type=1 3号板, nodule=3
⑤ B>0.5 且 C<0.5 → type=6 2号板, nodule=3
⑥ 其余   → type=4 合格600, nodule=3        ← 兜底，"等待最终处理方案"
```

## 1.4 输出编码

- `type`：0 无板 / 1 三号板 / 2 烫洗板 / 3 合格整版 / 4 合格600 / 5 合格不足600（仅划线链路产生）/ 6 二号板
- `nodule`：0 合格无结瘤 / 1 合格有结瘤 / 2 烫洗板 / 3 其他
- `defects`：`{label, cx, cy, w, h}` 原图坐标

## 1.5 决策树的漏洞

- **有结瘤且 ResNet 判 bad** 时不命中②，跌进③-⑥，结瘤事实被丢弃，可能返回"合格600"兜底。
- ①合格要求 B<0.1、C<0.03，③烫洗要求 B<0.05、C<0.05；A≥0.2 且 B∈[0.05,0.1) 之类的中间态直接掉到⑥。
- ③④⑤⑥ 完全不看 ResNet——**分类模型几乎形同虚设**（且 fc 被随机重置，见 §七红旗 1）。

---

# 二、电积镍分类 + 缺陷检测（`/dianjiclass`）

入口：`DianjiDetection.post()`（`start.py:236-308`）。

## 2.1 功能定位

电积镍板的牌号分类 + 缺陷框回传，**无面积占比决策树**，分类结果直接映射为 type。

## 2.2 逐步逻辑

1. **分类**（start.py:251-254）：预处理同上（224×224 + ImageNet 归一化）。注意 `# outputs = slf_resnet(image)` 被注释，实际 `outputs = resnet(image)`——**用的是电解镍的 2 分类 ResNet，而非电积镍的 4 分类 slf_resnet**（git 提交 3360ca0 自述："电积镍缺陷检测使用电积镍模型，分类仍暂时使用电解镍的"）。
2. **type 映射**（start.py:262-272）。设计语义（slf_resnet 4 类）：`3#=0, 9950=1, 9996=2, bad=3` → `type: bad=1, 9996=2, 9950=3, 3#=4`：
   - idx==2 → type=2（9996），**直接 return，不做缺陷检测**
   - idx==1 → type=3（9950）；idx==0 → type=4（3#）；else → type=1（bad）
   - ⚠️ 当前 resnet 只有 2 个输出，predicted_idx 只可能是 0/1 → **type=2(9996) 和 type=1(bad) 分支实际不可达**，实际行为退化为：good→9950、bad→3#。
3. **缺陷检测**（start.py:274）：`slf_yolo(image_path)`（电积镍专用 5 类模型），NMS 默认 conf=0.25/iou=0.45。
4. **label 映射**（start.py:281-287）。slf.pt 真实 names：`0 waterstain 水渍, 1 waterblock 深色水块, 2 colorstain 带颜色渍, 3 nodule 边部结瘤, 4 bubble 结粒`。映射到电解镍 label 空间：

| slf label | 电积镍含义 | → 电解镍 label | 电解镍含义 |
|---|---|---|---|
| 0 | waterstain 水渍 | 1 | waterstain 水渍 |
| 1 | waterblock 深色水块 | 0 | wuzi 污渍 |
| 2 | colorstain 带颜色渍 | 2 | yellowstain 黄渍 |
| 3 | nodule 边部结瘤 | 3 | nodule 边部结瘤 |
| 4 | bubble 结粒 | 5 | bubble 结粒 |

   映射原因（注释 start.py:280）：前端显示软件统一采用电解镍缺陷种类表，电积镍的类别必须翻译过去。
5. **输出**：`{"type": type, "defects": output}`，无 nodule 字段、无过滤、无面积统计。

---

# 三、镍条识别（`/nietiaoSeg` 与 `/isNieban`）

入口：`NietiaoSeg.post()`（`start.py:491-552`）、`IsNieban.post()`（554-574），核心 `segment/segpredict.py` 的 `infer`。

## 3.1 功能定位

对输送线上镍条图像做实例分割，提取最小外接矩形，换算成**物理坐标（毫米）**下的中心点与倾角，供机器人抓取；兼作"是否有板"判据。

## 3.2 模型与预处理

- YOLOv5-seg，权重 `best.pt`，输入 `imgsz=(640,640)`，letterbox 保宽高比缩放 + 灰边补齐，`/=255` 归一化（**无均值/方差归一化**，YOLOv5 惯例）。
- NMS：`non_max_suppression(conf_thres=0.25, iou_thres=0.45, max_det=1000, nm=32)`（segpredict.py:196，全是默认值）。
- 掩膜还原：`process_mask(proto, det[:,6:], det[:,:4], upsample=True)`（掩膜系数×原型掩膜，裁剪到框内，上采样到 640×640）。

## 3.3 从分割结果到输出的逐步计算

1. **掩膜转轮廓**：`masks2segments` 提取外轮廓，`scale_segments(..., normalize=True)` 缩回原图并归一化 0~1。
2. **尺寸过滤**：仅保留 bbox 宽>800 或 高>800（**原图尺度像素**，过滤前已 `scale_boxes` 还原到原图尺寸）的实例——只认大目标镍条。
3. **取第一个实例**：`segresults[0]`——多条镍条只取第一条。
4. **归一化→绝对坐标**：重新 `cv2.imread(source)` 拿原图 `w,h`，`(int(x*w), int(y*h))`。
5. **最小外接矩形**：`cv2.minAreaRect(点集)` → `rect=((cx,cy),(w,h),raw_angle)`；`cv2.boxPoints` 得 4 顶点。
6. **角度换算**（segpredict.py:301-317，对应提交 5cb215a "修复新版 OpenCV 角度不对"）：
   - 基于旧版 OpenCV `raw_angle ∈ [-90,0)` 约定；
   - `raw_angle >= -45` → `angle = raw_angle`（长边近水平）；
   - 否则 `angle = raw_angle + 90`（长边近垂直，统一为"长边相对水平线倾角"，收敛到 (-45,45]）；
   - `|angle|<0.01` 强制置 0.01（避免下游把 0 当错误）；钳制到 [-90,90]。
   - ⚠️ OpenCV ≥4.5 已改为 [0,90) 约定，需对照现场 OpenCV 版本实测。
7. **像素→毫米换算**（segpredict.py:321-326，硬编码现场标定参数）：
   - 比例系数 `k = 0.901` mm/像素；
   - 原点参考：x 方向 1224、y 方向 1024（**不是**图像几何中心 2048/1777，做过现场标定偏移）；
   - `center_x = round((cx_px - 1224)*k + 2.04, 2)`；`center_y = round(((cy_px - 1024)*k + 2.04)*(-1), 2)`——y 取负因图像 y 向下、机器人 y 向上；`+2.04` 标定补偿；
   - `mmWidth = width*k`，`mmHeight = height*k`（注意是旋转框两边长，非水平包围盒）。
8. 返回 `(mmWidth, mmHeight, angle, center_x, center_y, box)`；无检测返回 `(-1,0,0,0,0,0)`。

## 3.4 start.py 的抓取区域与完整性判断

- **抓取区域**：`x_range=[588,1828], y_range=[360,1688]`（原图像素坐标）。矩形 4 顶点逐点判断，任一点出界 → `"镍条位于抓取区域外"`。
- **完整性**：`width>500 or height>500`（**毫米**）→ 否则 `"镍条不完整"`。
- **输出字段**：`width/height`（mm）、`angle`（度，长边相对水平线）、`center_x/center_y`（mm，机器人坐标系）、`point1~point4`（原图像素四顶点）。
- `IsNieban`：`width==-1` → `type:1`（无板），否则 `type:0`，语义反直觉但为既定协议。

## 3.5 可疑点

- segpredict.py:331-339 有一套**冗余区域判断** `x_range=[980,2860], y_range=[470,1590]`，只用于打印 'ERROR'，是死代码，且与 start.py 的区域不一致。
- 每次 `infer` `cv2.imread(source)` **读两遍原图**（w、h 各一次），性能浪费。
- `box` 顶点顺序未统一（`cv2.boxPoints` 顺序随版本/角度变化），下游 point1~4 语义不稳定。
- 区域判断循环每次迭代都赋值 flag，效果等价于"最后一点决定"，写法易误读。

---

# 四、划线算法（`/cutting`，核心中的核心）

入口：`Huaxian.post()`（`start.py:372-489`），核心 `preprocess.py: cutmsg`（420-736 行）及其辅助函数。

## 4.1 坐标系与编码约定

- **工作坐标系**：1024×888（原图 ÷4）。缺陷矩形统一 `[x1,y1,x2,y2]`。
- **输出坐标系**：`positions` ×4 回原图。
- 产品类型：`type=0 无板; 1 三号板; 2 烫洗板; 3 合格整板; 4 合格600(小板); 5 合格不足600; 6 二号板`。
- 切割方向：`direction = 1/2/3/4 → 下/右/上/左`，0 不切。
- 条带状态：`states` 中 1=合格，2=坏。

## 4.2 入口流程（Huaxian.post）

接收正/反两面图 + 两份缺陷 JSON：

1. `findcent` 求每面板角度/宽/高/中心 → center±w/2、h/2 得板区域 `img_edge`。
2. **硬编码**：`center1 = center2 = (512,444)`（start.py:410-411）覆盖刚算出的真实中心——依赖上游 `xuanZhuan` 已把板居中，失效则划线整体偏移且不报警。
3. `readarray` / `readarray_C` / `read_nocorner` 三种规则解析 JSON。
4. `jianbian` 求结瘤贴边信息；`bioplist` 合并两面缺陷；`cutmsg` 决策。
5. `djtype` 合成（start.py:444-452）：默认 5；两面 nodule 都为 0 → 0；两面都不是 3 且至少一面是 2 → 2；其余保持 5。语义：`0 合格无结瘤, 1 合格有结瘤, 2 烫洗, 3 其他`。

## 4.3 JSON 解析与分类过滤

- **`readarray`（212 行）**：JSON `{"defects":[{label,cx,cy,w,h}...], "nodule":int}` → 中心+宽高算左上/右下，坐标 **÷4**。`label==3` → `nodule_info`；**`label!=0 且 !=1`**（除污渍/水渍外的一切，含结瘤本身）→ `plist`。返回 `(plist, nodule_info, ndtype)`。注意结瘤同时进两个列表（面积重复计入，见可疑点）。副作用：每次调用追加写 `plistdebug.txt`。
- **`readarray_C`（270 行）**：只取 label∈{7,8,11}（C 类），÷4。用于"C 类面积≥50% 判三号板"。
- **`read_nocorner`（307 行）**：只取 label==12（缺角）。缺角无法通过平行切割修复，只要存在就直接判三号板（cutmsg:448）。

## 4.4 公共基础函数

### `findcent(image)`（24 行）—— 板的角度/宽/高/中心

1. 灰度化 → `cv2.threshold(gray, 40, 130, THRESH_BINARY)` 固定阈值二值化（板亮背景暗；注释"90,125"与代码不符，已过期）。
2. `findContours(RETR_TREE)` → 取**面积最大**轮廓为镍板。
3. `minAreaRect` + `boxPoints` 得四角点；按 x+y 最小（左上）`np.roll` 重排为顺时针。
4. **下边两角点 y 各上移 55px**——去掉镍耳/吊耳区域。
5. 角度：`theta = atan((y1-y0)/(x1-x0))` 弧度转角度后 `int()` 截断（分辨率仅 1°）。
6. 宽 = `Δx/cos(|theta|)`，高 = `Δy/cos(|theta|)`；中心 = 四角点均值。
7. 返回 `(theta, width, height, (cx,cy))`。
- 🚩 **真实 bug**：`math.cos` 接收弧度，这里传入**角度制整数**（preprocess.py:48-49）。theta≈0 时 cos≈1 被掩盖，角度稍大宽高即出错。

### `scalculate(array)`（135 行）—— 缺陷并集面积

- 建 `np.zeros((4069,3554))` 掩码（4069≠4096 疑似笔误；实际输入是 ÷4 后坐标，远够用）。
- 每个矩形双重循环置 1 → `np.sum`。布尔置位保证**重叠只算一次**。
- 纯 Python 双重循环 + 每次分配约 115MB float64，是主要性能瓶颈。

## 4.5 两面缺陷合并 `bioplist`（394 行）

- 设计意图：上/下表面缺陷对齐合并（平移+缩放代码被注释）。
- **实际**：`xshift=0, yshift=0, scale=1` 全硬编码（上游已对齐），退化为 plist2 坐标 clamp 后直接 `extend` 拼接。
- 🚩 clamp 上下界张冠李戴（403-410 行）：y1 用 1023 截断、x2 用 887 截断，x/y 边界互换。

## 4.6 结瘤剪边 `jianbian(nodule_info, img_edge)`（739 行）

- 对每个结瘤，判断整条边是否贴在板某条边上（两端点都在 30px 容差内）：下→1，右→2，上→3，左→4。
- 输出如 `[1,2]` = "下边和右边有结瘤"，下游据此剪边。30px(工作系)=120px(原图)。
- 可疑：同一结瘤可命中多条边、列表可重复；容差硬编码。

## 4.7 切刀与条带求交工具

- **`bads(arrs, i, p)`（149 行）一刀切**：i=0 保留切线 x=p 左侧（跨界矩形裁剪到切线），i=1 右侧，i=2 上侧，i=3 下侧。配合 `scalculate` 得"沿此线切掉一侧后剩多少缺陷面积"。
- **`nietiaobads(arrs, p1, p2)`（183 行）**：水平条带 y∈[p1,p2] 内缺陷（y 向裁剪），用于横切分条评估。
- **`nietiaobadsy(arrs, p1, p2)`（197 行）**：竖直条带 x∈[p1,p2]，用于竖切方案。

## 4.8 核心 `cutmsg(center, array, array_C, noduleedges, nocorner, ndtype)`（420-736 行）

### 常量

- `snieban = 720×740 = 532800`（工作系标准整板标称面积）
- `wslice=120`、`hslice=125`（工作系"一刀条带"宽/高，×4 ≈ 480/500 原图像素，对应 600mm 小板的 1/6 分条）
- 条带好坏阈值：5000（五刀方案）/ 4000（一刀方案）像素²

### 决策流程（文字流程图）

```
areas=scalculate(array), areas_C=scalculate(array_C), areas_nocorner=scalculate(nocorner)
│
├─① 有边部结瘤(noduleedges非空) 或 areas_C ≥ 50%·snieban 或 有缺角
│     → type=1 (三号板)，不切割，cut_edges=noduleedges（下游剪边）
│
├─② areas ≥ 60%·snieban（缺陷太多）
│     → type=6 (二号板)，不切割
│
├─③ 10% ≤ areas < 60%（中等缺陷）→ 横切五刀方案（见 4.8.1）
│
├─④ 1% < areas < 10%（少量缺陷）→ 一刀/两刀方案（见 4.8.2）
│
└─⑤ areas ≤ 1%（几乎无缺陷）
      → ndtype==3 ? type=2(烫洗板) : type=3(合格整板)，不切割
```

### 4.8.1 分支③：横切五刀（461-569 行）

1. **切线**：以板中心 y 为基准，5 条水平切线 `slices[k] = (center_y + (k-2)·hslice)·4 + 20`（+20 是原图尺度 5px 经验偏移），把板分成 **6 条**镍条。
2. **逐条判好坏**：`nietiaobads` 求条带内缺陷 + `scalculate` 求面积；>5000 → `gdorbd[i]=2`（坏），否则 1（好）。
3. **升降级**（gdorbd 求和 = 6 + 坏条数）：
   - sum < 7（0 条坏）→ 不用切，升级 `type=3` 合格整板；
   - sum(gdorbd[0:4]) < 5（上 4 条全好）→ `type=4` 合格600：`direction=1`（下切），切两刀于 `center_y+hslice`、`center_y+2hslice`，保留下 3 条（600mm 小板 = 3 条×2 刀间距），`states=[1, gdorbd[4], gdorbd[5]]`；
   - sum(gdorbd[2:6]) < 5（下 4 条全好）→ `type=4`，`direction=3`（上切），切于 `center_y-2hslice`、`center_y-hslice`，`states=[gdorbd[0], gdorbd[1], 1]`；
   - sum > 9（≥4 条坏）→ 降级 `type=6` 二号板；
   - 都不满足 → `type=5` 合格不足600：5 刀 6 条，`positions=slices`，`states=gdorbd`（每条各自流转）。

### 4.8.2 分支④：少量缺陷一刀方案（571-727 行）

1. **选边**：`bads` 分别算从左/右/上/下切一刀能去掉多少缺陷：`s0` = `center_x - wslice` 以左面积、`s1` = `center_x + wslice` 以右、`s2` = `center_y - hslice` 以上、`s3` = `center_y + hslice` 以下；`nswe = argmax([s0,s1,s2,s3])`，朝缺陷最多侧切。
2. **切两刀出小板**：以选中边为基准，在 `center±wslice`、`center±2wslice`（或 h 方向）落两刀，初始 `states` 假设 `[2,2,1]` 或 `[1,2,2]`（最外条假设好），再用 `nietiaobads(y)+scalculate` 复核另两条：<4000 改判 1。
3. 若 `sum(states) < 4`（实际缺陷比预估少）→ 升级 `type=3` 合格整板不切。
4. 方向映射：nswe=0 左→4；1 右→2；2 上→3；3 下→1。

### 4.8.3 输出格式

```python
{'type': int,           # 0/1/2/3/4/5/6 产品等级
 'cutting_count': int,  # 切刀数
 'direction': int,      # 0不切 1下 2右 3上 4左
 'positions': [int...], # 切线坐标，已×4 回原图坐标系
 'states': [1/2...],    # 各条带好坏
 'cut_edges': [...]}    # 剪边信息（1下2右3上4左）
```

---

# 五、预处理算法（`/preprocess`，配套）

入口：`Preprocess.post()`（`start.py:310-345`）→ `xuanZhuan` / `xuanZhuan_and_flip`（preprocess.py:61/97 行）。

- `readimg`（9 行）：`cv2.imread` → resize 到 (1024,888)。`readimgtb`（16 行）先 `cv2.flip(img,1)` 水平镜像（用于底面相机 `bottom_0`）。⚠️ 均不兼容中文路径（项目其他地方用 `cv2.imdecode(np.fromfile(...))` 规避，这里没做）。
- **`xuanZhuan` 流程**：readimg 降采样 → `findcent` 求角度 → `imdecode` 重读原图（兼容中文路径）→ `getRotationMatrix2D + warpAffine` 绕图中心旋转 → resize 1024×888 → 再次 `findcent` → 平移使板中心到 (512,444) → 放大回 4096×3554 保存。返回的角度字段被替换回**旋转前的原始角度**（告诉调用方转了多少度）。
- `xuanZhuan_and_flip`：多一步水平镜像，用于底面图像（start.py:324-327 按文件名 `bottom_0` 分派）。
- 可疑：`kernel=np.ones((7,7))` 声明未用（死代码）；固定阈值二值化在光照变化下会失效；旋转后四角出现黑边，板靠图像边缘时可能把板角转出画布。

---

# 六、拍照质检（`/abdetect`，配套）

入口：`AbDetect.post()`（`start.py:347-370`）→ `abdetect.py: is_dark_light_clear(imagepath, t1, t2, t3)`，默认 t1=100, t2=200, t3=500。

纯传统图像统计，全局单值（**无分块**）：

1. 灰度化。
2. **平均灰度** `cv2.mean(gray)[0]` ∈ [0,255]。
3. **Laplacian 方差** `cv2.Laplacian(gray, CV_64F).var()`——经典无参考清晰度指标，方差大=边缘锐利=清晰。
4. 判定（优先级：清晰度 > 曝光）：
   - `imageVar ≤ t3` → type=3（模糊），直接返回；
   - `mean < t1` → type=2（过暗）；
   - `mean > t2` → type=1（过亮）；
   - 否则 type=0（正常）。

可疑：`is_exist_full`（abdetect.py:27）是空壳 TODO；阈值从 API 传入但无范围校验。

---

# 七、红旗与可疑点汇总（按严重度排序）

1. 🚩 **ResNet 分类头被随机重置**（start.py:74-75、89-90）：`torch.load` 还原训练好的完整模型后，又执行 `resnet.fc = nn.Linear(num_ftrs, 2/4)`——随机初始化覆盖了训练好的分类头，分类结果接近随机。正确写法只在"加载 state_dict 到新建模型"时才需替换 fc。这解释了为什么决策树几乎不敢依赖 ResNet。**接手后第一优先级实测验证**。
2. 🚩 **findcent 度/弧度混用**（preprocess.py:48-49）：角度制 theta 直接进 `math.cos`（弧度制），theta≥1° 时宽高计算失真；现场板子近水平所以未暴露。
3. 🚩 **板中心硬编码**（start.py:410-411）：划线链路 `center=(512,444)` 写死，`findcent` 结果被丢弃，对位失效无报警。
4. **bioplist 退化**（preprocess.py:399-410）：平移/缩放硬编码为 0/0/1，合并=直接拼接；clamp 的 x/y 边界互换。
5. **烫洗板分支不可达**（cutmsg:729）：判 `ndtype==3` 才给 type=2，但 Huaxian 合成的 djtype∈{0,2,5} 永远≠3；语义上 djtype==2（烫洗）反而映射不到烫洗板。
6. **states 双重 reverse 抵消**（preprocess.py:561-563）；600mm 小板第三条带"假设合格"从不复核；阈值 4000/5000 两套不一致；`cutting_count` 死赋值；竖切边界魔数 1000/900。
7. **结瘤面积重复计入**（preprocess.py:227-263）：label==3 同时进 nodule_info 和 plist（是否刻意需与工艺确认）。
8. **电积镍链路用错分类模型**（start.py:252-253）：2 类 resnet 替代 4 类 slf_resnet，type=1/2 分支不可达（git 注明"暂时"）。
9. **OpenCV 角度兼容**（segpredict.py:301-317）：基于旧版 [-90,0) 约定的换算，需对照现场 OpenCV 版本实测。
10. **数据集 yaml 不可信**：`data/nieban420*.yaml` 与生产权重 names 不一致，以 `ryw.pt`/`slf.pt` 内嵌 names 为准。
11. **工程隐患**：`scalculate` 每次分配 115MB + 双重循环；每次请求写 `debug.txt/plistdebug.txt/clistdebug.txt/tiaodebug.txt/leftbug.txt/'s%.txt'`（`'s%'` 是忘了格式化的字面文件名）堆积脏文件；`readimg` 不支持中文路径；`infer` 每帧读两遍原图；start.py 与 segpredict.py 两套不一致的抓取区域硬编码。
12. **魔数无配置化**：snieban=720×740、wslice=120/hslice=125、+20 偏移、阈值 5000/4000、剪边容差 30、二值化阈值 40、下边收缩 55、抓取区域坐标——现场改参只能改代码。

---

# 八、阅读路线建议

1. 先读 `preprocess.py` 的 `findcent`（24 行）和 `scalculate`（135 行）——所有链路的地基。
2. 再读 `start.py` 两个检测接口（118-308 行）的决策树——理解"深度学习出框、规则定级"的混合架构。
3. 重点攻坚 `cutmsg`（420-736 行）：按"分级(①②⑤) → 中等缺陷五刀分条(③) → 少量缺陷一刀切(④)"三层结构读；`bads/nietiaobads(y)` 只是"矩形裁剪到区间"的几何工具，`jianbian` 只是"贴边判定"，不要被代码量误导。
4. 最后读 `segment/segpredict.py` 的标定换算和 `abdetect.py`（30 行，最简单）。
