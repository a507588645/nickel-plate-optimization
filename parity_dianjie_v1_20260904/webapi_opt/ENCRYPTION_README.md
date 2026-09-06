# 模型加密使用说明

## 功能概述

本项目支持对模型文件进行加密保护，使用主板编号作为密钥种子，确保模型只能在特定机器上运行。

## 文件说明

### encrypt_models.py - 加密脚本
负责加密所有 `.pt` 和 `.pth` 模型文件。

**功能：**
- 自动获取主板编号（支持 Linux/Windows/macOS）
- 使用 SHA-256 生成 AES-256 密钥
- 两层加密：XOR 混淆 + AES-CBC 加密
- 输出 `.enc` 加密文件

**使用方法：**
```bash
# 加密所有模型文件
python encrypt_models.py encrypt

# 测试解密功能
python encrypt_models.py test
```

### model_decryptor.py - 解密加载模块
提供加密模型的内存加载功能。

**主要函数：**
- `get_machine_id()` - 获取机器唯一标识
- `load_encrypted_torch(filepath, device)` - 加载加密模型（兼容 torch.load）
- `load_encrypted_model(filepath, device)` - 核心解密函数
- `load_yolov5_with_encryption(weights_path, device, autoshape)` - YOLOv5 专用加载函数
- `is_encrypted_file(filepath)` - 检查文件是否加密

### test_encryption.py - 测试验证脚本
验证加密/解密功能正确性，包含 SHA1 哈希对比。

**主要测试：**
- ResNet 加密模型加载测试
- YOLOv5 加密模型加载测试
- XOR 参数解密正确性验证（通过 SHA1 对比）
- 所有模型文件 SHA1 汇总

### 启动文件 - 修改后的模型加载入口
自动检测并加载加密模型，支持同时兼容加密和未加密文件。

**新增功能：**
- 自动 GPU/CPU 检测
- 使用 `load_yolov5_with_encryption()` 包装函数
- 模型配置集中到 `models.yaml`
- 无需修改 models 文件夹下的文件

## 加密流程

```
原始文件 (.pt/.pth)
    ↓
  torch.load() 加载模型
    ↓
  参数级 XOR 混淆 (对每个模型参数的字节进行异或，key=0x55)
    ↓
  pickle 序列化
    ↓
  AES-256-CBC 加密
    ↓
  写入魔数 + XOR密钥 + IV + 加密数据
    ↓
加密文件 (.pt.enc/.pth.enc)
```

## 加载流程

```
加密文件 (.enc)
    ↓
  AES-256-CBC 解密
    ↓
  XOR 还原 (key=0x55)
    ↓
内存字节数据
    ↓
torch.load() 加载
    ↓
模型对象
```

## 机器ID获取方式

### Linux
优先级：
1. `/sys/class/dmi/id/board_serial` - 主板序列号
2. `/sys/class/dmi/id/product_uuid` - 产品UUID
3. `dmidecode -s baseboard-serial-number` - DMI信息
4. `/etc/machine-id` - 机器ID
5. MAC地址（备用）

### Windows
1. `wmic baseboard get serialnumber` - 主板序列号
2. 注册表 `MachineGuid` - 备用方案
3. MAC地址（最终备用）

### macOS
1. `ioreg -rd1 -c IOPlatformExpertDevice` - 硬件UUID
2. MAC地址（备用）

## 安全特性

1. **硬件绑定** - 密钥基于主板编号，模型无法在其他机器使用
2. **双重加密** - 参数级 XOR(0x55) + AES-CBC 两层保护
3. **参数混淆** - 对模型所有参数的字节级异或处理，而非文件级混淆
4. **内存加载** - 解密后模型不落盘，仅存在于内存
5. **自动检测** - 自动识别加密文件，兼容未加密文件
6. **SHA1 验证** - 提供文件和模型参数的 SHA1 哈希验证

## 使用示例

### 1. 加密模型

```bash
# 查看所有模型文件
ls -la *.pt *.pth

# 执行加密
python encrypt_models.py

# 输出示例：
# ============================================================
#         模型文件加密工具
# ============================================================
#
# [1/4] 获取机器标识...
# 机器ID: 0123456789ABCDEF
#
# [2/4] 生成AES密钥...
# 密钥(SHA-256): abc123...
#
# [3/4] 查找模型文件...
# 找到 4 个模型文件:
#   - ryw.pt
#   - slf.pt
#   - ryw.pth
#   - slf.pth
#
# [4/4] 开始加密...
# ✓ 已加密: ryw.pt -> ryw.pt.enc
#   原文件大小: 140,123,456 bytes
#   加密后大小: 140,123,488 bytes
# ...
#
# ============================================================
# 加密完成！
# ============================================================
```

### 2. 验证加密

```bash
# 测试解密功能
python encrypt_models.py test
```

### 3. 启动服务

```bash
# 首次先同步 uv 虚拟环境
uv sync

# 启动时会自动检测加密文件并自动识别 GPU/CPU
uv run python server.py

# 输出示例：
# 服务启动后可通过 /health 查看 ready 状态
```

### 4. 测试验证

```bash
# 运行测试套件（包含 SHA1 验证）
python test_encryption.py

# 输出示例：
# ============================================================
# 加密模型加载测试套件
# ============================================================
#
# 机器ID: 0123456789ABCDEF
#
# ============================================================
# 测试 1: ResNet 加密模型加载
# ============================================================
# ...
# ✓ SHA1 完全匹配，XOR 解密正确
```

## 重要说明

### 参数级 XOR vs 文件级 XOR
- **重要**: 本系统使用参数级 XOR 混淆，而非文件级异或
- 加密时: 先加载模型 → 对参数进行 XOR → 序列化 → AES 加密
- 解密时: AES 解密 → 反序列化 → 对参数进行 XOR 还原
- 这样可以确保模型参数的语义被保护，同时增加了逆向工程难度

### models 文件夹保护
- 本系统不修改 models 文件夹下的任何文件
- 使用 `load_yolov5_with_encryption()` 包装函数处理 YOLOv5 加载
- 保持原始代码库的完整性

### 注意事项
1. **备份原文件** - 加密前请备份原始模型文件
2. **记录机器ID** - 建议保存机器ID和密钥哈希用于验证
3. **跨平台兼容** - 不同操作系统机器ID获取方式可能不同
4. **权限要求** - Linux需要root权限读取某些硬件信息
5. **容器环境** - Docker容器需要映射 `/sys/class/dmi/` 目录
6. **SHA1 验证** - 使用 test_encryption.py 验证加密正确性

## 故障排查

### 问题：无法获取机器ID
**解决方案：**
- 检查系统权限（Linux需要sudo）
- 手动指定机器ID（修改代码）
- 使用MAC地址作为备用方案

### 问题：模型加载失败
**解决方案：**
```bash
# 测试解密功能
python test_encryption.py

# 检查文件完整性
ls -la *.enc

# 尝试重新加密
python encrypt_models.py encrypt

# 验证 SHA1 是否匹配
python test_encryption.py
```

### 问题：容器中无法运行
**解决方案：**
```bash
# Docker运行时添加挂载
docker run -v /sys/class/dmi:/sys/class/dmi ...
```

## 技术细节

### AES-256-CBC 参数
- 密钥长度: 256位 (32字节)
- 块大小: 128位 (16字节)
- IV: 16字节随机数
- 填充: PKCS7

### 参数级 XOR 混淆
- XOR 密钥: 0x55 (85)
- 作用方式: 对模型所有参数的浮点数/整数字节表示进行异或
- 浮点数处理: 转换为 uint8 视图后异或
- 整数处理: 直接异或运算

### 文件格式
```
[0-11]: 魔数 'MODEL_ENC_V1' (12字节)
[12]: XOR密钥 (1字节)
[13-16]: 参数数量 (4字节, unsigned int)
[17-32]: AES IV (16字节)
[33-]: AES-CBC 加密数据
```

### GPU/CPU 自动检测
- 使用 `torch.cuda.is_available()` 检测
- 自动选择最优设备
- 显示 GPU 数量和设备名称
