#!/usr/bin/env python3
"""
PyArmor 跨平台混淆脚本
混淆根目录下的 Python 源码和 segment 目录下的源码，其他文件直接复制
注意：hubconf.py 不混淆，因为 torch 需要动态导入
"""

import os
import sys
import shutil
import platform
import subprocess
from pathlib import Path

# 颜色输出（仅在支持的平台）
class Colors:
    def __init__(self):
        self.RED = '\033[0;31m' if platform.system() != 'Windows' else ''
        self.GREEN = '\033[0;32m' if platform.system() != 'Windows' else ''
        self.YELLOW = '\033[1;33m' if platform.system() != 'Windows' else ''
        self.NC = '\033[0m' if platform.system() != 'Windows' else ''

colors = Colors()

def print_green(text):
    print(f"{colors.GREEN}{text}{colors.NC}")

def print_red(text):
    print(f"{colors.RED}{text}{colors.NC}")

def print_yellow(text):
    print(f"{colors.YELLOW}{text}{colors.NC}")

def check_pyarmor():
    """检查并安装 pyarmor"""
    try:
        result = subprocess.run(['pyarmor', '--version'], capture_output=True, text=True)
        print_green(f"✓ pyarmor 已安装: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print_red("pyarmor 未安装，正在安装...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyarmor'], check=True)
        print_green("✓ pyarmor 安装成功")
        return True

def copy_files_without_py(src_dir, dst_dir):
    """复制目录，排除 .py 文件"""
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    if not src_path.exists():
        return
    
    dst_path.mkdir(parents=True, exist_ok=True)
    
    for item in src_path.iterdir():
        if item.is_file():
            if item.suffix != '.py':
                shutil.copy2(item, dst_path / item.name)
        elif item.is_dir():
            copy_files_without_py(item, dst_path / item.name)

def copy_files(src_dir, dst_dir):
    """复制目录"""
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    if not src_path.exists():
        return
    
    dst_path.mkdir(parents=True, exist_ok=True)
    
    for item in src_path.iterdir():
        if item.is_file():
            shutil.copy2(item, dst_path / item.name)
        elif item.is_dir():
            copy_files(item, dst_path / item.name)            

def copy_file_with_check(src, dst):
    """复制文件并打印信息"""
    src_path = Path(src)
    if src_path.exists():
        print(f"复制: {src_path.name}")
        shutil.copy2(src_path, dst)
    else:
        print_yellow(f"警告: 文件不存在 {src}")

def obfuscate_file(src, output_dir):
    """使用 pyarmor 混淆单个文件"""
    src_path = Path(src)
    if not src_path.exists():
        print_yellow(f"警告: 文件不存在 {src}")
        return False
    
    print(f"混淆: {src_path.name}")
    cmd = [
        'pyarmor', 'obfuscate',
        '--output', str(output_dir),
        '--exact',
        str(src_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print_red(f"混淆失败: {src_path.name}")
        print_red(f"错误: {e.stderr.decode() if e.stderr else str(e)}")
        return False

def main():
    print_green("=== PyArmor 跨平台混淆脚本 ===")
    print(f"平台: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print()
    
    # 工作目录和输出目录
    workspace = Path(__file__).parent.absolute()
    output_dir = workspace / "dist_pyarmor"
    
    print_yellow("清理旧的输出目录...")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义需要混淆的 Python 文件列表（不包括 hubconf.py）
    pyarmor_files = [
        "abdetect.py",
        "app_logging.py",
        "app_metrics.py",
        "app_paths.py",
        "app_settings.py",
        "encrypt_models.py",
        "export.py",
        "fastapi_app.py",
        "model_decryptor.py",
        "preprocess.py",
        "runtime_state.py",
        "server.py",
        "test_encryption.py",
        "resnet.py"
    ]
    
    # 定义不混淆但需要复制的 Python 文件
    copy_py_files = [
        "hubconf.py",
    ]
    
    # 定义需要直接复制的目录
    copy_dirs = [
        "data",
        "models",
        "utils",
    ]
    
    # 步骤 1: 检查 pyarmor
    print_green("步骤 1: 检查 pyarmor 是否安装...")
    check_pyarmor()
    
    # 步骤 2: 复制非 Python 文件
    print_green("\n步骤 2: 复制非 Python 文件...")
    
    # 复制配置文件和文档文件
    for filename in ["Dockerfile", "requirements.txt", "pyproject.toml", ".python-version", "uv.lock", "README.md", "SUMMARY.md", "ENCRYPTION_README.md", "TAURI_PACKAGING.md"]:
        copy_file_with_check(workspace / filename, output_dir)
    
    # 复制模型文件
    # model_extensions = ['.pt', '.pth']
    model_extensions = ['.enc']
    for ext in model_extensions:
        for file in workspace.glob(f"*{ext}"):
            copy_file_with_check(file, output_dir)
    
    # 复制不混淆的 Python 文件
    for filename in copy_py_files:
        copy_file_with_check(workspace / filename, output_dir)
    
    # 复制目录
    for dirname in copy_dirs:
        src = workspace / dirname
        dst = output_dir / dirname
        if src.exists():
            print(f"复制目录: {dirname} ")
            copy_files(src, dst)
    
    # 步骤 3: 混淆 Python 文件
    print_green("\n步骤 3: 使用 pyarmor obfuscate 混淆所有 Python 文件...")
    
    # 混淆根目录文件
    for filename in pyarmor_files:
        src = workspace / filename
        obfuscate_file(src, output_dir)
    
    # 混淆 segment 目录
    segment_dir = workspace / "segment"
    if segment_dir.exists():
        segment_output_dir = output_dir / "segment"
        segment_output_dir.mkdir(parents=True, exist_ok=True)

        for file in segment_dir.glob("*.py"):
            obfuscate_file(file, segment_output_dir)

    # 混淆 inference 目录
    inference_dir = workspace / "inference"
    if inference_dir.exists():
        inference_output_dir = output_dir / "inference"
        inference_output_dir.mkdir(parents=True, exist_ok=True)

        for file in inference_dir.glob("*.py"):
            obfuscate_file(file, inference_output_dir)
    
    # 步骤 4: 检查 pytransform 运行时
    print_green("\n步骤 4: 检查 pytransform 运行时...")
    
    pytransform_dir = output_dir / "pytransform"
    if pytransform_dir.exists():
        print_green("✓ pytransform 运行时已就位")
    else:
        print_yellow("警告: pytransform 目录不存在，可能需要手动复制")
    
    # 步骤 5: 生成运行说明
    print_green("\n步骤 5: 生成运行说明...")
    
    readme_content = """# PyArmor 混淆项目说明

## 项目结构

本项目已使用 PyArmor 进行代码混淆保护。

### 混淆的文件
以下 Python 文件已使用 PyArmor 混淆：
- abdetect.py
- app_logging.py
- app_metrics.py
- app_paths.py
- app_settings.py
- encrypt_models.py
- export.py
- fastapi_app.py
- model_decryptor.py
- preprocess.py
- runtime_state.py
- server.py
- test_encryption.py
- segment/segpredict.py
- inference/*.py
### 未混淆的文件（保持原样）
以下文件未混淆，因为需要被其他库动态导入：
- hubconf.py（torch 需要动态导入）

### 其他文件（原样复制）
- 模型文件 (*.pt, *.pth)
- 配置文件 (*.yaml, *.json)
- 文档文件 (*.md)
- pyproject.toml
- uv.lock
- .python-version
- requirements.txt
- Dockerfile
- data/, models/, utils/ 目录（但不包括其中的 .py 文件）

### PyArmor 运行时
- pytransform/ 目录包含 PyArmor 运行时库
- 包括 `__init__.py` 和平台特定的运行时模块

## 运行要求

1. 安装依赖：
```bash
uv sync
```

2. 确保 Python 环境正确：
   - 需要支持 C 扩展模块
   - 支持多平台（Linux, Windows, macOS）

## 启动服务

```bash
uv run python server.py
```

## 注意事项

1. 混淆后的代码在功能上与原始代码完全一致
2. 混淆提高了代码的反编译难度
3. hubconf.py 未混淆，因为 torch.hub.load 需要能够动态导入其中的函数
4. 如需修改功能，请编辑原始源码而非混淆后的代码
5. pytransform 运行时必须与混淆后的代码在同一目录

## 跨平台支持

本混淆脚本支持以下平台：
- Linux
- Windows
- macOS

## 平台差异

不同平台的 PyArmor 运行时文件：
- Linux: _pytransform.so
- Windows: _pytransform.pyd
- macOS: _pytransform.dylib

混淆会自动生成对应平台的运行时文件。
"""
    
    with open(output_dir / "OBFUSCATION_README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    # 完成
    print_green("\n=== 混淆完成！ ===")
    print_green(f"输出目录: {output_dir}")
    print("\n混淆后的文件列表:")
    
    # 列出混淆的 Python 文件
    for file in output_dir.glob("*.py"):
        print(f"  {file.name} ({file.stat().st_size} bytes)")
    
    segment_py_dir = output_dir / "segment"
    if segment_py_dir.exists():
        for file in segment_py_dir.glob("*.py"):
            print(f"  segment/{file.name} ({file.stat().st_size} bytes)")
    
    print()
    print_yellow("提示: 运行混淆后的代码需要安装相同的依赖包")
    print()
    print_green("使用方法:")
    print(f"  cd {output_dir}")
    print("  uv sync")
    print("  uv run python server.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_red("\n\n操作已取消")
        sys.exit(1)
    except Exception as e:
        print_red(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
