#!/usr/bin/env python3
"""
模型解密加载模块
支持从内存中加载加密的模型文件
包含 YOLOv5 的兼容加载函数
"""

import os
import platform
import subprocess
import uuid
import hashlib
import logging
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
import torch
import pickle
import struct


logger = logging.getLogger("jinchuan.model_decryptor")


def get_raw_machine_id():
    """获取原始机器唯一标识符，支持多平台"""
    system = platform.system()

    if system == 'Linux':
        try:
            # 读取主板序列号（尝试多种方式）
            for cmd in [
                ['cat', '/sys/class/dmi/id/board_serial'],
                ['cat', '/sys/class/dmi/id/product_uuid'],
                ['dmidecode', '-s', 'baseboard-serial-number'],
                ['cat', '/etc/machine-id'],
            ]:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    if result.returncode == 0 and result.stdout.strip():
                        return result.stdout.strip()
                except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
                    continue
        except Exception:
            pass

    elif system == 'Windows':
        try:
            # 使用wmic获取主板序列号
            result = subprocess.run(
                ['wmic', 'baseboard', 'get', 'serialnumber'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines[1:]:
                    if line.strip():
                        return line.strip()
        except Exception:
            pass

        try:
            # 备用方案：读取注册表
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r'SOFTWARE\Microsoft\Cryptography')
            guid, _ = winreg.QueryValueEx(key, 'MachineGuid')
            winreg.CloseKey(key)
            return guid
        except Exception:
            pass

    elif system == 'Darwin':  # macOS
        try:
            # 读取硬件UUID
            result = subprocess.run(
                ['ioreg', '-rd1', '-c', 'IOPlatformExpertDevice'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'IOPlatformUUID' in line:
                        return line.split('"')[-2]
        except Exception:
            pass

    # 最终备用方案：使用MAC地址
    return ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff)
                     for elements in range(0, 8*6, 8)][::-1])


def sha1_to_base62(input_string):
    """对输入字符串加盐后进行SHA1哈希，取前半部分转为Base62编码"""
    # 固定盐值
    SALT = "E4N8ONdKgl0XWoUqcentKgHLI7qFtMuv"

    # SHA1哈希
    salted_input = input_string + SALT
    sha1_hash = hashlib.sha1(salted_input.encode('utf-8')).digest()

    # 只保留SHA1结果的前半部分 (SHA1是20字节，取前10字节)
    hash_bytes = sha1_hash[:len(sha1_hash) // 2]

    # Base62字符集
    base62_chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

    # 将哈希字节转为大整数
    hash_int = int.from_bytes(hash_bytes, byteorder='big')

    # 转为Base62编码
    if hash_int == 0:
        return base62_chars[0]

    base62 = []
    while hash_int > 0:
        hash_int, remainder = divmod(hash_int, 62)
        base62.append(base62_chars[remainder])

    return ''.join(reversed(base62))


def get_machine_id():
    """获取机器唯一标识符，经过SHA1哈希和Base62编码处理"""
    raw_id = get_raw_machine_id()
    return sha1_to_base62(raw_id)


def generate_aes_key(seed_string):
    """根据种子字符串生成AES密钥"""
    key = hashlib.sha256(seed_string.encode('utf-8')).digest()
    return key


def xor_tensor(tensor, key_byte=0x55):
    """对Tensor参数进行异或处理（原地操作）"""
    if tensor.is_floating_point():
        # 对浮点数的字节表示进行异或
        tensor_view = tensor.view(torch.uint8)
        tensor_view ^= key_byte
    else:
        # 整数类型直接异或
        tensor ^= key_byte
    return tensor


def xor_model_parameters(model, key_byte=0x55):
    """对模型的所有参数进行异或处理"""
    for name, param in model.named_parameters():
        xor_tensor(param, key_byte)
    return


class BytesFile:
    """内存字节数据的文件对象封装"""
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def read(self, size=-1):
        if size == -1:
            result = self.data[self.pos:]
            self.pos = len(self.data)
        else:
            result = self.data[self.pos:self.pos + size]
            self.pos += size
        return result

    def seek(self, offset, whence=0):
        if whence == 0:  # 从文件开头
            self.pos = offset
        elif whence == 1:  # 从当前位置
            self.pos += offset
        elif whence == 2:  # 从文件末尾
            self.pos = len(self.data) + offset
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, buffer):
        n = len(buffer)
        data = self.read(n)
        buffer[:len(data)] = data
        return len(data)

    def readall(self):
        return self.read()

    def readline(self):
        line = b''
        while True:
            char = self.read(1)
            if not char or char == b'\n':
                break
            line += char
        return line


def is_encrypted_file(filepath):
    """检查文件是否为加密文件"""
    if filepath.endswith('.enc'):
        return True

    # 检查文件头魔数
    if os.path.exists(filepath):
        try:
            with open(filepath, 'rb') as f:
                magic = f.read(12)
                return magic == b'MODEL_ENC_V1'
        except Exception:
            pass

    return False


def get_original_filename(enc_file):
    """获取加密文件的原始文件名"""
    if enc_file.endswith('.enc'):
        return enc_file[:-4]
    return enc_file


def load_encrypted_model(filepath, device='cpu'):
    """
    加载加密的模型文件到内存

    Args:
        filepath: 加密文件路径（.enc）
        device: torch设备（'cpu' 或 'cuda'）

    Returns:
        解密后的模型对象
    """
    # 获取机器ID并生成密钥
    machine_id = get_machine_id()
    key = generate_aes_key(machine_id)

    # 读取加密文件
    with open(filepath, 'rb') as f:
        # 读取并验证魔数
        magic = f.read(12)
        if magic != b'MODEL_ENC_V1':
            raise ValueError(f"无效的加密文件格式: {filepath}")

        # 读取XOR密钥
        xor_key = struct.unpack('B', f.read(1))[0]

        # 读取异或参数信息数量
        params_count = struct.unpack('I', f.read(4))[0]

        # 读取IV和加密数据
        iv = f.read(16)
        encrypted_data = f.read()

    # AES解密
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted_data = unpad(cipher.decrypt(encrypted_data), AES.block_size)

    # 反序列化得到异或后的模型
    model = pickle.loads(decrypted_data)

    # 对模型参数进行异或还原
    if isinstance(model, dict):
        if 'model' in model and hasattr(model['model'], 'state_dict'):
            xor_model_parameters(model['model'], xor_key)
        elif 'state_dict' in model:
            for name, tensor in model['state_dict'].items():
                if isinstance(tensor, torch.Tensor):
                    xor_tensor(tensor, xor_key)
        elif all(isinstance(v, torch.Tensor) for v in model.values()):
            for name, tensor in model.items():
                if isinstance(tensor, torch.Tensor):
                    xor_tensor(tensor, xor_key)
    elif hasattr(model, 'state_dict'):
        xor_model_parameters(model, xor_key)

    return model


def load_encrypted_torch(filepath, device='cpu'):
    """
    兼容 torch.load 的加密模型加载函数
    自动检测加密文件或原始文件

    Args:
        filepath: 模型文件路径（.enc 或原始文件）
        device: torch设备

    Returns:
        解密后的模型对象
    """
    # 优先使用加密文件
    enc_path = filepath + '.enc'
    if os.path.exists(enc_path) and not is_encrypted_file(filepath):
        filepath = enc_path

    # 检查是否为加密文件
    if is_encrypted_file(filepath):
        model = load_encrypted_model(filepath, device)

        # 如果模型需要移动到设备
        if device != 'cpu' and isinstance(model, torch.nn.Module):
            model = model.to(device)
        elif device != 'cpu' and isinstance(model, dict):
            if 'model' in model and hasattr(model['model'], 'to'):
                model['model'] = model['model'].to(device)

        return model
    else:
        # 正常加载原始文件
        return torch.load(filepath, map_location=device, weights_only=False)


def load_yolov5_with_encryption(weights_path, device='cpu', autoshape=True):
    """
    加载 YOLOv5 模型，自动检测加密文件或原始文件
    避免修改 models 文件夹下的文件

    Args:
        weights_path: 模型文件路径（.pt 或 .enc）
        device: torch设备（'cpu' 或 'cuda'）
        autoshape: 是否应用 AutoShape 包装

    Returns:
        YOLOv5 模型
    """
    # 优先使用加密文件
    enc_path = weights_path + '.enc'
    if os.path.exists(enc_path):
        model_path = enc_path
    else:
        model_path = weights_path

    logger.info("load_model", extra={"event": "load_model"})

    # 检查是否为加密文件
    if is_encrypted_file(model_path):
        # 直接加载加密模型
        model = load_encrypted_torch(model_path, device=device)

        # 提取实际的模型对象（YOLOv5 保存的是字典格式）
        if isinstance(model, dict):
            if 'model' in model:
                model = model['model']
            elif 'ema' in model:
                model = model['ema']

        model.eval()
        # CPU 模式下确保使用 FP32 精度（CPU 不支持 FP16）
        if device == 'cpu':
            model.float()

        # 如果需要 AutoShape 包装
        if autoshape:
            from models.common import AutoShape
            model = AutoShape(model)

        if device != 'cpu':
            model = model.to(device)

        return model
    else:
        # 使用 torch.hub.load 加载原始模型
        model = torch.hub.load('.', 'custom', path=model_path, source='local', _verbose=False)

        # CPU 模式下确保使用 FP32 精度（CPU 不支持 FP16）
        if device == 'cpu':
            model.float()

        if device != 'cpu':
            model = model.to(device)

        return model


if __name__ == '__main__':
    print("=" * 60)
    print("测试加密模型加载")
    print("=" * 60)

    machine_id = get_machine_id()
    print(f"\n机器ID: {machine_id}")

    key = generate_aes_key(machine_id)
    print(f"AES密钥长度: {len(key)} bytes")

    # 查找加密文件
    enc_files = []
    for f in os.listdir('.'):
        if f.endswith('.enc'):
            enc_files.append(f)

    if not enc_files:
        print("\n未找到加密文件")
        exit(1)

    print(f"\n找到 {len(enc_files)} 个加密文件")

    # 测试加载加密模型
    for enc_file in enc_files[:3]:  # 只测试前3个
        try:
            print(f"\n加载: {enc_file}")
            model = load_encrypted_torch(enc_file, device='cpu')

            # 显示模型信息
            model_type = type(model).__name__
            print(f"  ✓ 加载成功")
            print(f"  模型类型: {model_type}")

            if isinstance(model, dict):
                if 'model' in model:
                    print(f"  包含模型")
                elif 'state_dict' in model:
                    print(f"  包含 state_dict，参数数: {len(model['state_dict'])}")
                else:
                    print(f"  字典键数: {len(model.keys())}")
            elif hasattr(model, 'state_dict'):
                state_dict = model.state_dict()
                print(f"  参数数: {len(state_dict)}")
                if hasattr(model, 'fc'):
                    try:
                        print(f"  输出特征数: {model.fc.out_features}")
                    except:
                        pass

        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
            import traceback
            traceback.print_exc()

    # 测试 YOLOv5 包装函数
    print("\n" + "=" * 60)
    print("测试 YOLOv5 包装函数")
    print("=" * 60)

    try:
        model = load_yolov5_with_encryption('slf.pt', device='cpu')
        print(f"✓ load_yolov5_with_encryption 加载成功")
        print(f"  模型类型: {type(model).__name__}")
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
