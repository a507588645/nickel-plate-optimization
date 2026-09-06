#!/usr/bin/env python3
"""
模型文件加密脚本
获取主板编号作为AES加密密钥生成的种子参数，支持多平台。
对pt/pth文件进行参数异或处理后，再使用AES加密保存到pt.enc/pth.enc文件。
"""

import os
import platform
import subprocess
import uuid
import hashlib
import base64
import argparse
import struct
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import torch
import pickle


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
    # 使用SHA-256生成32字节密钥
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
    xor_params_info = []
    for name, param in model.named_parameters():
        original_shape = param.shape
        xor_tensor(param, key_byte)
        xor_params_info.append({
            'name': name,
            'shape': list(original_shape),
            'dtype': str(param.dtype)
        })
    return xor_params_info


def encrypt_model(input_path, output_path, key, xor_key=0x55):
    """加密模型文件：参数异或 + AES加密"""
    print(f"\n正在加密: {input_path}")

    # 加载原始模型
    model = torch.load(input_path, map_location='cpu')

    # 判断模型类型
    model_type = None
    if isinstance(model, dict):
        # 可能是完整模型检查点或state_dict
        if 'model' in model:
            model_type = 'checkpoint_with_model'
            target = model['model']
        elif 'state_dict' in model:
            model_type = 'checkpoint_with_state_dict'
            target = model['state_dict']
        elif all(isinstance(v, torch.Tensor) for v in model.values()):
            model_type = 'state_dict'
            target = model
        else:
            model_type = 'checkpoint'
            target = None
    elif hasattr(model, 'state_dict'):
        model_type = 'model'
        target = model
    else:
        model_type = 'unknown'
        target = None

    # 对模型参数进行异或处理
    xor_params_info = []
    if model_type == 'model':
        xor_params_info = xor_model_parameters(model, xor_key)
        print(f"  对 {len(xor_params_info)} 个参数进行异或处理")
        for info in xor_params_info[:3]:  # 显示前3个
            print(f"    - {info['name']}: {info['shape']} ({info['dtype']})")
        if len(xor_params_info) > 3:
            print(f"    ... 还有 {len(xor_params_info) - 3} 个参数")
    elif model_type == 'state_dict' or model_type == 'checkpoint_with_state_dict':
        count = 0
        for name, tensor in target.items():
            if isinstance(tensor, torch.Tensor):
                xor_tensor(tensor, xor_key)
                xor_params_info.append({
                    'name': name,
                    'shape': list(tensor.shape),
                    'dtype': str(tensor.dtype)
                })
                count += 1
        print(f"  对 {count} 个张量参数进行异或处理")
        for info in xor_params_info[:3]:
            print(f"    - {info['name']}: {info['shape']} ({info['dtype']})")
        if len(xor_params_info) > 3:
            print(f"    ... 还有 {len(xor_params_info) - 3} 个参数")
    elif model_type == 'checkpoint_with_model':
        if hasattr(target, 'state_dict'):
            xor_params_info = xor_model_parameters(target, xor_key)
            print(f"  对 {len(xor_params_info)} 个参数进行异或处理")
        else:
            print(f"  警告: 不支持对此模型类型进行参数异或")
    elif model_type == 'checkpoint':
        print(f"  警告: 检查点类型，不支持参数异或")
    else:
        print(f"  警告: 未知模型类型，跳过参数异或")

    # 将异或后的模型序列化为字节
    model_bytes = pickle.dumps(model)

    # AES加密
    cipher = AES.new(key, AES.MODE_CBC)
    iv = cipher.iv
    encrypted_data = cipher.encrypt(pad(model_bytes, AES.block_size))

    # 保存：IV + 加密数据
    with open(output_path, 'wb') as f:
        # 写入魔数（用于标识加密文件）
        f.write(b'MODEL_ENC_V1')
        # 写入XOR密钥（可选，用于验证）
        f.write(struct.pack('B', xor_key))
        # 写入异或参数信息数量
        f.write(struct.pack('I', len(xor_params_info)))
        # 写入IV和加密数据
        f.write(iv + encrypted_data)

    original_size = len(model_bytes)
    encrypted_size = len(iv) + len(encrypted_data)
    print(f"  原模型大小: {original_size:,} bytes")
    print(f"  加密后大小: {encrypted_size:,} bytes")
    print(f"✓ 已加密: {output_path}")


def decrypt_model(input_path, key, device='cpu'):
    """解密模型文件，返回模型对象"""
    with open(input_path, 'rb') as f:
        # 读取并验证魔数
        magic = f.read(12)
        if magic != b'MODEL_ENC_V1':
            raise ValueError(f"无效的加密文件格式: {magic}")

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


def show_help():
    """显示帮助信息"""
    help_text = """
模型文件加密/解密工具
======================

功能说明:
  - 使用机器码作为AES加密密钥种子，对.pt/.pth模型文件进行加密
  - 加密过程包含参数异或处理 + AES加密双重保护
  - 解密需要使用相同的机器码

用法:
  python encrypt_models.py <命令> [选项]

命令:
  encrypt [选项]      加密模型文件 (默认命令)
  decrypt <文件> [选项]  解密指定的加密文件
  machine-id          显示当前机器的机器码
  test                测试加密解密功能
  help                显示此帮助信息

加密选项:
  --input <文件>       指定输入文件 (不指定则加密当前目录所有模型文件)
  --output <文件>      指定输出文件
  --machine-id <码>    指定机器码用于加密 (默认使用本机机器码)

解密选项:
  --machine-id <码>    指定机器码用于解密 (默认使用本机机器码)
  --output <文件>      指定输出文件 (不指定则输出到原文件名去掉.enc)

示例:
  # 显示本机机器码
  python encrypt_models.py machine-id

  # 加密所有模型文件 (使用本机机器码)
  python encrypt_models.py encrypt

  # 加密指定文件 (使用本机机器码)
  python encrypt_models.py encrypt --input best.pt

  # 加密文件，指定机器码
  python encrypt_models.py encrypt --input best.pt --machine-id AbCdEf1234

  # 解密文件 (使用本机机器码)
  python encrypt_models.py decrypt best.pt.enc

  # 解密文件，指定机器码
  python encrypt_models.py decrypt best.pt.enc --machine-id AbCdEf1234

  # 测试功能
  python encrypt_models.py test

注意事项:
  - 加密的模型只能在相同机器码的设备上解密使用
  - 建议在加密前备份原始模型文件
"""
    print(help_text)


def main():
    import sys

    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description='模型文件加密/解密工具',
        add_help=False  # 自定义帮助
    )
    parser.add_argument('command', nargs='?', default='encrypt', 
                        help='命令: encrypt, decrypt, machine-id, test, help')
    parser.add_argument('--input', '-i', help='输入文件路径')
    parser.add_argument('--output', '-o', help='输出文件路径')
    parser.add_argument('--machine-id', '-m', dest='custom_machine_id',
                        help='指定机器码 (SHA1+Base62编码)')

    # 解析参数
    args = parser.parse_args()

    # 处理帮助命令
    if args.command == 'help':
        show_help()
        return

    # 处理machine-id命令
    if args.command == 'machine-id':
        machine_id = get_machine_id() if not args.custom_machine_id else args.custom_machine_id
        print(f"机器码: {machine_id}")
        return

    # 处理test命令
    if args.command == 'test':
        print("=" * 60)
        print("测试加密解密功能")
        print("=" * 60)

        machine_id = args.custom_machine_id if args.custom_machine_id else get_machine_id()
        print(f"\n使用机器码: {machine_id}")

        key = generate_aes_key(machine_id)
        print(f"生成的AES密钥长度: {len(key)} bytes")

        # 找一个模型文件进行测试
        test_files = ['weights/slf.pth', 'weights/ryw.pth', 'weights/best.pt', 'weights/unused/seg.pt']
        test_file = None
        for f in test_files:
            if os.path.exists(f):
                test_file = f
                break

        if not test_file:
            print("\n错误: 找不到测试文件")
            return

        print(f"\n使用测试文件: {test_file}")

        # 备份原始文件
        backup_file = test_file + '.backup'
        if not os.path.exists(backup_file):
            import shutil
            shutil.copy(test_file, backup_file)
            print(f"已创建备份: {backup_file}")

        # 加密
        enc_file = test_file + '.enc'
        encrypt_model(test_file, enc_file, key)

        # 解密验证
        print("\n解密验证...")
        dec_model = decrypt_model(enc_file, key)
        print(f"✓ 解密成功，模型类型: {type(dec_model).__name__}")

        # 比较原始模型和解密后的模型
        original_model = torch.load(test_file, map_location='cpu')
        if isinstance(original_model, dict) and isinstance(dec_model, dict):
            for key_name in original_model:
                if key_name in dec_model:
                    orig_tensor = original_model[key_name]
                    dec_tensor = dec_model[key_name]
                    if isinstance(orig_tensor, torch.Tensor) and isinstance(dec_tensor, torch.Tensor):
                        if torch.equal(orig_tensor, dec_tensor):
                            print(f"  ✓ {key_name}: 匹配")
                        else:
                            max_diff = torch.max(torch.abs(orig_tensor - dec_tensor)).item()
                            print(f"  ✗ {key_name}: 不匹配 (最大差异: {max_diff})")
        elif hasattr(original_model, 'state_dict') and hasattr(dec_model, 'state_dict'):
            orig_dict = original_model.state_dict()
            dec_dict = dec_model.state_dict()
            for key_name in list(orig_dict.keys())[:5]:  # 只检查前5个
                if key_name in dec_dict:
                    orig_tensor = orig_dict[key_name]
                    dec_tensor = dec_dict[key_name]
                    if torch.equal(orig_tensor, dec_tensor):
                        print(f"  ✓ {key_name}: 匹配")
                    else:
                        max_diff = torch.max(torch.abs(orig_tensor - dec_tensor)).item()
                        print(f"  ✗ {key_name}: 不匹配 (最大差异: {max_diff})")
            print("  ...")

        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)
        return

    # 处理decrypt命令
    if args.command == 'decrypt':
        if not args.input:
            print("错误: 解密命令需要指定输入文件")
            print("用法: python encrypt_models.py decrypt <文件> [--machine-id <码>]")
            return

        input_path = args.input
        if not os.path.exists(input_path):
            print(f"错误: 文件不存在: {input_path}")
            return

        output_path = args.output
        if not output_path:
            # 默认输出到原文件名去掉.enc
            if input_path.endswith('.enc'):
                output_path = input_path[:-4]
            else:
                output_path = input_path + '.dec'

        print("=" * 60)
        print("模型文件解密")
        print("=" * 60)

        machine_id = args.custom_machine_id if args.custom_machine_id else get_machine_id()
        print(f"\n使用机器码: {machine_id}")
        key = generate_aes_key(machine_id)

        print(f"\n正在解密: {input_path}")
        try:
            model = decrypt_model(input_path, key)
            print(f"✓ 解密成功，模型类型: {type(model).__name__}")
            
            # 保存解密后的模型
            torch.save(model, output_path)
            print(f"✓ 已保存: {output_path}")
            
            print("\n" + "=" * 60)
            print("解密完成！")
            print("=" * 60)
        except Exception as e:
            print(f"✗ 解密失败: {e}")
        return

    # 处理encrypt命令 (默认)
    print("=" * 60)
    print("模型文件加密工具")
    print("=" * 60)

    machine_id = args.custom_machine_id if args.custom_machine_id else get_machine_id()
    print(f"\n使用机器码: {machine_id}")
    if args.custom_machine_id:
        print(f"  注意: 使用自定义机器码加密，解密时需指定相同的机器码")
    else:
        print(f"  使用本机机器码，加密后的文件只能在相同机器上解密")

    key = generate_aes_key(machine_id)
    print(f"生成的AES密钥长度: {len(key)} bytes")

    # 如果指定了输入文件，只加密该文件
    if args.input:
        input_file = args.input
        if not os.path.exists(input_file):
            print(f"\n错误: 文件不存在: {input_file}")
            return
        
        output_file = args.output
        if not output_file:
            output_file = input_file + '.enc'
        
        try:
            encrypt_model(input_file, output_file, key)
            print("\n" + "=" * 60)
            print("加密完成！")
            print("=" * 60)
        except Exception as e:
            print(f"✗ 加密失败: {e}")
        return

    # 查找所有模型文件
    model_files = []
    for root, dirs, files in os.walk('.'):
        for f in files:
            if f.endswith('.pt') or f.endswith('.pth'):
                # 跳过加密文件
                if f.endswith('.enc'):
                    continue
                full_path = os.path.join(root, f)
                model_files.append(full_path)

    if not model_files:
        print("\n未找到模型文件 (.pt/.pth)")
        return

    print(f"\n找到 {len(model_files)} 个模型文件")

    # 加密所有模型文件
    success_count = 0
    for model_file in sorted(model_files):
        try:
            output_file = model_file + '.enc'
            encrypt_model(model_file, output_file, key)
            success_count += 1
        except Exception as e:
            print(f"✗ 加密失败 {model_file}: {e}")

    print("\n" + "=" * 60)
    print(f"加密完成！成功: {success_count}/{len(model_files)}")
    print("=" * 60)
    print("\n提示:")
    print("  - 使用 'python encrypt_models.py machine-id' 查看机器码")
    print("  - 使用 'python encrypt_models.py decrypt <文件>' 解密文件")
    print("  - 使用 'python encrypt_models.py test' 测试加密解密功能")


if __name__ == '__main__':
    main()
