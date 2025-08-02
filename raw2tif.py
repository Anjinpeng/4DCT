import numpy as np
import tifffile
import os

def convert_raw_to_tiff(input_path, output_path, shape, dtype=np.float32):
    # 添加路径标准化处理
    input_path = os.path.normpath(input_path)
    output_path = os.path.normpath(output_path)

    if not os.path.exists(input_path):
        print(f"错误: 输入文件不存在 - {input_path}")
        return

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    try:
        data = np.fromfile(input_path, dtype=dtype)
    except Exception as e:
        print(f"读取文件失败: {input_path}")
        print(f"错误详情: {str(e)}")
        return
    
    expected_size = np.prod(shape)
    if data.size != expected_size:
        print(f"数据尺寸不匹配: {input_path}")
        print(f"预期大小: {expected_size} 元素")
        print(f"实际大小: {data.size} 元素")
        return

    try:
        data = data.reshape(shape)
        tifffile.imwrite(
            output_path,
            data,
            metadata={'axes': 'ZYX'},
            compression='zlib'
        )
        print(f"成功转换: {input_path} -> {output_path}")
    except Exception as e:
        print(f"处理文件失败: {input_path}")
        print(f"错误详情: {str(e)}")

angle = "180"

# 调试步骤：打印当前工作目录
print(f"当前工作目录: {os.getcwd()}")

# 使用os.path.join构建跨平台路径
base_dir = os.path.dirname(__file__)  # 获取脚本所在目录

for idx in [f"{i:02d}" for i in range(10)]:
    for frame in [f"{i:03d}" for i in range(10)]:
        # 使用os.path.join构建路径
        input_path = os.path.join(
            base_dir,
            "raw_data",
            "S03_021",
            f"{idx}_080_{angle}_090_gt_f{frame}.raw"
        )
        
        output_path = os.path.join(
            base_dir,
            "volume_data",
            "S03_021",
            f"{idx}_080_{angle}_090_gt_f{frame}.tiff"
        )
        
        # 调试步骤：验证路径
        print(f"尝试访问输入文件: {input_path}")
        if not os.path.exists(input_path):
            print(f"!! 文件确实不存在: {input_path}")
            continue
            
        convert_raw_to_tiff(
            input_path=input_path,
            output_path=output_path,
            shape=(80, 80, 80)
        )