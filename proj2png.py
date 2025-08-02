import numpy as np
from PIL import Image
import os

def npy_to_png(npy_file, png_file):
    # 加载 .npy 文件
    data = np.load(npy_file)
    # 保存为 .png 文件
    
    if data.dtype == np.float32 or data.dtype == np.float64:
        # 将浮点数数据转换到 0-255 的范围
        data = (data * 255).astype(np.uint8)
    image = Image.fromarray(data)
    image.save(png_file)
    print(f"Converted {npy_file} to {png_file}")

def convert_all_npy_to_png(input_folder, output_folder):
    # 创建输出目录（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 遍历文件夹中的所有 .npy 文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.npy'):
            npy_file = os.path.join(input_folder, filename)
            png_file = os.path.join(output_folder, filename.replace('.npy', '.png'))
            npy_to_png(npy_file, png_file)

# 示例用法
if __name__ == "__main__":
    input_folder = "data\synthetic_dataset\DFM\MP9_cone\proj_train"  # 输入文件夹路径
    output_folder = "output_proj_MP9"  # 输出文件夹路径
    convert_all_npy_to_png(input_folder, output_folder)