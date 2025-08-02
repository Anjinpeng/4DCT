import numpy as np
from PIL import Image
import os

# 加载 .npy 文件
npy_file_path = 'ct_vol_npy/vol_pred.npy'  # 确保这个文件存在于当前目录
ct_volume = np.load(npy_file_path)



# 保存为多页 TIFF 文件
tif_file_path = 'ct_vol_npy/vol_pred.tif'  # 保存的文件名
images = []

for i in range(ct_volume.shape[2]):
    slice_image = ct_volume[:,:,i]
    images.append(Image.fromarray(slice_image))

# 保存为多页 TIFF
#images[0].save(tif_file_path, save_all=True, append_images=images[1:])
images[0].save(tif_file_path, save_all=True, append_images=images[1:], compression='tiff_lzw')
print(f"3D CT 数据已保存为 {tif_file_path} 的多页 TIFF 文件。")