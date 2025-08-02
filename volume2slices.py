import numpy as np
import matplotlib.pyplot as plt
import os
from utils.image_utils import metric_vol
import yaml

# Path to the .npy file
npy_gt_path = "output/4d_lung/point_cloud/iteration_10000/vol_gt_9.npy"
# Output directory for slices
output_gt_dir = "output/4d_lung/point_cloud/iteration_10000/slice_gt_9"

# Path to the .npy file
npy_pred_path = "output/4d_lung/point_cloud/iteration_10000/vol_pred_9.npy"
# Output directory for slices
output_pred_dir = "output/4d_lung/point_cloud/iteration_10000/slice_pred_9"

eval_path = "output/4d_lung/point_cloud/iteration_10000/eval_3d_9"
# Create output directory if it doesn't exist
os.makedirs(output_gt_dir, exist_ok=True)
os.makedirs(output_pred_dir, exist_ok=True)
os.makedirs(eval_path, exist_ok=True)
# Load the volume data
vol_gt = np.load(npy_gt_path)
# Print the original data type of the volume
print("GT data type:", vol_gt.dtype)

#Load the volume data
vol_pred = np.load(npy_pred_path)
# Print the original data type of the volume
print("PRED data type:", vol_pred.dtype)


# Iterate over slices in the z-direction
for i in range(vol_gt.shape[2]):  # Assuming z-axis is the last dimension
    slice_data = vol_gt[:, :, i]  # Extract the i-th slice

    # Plot the slice
    plt.imshow(slice_data, cmap='gray')
    plt.axis('off')  # Turn off axis

    # Save the slice as an image
    slice_file_path = os.path.join(output_gt_dir, f"slice_gt_{i:03d}.png")
    plt.savefig(slice_file_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the plot to free memory

print(f"Saved {vol_gt.shape[2]} slices to {output_gt_dir}")

# Iterate over slices in the z-direction
for i in range(vol_pred.shape[2]):  # Assuming z-axis is the last dimension
    slice_data = vol_pred[:, :, i]  # Extract the i-th slice

    # Plot the slice
    plt.imshow(slice_data, cmap='gray')
    plt.axis('off')  # Turn off axis

    # Save the slice as an image
    slice_file_path = os.path.join(output_pred_dir, f"slice_pred_{i:03d}.png")
    plt.savefig(slice_file_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the plot to free memory

print(f"Saved {vol_gt.shape[2]} slices to {output_gt_dir}")

psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
ssim_3d, ssim_3d_axis = metric_vol(vol_gt, vol_pred, "ssim")
eval_dict = {
            "psnr_3d": psnr_3d,
            "ssim_3d": ssim_3d,
            "ssim_3d_x": ssim_3d_axis[0],
            "ssim_3d_y": ssim_3d_axis[1],
            "ssim_3d_z": ssim_3d_axis[2],
        }
with open(os.path.join(eval_path, "eval3d.yml"), "w") as f:
            yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)

print(f"Saved eval_3d to {eval_path}")