import numpy as np
import pyvista as pv

vol_path = "output/DFM1/point_cloud/iteration_10000/vol_pred_9.npy"
#vol_path = "data/synthetic_dataset/DFM/MP9_cone/vol_gt.npy"
#vol_path ="ct_vol_npy/vol_pred.npy"
vol = np.load(vol_path)

plotter = pv.Plotter(window_size=[1000, 1000], line_smoothing=True, off_screen=False)
plotter.add_volume(vol, cmap="viridis", opacity="linear")
plotter.show()
