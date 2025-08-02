#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from utils.camera_utils import cameraList_from_camInfos


from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.general_utils import t2a
import numpy as np
import random
import torch

class Scene:

    gaussians : GaussianModel

    def __init__(
        self, 
        args : ModelParams,
        shuffle=True
        ):
       
        self.model_path = args.model_path
        
        
        self.train_cameras_4d = {}
        self.test_cameras_4d = {}
       
         # Read scene info
        scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path,
                args.eval,
            )
        
       
        
       # Load cameras
        print("Loading Training Cameras")
        self.train_cameras_4d = cameraList_from_camInfos(scene_info.train_cameras_4d, args)
        print("Loading Test Cameras")
        self.test_cameras_4d = cameraList_from_camInfos(scene_info.test_cameras_4d, args)
        
       
        # Set up some parameters
        self.vol_gt_4d = scene_info.vol_4d
        self.scanner_cfg = scene_info.scanner_cfg
        self.scene_scale = scene_info.scene_scale
        self.bbox = torch.stack(
            [
                torch.tensor(self.scanner_cfg["offOrigin"])
                - torch.tensor(self.scanner_cfg["sVoxel"]) / 2,
                torch.tensor(self.scanner_cfg["offOrigin"])
                + torch.tensor(self.scanner_cfg["sVoxel"]) / 2,
            ],
            dim=0,
        )

    def save(self, iteration, stage,queryfunc):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
            if queryfunc is not None:
                vol_pred = queryfunc(self.gaussians)["vol"]
                vol_gt = self.vol_gt_4d[0]
                np.save(os.path.join(point_cloud_path, "vol_gt.npy"), t2a(vol_gt))
                np.save(
                    os.path.join(point_cloud_path, "vol_pred.npy"),
                    t2a(vol_pred),
                )

        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
            self.gaussians.save_deformation(point_cloud_path)
            if queryfunc is not None:
                tol_num=len(self.vol_gt_4d)
                for idx , vol_gt in enumerate(self.vol_gt_4d):
                    vol_pred = queryfunc(self.gaussians,float(idx/tol_num))["vol"]
                
                    np.save(os.path.join(point_cloud_path, f"vol_gt_{idx}.npy"), t2a(vol_gt))
                    np.save(
                        os.path.join(point_cloud_path, f"vol_pred_{idx}.npy"),
                        t2a(vol_pred),
                )
    
    
    def getTrainCameras(self,stage):
        if stage=="fine":
            return self.train_cameras_4d
        else :
            train_cameras=self.train_cameras_4d
            return train_cameras[0]

    def getTestCameras(self):
        return self.test_cameras_4d
    