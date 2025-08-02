#终于写道tain了意味着这个项目就快能跑起来了，希望不要有太多bug
import torch
import os
import numpy as np
import random
from random import randint
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from utils.general_utils import safe_state
from gaussian_renderer import network_gui, render, query
from utils.loss_utils import l1_loss, tv_3d_loss
from utils.image_utils import metric_vol, metric_proj, ssim 
from utils.cfg_utils import load_config

from scene import Scene
from scene.gaussian_model import GaussianModel
from scene.initialize import initialize_gaussian
import sys
from tqdm import tqdm
import copy

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    
def scene_reconstruction_3d(dataset, opt, hyper, pipe, 
                         testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint,
                         gaussians, scene, 
                         stage, tb_writer, train_iter
):
    
    first_iter = 0
    final_iter = train_iter
    print("final_iter:",train_iter)
    gaussians.training_setup(opt)
    
    scanner_cfg = scene.scanner_cfg
    bbox = scene.bbox
    volume_to_world = max(scanner_cfg["sVoxel"])
    max_scale = opt.max_scale * volume_to_world if opt.max_scale else None
    densify_scale_threshold = (
        opt.densify_scale_threshold * volume_to_world
        if opt.densify_scale_threshold
        else None
    )
    
    queryfunc = lambda x: query(
        x,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipe,
        stage=stage
    )
    
    if checkpoint:
        if  stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            # process is in the coarse stage, but start from fine stage
            return
        else: 
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)
            print(f"Load checkpoint {os.path.basename(checkpoint)}.")
    
    # Set up loss
    use_tv = opt.lambda_tv > 0
    if use_tv:
        print("Use total variation loss")
        tv_vol_size = opt.tv_vol_size
        tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
        tv_vol_sVoxel = torch.tensor(scanner_cfg["dVoxel"]) * tv_vol_nVoxel
            
    # Train
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    ckpt_save_path = os.path.join(scene.model_path, "ckpt")
    os.makedirs(ckpt_save_path, exist_ok=True)
    viewpoint_stack = None
    progress_bar = tqdm(range(0, train_iter), desc="Train", leave=False)
    progress_bar.update(first_iter)
    first_iter += 1
          
    for iteration in range(first_iter, final_iter+1): 
        
        
        iter_start.record()
        # Update learning rate
        gaussians.update_learning_rate(iteration)
        
        
        
        
         # Get one camera for training
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras(stage=stage).copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render X-ray projection
        render_pkg = render(viewpoint_cam, gaussians, pipe,stage=stage)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        
        # Compute loss
        gt_image = viewpoint_cam.original_image.cuda()
        
        loss = {"total": 0.0}
        render_loss = l1_loss(image, gt_image)
        loss["render"] = render_loss
        loss["total"] += loss["render"]
        if opt.lambda_dssim > 0:
            loss_dssim = 1.0 - ssim(image, gt_image)
            loss["dssim"] = loss_dssim
            loss["total"] = loss["total"] + opt.lambda_dssim * loss_dssim
        # 3D TV loss
        if use_tv:
            # Randomly get the tiny volume center
            tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (
                bbox[1] - tv_vol_sVoxel - bbox[0]
            ) * torch.rand(3)
            vol_pred = query(
                gaussians,
                tv_vol_center,
                tv_vol_nVoxel,
                tv_vol_sVoxel,
                pipe,
                stage=stage
            )["vol"]
            loss_tv = tv_3d_loss(vol_pred, reduction="mean")
            loss["tv"] = loss_tv
            loss["total"] = loss["total"] + opt.lambda_tv * loss_tv
            
        loss["total"].backward()
        
        
        iter_end.record()
        torch.cuda.synchronize()
        
        with torch.no_grad():
            
            if stage=="coarse":
                 # Adaptive control
                gaussians.max_radii2D[visibility_filter] = torch.max(
                          gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration < opt.densify_until_iter:
                    if (
                        iteration > opt.densify_from_iter
                        and iteration % opt.densification_interval == 0
                    ):
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold,
                            opt.density_min_threshold,
                            opt.max_screen_size,
                            max_scale,
                            opt.max_num_gaussians,
                            densify_scale_threshold,
                            bbox,
                    )
                if gaussians.get_density.shape[0] == 0:
                    raise ValueError(
                    "No Gaussian left. Change adaptive control hyperparameters!"
                )
            
            
             # Progress bar
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss['total'].item():.1e}",
                        "pts": f"{gaussians.get_density.shape[0]:2.1e}",
                    }
                )
                progress_bar.update(10)
            if iteration == train_iter:
                progress_bar.close()
            
            # Optimization
            if iteration < train_iter:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                
             # Save checkpoints
            if iteration in checkpoint_iterations:
                tqdm.write(f"[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    ckpt_save_path + "/chkpnt"  +f"_{stage}_" + str(iteration) + ".pth",
                )
                
             # Save gaussians
            if iteration in saving_iterations or iteration == train_iter:
                tqdm.write(f"[ITER {iteration}] Saving Gaussians")
                scene.save(iteration,stage,queryfunc)
                
            
             # Logging
            
  
            
def scene_reconstruction_4d(dataset, opt, hyper, pipe, 
                         testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint,
                         gaussians, scene, 
                         stage, tb_writer, train_iter
):
    
    first_iter = 0
    final_iter = train_iter

    gaussians.training_setup(opt)
    
    scanner_cfg = scene.scanner_cfg
    
    queryfunc = lambda x,y: query(
        x,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipe,
        stage=stage,
        time=y
    )
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print(f"Load checkpoint {os.path.basename(checkpoint)}.")
        
            
    # Train
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    ckpt_save_path = os.path.join(scene.model_path, "ckpt")
    os.makedirs(ckpt_save_path, exist_ok=True)
    
    progress_bar = tqdm(range(0, opt.iterations), desc="Train", leave=False)
    progress_bar.update(first_iter)
    first_iter += 1
    
    
    temp_list = copy.deepcopy(   scene.getTrainCameras(stage=stage)  )
    num_idx = len(temp_list)   
    viewpoint_stack=[[] for _ in range(num_idx)]
    for iteration in range(first_iter, final_iter+1): 
        
        
        iter_start.record()
        # Update learning rate
        gaussians.update_learning_rate(iteration)
        loss = {"total": 0.0, "render": 0.0 }
        for idx in range(num_idx):
            # Get one camera for training
            if not viewpoint_stack[idx]:
                viewpoint_stack[idx] = temp_list[idx].copy()
            viewpoint_cam = viewpoint_stack[idx].pop(randint(0, len(viewpoint_stack[idx]) - 1))
            
            # Render X-ray projection
            render_pkg = render(viewpoint_cam, gaussians, pipe,stage=stage)
            image, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )
        
             # Compute loss
            gt_image = viewpoint_cam.original_image.cuda()
        
            
            render_loss = l1_loss(image, gt_image)
            
            loss["render"] += render_loss
            loss["total"] += loss["render"]
           # if opt.lambda_dssim > 0:
            #    loss_dssim = 1.0 - ssim(image, gt_image)
             #   loss["dssim"] += loss_dssim
              #  loss["total"] = loss["total"] + opt.lambda_dssim * loss_dssim
        
        if  hyper.time_smoothness_weight != 0:
            regula_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
            loss["regulation"]=regula_loss
            loss["total"]+=regula_loss
        loss["total"].backward()
        
        
        iter_end.record()
        torch.cuda.synchronize()
        
        with torch.no_grad():
            
             # Progress bar
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss['total'].item():.1e}",
                        "pts": f"{gaussians.get_density.shape[0]:2.1e}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            
            # Optimization
            if iteration < opt.iterations:
                gaussians.optimizer_4d.step()
                gaussians.optimizer_4d.zero_grad(set_to_none=True)
                
             # Save checkpoints
            if iteration in checkpoint_iterations:
                tqdm.write(f"[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    ckpt_save_path + "/chkpnt"  +f"_{stage}_" + str(iteration) + ".pth",
                )
                
             # Save gaussians
            if iteration in saving_iterations or iteration == train_iter:
                tqdm.write(f"[ITER {iteration}] Saving Gaussians")
                scene.save(iteration,stage,queryfunc)
                
            
             # Logging
            
                        


def training(
    dataset:ModelParams,
    hyper:ModelHiddenParams,
    opt:OptimizationParams,
    pipe:PipelineParams,
    testing_iterations, 
    saving_iterations, 
    checkpoint_iterations, 
    checkpoint,
    expname
):
    tb_writer = prepare_output_and_logger(expname)
    # Set up dataset
    scene = Scene(dataset,shuffle=False)
    scanner_cfg = scene.scanner_cfg
    volume_to_world = max(scanner_cfg["sVoxel"])
    scale_bound = None
    if dataset.scale_min > 0 and dataset.scale_max > 0:
        scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world
    
    # Set up Gaussians
    gaussians = GaussianModel(scale_bound, hyper)
    initialize_gaussian(gaussians, dataset, None)
    scene.gaussians = gaussians
    
    scene_reconstruction_3d(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                             checkpoint_iterations, checkpoint, 
                             gaussians, scene, "coarse", tb_writer, opt.coarse_iterations)
    print("\n Reconstruction_3d Training complete!!!")
    scene_reconstruction_4d(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, 
                         gaussians, scene, "fine", tb_writer, opt.iterations)
    


def prepare_output_and_logger(expname):    
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(
    tb_writer, 
    iteration,
    metrics_train,
    elapsed, 
    testing_iterations, 
    scene : Scene, 
    renderFunc, 
    stage
):
    if tb_writer:
        for key in list(metrics_train.keys()):
            tb_writer.add_scalar(f"{stage}/train/{key}", metrics_train[key], iteration)
        
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)
        tb_writer.add_scalar(
            f"{stage}/train/total_points", scene.gaussians.get_xyz.shape[0], iteration
        )
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # 
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, cam_type=dataset_type, *renderArgs)["render"], 0.0, 1.0)
                    if dataset_type == "PanopticSports":
                        gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    except:
                        pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    # mask=viewpoint.mask
                    
                    psnr_test += psnr(image, gt_image, mask=None).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                # print("sh feature",scene.gaussians.get_features.shape)
                if tb_writer:
                    tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_density, iteration)
            
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate', scene.gaussians._deformation_table.sum()/scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_histogram(f"{stage}/scene/motion_histogram", scene.gaussians._deformation_accum.mean(dim=-1)/100, iteration,max_bins=500)
        
        torch.cuda.empty_cache()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    
    args.test_iterations.append(args.iterations)
    args.test_iterations.append(1)
    
    args_dict = vars(args)
    if args.config is not None:
        print(f"Loading configuration file from {args.config}")
        cfg = load_config(args.config)
        for key in list(cfg.keys()):
            args_dict[key] = cfg[key]
    
    # Initialize system state (RNG)
    safe_state(args.quiet)
    
        
    print("Optimizing " + args.model_path)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(
        lp.extract(args), 
        hp.extract(args), 
        op.extract(args), 
        pp.extract(args), 
        args.test_iterations, 
        args.save_iterations, 
        args.checkpoint_iterations, 
        args.start_checkpoint,
        args.expname
    )
    
    
    # All done
    print("\nTraining complete!!!")





