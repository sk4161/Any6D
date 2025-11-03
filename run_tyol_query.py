import copy
import yaml as pyyaml
import wandb

from foundationpose.datareader import TyolReader
from foundationpose.Utils import visualize_frame_results_gt
from estimater import *
from bop_toolkit_lib.pose_error_custom import mssd, mspd, vsd

from metrics import *
import json
from bop_toolkit_lib.renderer_vispy import RendererVispy
from pytorch_lightning import seed_everything
from datetime import datetime


class MissingGroundTruthError(RuntimeError):
    """Raised when a query frame lacks ground-truth pose."""

    def __init__(self, frame_index: int):
        super().__init__(f"Ground-truth pose is unavailable for frame {frame_index}")
        self.frame_index = frame_index


if __name__ == "__main__":
    seed_everything(0)

    parser = argparse.ArgumentParser(description="Set experiment name and paths")

    parser.add_argument("--name", type=str, default="any6d", help="Experiment name")

    # Anchor reference arguments (simplified)
    parser.add_argument(
        "--anchor-scene-id",
        type=str,
        required=True,
        help="Scene ID for anchor (e.g., 000001)",
    )
    parser.add_argument(
        "--anchor-frame-index",
        type=int,
        required=True,
        help="Frame index for anchor",
    )
    parser.add_argument(
        "--anchor-save-dir",
        type=str,
        required=True,
        help="Directory where anchor results are saved (e.g., tyol_anchor_results_000001)",
    )
    parser.add_argument(
        "--save-dir", type=str, default="./results", help="Directory to save results"
    )

    # TYO-L specific arguments
    parser.add_argument(
        "--tyol_data_root",
        type=str,
        required=True,
        help="Path to TYO-L dataset root",
    )
    parser.add_argument(
        "--tyol_model_path",
        type=str,
        required=True,
        help="Path to TYO-L models",
    )
    parser.add_argument(
        "--tyol_models_info_path",
        type=str,
        required=True,
        help="Path to TYO-L models_info.json",
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        default=None,
        help="Specific scene to process (e.g., 000001)",
    )
    parser.add_argument(
        "--running_stride",
        type=int,
        default=10,
        help="Process every N frames",
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="Starting frame index for processing",
    )

    args = parser.parse_args()

    # Create save directories
    save_results_est_path = os.path.join(args.save_dir, f"{args.name}_est")
    os.makedirs(save_results_est_path, exist_ok=True)

    # Setup wandb
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb_run_name = f"{args.name}_{current_time}"
    wandb.init(
        project="any6d-tyol-query",
        name=wandb_run_name,
        config=vars(args),
        tags=["tyol", "query", args.name],
    )
    wandb.config.update(args)

    # Get list of scenes to process
    if args.scene_id:
        scene_dirs = [os.path.join(args.tyol_data_root, "test", args.scene_id)]
    else:
        # Process all available scenes
        test_dir = os.path.join(args.tyol_data_root, "test")
        scene_names = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
        scene_names = sorted(scene_names)
        scene_dirs = [
            os.path.join(test_dir, name)
            for name in scene_names
            if os.path.exists(os.path.join(test_dir, name))
        ]

    # Load anchor data from TYO-L dataset and anchor results
    print(f"Loading anchor data from scene {args.anchor_scene_id}, frame {args.anchor_frame_index}...")
    
    # Initialize anchor scene reader
    anchor_scene_dir = os.path.join(args.tyol_data_root, "test", args.anchor_scene_id)
    anchor_object_id = int(args.anchor_scene_id)
    anchor_reader = TyolReader(anchor_scene_dir, object_id=anchor_object_id)
    
    # Load anchor data from TYO-L dataset
    anchor_image = anchor_reader.get_color(args.anchor_frame_index)
    anchor_depth = anchor_reader.get_depth(args.anchor_frame_index)
    anchor_mask = anchor_reader.get_mask(args.anchor_frame_index)
    K_anchor = anchor_reader.K
    anchor_gt_pose = anchor_reader.get_gt_pose(args.anchor_frame_index)
    
    # Debug: Print anchor GT pose
    print(f"Anchor frame {args.anchor_frame_index}: GT translation = {anchor_gt_pose[:3, 3]}")
    
    # Load anchor prediction results from anchor_save_dir
    anchor_pred_pose = np.loadtxt(
        os.path.join(args.anchor_save_dir, f"{args.anchor_scene_id}_predicted_pose.txt")
    ).reshape(4, 4)
    mesh_anchor = trimesh.load(
        os.path.join(args.anchor_save_dir, f"final_mesh_{args.anchor_scene_id}.obj"),
        force="mesh"
    )

    # Initialize results storage
    result = {
        "ADD": [],
        "ADDS": [],
        "MSSD": [],
        "MSPD": [],
        "VSD": [],
        "ADD_GT": [],
        "ADDS_GT": [],
        "AR": [],
        "R_error": [],
        "T_error": [],
    }

    excel_files = []
    glctx = dr.RasterizeCudaContext()
    mesh_tmp = copy.deepcopy(
        trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4))
    )
    mesh = trimesh.Trimesh(
        vertices=mesh_tmp.vertices.copy(), faces=mesh_tmp.faces.copy()
    )
    est = Any6D(
        mesh=mesh,
        scorer=ScorePredictor(),
        refiner=PoseRefinePredictor(),
        debug_dir=save_results_est_path,
        debug=0,
        glctx=glctx,
    )

    renderer = RendererVispy(640, 480, mode="depth")
    obj_count = 0
    
    data = []

    for scene_dir in tqdm(scene_dirs, desc="Evaluating Scenes"):
        if not os.path.exists(scene_dir):
            print(f"Scene directory {scene_dir} not found, skipping...")
            continue

        scene_name = os.path.basename(scene_dir)
        print(f"\nProcessing scene: {scene_name}")

        # Initialize TyolReader
        object_id = int(scene_name)
        reader = TyolReader(scene_dir, object_id=object_id)

        # Create list of frame indices to process
        original_color_files = reader.color_files
        frame_indices = list(range(args.start_frame, len(reader.color_files), args.running_stride))
        
        # Filter for frames with available GT poses
        valid_frame_indices = []
        for frame_idx in frame_indices:
            if frame_idx < len(reader):
                gt_pose = reader.get_gt_pose(frame_idx, object_id)
                if gt_pose is not None:
                    valid_frame_indices.append(frame_idx)
        
        print(
            f"Processing {len(valid_frame_indices)} out of {len(original_color_files)} frames "
            f"(start_frame={args.start_frame}, stride={args.running_stride}, with GT poses available)"
        )

        # Load the object mesh for this scene
        gt_mesh = reader.get_gt_mesh(args.tyol_model_path, object_id)
        gt_diameter = calc_pts_diameter(np.array(gt_mesh.vertices))
        
        # Setup renderer object
        gt_mesh_dict = {
            "pts": np.asarray(gt_mesh.vertices) * 1e3,  # Convert back to mm for renderer
            "normals": np.asarray(gt_mesh.face_normals),
            "faces": np.asarray(gt_mesh.faces),
        }
        renderer.my_add_object(gt_mesh_dict, object_id)
        
        # Reset object in estimator
        est.reset_object(mesh=gt_mesh, symmetry_tfs=None)

        # Load symmetry information
        with open(args.tyol_models_info_path, "r") as f:
            model_info = json.load(f)
        trans_disc = [{"R": np.eye(3), "t": np.array([[0, 0, 0]]).T}]  # Identity.
        if "symmetries_discrete" in model_info.get(f"{object_id}", {}):
            for sym in model_info[f"{object_id}"]["symmetries_discrete"]:
                sym_4x4 = np.reshape(sym, (4, 4))
                R = sym_4x4[:3, :3]
                t = sym_4x4[:3, 3].reshape((3, 1))
                trans_disc.append({"R": R, "t": t})

        # Process each frame
        for i, actual_frame_idx in enumerate(tqdm(valid_frame_indices, desc=f"Processing {scene_name}")):
            try:
                # Get query data by directly constructing file paths
                # RGB image
                color_file = f"{scene_dir}/rgb/{actual_frame_idx:06d}.png"
                color_query = cv2.imread(color_file)
                color_query = cv2.cvtColor(color_query, cv2.COLOR_BGR2RGB)
                
                # Depth image
                depth_file = f"{scene_dir}/depth/{actual_frame_idx:06d}.png"
                depth_query = cv2.imread(depth_file, -1).astype(np.float32) / 1000.0
                depth_query = depth_query / reader.bop_depth_scale
                
                # Mask image
                mask_file = f"{scene_dir}/mask_visib/{actual_frame_idx:06d}_000000.png"
                mask_query = cv2.imread(mask_file, -1)
                
                K_query = reader.K

                # Get ground truth pose directly from scene_gt
                frame_key = str(actual_frame_idx)
                if frame_key in reader.scene_gt:
                    objects_in_frame = reader.scene_gt[frame_key]
                    gt_pose = None
                    for obj_data in objects_in_frame:
                        if obj_data["obj_id"] == object_id:
                            # Convert BOP format to 4x4 matrix
                            R = np.array(obj_data["cam_R_m2c"]).reshape(3, 3)
                            t = np.array(obj_data["cam_t_m2c"]).reshape(3, 1) / 1000.0  # Convert mm to meters
                            
                            gt_pose = np.eye(4)
                            gt_pose[:3, :3] = R
                            gt_pose[:3, 3:4] = t
                            break
                    
                    if gt_pose is not None:
                        has_gt = True
                        # Debug: Print GT pose for verification
                        print(f"Frame {actual_frame_idx}: GT translation = {gt_pose[:3, 3]}")
                    else:
                        print(f"No GT pose for object {object_id} in frame {actual_frame_idx}")
                        has_gt = False
                else:
                    print(f"No GT data for frame {actual_frame_idx}, skipping metrics...")
                    has_gt = False

                # Initialize default values for metrics
                err_R = err_T = np.array([0.0])
                add = adds = 0.0
                add_thres = adds_thres = 0.0
                mean_ar = mean_vsd = mean_mssd = mean_mspd = 0.0

                # Run pose estimation using register
                ob_pose_pred = est.register(
                    K_query,
                    color_query,
                    depth_query,
                    mask_query,
                    iteration=5,
                )

                # Calculate metrics if ground truth is available
                if has_gt:
                    err_R, err_T = compute_RT_distances(ob_pose_pred, gt_pose)

                    pose_recall_th = [(5, 5), (5, 10), (10, 10)]

                    for r_th, t_th in pose_recall_th:
                        succ_r, succ_t = err_R <= r_th, err_T <= t_th
                        succ_pose = np.logical_and(succ_r, succ_t).astype(float)

                    add = compute_add(gt_mesh.vertices, ob_pose_pred, gt_pose)
                    adds = compute_adds(gt_mesh.vertices, ob_pose_pred, gt_pose)

                    add_thres = float(add <= gt_diameter * 0.1)
                    adds_thres = float(adds <= gt_diameter * 0.1)

                    ob_pose_pred, gt_pose = (
                        ob_pose_pred.astype(np.float16),
                        gt_pose.astype(np.float16),
                    )

                    pred_r, pred_t = (
                        ob_pose_pred[:3, :3],
                        np.expand_dims(ob_pose_pred[:3, 3], axis=1) * 1e3,  # Convert to mm
                    )
                    gt_r, gt_t = (
                        gt_pose[:3, :3],
                        np.expand_dims(gt_pose[:3, 3], axis=1) * 1e3,  # Convert to mm
                    )

                    mssd_err = (
                        mssd(
                            pose_est=ob_pose_pred,
                            pose_gt=gt_pose,
                            pts=gt_mesh.vertices,
                            syms=trans_disc,
                        )
                        * 1e3  # Convert to mm
                    )
                    mspd_err = mspd(
                        pose_est=ob_pose_pred,
                        pose_gt=gt_pose,
                        pts=gt_mesh.vertices,
                        K=K_query,
                        syms=trans_disc,
                    )

                    mssd_rec = np.array(
                        [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
                    )
                    mspd_rec = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

                    vsd_delta = 15.0
                    vsd_taus = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
                    vsd_rec = np.array(
                        [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
                    )

                    vsd_errs = vsd(
                        pred_r,
                        pred_t,
                        gt_r,
                        gt_t,
                        (depth_query * 1e3),  # Convert to mm
                        K_query.reshape(3, 3),
                        vsd_delta,
                        vsd_taus,
                        True,
                        (gt_diameter * 1e3),  # Convert to mm
                        renderer,
                        object_id,
                    )
                    vsd_errs = np.asarray(vsd_errs)
                    all_vsd_recs = np.stack([vsd_errs < rec_i for rec_i in vsd_rec], axis=1)
                    mean_vsd = all_vsd_recs.mean()

                    mssd_cur_rec = mssd_rec * (gt_diameter * 1e3)  # Convert to mm
                    mean_mssd = (mssd_err < mssd_cur_rec).mean()
                    mean_mspd = (mspd_err < mspd_rec).mean()

                    mean_ar = (mean_mssd + mean_mspd + mean_vsd) / 3.0

                    result["ADD"].append(float(add_thres))
                    result["ADDS"].append(float(adds_thres))
                    result["AR"].append(float(mean_ar))
                    result["VSD"].append(float(mean_vsd))
                    result["MSSD"].append(float(mean_mssd))
                    result["MSPD"].append(float(mean_mspd))
                    result["R_error"].append(float(err_R.tolist()[0]))
                    result["T_error"].append(float(err_T.tolist()[0]))

                    # Log frame-level metrics to wandb
                    wandb.log({
                        "frame_idx": actual_frame_idx,
                        "scene": scene_name,
                        "frame_count": obj_count,
                        "ADD": add_thres,
                        "ADD-S": adds_thres,
                        "AR": mean_ar,
                        "VSD": mean_vsd,
                        "MSSD": mean_mssd,
                        "MSPD": mean_mspd,
                        "R_error": err_R.item(),
                        "T_error": err_T.item(),
                        "ADD_distance": add,
                        "ADD-S_distance": adds,
                    })

                    # Store frame data
                    frame_data = {
                        "scene": scene_name,
                        "frame": actual_frame_idx,
                        "object_id": object_id,
                        "ADD": float(add_thres),
                        "ADD-S": float(adds_thres),
                        "AR": float(mean_ar),
                        "VSD": float(mean_vsd),
                        "MSSD": float(mean_mssd),
                        "MSPD": float(mean_mspd),
                        "R_error": float(err_R.item()),
                        "T_error": float(err_T.item()),
                        "ADD_distance": float(add),
                        "ADD-S_distance": float(adds),
                        "pred_pose": [[float(x) for x in row] for row in ob_pose_pred],
                        "gt_pose": [[float(x) for x in row] for row in gt_pose],
                    }
                    data.append(frame_data)

                # Save predicted pose
                pose_save_path = os.path.join(
                    save_results_est_path,
                    f"{scene_name}_frame_{actual_frame_idx:06d}_pred_pose.txt",
                )
                np.savetxt(pose_save_path, ob_pose_pred)

                # Visualization
                try:
                    # Prepare metrics for visualization
                    frame_metrics = {
                        "instance_id": [obj_count],
                        "ADD-S": [adds_thres] if has_gt else [0],
                        "ADD": [add_thres] if has_gt else [0],
                        "AR": [mean_ar] if has_gt else [0],
                        "VSD": [mean_vsd] if has_gt else [0],
                        "MSSD": [mean_mssd] if has_gt else [0],
                        "MSPD": [mean_mspd] if has_gt else [0],
                        "R_error": [err_R.tolist()[0]] if has_gt else [0],
                        "T_error": [err_T.tolist()[0]] if has_gt else [0],
                        "cls_id": [scene_name],
                    }
                    
                    visualize_frame_results_gt(
                        color=color_query,
                        gt_mesh=gt_mesh,
                        K=reader.K,
                        gt_pose=gt_pose if has_gt else np.eye(4),
                        pred_pose=ob_pose_pred,
                        metric=frame_metrics,
                        obj_f=f"{scene_name}",
                        frame_idx=actual_frame_idx,
                        save_path=save_results_est_path,
                        glctx=glctx,
                        name=f"{len(reader.color_files)}_{args.name}",
                        nocs_metric=True,
                        est_mesh=gt_mesh,
                    )
                except Exception as viz_error:
                    print(f"Warning: Visualization failed for frame {actual_frame_idx}: {viz_error}")

            except Exception as e:
                print(f"Error processing frame {actual_frame_idx} in {scene_name}: {e}")
                continue

        obj_count += 1

    # Calculate and log summary statistics
    if result["ADD"]:
        summary = {
            "mean_ADD": np.mean(result["ADD"]),
            "mean_ADDS": np.mean(result["ADDS"]),
            "mean_AR": np.mean(result["AR"]),
            "mean_VSD": np.mean(result["VSD"]),
            "mean_MSSD": np.mean(result["MSSD"]),
            "mean_MSPD": np.mean(result["MSPD"]),
            "mean_R_error": np.mean(result["R_error"]),
            "mean_T_error": np.mean(result["T_error"]),
            "std_ADD": np.std(result["ADD"]),
            "std_ADDS": np.std(result["ADDS"]),
            "std_AR": np.std(result["AR"]),
            "std_VSD": np.std(result["VSD"]),
            "std_MSSD": np.std(result["MSSD"]),
            "std_MSPD": np.std(result["MSPD"]),
            "std_R_error": np.std(result["R_error"]),
            "std_T_error": np.std(result["T_error"]),
        }

        print("\nSummary Results:")
        for key, value in summary.items():
            print(f"{key}: {value:.4f}")

        wandb.log(summary)

        # Save results to JSON
        results_file = os.path.join(save_results_est_path, "results.json")
        with open(results_file, "w") as f:
            json.dump({"summary": summary, "data": data}, f, indent=2)

        print(f"\nResults saved to: {results_file}")
    else:
        print("\nNo frames with ground truth were processed.")

    wandb.finish()
    print(f"\nProcessed {obj_count} scenes")