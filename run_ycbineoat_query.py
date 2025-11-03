import copy
import yaml as pyyaml
import wandb

from foundationpose.datareader import YcbineoatReader
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

    # Anchor reference arguments (matching run_ho3d_anchor.py)
    parser.add_argument(
        "--anchor-image",
        type=str,
        required=True,
        help="Path to the anchor image",
    )
    parser.add_argument(
        "--anchor-mask",
        type=str,
        required=True,
        help="Path to the anchor mask image file",
    )
    parser.add_argument(
        "--anchor-depth",
        type=str,
        required=True,
        help="Path to the anchor depth image",
    )
    parser.add_argument(
        "--anchor-calibration",
        type=str,
        required=True,
        help="Path to calibration file",
    )
    parser.add_argument(
        "--anchor-object-pose",
        type=str,
        required=True,
        help="Path to text file containing ground-truth object pose for anchor",
    )
    parser.add_argument(
        "--anchor-pred-pose",
        type=str,
        required=True,
        help="Path to predicted anchor pose matrix",
    )
    parser.add_argument(
        "--anchor-mesh", type=str, required=True, help="Path to the anchor mesh file"
    )
    parser.add_argument(
        "--save-dir", type=str, default="./results", help="Directory to save results"
    )

    # YCBInEOAT specific arguments
    parser.add_argument(
        "--ycbineoat_data_root",
        type=str,
        required=True,
        help="Path to YCBInEOAT dataset root",
    )
    parser.add_argument(
        "--ycb_model_path",
        type=str,
        required=True,
        help="Path to YCB models",
    )
    parser.add_argument(
        "--ycbv_models_info_path",
        type=str,
        required=True,
        help="Path to YCB-V models_info.json",
    )
    parser.add_argument(
        "--video_name",
        type=str,
        default=None,
        help="Specific video to process (e.g., bleach0)",
    )
    parser.add_argument(
        "--running_stride",
        type=int,
        default=10,
        help="Process every N frames",
    )

    args = parser.parse_args()

    # Create save directories
    save_results_est_path = os.path.join(args.save_dir, f"{args.name}_est")
    os.makedirs(save_results_est_path, exist_ok=True)

    # Setup wandb
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb_run_name = f"{args.name}_{current_time}"
    wandb.init(
        project="any6d-ycbineoat-query",
        name=wandb_run_name,
        config=vars(args),
        tags=["ycbineoat", "query", args.name],
    )
    wandb.config.update(args)

    # Get list of videos to process
    if args.video_name:
        video_dirs = [os.path.join(args.ycbineoat_data_root, args.video_name)]
    else:
        # Process all available videos
        video_names = [
            "bleach0",
            "bleach_hard_00_03_chaitanya",
            "cracker_box_reorient",
            "cracker_box_yalehand0",
            "mustard0",
            "mustard_easy_00_02",
            "sugar_box1",
            "sugar_box_yalehand0",
            "tomato_soup_can_yalehand0",
        ]
        video_dirs = [
            os.path.join(args.ycbineoat_data_root, name)
            for name in video_names
            if os.path.exists(os.path.join(args.ycbineoat_data_root, name))
        ]

    # Load anchor data
    print(f"Loading anchor data...")
    anchor_depth = cv2.imread(args.anchor_depth, -1) / 1000.0
    anchor_mask = cv2.imread(args.anchor_mask, -1)
    anchor_image = cv2.imread(args.anchor_image)
    anchor_gt_pose = np.loadtxt(args.anchor_object_pose).reshape(4, 4)
    anchor_pred_pose = np.loadtxt(args.anchor_pred_pose).reshape(4, 4)
    mesh_anchor = trimesh.load(args.anchor_mesh, force="mesh")

    # Load anchor calibration from yml file
    with open(args.anchor_calibration, "r") as file:
        calib_data = pyyaml.unsafe_load(file)

    # Extract intrinsic matrix (using color camera parameters)
    K_anchor = np.array(
        [
            [calib_data["color"]["fx"], 0, calib_data["color"]["ppx"]],
            [0, calib_data["color"]["fy"], calib_data["color"]["ppy"]],
            [0, 0, 1],
        ]
    )

    # Initialize results storage (same as HO3D)
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

    # YCBInEOAT video name to object mapping
    videoname_to_object = {
        "bleach0": "021_bleach_cleanser",
        "bleach_hard_00_03_chaitanya": "021_bleach_cleanser",
        "cracker_box_reorient": "003_cracker_box",
        "cracker_box_yalehand0": "003_cracker_box",
        "mustard0": "006_mustard_bottle",
        "mustard_easy_00_02": "006_mustard_bottle",
        "sugar_box1": "004_sugar_box",
        "sugar_box_yalehand0": "004_sugar_box",
        "tomato_soup_can_yalehand0": "005_tomato_soup_can",
    }

    # Object name to ID mapping for YCB
    name_to_id = {
        "003_cracker_box": 3,
        "004_sugar_box": 4,
        "005_tomato_soup_can": 5,
        "006_mustard_bottle": 6,
        "021_bleach_cleanser": 21,
    }

    for video_dir in tqdm(video_dirs, desc="Evaluating Videos"):
        if not os.path.exists(video_dir):
            print(f"Video directory {video_dir} not found, skipping...")
            continue

        video_name = os.path.basename(video_dir)
        print(f"\nProcessing video: {video_name}")

        # Initialize YcbineoatReader
        reader = YcbineoatReader(video_dir)

        # Apply stride to process fewer frames
        original_color_files = reader.color_files
        reader.color_files = reader.color_files[:: args.running_stride]
        print(
            f"Processing {len(reader.color_files)} out of {len(original_color_files)} frames"
        )

        # Get object information
        obj_name = videoname_to_object.get(video_name)
        if obj_name is None:
            print(f"Unknown object for video {video_name}, skipping...")
            continue

        ob_id = name_to_id.get(obj_name)
        if ob_id is None:
            print(f"Unknown object ID for {obj_name}, skipping...")
            continue

        # Load the object mesh for this video
        gt_mesh = reader.get_gt_mesh(args.ycb_model_path)
        gt_diameter = calc_pts_diameter(np.array(gt_mesh.vertices))
        
        # Setup renderer object (same as HO3D)
        gt_mesh_dict = {
            "pts": np.asarray(gt_mesh.vertices) * 1e3,
            "normals": np.asarray(gt_mesh.face_normals),
            "faces": np.asarray(gt_mesh.faces),
        }
        renderer.my_add_object(gt_mesh_dict, ob_id)
        
        # Reset object in estimator (like HO3D script does)
        est.reset_object(mesh=gt_mesh, symmetry_tfs=None)

        # Load symmetry information
        with open(args.ycbv_models_info_path, "r") as f:
            model_info = json.load(f)
        trans_disc = [{"R": np.eye(3), "t": np.array([[0, 0, 0]]).T}]  # Identity.
        if "symmetries_discrete" in model_info[f"{ob_id}"]:
            for sym in model_info[f"{ob_id}"]["symmetries_discrete"]:
                sym_4x4 = np.reshape(sym, (4, 4))
                R = sym_4x4[:3, :3]
                t = sym_4x4[:3, 3].reshape((3, 1))
                trans_disc.append({"R": R, "t": t})

        # Process each frame
        for i_frame in tqdm(range(len(reader)), desc=f"Processing {video_name}"):
            try:
                # Get query data
                color_query = reader.get_color(i_frame)
                depth_query = reader.get_depth(i_frame)
                mask_query = reader.get_mask(i_frame)
                K_query = reader.K

                # Get ground truth pose if available
                gt_pose = reader.get_gt_pose(i_frame)
                if gt_pose is None:
                    print(f"No GT pose for frame {i_frame}, skipping metrics...")
                    has_gt = False
                else:
                    has_gt = True

                # Run pose estimation using register (same as HO3D script)
                ob_pose_pred = est.register(
                    K_query,
                    color_query,
                    depth_query,
                    mask_query,
                    iteration=5,
                )

                # Initialize default values for metrics
                err_R = err_T = np.array([0.0])
                add = adds = 0.0
                add_thres = adds_thres = 0.0
                mean_ar = mean_vsd = mean_mssd = mean_mspd = 0.0

                # Calculate metrics if ground truth is available (exactly same as HO3D)
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
                        np.expand_dims(ob_pose_pred[:3, 3], axis=1) * 1e3,
                    )
                    gt_r, gt_t = (
                        gt_pose[:3, :3],
                        np.expand_dims(gt_pose[:3, 3], axis=1) * 1e3,
                    )

                    mssd_err = (
                        mssd(
                            pose_est=ob_pose_pred,
                            pose_gt=gt_pose,
                            pts=gt_mesh.vertices,
                            syms=trans_disc,
                        )
                        * 1e3
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
                        (depth_query * 1e3),
                        K_query.reshape(3, 3),
                        vsd_delta,
                        vsd_taus,
                        True,
                        (gt_diameter * 1e3),
                        renderer,
                        ob_id,
                    )
                    vsd_errs = np.asarray(vsd_errs)
                    all_vsd_recs = np.stack([vsd_errs < rec_i for rec_i in vsd_rec], axis=1)
                    mean_vsd = all_vsd_recs.mean()

                    mssd_cur_rec = mssd_rec * (gt_diameter * 1e3)
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

                    # Log frame-level metrics to wandb (same as HO3D)
                    wandb.log({
                        "frame_idx": i_frame,
                        "video": video_name,
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

                    # Store frame data (same as HO3D)
                    frame_data = {
                        "video": video_name,
                        "frame": i_frame,
                        "object": obj_name,
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
                    f"{video_name}_frame_{i_frame:06d}_pred_pose.txt",
                )
                np.savetxt(pose_save_path, ob_pose_pred)

                # Visualization (same as HO3D script)
                try:
                    # Prepare metrics for visualization (same as HO3D)
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
                        "cls_id": [video_name],
                    }
                    
                    visualize_frame_results_gt(
                        color=color_query,
                        gt_mesh=gt_mesh,
                        K=reader.K,
                        gt_pose=gt_pose if has_gt else np.eye(4),
                        pred_pose=ob_pose_pred,
                        metric=frame_metrics,
                        obj_f=f"{video_name}",
                        frame_idx=i_frame,
                        save_path=save_results_est_path,
                        glctx=glctx,
                        name=f"{len(reader.color_files)}_{args.name}",
                        nocs_metric=True,
                        est_mesh=gt_mesh,
                    )
                except Exception as viz_error:
                    print(f"Warning: Visualization failed for frame {i_frame}: {viz_error}")

            except Exception as e:
                print(f"Error processing frame {i_frame} in {video_name}: {e}")
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
    print(f"\nProcessed {obj_count} videos")
