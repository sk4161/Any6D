import glob
import copy
import yaml as pyyaml
import wandb

from foundationpose.datareader import TlessReader
from foundationpose.Utils import visualize_frame_results_gt
from estimater import *
from bop_toolkit_lib.pose_error_custom import mssd, mspd, vsd

from metrics import *
import json
from bop_toolkit_lib.renderer_vispy import RendererVispy
from pytorch_lightning import seed_everything
from datetime import datetime
from typing import Optional


class MissingGroundTruthError(RuntimeError):
    """Raised when a query frame lacks ground-truth pose."""

    def __init__(self, frame_index: int):
        super().__init__(f"Ground-truth pose is unavailable for frame {frame_index}")
        self.frame_index = frame_index


def find_reconstructed_mesh(anchor_dir: str, object_id: int, scene_hint: Optional[str] = None) -> str:
    """
    Locate the reconstructed mesh produced during the anchor stage.
    Prefers meshes that match the anchor scene hint, but falls back to any matching object file.
    """
    candidates = []
    if scene_hint is not None:
        candidates.append(
            os.path.join(anchor_dir, f"final_mesh_scene{scene_hint}_obj{object_id:02d}.obj")
        )
    candidates.extend(
        sorted(glob.glob(os.path.join(anchor_dir, f"final_mesh_*_obj{object_id:02d}.obj")))
    )

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"No reconstructed mesh found for object {object_id} in {anchor_dir}. "
        "Run the anchor script first to generate final_mesh_scene*_objXX.obj."
    )


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
        "--anchor-object-id",
        type=int,
        default=None,
        help="Object ID for anchor (e.g., 29)",
    )
    parser.add_argument(
        "--anchor-mask-id",
        type=int,
        default=None,
        help="Mask ID for anchor (0, 1, 2, 3...) - alternative to --anchor-object-id",
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
        help="Directory where anchor results are saved (e.g., tless_anchor_results/scene000001_obj29_frame0)",
    )
    parser.add_argument(
        "--save-dir", type=str, default="./results", help="Directory to save results"
    )

    # TLESS specific arguments
    parser.add_argument(
        "--tless_data_root",
        type=str,
        required=True,
        help="Path to TLESS dataset root",
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        default=None,
        help="Specific scene to process (e.g., 000001)",
    )
    parser.add_argument(
        "--object_id",
        type=int,
        default=None,
        help="Specific object to process (e.g., 29)",
    )
    parser.add_argument(
        "--mask_id",
        type=int,
        default=None,
        help="Specific mask ID to process (0, 1, 2, 3...) - alternative to --object_id",
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
        project="any6d-tless-query",
        name=wandb_run_name,
        config=vars(args),
        tags=["tless", "query", args.name],
    )
    wandb.config.update(args)

    # Get list of scenes to process
    if args.scene_id:
        scene_dirs = [os.path.join(args.tless_data_root, "test_primesense", args.scene_id)]
    else:
        # Process all available scenes
        test_dir = os.path.join(args.tless_data_root, "test_primesense")
        scene_names = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
        scene_names = sorted(scene_names)
        scene_dirs = [
            os.path.join(test_dir, name)
            for name in scene_names
            if os.path.exists(os.path.join(test_dir, name))
        ]

    # Initialize anchor scene reader
    anchor_scene_dir = os.path.join(args.tless_data_root, "test_primesense", args.anchor_scene_id)
    anchor_reader = TlessReader(anchor_scene_dir)
    
    # Determine anchor object ID
    if args.anchor_object_id is None and args.anchor_mask_id is None:
        raise ValueError("Must specify either --anchor-object-id or --anchor-mask-id")
    if args.anchor_object_id is not None and args.anchor_mask_id is not None:
        raise ValueError("Cannot specify both --anchor-object-id and --anchor-mask-id")
    
    if args.anchor_object_id is None:
        # Use anchor_mask_id to determine anchor_object_id
        anchor_object_ids = anchor_reader.get_instance_ids_in_image(args.anchor_frame_index)
        if args.anchor_mask_id >= len(anchor_object_ids):
            raise ValueError(f"Anchor mask ID {args.anchor_mask_id} out of range. Frame has {len(anchor_object_ids)} objects")
        anchor_object_id = anchor_object_ids[args.anchor_mask_id]
        anchor_mask_id = args.anchor_mask_id
        print(f"Using anchor mask ID {anchor_mask_id} which corresponds to object ID {anchor_object_id}")
    else:
        anchor_object_id = args.anchor_object_id
        anchor_object_ids = anchor_reader.get_instance_ids_in_image(args.anchor_frame_index)
        if anchor_object_id not in anchor_object_ids:
            raise ValueError(f"Anchor object {anchor_object_id} not found in frame {args.anchor_frame_index}")
        anchor_mask_id = list(anchor_object_ids).index(anchor_object_id)
        print(f"Using anchor object ID {anchor_object_id} which corresponds to mask ID {anchor_mask_id}")

    # Load anchor data from TLESS dataset and anchor results
    print(f"Loading anchor data from scene {args.anchor_scene_id}, object {anchor_object_id}, frame {args.anchor_frame_index}...")
    
    # Load anchor data from TLESS dataset
    anchor_image = anchor_reader.get_color(args.anchor_frame_index)
    anchor_depth = anchor_reader.get_depth(args.anchor_frame_index)
    anchor_mask = anchor_reader.get_mask(args.anchor_frame_index, anchor_object_id)
    K_anchor = anchor_reader.get_K(args.anchor_frame_index)
    anchor_gt_pose = anchor_reader.get_gt_pose(args.anchor_frame_index, anchor_object_id)
    
    # Debug: Print anchor GT pose
    print(f"Anchor frame {args.anchor_frame_index}: GT translation = {anchor_gt_pose[:3, 3]}")
    
    find_reconstructed_mesh(args.anchor_save_dir, anchor_object_id, args.anchor_scene_id)
    mesh_cache = {}
    anchor_pose_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def get_anchor_reference(obj_id: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (anchor_gt_pose, anchor_pred_pose) for the requested object.
        """
        if obj_id in anchor_pose_cache:
            return anchor_pose_cache[obj_id]

        anchor_gt_obj = anchor_reader.get_gt_pose(args.anchor_frame_index, obj_id)
        if anchor_gt_obj is None:
            raise MissingGroundTruthError(args.anchor_frame_index)

        pose_file = os.path.join(
            args.anchor_save_dir,
            f"scene{args.anchor_scene_id}_obj{obj_id:02d}_predicted_pose.txt",
        )
        if not os.path.exists(pose_file):
            raise FileNotFoundError(
                f"Anchor predicted pose file missing for object {obj_id}: {pose_file}"
            )
        anchor_pred_obj = np.loadtxt(pose_file).reshape(4, 4)
        anchor_pose_cache[obj_id] = (anchor_gt_obj, anchor_pred_obj)
        return anchor_pose_cache[obj_id]

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

    # Get image dimensions from TLESS dataset
    sample_color = anchor_reader.get_color(0)
    img_height, img_width = sample_color.shape[:2]
    renderer = RendererVispy(img_width, img_height, mode="depth")
    obj_count = 0
    
    data = []

    for scene_dir in tqdm(scene_dirs, desc="Evaluating Scenes"):
        if not os.path.exists(scene_dir):
            print(f"Scene directory {scene_dir} not found, skipping...")
            continue

        scene_name = os.path.basename(scene_dir)
        print(f"\nProcessing scene: {scene_name}")

        # Initialize TlessReader
        reader = TlessReader(scene_dir)

        # Determine which objects to process in this scene
        objects_to_process = []
        if args.object_id is not None:
            # Process specific object only
            objects_to_process = [args.object_id]
        elif args.mask_id is not None:
            # Process specific mask ID only
            frame_0_objects = reader.get_instance_ids_in_image(0)
            if args.mask_id >= len(frame_0_objects):
                print(f"Mask ID {args.mask_id} out of range for scene {scene_name}. Available mask IDs: 0-{len(frame_0_objects)-1}")
                continue
            target_object_id = frame_0_objects[args.mask_id]
            objects_to_process = [target_object_id]
            print(f"Using mask ID {args.mask_id} which corresponds to object ID {target_object_id}")
        else:
            # Get all objects that appear in the scene
            frame_0_objects = reader.get_instance_ids_in_image(0)
            objects_to_process = frame_0_objects
        
        print(f"Processing objects: {objects_to_process}")

        for object_id in objects_to_process:
            print(f"\nProcessing object {object_id} in scene {scene_name}")
            
            # Load anchor reference (GT + predicted pose) for HO3D-style relative transform
            try:
                anchor_gt_pose_obj, anchor_pred_pose_obj = get_anchor_reference(object_id)
            except (FileNotFoundError, MissingGroundTruthError) as anchor_err:
                print(f"Skipping object {object_id}: {anchor_err}")
                continue

            # Load the reconstructed mesh from anchor stage for estimation
            if object_id not in mesh_cache:
                try:
                    mesh_path = find_reconstructed_mesh(
                        args.anchor_save_dir, object_id, args.anchor_scene_id
                    )
                except FileNotFoundError as missing_mesh_err:
                    print(missing_mesh_err)
                    print(f"Skipping object {object_id} due to missing reconstructed mesh.")
                    continue
                mesh_cache[object_id] = trimesh.load(mesh_path, force="mesh")
            reconstructed_mesh = mesh_cache[object_id].copy()

            # Load the ground-truth CAD mesh for evaluation metrics only
            models_dir = os.path.join(args.tless_data_root, "models_cad")
            gt_mesh_path = os.path.join(models_dir, f"obj_{object_id:06d}.ply")
            if not os.path.exists(gt_mesh_path):
                print(f"Model file not found: {gt_mesh_path}, skipping object {object_id}")
                continue

            gt_mesh = trimesh.load(gt_mesh_path)
            # Apply same scaling as in anchor script (TLESS meshes are in mm)
            gt_mesh.vertices *= 0.001

            gt_diameter = calc_pts_diameter(np.array(gt_mesh.vertices))

            # Setup renderer object
            gt_mesh_dict = {
                "pts": np.asarray(gt_mesh.vertices) * 1e3,  # Convert back to mm for renderer
                "normals": np.asarray(gt_mesh.face_normals),
                "faces": np.asarray(gt_mesh.faces),
            }
            renderer.my_add_object(gt_mesh_dict, object_id)

            # Reset object in estimator using reconstructed mesh
            est.reset_object(mesh=reconstructed_mesh, symmetry_tfs=None)

            # TLESS objects are textureless, no specific symmetry info available
            trans_disc = [{"R": np.eye(3), "t": np.array([[0, 0, 0]]).T}]  # Identity only

            # Create list of frame indices to process
            frame_indices = list(range(args.start_frame, len(reader.color_files), args.running_stride))
            
            # Filter for frames with this specific object
            valid_frame_indices = []
            for frame_idx in frame_indices:
                if frame_idx < len(reader.color_files):
                    objects_in_frame = reader.get_instance_ids_in_image(frame_idx)
                    if object_id in objects_in_frame:
                        gt_pose = reader.get_gt_pose(frame_idx, object_id)
                        if gt_pose is not None:
                            valid_frame_indices.append(frame_idx)
            
            print(
                f"Processing {len(valid_frame_indices)} frames for object {object_id} "
                f"(start_frame={args.start_frame}, stride={args.running_stride}, with GT poses available)"
            )

            # Process each frame
            for i, actual_frame_idx in enumerate(tqdm(valid_frame_indices, desc=f"Processing object {object_id}")):
                try:
                    # Get query data
                    color_query = reader.get_color(actual_frame_idx)
                    depth_query = reader.get_depth(actual_frame_idx)
                    mask_query = reader.get_mask(actual_frame_idx, object_id)
                    
                    if mask_query is None:
                        print(f"No mask for object {object_id} in frame {actual_frame_idx}, skipping")
                        continue
                    
                    K_query = reader.get_K(actual_frame_idx)
                    gt_pose = reader.get_gt_pose(actual_frame_idx, object_id)
                    
                    if gt_pose is None:
                        print(f"No GT pose for object {object_id} in frame {actual_frame_idx}, skipping")
                        continue
                    
                    has_gt = True
                    # Debug: Print GT pose for verification
                    print(f"Frame {actual_frame_idx}, Object {object_id}: GT translation = {gt_pose[:3, 3]}")

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

                    pose_aq = ob_pose_pred @ np.linalg.inv(anchor_pred_pose_obj)
                    pred_pose = pose_aq @ anchor_gt_pose_obj

                    # Calculate metrics if ground truth is available
                    if has_gt:
                        err_R, err_T = compute_RT_distances(pred_pose, gt_pose)

                        pose_recall_th = [(5, 5), (5, 10), (10, 10)]

                        for r_th, t_th in pose_recall_th:
                            succ_r, succ_t = err_R <= r_th, err_T <= t_th
                            succ_pose = np.logical_and(succ_r, succ_t).astype(float)

                        add = compute_add(gt_mesh.vertices, pred_pose, gt_pose)
                        adds = compute_adds(gt_mesh.vertices, pred_pose, gt_pose)

                        add_thres = float(add <= gt_diameter * 0.1)
                        adds_thres = float(adds <= gt_diameter * 0.1)

                        pred_pose = pred_pose.astype(np.float16)
                        gt_pose = gt_pose.astype(np.float16)

                        pred_r, pred_t = (
                            pred_pose[:3, :3],
                            np.expand_dims(pred_pose[:3, 3], axis=1) * 1e3,  # Convert to mm
                        )
                        gt_r, gt_t = (
                            gt_pose[:3, :3],
                            np.expand_dims(gt_pose[:3, 3], axis=1) * 1e3,  # Convert to mm
                        )

                        mssd_err = (
                            mssd(
                                pose_est=pred_pose,
                                pose_gt=gt_pose,
                                pts=gt_mesh.vertices,
                                syms=trans_disc,
                            )
                            * 1e3  # Convert to mm
                        )
                        mspd_err = mspd(
                            pose_est=pred_pose,
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

                        try:
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
                        except Exception as vsd_error:
                            print(f"Warning: VSD calculation failed for frame {actual_frame_idx}: {vsd_error}")
                            mean_vsd = 0.0  # Use default value when VSD calculation fails

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
                            "object_id": object_id,
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
                            "pred_pose": [[float(x) for x in row] for row in pred_pose],
                            "gt_pose": [[float(x) for x in row] for row in gt_pose],
                        }
                        data.append(frame_data)

                    # Save predicted pose
                    pose_save_path = os.path.join(
                        save_results_est_path,
                        f"{scene_name}_obj{object_id:02d}_frame_{actual_frame_idx:06d}_pred_pose.txt",
                    )
                    np.savetxt(pose_save_path, pred_pose)

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
                            "cls_id": [f"{scene_name}_obj{object_id}"],
                        }
                        
                        # For visualization: est_mesh should be the mesh without pose applied
                        # The visualize function will apply pred_pose to it
                        visualize_frame_results_gt(
                            color=color_query,
                            gt_mesh=gt_mesh,
                            K=K_query,
                            gt_pose=gt_pose if has_gt else np.eye(4),
                            pred_pose=pred_pose,
                            metric=frame_metrics,
                            obj_f=f"{scene_name}_obj{object_id:02d}",
                            frame_idx=actual_frame_idx,
                            save_path=save_results_est_path,
                            glctx=glctx,
                            name=f"{len(reader.color_files)}_{args.name}",
                            nocs_metric=True,
                            est_mesh=gt_mesh,  # Use GT mesh, pred_pose will be applied during rendering
                        )
                    except Exception as viz_error:
                        print(f"Warning: Visualization failed for frame {actual_frame_idx}: {viz_error}")

                except Exception as e:
                    print(f"Error processing frame {actual_frame_idx} in {scene_name} for object {object_id}: {e}")
                    import traceback
                    traceback.print_exc()
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
    print(f"\nProcessed {obj_count} object instances")
