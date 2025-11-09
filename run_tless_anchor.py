import os
import trimesh
import numpy as np

import nvdiffrast.torch as dr
import argparse

from estimater import Any6D
from foundationpose.datareader import TlessReader
from foundationpose.Utils import (
    visualize_frame_results,
    calculate_chamfer_distance_gt_mesh,
    align_mesh_to_coordinate,
)
from sam2_instantmesh import (
    get_bounding_box,
    running_sam_box,
    preprocess_image,
    diffusion_image_generation,
    instant_mesh_process,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TLESS anchor pose estimation")
    parser.add_argument(
        "--tless-data-root",
        type=str,
        required=True,
        help="Path to TLESS dataset root directory",
    )
    parser.add_argument(
        "--scene-id",
        type=str,
        required=True,
        help="Scene ID (e.g., 000001)",
    )
    parser.add_argument(
        "--object-id",
        type=int,
        default=None,
        help="Object ID (e.g., 2, 25, 29, 30). If not specified, uses the first object in the scene",
    )
    parser.add_argument(
        "--mask-id",
        type=int,
        default=None,
        help="Mask ID (0, 1, 2, 3...) - position of object in scene_gt.json for the frame. Alternative to --object-id",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        required=True,
        help="Frame index to use as anchor",
    )
    parser.add_argument(
        "--anchor-mesh",
        type=str,
        required=True,
        help="Path to the anchor mesh file (.obj)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Directory to save output files",
    )
    parser.add_argument(
        "--tless-model-path",
        type=str,
        default=None,
        help="Path to TLESS models directory (optional, not used)",
    )
    parser.add_argument(
        "--img_to_3d", 
        action="store_true", 
        help="Running with InstantMesh+SAM2"
    )
    args = parser.parse_args()

    # Parse arguments
    tless_data_root = args.tless_data_root
    scene_id = args.scene_id
    object_id = args.object_id
    mask_id = args.mask_id
    frame_index = args.frame_index
    anchor_mesh = args.anchor_mesh
    save_dir = args.save_dir
    tless_model_path = args.tless_model_path
    img_to_3d = args.img_to_3d
    
    # Validate that only one of object_id or mask_id is specified
    if object_id is not None and mask_id is not None:
        raise ValueError("Cannot specify both --object-id and --mask-id. Choose one.")
    if object_id is None and mask_id is None:
        mask_id = 0  # Default to first object (mask_id=0)
    
    # Create save directory
    save_path = save_dir
    os.makedirs(save_path, exist_ok=True)

    # Initialize TLESS reader (following TYO-L pattern)
    scene_dir = os.path.join(tless_data_root, "test_primesense", scene_id)
    if not os.path.exists(scene_dir):
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
    
    reader = TlessReader(scene_dir)
    
    # Validate frame index
    if frame_index >= len(reader.color_files):
        raise ValueError(f"Frame index {frame_index} out of range. Scene has {len(reader.color_files)} frames.")
    
    # Handle object ID selection
    if object_id is None:
        # Use mask_id to determine object_id
        object_ids = reader.get_instance_ids_in_image(frame_index)
        if len(object_ids) == 0:
            raise ValueError(f"No objects found in frame {frame_index}")
        if mask_id >= len(object_ids):
            raise ValueError(f"Mask ID {mask_id} out of range. Frame {frame_index} has {len(object_ids)} objects (mask IDs: 0-{len(object_ids)-1})")
        object_id = object_ids[mask_id]
        print(f"Using mask ID {mask_id} which corresponds to object ID {object_id}")
        print(f"Available objects in frame {frame_index}: {object_ids}")
    else:
        # Validate specified object ID exists in the frame
        object_ids = reader.get_instance_ids_in_image(frame_index)
        if object_id not in object_ids:
            raise ValueError(f"Object {object_id} not found in frame {frame_index}. Available objects: {object_ids}")
        # Find mask_id for this object_id for informational purposes
        mask_id = list(object_ids).index(object_id)
        print(f"Using object ID {object_id} which corresponds to mask ID {mask_id}")
    
    print(f"Loading TLESS scene {scene_id}, object {object_id}, frame {frame_index}")

    glctx = dr.RasterizeCudaContext()

    # Load input data from TLESS dataset (following TYO-L pattern)
    color = reader.get_color(frame_index)
    # Note: TlessReader.get_color() returns RGB format (via imageio.imread)
    # No color conversion needed
    
    depth = reader.get_depth(frame_index)
    
    # Get object mask
    mask_img = reader.get_mask(frame_index, object_id)
    if mask_img is None:
        raise ValueError(f"No mask found for object {object_id} in frame {frame_index}")
    
    # Convert mask to boolean (same as TYO-L)
    mask = (mask_img > 0).astype(np.bool_)
    
    # Get camera intrinsics for this frame
    K = reader.get_K(frame_index)
    
    # Get ground truth pose
    gt_pose = reader.get_gt_pose(frame_index, object_id)
    if gt_pose is None:
        raise ValueError(f"No ground truth pose found for object {object_id} in frame {frame_index}")

    # Use provided mesh path
    mesh_path = anchor_mesh

    # Process with or without InstantMesh (same as TYO-L)
    if img_to_3d:
        cmin, rmin, cmax, rmax = get_bounding_box(mask).astype(np.int32)
        input_box = np.array([cmin, rmin, cmax, rmax])[None, :]
        mask_refine = running_sam_box(color, input_box)
        
        # Preprocess image for InstantMesh
        image_processed = preprocess_image(color, mask_refine)
        
        # Generate multi-view images
        image_mv = diffusion_image_generation(image_processed)
        
        # Process with InstantMesh
        mesh_path = instant_mesh_process(
            image_processed, 
            image_mv, 
            f"{save_path}/scene_{scene_id}_obj{object_id:02d}_instantmesh.obj"
        )
        
        # Load the generated mesh
        mesh = trimesh.load(mesh_path)
        print(f"Generated mesh with InstantMesh: {mesh_path}")
    else:
        # Load mesh from file
        if os.path.exists(mesh_path):
            mesh = trimesh.load(mesh_path)
        else:
            raise FileNotFoundError(f"Mesh file not found at {mesh_path}")

    # Align mesh and save center mesh (TLESS meshes are in mm, convert to meters)
    mesh = align_mesh_to_coordinate(mesh, scale=0.001)
    mesh.export(os.path.join(save_path, f"center_mesh_scene{scene_id}_obj{object_id:02d}.obj"))

    # Initialize estimator (same as TYO-L)
    est = Any6D(symmetry_tfs=None, mesh=mesh, debug_dir=save_path, debug=0)

    # Run pose estimation (same method as TYO-L)
    pred_pose = est.register_any6d(
        K=K, rgb=color, depth=depth, ob_mask=mask, iteration=5, name="demo"
    )

    # Load ground truth mesh for comparison
    # In TLESS, the model files are named obj_000001.ply to obj_000030.ply
    # The object_id from scene_gt.json corresponds to these model numbers
    models_dir = os.path.join(tless_data_root, "models_cad")
    gt_mesh_path = os.path.join(models_dir, f"obj_{object_id:06d}.ply")
    if not os.path.exists(gt_mesh_path):
        raise FileNotFoundError(f"Ground truth mesh not found: {gt_mesh_path}")
    gt_mesh = trimesh.load(gt_mesh_path)
    # TLESS meshes are in mm, convert to meters like Hope dataset
    gt_mesh = align_mesh_to_coordinate(gt_mesh, scale=0.001)

    # Visualization (exactly same as TYO-L)
    visualize_frame_results(
        color=color,
        gt_mesh=gt_mesh,
        est=est,
        K=K,
        gt_pose=gt_pose,
        pred_pose=pred_pose,
        metric=None,
        obj_f=f"scene{scene_id}_obj{object_id:02d}",
        frame_idx=frame_index,
        save_path=save_path,
        glctx=glctx,
        name="demo_data",
        mesh_index=0,
        init=False,
        save_on_folder=True,
    )

    # Calculate Chamfer distance (same as TYO-L)
    chamfer_dis = calculate_chamfer_distance_gt_mesh(
        gt_pose, gt_mesh, pred_pose, est.mesh
    )
    print(f"Chamfer Distance: {chamfer_dis}")

    # Save results (exactly same format as TYO-L)
    np.savetxt(os.path.join(save_path, f"scene{scene_id}_obj{object_id:02d}_predicted_pose.txt"), pred_pose)
    est.mesh.export(os.path.join(save_path, f"final_mesh_scene{scene_id}_obj{object_id:02d}.obj"))
    np.savetxt(os.path.join(save_path, f"scene{scene_id}_obj{object_id:02d}_cd.txt"), [chamfer_dis])

    # Final output (same as TYO-L)
    print(f"\nResults saved to: {save_path}")
    print(f"Scene: {scene_id}, Object: {object_id}, Mask ID: {mask_id}")
    print(f"Chamfer Distance: {chamfer_dis:.6f}")