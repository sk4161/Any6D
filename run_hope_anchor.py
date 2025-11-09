import os
import trimesh
import numpy as np
import cv2
import json

import nvdiffrast.torch as dr
import argparse

from estimater import Any6D
from foundationpose.datareader import HopeReader
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
    parser = argparse.ArgumentParser(description="Run Hope anchor pose estimation")
    parser.add_argument(
        "--hope-data-root",
        type=str,
        required=True,
        help="Path to Hope dataset root directory",
    )
    parser.add_argument(
        "--object-id",
        type=str,
        required=True,
        help="Object ID (e.g., obj_000001)",
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
        "--hope-model-path",
        type=str,
        default="/raid/kanazawa/datasets/hope/models",
        help="Path to Hope models directory",
    )
    parser.add_argument(
        "--img_to_3d", 
        action="store_true", 
        help="Running with InstantMesh+SAM2"
    )
    args = parser.parse_args()

    # Parse arguments
    hope_data_root = args.hope_data_root
    object_id = args.object_id
    frame_index = args.frame_index
    anchor_mesh = args.anchor_mesh
    save_dir = args.save_dir
    hope_model_path = args.hope_model_path
    img_to_3d = args.img_to_3d
    
    # Create save directory
    save_path = save_dir
    os.makedirs(save_path, exist_ok=True)

    # Initialize Hope reader
    scene_dir = os.path.join(hope_data_root, "onboarding_dynamic", object_id)
    if not os.path.exists(scene_dir):
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
    
    # Extract numeric object ID from object_id string (e.g., "obj_000001" -> 1)
    if object_id.startswith('obj_'):
        numeric_object_id = int(object_id.split('_')[1])
    else:
        numeric_object_id = int(object_id)
        
    reader = HopeReader(scene_dir, object_id=numeric_object_id)
    
    # Validate frame index
    if frame_index >= len(reader):
        raise ValueError(f"Frame index {frame_index} out of range. Scene has {len(reader)} frames.")

    print(f"Loading Hope object {object_id}, frame {frame_index}")

    glctx = dr.RasterizeCudaContext()

    # Load input data from Hope dataset
    color = reader.get_color(frame_index)
    # Note: HopeReader.get_color() already returns RGB format (via imageio.imread)
    # No color conversion needed
    
    depth = reader.get_depth(frame_index)
    # Fix Hope dataset depth values: they are in mm but processed as if in meters
    # Convert from mm to meters by dividing by 1000
    if 'hope' in hope_data_root.lower():
        depth = depth / 1000.0  # Convert mm to meters
    
    # Get object mask
    mask_img = reader.get_mask(frame_index, numeric_object_id)
    if mask_img is None:
        raise ValueError(f"No mask found for object {object_id} in frame {frame_index}")
    
    # Convert mask to boolean
    mask = (mask_img > 0).astype(np.bool_)
    
    # Get camera intrinsics
    K = reader.K
    
    # Get ground truth pose (only available for frame 0 in Hope dataset)
    gt_pose = reader.get_gt_pose(frame_index, numeric_object_id)
    if gt_pose is None and frame_index == 0:
        raise ValueError(f"No ground truth pose found for object {object_id} in frame {frame_index}")
    elif gt_pose is None:
        print(f"Warning: No ground truth pose available for frame {frame_index}. Only frame 0 has GT in Hope dataset.")

    # Use provided mesh path
    mesh_path = anchor_mesh

    # Process with or without InstantMesh
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
            f"{save_path}/{object_id}_instantmesh.obj"
        )
        
        # Load the generated mesh
        mesh = trimesh.load(mesh_path)
        print(f"Generated mesh with InstantMesh: {mesh_path}")
    else:
        # Load mesh from file
        if os.path.exists(mesh_path):
            mesh = trimesh.load(mesh_path)
            # Apply same scaling as Hope dataset meshes (mm to meters conversion)
            # Hope dataset meshes are stored in mm and need 1e-3 scaling
            if 'hope' in hope_data_root.lower() or 'hope' in mesh_path.lower():
                mesh.vertices *= 1e-3  # Convert from mm to meters
        else:
            raise FileNotFoundError(f"Mesh file not found at {mesh_path}")

    # Align mesh and save center mesh
    mesh = align_mesh_to_coordinate(mesh, scale=1.0)
    mesh.export(os.path.join(save_path, f"center_mesh_{object_id}.obj"))

    # Initialize estimator
    est = Any6D(symmetry_tfs=None, mesh=mesh, debug_dir=save_path, debug=0)

    # Run pose estimation
    pred_pose = est.register_any6d(
        K=K, rgb=color, depth=depth, ob_mask=mask, iteration=5, name="demo"
    )

    # Load ground truth mesh for comparison (if GT pose exists)
    if gt_pose is not None:
        gt_mesh = reader.get_gt_mesh(numeric_object_id)
    else:
        gt_mesh = None

    # Visualization
    visualize_frame_results(
        color=color,
        gt_mesh=gt_mesh,
        est=est,
        K=K,
        gt_pose=gt_pose,
        pred_pose=pred_pose,
        metric=None,
        obj_f=object_id,
        frame_idx=frame_index,
        save_path=save_path,
        glctx=glctx,
        name="demo_data",
        mesh_index=0,
        init=False,
        save_on_folder=True,
    )

    # Calculate Chamfer distance (only if GT is available)
    if gt_pose is not None and gt_mesh is not None:
        chamfer_dis = calculate_chamfer_distance_gt_mesh(
            gt_pose, gt_mesh, pred_pose, est.mesh
        )
        print(f"Chamfer Distance: {chamfer_dis}")
        np.savetxt(os.path.join(save_path, f"{object_id}_cd.txt"), [chamfer_dis])
    else:
        chamfer_dis = None
        print("Chamfer Distance: N/A (no ground truth available)")

    # Save results
    np.savetxt(os.path.join(save_path, f"{object_id}_predicted_pose.txt"), pred_pose)
    est.mesh.export(os.path.join(save_path, f"final_mesh_{object_id}.obj"))

    # Final output
    print(f"\nResults saved to: {save_path}")
    print(f"Object: {object_id}")
    if chamfer_dis is not None:
        print(f"Chamfer Distance: {chamfer_dis:.6f}")
    else:
        print("Chamfer Distance: N/A (no ground truth available)")