import os
import trimesh
import numpy as np
import cv2
import json

import nvdiffrast.torch as dr
import argparse

from estimater import Any6D
from foundationpose.datareader import TyolReader
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
    parser = argparse.ArgumentParser(description="Run TYO-L anchor pose estimation")
    parser.add_argument(
        "--tyol-data-root",
        type=str,
        required=True,
        help="Path to TYO-L dataset root directory",
    )
    parser.add_argument(
        "--scene-id",
        type=str,
        required=True,
        help="Scene ID (e.g., 000001)",
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
        "--tyol-model-path",
        type=str,
        default="/raid/kanazawa/datasets/TYO-L",
        help="Path to TYO-L models directory",
    )
    parser.add_argument(
        "--img_to_3d", 
        action="store_true", 
        help="Running with InstantMesh+SAM2"
    )
    args = parser.parse_args()

    # Parse arguments
    tyol_data_root = args.tyol_data_root
    scene_id = args.scene_id
    frame_index = args.frame_index
    anchor_mesh = args.anchor_mesh
    save_dir = args.save_dir
    tyol_model_path = args.tyol_model_path
    img_to_3d = args.img_to_3d
    
    # Create save directory
    save_path = save_dir
    os.makedirs(save_path, exist_ok=True)

    # Initialize TYO-L reader
    scene_dir = os.path.join(tyol_data_root, "test", scene_id)
    if not os.path.exists(scene_dir):
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
    
    # For TYO-L, object ID is the same as scene ID
    object_id = int(scene_id)
    reader = TyolReader(scene_dir, object_id=object_id)
    
    # Validate frame index
    if frame_index >= len(reader):
        raise ValueError(f"Frame index {frame_index} out of range. Scene has {len(reader)} frames.")

    print(f"Loading TYO-L scene {scene_id}, frame {frame_index}")

    glctx = dr.RasterizeCudaContext()

    # Load input data from TYO-L dataset (same format as HO3D)
    color = reader.get_color(frame_index)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    depth = reader.get_depth(frame_index)
    
    # Get object mask
    mask_img = reader.get_mask(frame_index, object_id)
    if mask_img is None:
        raise ValueError(f"No mask found for scene {scene_id} in frame {frame_index}")
    
    # Convert mask to boolean (same as HO3D)
    mask = (mask_img > 0).astype(np.bool_)
    
    # Get camera intrinsics
    K = reader.K
    
    # Get ground truth pose
    gt_pose = reader.get_gt_pose(frame_index, object_id)
    if gt_pose is None:
        raise ValueError(f"No ground truth pose found for scene {scene_id} in frame {frame_index}")

    # Use provided mesh path
    mesh_path = anchor_mesh

    # Process with or without InstantMesh (same as HO3D)
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
            f"{save_path}/scene_{scene_id}_instantmesh.obj"
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

    # Align mesh and save center mesh (same as HO3D)
    mesh = align_mesh_to_coordinate(mesh)
    mesh.export(os.path.join(save_path, f"center_mesh_{scene_id}.obj"))

    # Initialize estimator (same as HO3D)
    est = Any6D(symmetry_tfs=None, mesh=mesh, debug_dir=save_path, debug=0)

    # Run pose estimation (same method as HO3D)
    pred_pose = est.register_any6d(
        K=K, rgb=color, depth=depth, ob_mask=mask, iteration=5, name="demo"
    )

    # Load ground truth mesh for comparison
    gt_mesh = reader.get_gt_mesh(tyol_model_path, object_id)

    # Visualization (exactly same as HO3D)
    visualize_frame_results(
        color=color,
        gt_mesh=gt_mesh,
        est=est,
        K=K,
        gt_pose=gt_pose,
        pred_pose=pred_pose,
        metric=None,
        obj_f=scene_id,
        frame_idx=frame_index,
        save_path=save_path,
        glctx=glctx,
        name="demo_data",
        mesh_index=0,
        init=False,
        save_on_folder=True,
    )

    # Calculate Chamfer distance (same as HO3D)
    chamfer_dis = calculate_chamfer_distance_gt_mesh(
        gt_pose, gt_mesh, pred_pose, est.mesh
    )
    print(f"Chamfer Distance: {chamfer_dis}")

    # Save results (exactly same format as HO3D)
    np.savetxt(os.path.join(save_path, f"{scene_id}_predicted_pose.txt"), pred_pose)
    est.mesh.export(os.path.join(save_path, f"final_mesh_{scene_id}.obj"))
    np.savetxt(os.path.join(save_path, f"{scene_id}_cd.txt"), [chamfer_dis])

    # Final output (same as HO3D)
    print(f"\nResults saved to: {save_path}")
    print(f"Scene: {scene_id}")
    print(f"Chamfer Distance: {chamfer_dis:.6f}")