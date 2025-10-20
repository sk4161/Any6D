import os
import trimesh
import numpy as np
import cv2
import yaml

import nvdiffrast.torch as dr
import argparse

from estimater import Any6D
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
    parser = argparse.ArgumentParser(description="Set experiment name and paths")
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
        help="Path to the anchor calibration yml file",
    )
    parser.add_argument(
        "--anchor-object-pose",
        type=str,
        required=True,
        help="Path to the anchor object pose txt file",
    )
    parser.add_argument(
        "--anchor-mesh",
        type=str,
        required=True,
        help="Path to the anchor mesh file",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Directory to save output files",
    )
    parser.add_argument(
        "--ycb_model_path",
        type=str,
        default="/home/miruware/ssd_4tb/dataset/ho3d/YCB_Video_Models",
        help="Path to the YCB Video Models",
    )
    parser.add_argument(
        "--img_to_3d", action="store_true", help="Running with InstantMesh+SAM2"
    )
    args = parser.parse_args()

    # Parse arguments
    anchor_image = args.anchor_image
    anchor_mask = args.anchor_mask
    anchor_depth = args.anchor_depth
    anchor_calibration = args.anchor_calibration
    anchor_object_pose = args.anchor_object_pose
    anchor_mesh = args.anchor_mesh
    save_dir = args.save_dir
    ycb_model_path = args.ycb_model_path
    img_to_3d = args.img_to_3d

    # Parse object name from the pose file path
    # The pose file is in format: .../pose_000000.txt
    # We need to extract the object name from the parent directory structure
    pose_dir = os.path.dirname(anchor_object_pose)

    # Try to find object name from the path
    # Look for YCB object names in the path
    obj_mapping = {
        "006_mustard_bottle": 5,
        "021_bleach_cleanser": 12,
        "019_pitcher_base": 11,
        "004_sugar_box": 3,
        "005_tomato_soup_can": 4,
        "010_potted_meat_can": 9,
    }

    obj = None
    for obj_name in obj_mapping.keys():
        if obj_name in pose_dir:
            obj = obj_name
            break

    if obj is None:
        # Default to mustard bottle if not found
        print(
            "Warning: Could not determine object from path, defaulting to 019_pitcher_base"
        )
        obj = "019_pitcher_base"

    obj_num = obj_mapping[obj]

    # Create save directory
    save_path = save_dir
    os.makedirs(save_path, exist_ok=True)

    glctx = dr.RasterizeCudaContext()

    # Load input data
    color = cv2.cvtColor(cv2.imread(anchor_image), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(anchor_depth, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0

    # Load mask from image file
    mask_img = cv2.imread(anchor_mask, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise FileNotFoundError(f"Mask file not found at {anchor_mask}")

    # Convert to boolean mask (assuming non-zero values are mask)
    mask = (mask_img > 0).astype(np.bool_)

    # Use provided mesh path
    mesh_path = anchor_mesh

    # Process with or without InstantMesh

    if img_to_3d:
        cmin, rmin, cmax, rmax = get_bounding_box(mask).astype(np.int32)
        input_box = np.array([cmin, rmin, cmax, rmax])[None, :]
        mask_refine = running_sam_box(color, input_box)

        input_image = preprocess_image(color, mask_refine, save_path, obj)
        images = diffusion_image_generation(
            save_path, save_path, obj, input_image=input_image
        )
        instant_mesh_process(images, save_path, obj)

        mesh = trimesh.load(os.path.join(save_path, f"mesh_{obj}.obj"))
    else:
        # Load the provided mesh file
        if os.path.exists(mesh_path):
            mesh = trimesh.load(mesh_path)
        else:
            raise FileNotFoundError(f"Mesh file not found at {mesh_path}")

    mesh = align_mesh_to_coordinate(mesh)
    mesh.export(os.path.join(save_path, f"center_mesh_{obj}.obj"))

    est = Any6D(symmetry_tfs=None, mesh=mesh, debug_dir=save_path, debug=0)

    # Load intrinsic calibration from yml file
    with open(anchor_calibration, "r") as file:
        calib_data = yaml.load(file, Loader=yaml.FullLoader)

    # Extract intrinsic matrix
    intrinsic = np.array(
        [
            [calib_data["color"]["fx"], 0.0, calib_data["color"]["ppx"]],
            [0.0, calib_data["color"]["fy"], calib_data["color"]["ppy"]],
            [0.0, 0.0, 1.0],
        ]
    )

    pred_pose = est.register_any6d(
        K=intrinsic, rgb=color, depth=depth, ob_mask=mask, iteration=5, name="demo"
    )

    # Load ground truth pose
    gt_pose = np.loadtxt(anchor_object_pose)

    # Load ground truth mesh
    gt_mesh = trimesh.load(f"{ycb_model_path}/models/{obj}/textured_simple.obj")

    visualize_frame_results(
        color=color,
        gt_mesh=gt_mesh,
        est=est,
        K=intrinsic,
        gt_pose=gt_pose,
        pred_pose=pred_pose,
        metric=None,
        obj_f=obj,
        frame_idx=0,
        save_path=save_path,
        glctx=glctx,
        name="demo_data",
        mesh_index=0,
        init=False,
        save_on_folder=True,
    )

    chamfer_dis = calculate_chamfer_distance_gt_mesh(
        gt_pose, gt_mesh, pred_pose, est.mesh
    )
    print(f"Chamfer Distance: {chamfer_dis}")

    # Save results
    np.savetxt(os.path.join(save_path, f"{obj}_predicted_pose.txt"), pred_pose)
    est.mesh.export(os.path.join(save_path, f"final_mesh_{obj}.obj"))
    np.savetxt(os.path.join(save_path, f"{obj}_cd.txt"), [chamfer_dis])

    print(f"\nResults saved to: {save_path}")
    print(f"Object: {obj}")
    print(f"Chamfer Distance: {chamfer_dis:.6f}")
