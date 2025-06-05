import os
import trimesh
import numpy as np
import cv2

import nvdiffrast.torch as dr
import argparse
import pandas as pd

from estimater import Any6D
from foundationpose.Utils import visualize_frame_results, calculate_chamfer_distance_gt_mesh, align_mesh_to_coordinate
from tqdm import tqdm
from sam2_instantmesh import *



if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Set experiment name and paths")
    parser.add_argument("--anchor_folder", type=str, default="/home/miruware/ssd_4tb/cvpr_2025_results/anchor_results/dexycb_reference_view_ours", help="Path to the YCB-V model info JSON")
    parser.add_argument("--ycb_model_path", type=str, default="/home/miruware/ssd_4tb/dataset/ho3d/YCB_Video_Models", help="Path to the YCB Video Models")
    parser.add_argument("--img_to_3d", action="store_true",help="Running with InstantMesh+SAM2")
    args = parser.parse_args()


    anchor_folder = args.anchor_folder
    ycb_model_path = args.ycb_model_path
    img_to_3d = args.img_to_3d


    results = []

    obj_list = [f for f in os.listdir(anchor_folder) if not f.endswith('.xlsx')]

    glctx = dr.RasterizeCudaContext()

    for obj in tqdm(obj_list, desc='Object'):

        if obj == '006_mustard_bottle':
            obj_num = 5
        elif obj == '021_bleach_cleanser':
            obj_num = 12
        elif obj == '019_pitcher_base':
            obj_num = 11
        elif obj == '004_sugar_box':
            obj_num = 3
        elif obj == '005_tomato_soup_can':
            obj_num = 4
        elif obj == '010_potted_meat_can':
            obj_num = 9



        save_path = f'{anchor_folder}/{obj}'
        mesh_path = os.path.join(f'{anchor_folder}/{obj}/mesh_{obj}.obj')


        color = cv2.cvtColor(cv2.imread(os.path.join(save_path, 'color.png')), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(os.path.join(save_path, 'depth.png'), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
        mask = cv2.cvtColor(cv2.imread(os.path.join(save_path, 'mask.png')),cv2.COLOR_BGR2RGB)[...,0].astype(np.bool_)

        if img_to_3d:
            cmin, rmin, cmax, rmax = get_bounding_box(mask).astype(np.int32)
            input_box = np.array([cmin, rmin, cmax, rmax])[None, :]
            mask_refine = running_sam_box(color, input_box)

            input_image = preprocess_image(color, mask_refine, save_path, obj)
            images = diffusion_image_generation(save_path, save_path, obj, input_image=input_image)
            instant_mesh_process(images, save_path, obj)

            mesh = trimesh.load(os.path.join(save_path, f'mesh_{obj}.obj'))
        else:
            mesh = trimesh.load(mesh_path)


        mesh = align_mesh_to_coordinate(mesh)
        mesh.export(os.path.join(save_path, f'center_mesh_{obj}.obj'))

        est = Any6D(symmetry_tfs=None, mesh=mesh, debug_dir=save_path, debug=0)

        intrinsic = np.loadtxt(f'{anchor_folder}/{obj}/K.txt')


        pred_pose = est.register_any6d(K=intrinsic, rgb=color, depth=depth, ob_mask=mask, iteration=5, name=f'demo')


        gt_pose = np.loadtxt(os.path.join(save_path, f'{obj}_gt_pose.txt'))


        gt_mesh = trimesh.load(f'{ycb_model_path}/models/{obj}/textured_simple.obj')

        visualize_frame_results(color=color, gt_mesh=gt_mesh, est=est, K=intrinsic, gt_pose=gt_pose, pred_pose=pred_pose,
                                metric=None, obj_f=obj, frame_idx=0, save_path=save_path, glctx=glctx,
                                name=f'demo_data', mesh_index=0, init=False, save_on_folder=True)

        chamfer_dis = calculate_chamfer_distance_gt_mesh(gt_pose, gt_mesh, pred_pose, est.mesh)
        print(chamfer_dis)

        np.savetxt(os.path.join(save_path, f'{obj}_initial_pose.txt'), pred_pose)
        est.mesh.export(os.path.join(save_path, f'final_mesh_{obj}.obj'))

        np.savetxt(os.path.join(save_path, f'{obj}_cd.txt'), [chamfer_dis])

        results.append({
            'Object': obj,
            'Object_Number': obj_num,
            'Chamfer_Distance': float(chamfer_dis)
            })

    df = pd.DataFrame(results)

    df = df.sort_values('Object')

    excel_path = os.path.join(anchor_folder, 'chamfer_distances.xlsx')
    df.to_excel(excel_path, index=False)

    print("\nChamfer Distance Summary Statistics:")
    print(df['Chamfer_Distance'].describe())
    print(f"\nResults saved to: {excel_path}")



