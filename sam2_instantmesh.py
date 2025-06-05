import torch
import numpy as np
import rembg
import os
import torchvision
import trimesh

from sam2.sam2.build_sam import build_sam2
from sam2.sam2.sam2_image_predictor import SAM2ImagePredictor
from instantmesh.src.utils.infer_util import remove_background, resize_foreground
from instantmesh.src.utils.train_util import instantiate_from_config
from instantmesh.src.utils.camera_util import FOV_to_intrinsics, get_zero123plus_input_cameras, get_circular_camera_poses
from PIL import Image
from omegaconf import OmegaConf

from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, StableDiffusionPipeline, StableDiffusionUpscalePipeline
from einops import rearrange

def save_obj(pointnp_px3, facenp_fx3, colornp_px3, fpath):

    pointnp_px3 = pointnp_px3 @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    facenp_fx3 = facenp_fx3[:, [2, 1, 0]]

    mesh = trimesh.Trimesh(
        vertices=pointnp_px3,
        faces=facenp_fx3,
        vertex_colors=colornp_px3,
    )
    mesh.export(fpath, 'obj')

def get_bounding_box(d_model, pad_rel=0.00, return_torch=False):
    """
    Get the bounding box of the non-zero elements in the input array with optional padding.

    Parameters:
    d_model (np.ndarray): 2D input array.
    pad_rel (float): Relative padding (default: 5%)

    Returns:
    torch.Tensor: Padded bounding box in xyxy format on CUDA.
    """
    # Find the indices of non-zero elements
    non_zero_indices = np.nonzero(d_model)

    if len(non_zero_indices[0]) == 0 or len(non_zero_indices[1]) == 0:
        x_min = y_min = x_max = y_max = 0
    else:
        # Get the bounding box coordinates
        y_min, x_min = np.min(non_zero_indices, axis=1)
        y_max, x_max = np.max(non_zero_indices, axis=1)

    # Calculate padding
    x_pad = pad_rel * (x_max - x_min)
    y_pad = pad_rel * (y_max - y_min)

    # Get image dimensions
    height, width = d_model.shape

    # Create padded bounding box
    x1 = max(0, x_min - x_pad)
    y1 = max(0, y_min - y_pad)
    x2 = min(width - 1, x_max + x_pad)
    y2 = min(height - 1, y_max + y_pad)

    # Create the bounding box in xyxy format
    bounding_box = np.array([x1, y1, x2, y2])
    if return_torch:
        return torch.from_numpy(bounding_box[None]).cuda()
    else:
        return bounding_box


def running_sam_box(color, box=None,checkpoint="./sam2/checkpoints/sam2.1_hiera_large.pt", model_cfg = "./sam2/configs/sam2.1/sam2.1_hiera_l.yaml"):
    sam_predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam_predictor.set_image(color)
        masks, scores, _ = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=False,
            )
        mask = masks[0].astype(np.bool_)

    del sam_predictor
    torch.cuda.empty_cache()  # Clear the GPU memory
    return mask


def preprocess_image(color, mask, debug_dir, name=None, rem_bg=True, target_bright = 999, flip=True):
    """
    Preprocess input image by applying mask, removing background and resizing.

    Args:
        color: Input color image (numpy array)
        mask: Boolean mask array
        debug_dir: Directory to save debug images
        name: Optional name for the saved image

    Returns:
        PIL Image: Preprocessed image
    """
    # Create a copy and apply white background where mask is False
    input_image = color.copy()
    input_image[mask == False] = [255, 255, 255]

    # init_bright = input_image[mask].mean()
    # if init_bright > target_bright:
    #      ratio = target_bright / init_bright
    #      input_image = (input_image * ratio).astype(np.uint8)
    #      input_image[mask == False] = [255, 255, 255]


    # Remove background and resize
    if rem_bg:
        rembg_session = rembg.new_session()
        input_image = remove_background(Image.fromarray(input_image), rembg_session)
    else:
        x1, y1, x2, y2 = get_bounding_box(mask, pad_rel=0.1)
        input_image = Image.fromarray(input_image[int(y1):int(y2), int(x1):int(x2)])
        input_image = input_image.convert("RGBA")
    if flip:
        input_image = input_image.transpose(Image.FLIP_LEFT_RIGHT)
    input_image = resize_foreground(input_image.convert("RGBA"), 0.80)

    # Save debug image if directory is provided
    if debug_dir:
        input_image.save(os.path.join(debug_dir, f'input_{name}.png'))

    return input_image


def diffusion_image_generation(debug_dir, debug_input_dir, name=None,input_image=None,config_path = "./instantmesh/configs/instant-mesh-large.yaml"):
    if input_image == None:
        input_image = Image.open(os.path.join(debug_input_dir, f'input_{name}.png'))

    config = OmegaConf.load(config_path)

    device = torch.device('cuda')
    print('Loading diffusion model ...')
    multiview_diffusion_model = DiffusionPipeline.from_pretrained("sudo-ai/zero123plus-v1.2", custom_pipeline="instantmesh/zero123plus", torch_dtype=torch.float16)
    multiview_diffusion_model.scheduler = EulerAncestralDiscreteScheduler.from_config(multiview_diffusion_model.scheduler.config, timestep_spacing='trailing')

    # load custom white-background UNet
    print('Loading custom white-background unet ...')
    state_dict = torch.load(config.infer_config.unet_path, map_location='cpu', weights_only=True)
    multiview_diffusion_model.unet.load_state_dict(state_dict, strict=True)
    multiview_diffusion_model = multiview_diffusion_model.to(device)

    output_image = multiview_diffusion_model(input_image, num_inference_steps=75, ).images[0]
    output_image.save(os.path.join(debug_dir, f'6_views_{name}.png'))
    print(f"Image saved to {os.path.join(debug_dir, f'6_views_{name}.png')}")

    images = np.asarray(output_image, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()  # (3, 960, 640)
    images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)  # (6, 3, 320, 320)
    del multiview_diffusion_model
    torch.cuda.empty_cache()
    return images


def instant_mesh_process(images, debug_dir, name=None):
    device = torch.device('cuda')

    config_path = "./instantmesh/configs/instant-mesh-large.yaml"
    config = OmegaConf.load(config_path)
    IS_FLEXICUBES = True

    instant_mesh = instantiate_from_config(config.model_config)
    model_ckpt_path = config.infer_config.model_path
    state_dict = torch.load(model_ckpt_path, weights_only=True)['state_dict']
    state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
    instant_mesh.load_state_dict(state_dict, strict=True)

    instant_mesh = instant_mesh.to(device)
    instant_mesh.init_flexicubes_geometry(device, fovy=30.0)
    instant_mesh = instant_mesh.eval()

    images = images.unsqueeze(0).to(device)
    images = torchvision.transforms.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0 * 1.0).to(device)
    chunk_size = 5 if IS_FLEXICUBES else 1
    with torch.no_grad():
        planes = instant_mesh.forward_planes(images, input_cameras)
        mesh_path_idx = os.path.join(debug_dir, f'mesh_{name}.obj')

        mesh_out = instant_mesh.extract_mesh(planes, use_texture_map=False, **config.infer_config, )
        vertices, faces, vertex_colors = mesh_out
        save_obj(vertices, faces, vertex_colors, mesh_path_idx)
        print(f"Mesh saved to {mesh_path_idx}")
        # try:
        #     # get video
        #     video_path_idx = os.path.join(debug_dir, f'rendering_mesh_{name}.mp4')
        #     render_size = config.infer_config.render_resolution
        #     render_cameras = get_render_cameras(batch_size=1, M=120, radius=4.5, elevation=40.0, is_flexicubes=IS_FLEXICUBES, ).to(device)
        #
        #     frames = render_frames(instant_mesh, planes, render_cameras=render_cameras, render_size=render_size, chunk_size=chunk_size, is_flexicubes=IS_FLEXICUBES, )
        #
        #     save_video(frames, video_path_idx, fps=30, )
        #     print(f"Video saved to {video_path_idx}")
        #
        #     per_frame_path = os.path.join(debug_dir, f'rendering_mesh_per_frame_{name}')
        #     save_frames_per_image(frames, per_frame_path)
        #     print(f"Per Frame Video saved to {per_frame_path}")
        # except:
        #     pass
    del instant_mesh
    torch.cuda.empty_cache()  # Clear the GPU memory