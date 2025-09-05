import argparse
import os
from typing import Tuple

import cv2
import numpy as np
import trimesh


def load_intrinsics(path: str) -> np.ndarray:
    K = np.loadtxt(path).astype(np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {K.shape} from {path}")
    return K


def load_pose(path: str) -> np.ndarray:
    pose = np.loadtxt(path).astype(np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"Pose must be 4x4, got {pose.shape} from {path}")
    return pose


def project_points_cam_to_image(x_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project 3D camera-frame points to image pixels using K.

    Args:
        x_cam: (N, 3) points in camera coordinates.
        K: (3, 3) intrinsics matrix.

    Returns:
        (N, 2) pixel coordinates (u, v) as float64.
    """
    z = x_cam[:, 2]
    eps = 1e-9
    z = np.where(z == 0.0, eps, z)
    x = x_cam[:, 0] / z
    y = x_cam[:, 1] / z
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    return np.stack([u, v], axis=1)


def transform_obj_to_cam(points_obj: np.ndarray, cam_T_obj: np.ndarray) -> np.ndarray:
    """Apply 4x4 cam_T_obj to object-frame points.

    Args:
        points_obj: (N, 3)
        cam_T_obj: (4, 4)

    Returns:
        (N, 3) points in camera frame.
    """
    ones = np.ones((points_obj.shape[0], 1), dtype=points_obj.dtype)
    homo = np.hstack([points_obj, ones])  # (N, 4)
    x_cam_h = (cam_T_obj @ homo.T).T  # (N, 4)
    return x_cam_h[:, :3]


def draw_points(
    image: np.ndarray,
    pixels: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 1,
    stride: int = 10,
) -> np.ndarray:
    h, w = image.shape[:2]
    out = image.copy()
    pts = pixels[:: max(1, stride)]
    for u, v in pts:
        ui = int(round(u))
        vi = int(round(v))
        if 0 <= ui < w and 0 <= vi < h:
            cv2.circle(out, (ui, vi), radius, color, -1, lineType=cv2.LINE_AA)
    return out


def draw_wireframe(
    image: np.ndarray,
    pixels: np.ndarray,
    faces: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
    stride_faces: int = 20,
) -> np.ndarray:
    h, w = image.shape[:2]
    out = image.copy()
    for f in faces[:: max(1, stride_faces)]:
        i0, i1, i2 = f
        tri = [pixels[i0], pixels[i1], pixels[i2]]
        pts = []
        for u, v in tri:
            ui = int(round(u))
            vi = int(round(v))
            pts.append((ui, vi))
        # Draw edges if any endpoint is within image bounds (simple visibility)
        def in_bounds(p):
            return 0 <= p[0] < w and 0 <= p[1] < h

        if in_bounds(pts[0]) or in_bounds(pts[1]):
            cv2.line(out, pts[0], pts[1], color, thickness, lineType=cv2.LINE_AA)
        if in_bounds(pts[1]) or in_bounds(pts[2]):
            cv2.line(out, pts[1], pts[2], color, thickness, lineType=cv2.LINE_AA)
        if in_bounds(pts[2]) or in_bounds(pts[0]):
            cv2.line(out, pts[2], pts[0], color, thickness, lineType=cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser(description="Reproject mesh overlay for Any6D pose")
    parser.add_argument("--image", required=True, help="Path to RGB image (BGR or RGB ok)")
    parser.add_argument("--K", required=True, help="Path to 3x3 intrinsics text file")
    parser.add_argument("--pose", required=True, help="Path to 4x4 cam_T_obj text file")
    parser.add_argument("--mesh", required=True, help="Path to mesh (.obj/.ply) in object frame used by pose")
    parser.add_argument("--out", required=True, help="Path to save overlay image")
    parser.add_argument("--mode", choices=["points", "wireframe", "both"], default="both")
    parser.add_argument("--vertex_stride", type=int, default=10, help="Subsample factor for points")
    parser.add_argument("--face_stride", type=int, default=20, help="Subsample factor for faces")
    parser.add_argument("--color", type=str, default="0,255,0", help="Overlay color as R,G,B (0-255)")
    args = parser.parse_args()

    # Load inputs
    img_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    color_vals = tuple(int(x) for x in args.color.split(","))
    if len(color_vals) != 3:
        raise ValueError("--color must be R,G,B")

    K = load_intrinsics(args.K)
    pose = load_pose(args.pose)

    mesh = trimesh.load(args.mesh, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        # In case it's a scene, try to merge
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(g for g in mesh.dump().geometry.values()))
        else:
            raise ValueError(f"Unsupported mesh type: {type(mesh)}")

    verts_obj = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32) if mesh.faces is not None else np.zeros((0, 3), dtype=np.int32)

    # Transform and project
    verts_cam = transform_obj_to_cam(verts_obj, pose)
    # Keep only points in front of camera
    front_mask = verts_cam[:, 2] > 1e-6
    verts_cam_front = verts_cam[front_mask]
    pixels = project_points_cam_to_image(verts_cam_front, K)

    # Map faces to surviving vertex indices if drawing wireframe
    overlay = img_bgr
    if args.mode in ("points", "both"):
        overlay = draw_points(overlay, pixels, color=color_vals[::-1], radius=1, stride=args.vertex_stride)

    if args.mode in ("wireframe", "both") and len(faces) > 0:
        # Build index mapping from original vertex indices to compacted indices
        orig_to_compact = -np.ones(verts_cam.shape[0], dtype=np.int64)
        orig_indices = np.flatnonzero(front_mask)
        orig_to_compact[orig_indices] = np.arange(pixels.shape[0])
        faces_compact = orig_to_compact[faces]
        # Drop faces with any vertex behind camera
        valid_faces = np.all(faces_compact >= 0, axis=1)
        faces_compact = faces_compact[valid_faces]
        overlay = draw_wireframe(overlay, pixels, faces_compact, color=color_vals[::-1], thickness=1, stride_faces=args.face_stride)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, overlay)
    print(f"Saved overlay: {args.out}")


if __name__ == "__main__":
    main()

