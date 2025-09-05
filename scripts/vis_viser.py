import argparse
import os
import sys
from typing import Optional

import numpy as np
import cv2
import trimesh


def ensure_viser() -> None:
    try:
        import viser  # noqa: F401
    except Exception:
        print(
            "[ERROR] viser が見つかりませんでした。\n"
            "pip install viser でインストールしてから再実行してください。",
            file=sys.stderr,
        )
        raise


def load_K(path: str) -> np.ndarray:
    K = np.loadtxt(path).astype(np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"K は 3x3 である必要があります: shape={K.shape}")
    return K


def load_pose(path: str) -> np.ndarray:
    pose = np.loadtxt(path).astype(np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"pose は 4x4 である必要があります: shape={pose.shape}")
    return pose


def depth_to_points(depth: np.ndarray, K: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    H, W = depth.shape[:2]
    vs, us = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    if mask is None:
        valid = depth > 0.0
    else:
        valid = (depth > 0.0) & (mask.astype(bool))
    us = us[valid].astype(np.float64)
    vs = vs[valid].astype(np.float64)
    zs = depth[valid].astype(np.float64)
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack([xs, ys, zs], axis=1)
    return pts


def transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    homo = np.hstack([points, np.ones((points.shape[0], 1), dtype=points.dtype)])
    out = (T @ homo.T).T
    return out[:, :3]


def main() -> None:
    parser = argparse.ArgumentParser(description="Viser で 3D 可視化 (シーン点群 + 予測メッシュ)")
    parser.add_argument("--image", required=True, help="RGB 画像 (色用)")
    parser.add_argument("--depth", required=True, help="深度画像 (16bit PNG など)")
    parser.add_argument("--depth_scale", type=float, default=1000.0, help="深度スケール (例: 1000 で m)"
                        )
    parser.add_argument("--K", required=True, help="3x3 内部行列の txt")
    parser.add_argument("--pose", required=True, help="4x4 cam_T_obj の txt")
    parser.add_argument("--mesh", required=True, help="メッシュ (obj/ply)")
    parser.add_argument("--mask", default=None, help="オプション: バイナリマスク画像 (同解像度)")
    parser.add_argument("--max_scene_points", type=int, default=200000, help="シーン点群の最大点数")
    parser.add_argument("--mesh_samples", type=int, default=80000, help="メッシュ表面サンプル点数")
    parser.add_argument("--host", default="0.0.0.0", help="Viser ホスト")
    parser.add_argument("--port", type=int, default=8080, help="Viser ポート")
    args = parser.parse_args()

    ensure_viser()
    import viser

    # 入力読み込み
    img_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"画像が読めません: {args.image}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    depth_raw = cv2.imread(args.depth, cv2.IMREAD_ANYDEPTH)
    if depth_raw is None:
        raise FileNotFoundError(f"深度が読めません: {args.depth}")
    depth = depth_raw.astype(np.float32) / float(args.depth_scale)

    K = load_K(args.K)
    pose = load_pose(args.pose)

    mask = None
    if args.mask is not None and os.path.isfile(args.mask):
        m = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        mask = (m > 0)

    # シーン点群作成（カメラ座標）
    scene_pts = depth_to_points(depth, K, mask)

    # 色 (0..1)
    H, W = depth.shape[:2]
    if mask is None:
        valid = depth > 0.0
    else:
        valid = (depth > 0.0) & mask
    colors = img_rgb.reshape(-1, 3)[valid.reshape(-1)].astype(np.float32) / 255.0

    # 点数制限
    if scene_pts.shape[0] > args.max_scene_points:
        idx = np.random.choice(scene_pts.shape[0], args.max_scene_points, replace=False)
        scene_pts = scene_pts[idx]
        colors = colors[idx]

    # 予測メッシュをサンプリングし、カメラ座標へ
    mesh = trimesh.load(args.mesh, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(g for g in mesh.dump().geometry.values()))
    mesh_pts, _ = trimesh.sample.sample_surface(mesh, args.mesh_samples)
    mesh_pts_cam = transform(mesh_pts.astype(np.float64), pose)
    mesh_cols = np.tile(np.array([[1.0, 0.2, 0.2]], dtype=np.float32), (mesh_pts_cam.shape[0], 1))

    # Viser で表示
    server = viser.ViserServer(host=args.host, port=args.port)
    server.add_point_cloud(
        name="scene",
        points=scene_pts.astype(np.float32),
        colors=colors.astype(np.float32),
        point_size=0.003,
    )
    server.add_point_cloud(
        name="mesh_pred",
        points=mesh_pts_cam.astype(np.float32),
        colors=mesh_cols,
        point_size=0.004,
    )

    print(f"Viser server is running at http://{args.host}:{args.port}")
    print("[Hint] Ctrl+C で停止")

    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Server stopped.")


if __name__ == "__main__":
    main()

