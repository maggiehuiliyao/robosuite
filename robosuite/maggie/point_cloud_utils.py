"""
Point cloud extraction utilities for Robosuite environments.
"""

import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix


def depth_to_point_cloud(depth, camera_intrinsics, camera_extrinsics=None):
    h, w = depth.shape

    fx = camera_intrinsics['fx']
    fy = camera_intrinsics['fy']
    cx = camera_intrinsics['cx']
    cy = camera_intrinsics['cy']

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    valid_mask = (depth > 0.01) & (depth < 10.0)

    # Convert to camera coordinates
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points_cam = np.stack([x, y, z], axis=-1)
    points = points_cam

    return points, valid_mask


def get_table_bounds(env, margin=0.05):
    """Get workspace bounds around table."""
    table_body_id = env.sim.model.body_name2id('table')
    table_pos = env.sim.data.body_xpos[table_body_id]

    # Get table size (assume square table, hardcode for Lift task)
    table_half_size = 2.5

    bounds = np.array([
        [table_pos[0] - table_half_size, table_pos[0] + table_half_size],  # x
        [table_pos[1] - table_half_size, table_pos[1] + table_half_size],  # y
        [table_pos[2], table_pos[2] + 0.3]  # z: table surface to 30cm above
    ])
    return bounds


def extract_point_cloud_from_obs(obs, camera_name='agentview', env=None):
    rgb_key = f"{camera_name}_image"
    depth_key = f"{camera_name}_depth"
    rgb_img = obs[rgb_key]  # (H, W, 3)
    depth_img = obs[depth_key]  # (H, W) or (H, W, 1)
    if len(depth_img.shape) == 3:
        depth_img = depth_img.squeeze(-1)  # Remove channel dimension if present

    h, w = depth_img.shape

    camera_extrinsics = get_camera_extrinsic_matrix(env.sim, camera_name)
    camera_intrinsics_matrix = get_camera_intrinsic_matrix(env.sim, camera_name, h, w)

    # Convert intrinsics matrix to dict format
    camera_intrinsics = {
        'fx': camera_intrinsics_matrix[0, 0],
        'fy': camera_intrinsics_matrix[1, 1],
        'cx': camera_intrinsics_matrix[0, 2],
        'cy': camera_intrinsics_matrix[1, 2]
    }

    extent = env.sim.model.stat.extent
    near = env.sim.model.vis.map.znear * extent
    far = env.sim.model.vis.map.zfar * extent

    depth_img = near / (1.0 - depth_img * (1.0 - near / far))
    points, valid_mask = depth_to_point_cloud(
        depth_img, camera_intrinsics, camera_extrinsics=None
    )

    # Filter valid points
    xyz = points[valid_mask]
    rgb = rgb_img[valid_mask] / 255.0  # Normalize to 0-1

    # Transform to world frame for filtering
    xyz_homog = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
    xyz_world = (camera_extrinsics @ xyz_homog.T).T[:, :3]

    # Filter to table workspace
    bounds = get_table_bounds(env)
    xyz_world, rgb = filter_point_cloud_workspace(xyz_world, rgb, bounds)

    # Transform back to camera frame
    world_to_cam = np.linalg.inv(camera_extrinsics)
    xyz_homog = np.concatenate([xyz_world, np.ones((xyz_world.shape[0], 1))], axis=1)
    xyz = (world_to_cam @ xyz_homog.T).T[:, :3]

    return xyz, rgb


def downsample_point_cloud(xyz, rgb, num_points):
    """
    Downsample point cloud to fixed number of points.

    Args:
        xyz: (N, 3) point coordinates
        rgb: (N, 3) RGB colors
        num_points: Target number of points

    Returns:
        xyz_down: (num_points, 3) downsampled coordinates
        rgb_down: (num_points, 3) downsampled colors
    """
    n = xyz.shape[0]

    if n >= num_points:
        # Random sampling
        indices = np.random.choice(n, num_points, replace=False)
    else:
        # Oversample if not enough points
        indices = np.random.choice(n, num_points, replace=True)

    return xyz[indices], rgb[indices]


def filter_point_cloud_workspace(xyz, rgb, bounds):
    """
    Filter point cloud to only include points within workspace bounds.

    Args:
        xyz: (N, 3) point coordinates
        rgb: (N, 3) RGB colors
        bounds: dict with 'x', 'y', 'z' each containing [min, max]
                or (3, 2) array [[x_min, x_max], [y_min, y_max], [z_min, z_max]]

    Returns:
        xyz_filt: Filtered point coordinates
        rgb_filt: Filtered RGB colors
    """
    if isinstance(bounds, dict):
        x_min, x_max = bounds['x']
        y_min, y_max = bounds['y']
        z_min, z_max = bounds['z']
    else:
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]
        z_min, z_max = bounds[2]

    mask = (
        (xyz[:, 0] >= x_min) & (xyz[:, 0] <= x_max) &
        (xyz[:, 1] >= y_min) & (xyz[:, 1] <= y_max) &
        (xyz[:, 2] >= z_min) & (xyz[:, 2] <= z_max)
    )

    return xyz[mask], rgb[mask]


def normalize_point_cloud_rgb(rgb):
    """
    Normalize RGB values as M2T2 expects.
    M2T2 uses ImageNet normalization.

    Args:
        rgb: (N, 3) RGB values in [0, 1] range

    Returns:
        rgb_norm: (N, 3) normalized RGB values
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return (rgb - mean) / std


def prepare_point_cloud_for_m2t2(xyz, rgb, num_points=16384, center=True):
    """
    Prepare point cloud in M2T2 expected format.

    Args:
        xyz: (N, 3) point coordinates
        rgb: (N, 3) RGB colors in [0, 1] range
        num_points: Number of points to sample
        center: Whether to center the point cloud

    Returns:
        inputs: (num_points, 6) concatenated [xyz, rgb] for M2T2
        points: (num_points, 3) original xyz coordinates
        center_offset: (3,) offset used for centering (or zeros if not centered)
    """
    # Downsample to target number of points
    xyz_sample, rgb_sample = downsample_point_cloud(xyz, rgb, num_points)

    # Center point cloud if requested
    if center:
        center_offset = xyz_sample.mean(axis=0)
        xyz_centered = xyz_sample - center_offset
    else:
        center_offset = np.zeros(3)
        xyz_centered = xyz_sample

    # Normalize RGB
    # rgb_norm = normalize_point_cloud_rgb(rgb_sample)
    rgb_norm = rgb_sample  # M2T2 now expects raw RGB in [0,1]
    # Concatenate into M2T2 input format
    inputs = np.concatenate([xyz_centered, rgb_norm], axis=1)

    return inputs, xyz_sample, center_offset
