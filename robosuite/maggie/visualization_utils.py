"""
Visualization utilities for saving and loading RGB, depth, and point cloud data.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle


def save_rgb_image(rgb, save_path):
    """
    Save RGB image to file.

    Args:
        rgb: (H, W, 3) RGB image (0-255 or 0-1 range)
        save_path: Path to save image
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize to 0-255 if needed
    if rgb.max() <= 1.0:
        rgb = (rgb * 255).astype(np.uint8)

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.title("RGB Image")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved RGB image to {save_path}")


def save_depth_image(depth, save_path):
    """
    Save depth image with colormap.

    Args:
        depth: (H, W) depth image in meters
        save_path: Path to save image
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 10))
    plt.imshow(depth, cmap='turbo')
    plt.colorbar(label='Depth (m)')
    plt.title("Depth Image")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved depth image to {save_path}")


def save_point_cloud(xyz, rgb, save_path):
    """
    Save point cloud as pickle file and 3D visualization.

    Args:
        xyz: (N, 3) point coordinates
        rgb: (N, 3) RGB colors (0-1 range)
        save_path: Path to save (without extension)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as pickle for later loading
    pkl_path = save_path.with_suffix('.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({'xyz': xyz, 'rgb': rgb}, f)
    print(f"Saved point cloud data to {pkl_path}")

    # Create 3D visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Downsample for visualization if too many points
    if xyz.shape[0] > 10000:
        indices = np.random.choice(xyz.shape[0], 10000, replace=False)
        xyz_vis = xyz[indices]
        rgb_vis = rgb[indices]
    else:
        xyz_vis = xyz
        rgb_vis = rgb

    # Plot point cloud
    ax.scatter(xyz_vis[:, 0], xyz_vis[:, 1], xyz_vis[:, 2],
               c=rgb_vis, s=1, alpha=0.6)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Point Cloud ({xyz.shape[0]} points)')

    # Equal aspect ratio
    max_range = np.array([
        xyz_vis[:, 0].max() - xyz_vis[:, 0].min(),
        xyz_vis[:, 1].max() - xyz_vis[:, 1].min(),
        xyz_vis[:, 2].max() - xyz_vis[:, 2].min()
    ]).max() / 2.0

    mid_x = (xyz_vis[:, 0].max() + xyz_vis[:, 0].min()) * 0.5
    mid_y = (xyz_vis[:, 1].max() + xyz_vis[:, 1].min()) * 0.5
    mid_z = (xyz_vis[:, 2].max() + xyz_vis[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    img_path = save_path.with_suffix('.png')
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved point cloud visualization to {img_path}")


def save_all_visualizations(obs, xyz, rgb, camera_name, output_dir):
    """
    Save all visualizations (RGB, depth, point cloud) for a given observation.

    Args:
        obs: Observation dictionary from Robosuite (raw, unflipped)
        xyz: (N, 3) point cloud coordinates (generated from flipped images)
        rgb: (N, 3) point cloud RGB colors (0-1 range)
        camera_name: Name of camera
        output_dir: Directory to save visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save RGB (flip to match point cloud orientation)
    rgb_key = f"{camera_name}_image"
    if rgb_key in obs:
        rgb_img = np.flipud(obs[rgb_key])  # Flip to match point cloud
        save_rgb_image(rgb_img, output_dir / f"{camera_name}_rgb.png")

    # Save depth (flip to match point cloud orientation)
    depth_key = f"{camera_name}_depth"
    if depth_key in obs:
        depth = obs[depth_key]
        if len(depth.shape) == 3:
            depth = depth.squeeze(-1)
        depth = np.flipud(depth)  # Flip to match point cloud
        save_depth_image(depth, output_dir / f"{camera_name}_depth.png")

    # Save point cloud
    save_point_cloud(xyz, rgb, output_dir / f"{camera_name}_pointcloud")

    print(f"Saved all visualizations to {output_dir}")


def load_point_cloud(pkl_path):
    """
    Load point cloud from pickle file.

    Args:
        pkl_path: Path to pickle file

    Returns:
        xyz: (N, 3) point coordinates
        rgb: (N, 3) RGB colors
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    return data['xyz'], data['rgb']


def visualize_point_cloud_interactive(xyz, rgb):
    """
    Create interactive 3D visualization of point cloud.

    Args:
        xyz: (N, 3) point coordinates
        rgb: (N, 3) RGB colors (0-1 range)
    """
    try:
        import open3d as o3d

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)

        # Visualize
        o3d.visualization.draw_geometries([pcd],
                                          window_name="Point Cloud",
                                          width=1024, height=768)

    except ImportError:
        print("Open3D not installed. Install with: pip install open3d")
        print("Falling back to matplotlib visualization...")

        # Matplotlib fallback
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Downsample for visualization
        if xyz.shape[0] > 10000:
            indices = np.random.choice(xyz.shape[0], 10000, replace=False)
            xyz = xyz[indices]
            rgb = rgb[indices]

        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                   c=rgb, s=1, alpha=0.6)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        plt.show()


def save_grasp_visualization(xyz, rgb, grasp_pose, save_path, grasp_length=0.1):
    """
    Save point cloud with grasp pose visualization.

    Args:
        xyz: (N, 3) point cloud coordinates
        rgb: (N, 3) RGB colors (0-1 range)
        grasp_pose: 4x4 grasp transformation matrix
        save_path: Path to save image
        grasp_length: Length of grasp axes to visualize
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Downsample point cloud for visualization
    if xyz.shape[0] > 10000:
        indices = np.random.choice(xyz.shape[0], 10000, replace=False)
        xyz_vis = xyz[indices]
        rgb_vis = rgb[indices]
    else:
        xyz_vis = xyz
        rgb_vis = rgb

    # Plot point cloud
    ax.scatter(xyz_vis[:, 0], xyz_vis[:, 1], xyz_vis[:, 2],
               c=rgb_vis, s=1, alpha=0.3)

    # Plot grasp pose axes
    origin = grasp_pose[:3, 3]
    x_axis = grasp_pose[:3, 0] * grasp_length
    y_axis = grasp_pose[:3, 1] * grasp_length
    z_axis = grasp_pose[:3, 2] * grasp_length

    # X-axis (red)
    ax.quiver(origin[0], origin[1], origin[2],
              x_axis[0], x_axis[1], x_axis[2],
              color='red', linewidth=3, arrow_length_ratio=0.3, label='X-axis')

    # Y-axis (green)
    ax.quiver(origin[0], origin[1], origin[2],
              y_axis[0], y_axis[1], y_axis[2],
              color='green', linewidth=3, arrow_length_ratio=0.3, label='Y-axis')

    # Z-axis (blue - approach direction)
    ax.quiver(origin[0], origin[1], origin[2],
              z_axis[0], z_axis[1], z_axis[2],
              color='blue', linewidth=3, arrow_length_ratio=0.3, label='Z-axis (approach)')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Point Cloud with Predicted Grasp')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved grasp visualization to {save_path}")


if __name__ == '__main__':
    """Example usage and testing."""
    print("Visualization utilities loaded successfully!")
    print("\nExample usage:")
    print("  from visualization_utils import save_all_visualizations, load_point_cloud")
    print("  save_all_visualizations(obs, xyz, rgb, 'agentview', './debug_output')")
    print("  xyz, rgb = load_point_cloud('./debug_output/agentview_pointcloud.pkl')")
