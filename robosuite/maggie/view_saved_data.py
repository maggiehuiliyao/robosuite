#!/usr/bin/env python
"""
Script to load and view saved visualization data.

Usage:
    python view_saved_data.py --dir ./debug_output --camera agentview
"""

import argparse
from pathlib import Path
from visualization_utils import load_point_cloud, visualize_point_cloud_interactive
from PIL import Image
import matplotlib.pyplot as plt


def view_rgb(image_path):
    """View RGB image."""
    img = Image.open(image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title(f"RGB Image: {image_path.name}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def view_depth(image_path):
    """View depth image."""
    img = Image.open(image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title(f"Depth Image: {image_path.name}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def view_point_cloud_2d(image_path):
    """View 2D point cloud visualization."""
    img = Image.open(image_path)
    plt.figure(figsize=(12, 10))
    plt.imshow(img)
    plt.title(f"Point Cloud: {image_path.name}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def view_grasp(image_path):
    """View grasp visualization."""
    img = Image.open(image_path)
    plt.figure(figsize=(12, 10))
    plt.imshow(img)
    plt.title(f"Grasp Visualization: {image_path.name}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='View saved visualization data')
    parser.add_argument('--dir', type=str, required=True,
                        help='Directory containing saved visualizations')
    parser.add_argument('--camera', type=str, default='agentview',
                        help='Camera name')
    parser.add_argument('--interactive', action='store_true',
                        help='Show interactive 3D point cloud (requires Open3D)')
    parser.add_argument('--show-all', action='store_true',
                        help='Show all visualizations sequentially')

    args = parser.parse_args()

    data_dir = Path(args.dir)
    if not data_dir.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        return

    # File paths
    rgb_path = data_dir / f"{args.camera}_rgb.png"
    depth_path = data_dir / f"{args.camera}_depth.png"
    pcd_pkl_path = data_dir / f"{args.camera}_pointcloud.pkl"
    pcd_img_path = data_dir / f"{args.camera}_pointcloud.png"
    grasp_path = data_dir / f"{args.camera}_grasp_visualization.png"

    print(f"Looking for visualizations in: {data_dir}")
    print(f"Camera: {args.camera}")
    print()

    # Check what's available
    available = []
    if rgb_path.exists():
        available.append(('RGB', rgb_path, view_rgb))
    if depth_path.exists():
        available.append(('Depth', depth_path, view_depth))
    if pcd_img_path.exists():
        available.append(('Point Cloud (2D)', pcd_img_path, view_point_cloud_2d))
    if grasp_path.exists():
        available.append(('Grasp', grasp_path, view_grasp))
    if pcd_pkl_path.exists() and args.interactive:
        available.append(('Point Cloud (3D)', pcd_pkl_path, None))

    if not available:
        print("No visualization files found!")
        print(f"Expected files:")
        print(f"  {rgb_path}")
        print(f"  {depth_path}")
        print(f"  {pcd_pkl_path}")
        print(f"  {grasp_path}")
        return

    print("Found visualizations:")
    for i, (name, path, _) in enumerate(available):
        print(f"  {i+1}. {name}: {path.name}")
    print()

    if args.show_all:
        # Show all visualizations
        for name, path, view_func in available:
            if name == 'Point Cloud (3D)':
                print(f"Loading interactive 3D point cloud...")
                xyz, rgb = load_point_cloud(path)
                print(f"Loaded {xyz.shape[0]} points")
                visualize_point_cloud_interactive(xyz, rgb)
            else:
                print(f"Showing {name}...")
                view_func(path)
    else:
        # Interactive menu
        while True:
            print("\nSelect visualization to view:")
            for i, (name, path, _) in enumerate(available):
                print(f"  {i+1}. {name}")
            print(f"  {len(available)+1}. Exit")

            try:
                choice = int(input("\nChoice: "))
                if choice == len(available) + 1:
                    break
                if 1 <= choice <= len(available):
                    name, path, view_func = available[choice - 1]
                    if name == 'Point Cloud (3D)':
                        print(f"Loading interactive 3D point cloud...")
                        xyz, rgb = load_point_cloud(path)
                        print(f"Loaded {xyz.shape[0]} points")
                        visualize_point_cloud_interactive(xyz, rgb)
                    else:
                        print(f"Showing {name}...")
                        view_func(path)
                else:
                    print("Invalid choice!")
            except ValueError:
                print("Invalid input!")
            except KeyboardInterrupt:
                break

    print("\nDone!")


if __name__ == '__main__':
    main()
