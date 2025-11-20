import numpy as np
import torch
from mayavi import mlab

def draw_gripper_wireframe(grasp_pose, color=(0, 1, 0), tube_radius=0.001, gripper_width=0.08):
    """Draw parallel plate gripper as wireframe"""

    # Gripper dimensions (Panda gripper approximate)
    finger_length = 0.04
    finger_width = 0.01
    palm_depth = 0.03

    # Define gripper geometry in gripper frame
    # Two parallel plates (fingers)
    hw = gripper_width / 2  # half width

    # Left finger
    left_finger = np.array([
        [-hw, -finger_width/2, 0],
        [-hw, -finger_width/2, finger_length],
        [-hw, finger_width/2, finger_length],
        [-hw, finger_width/2, 0],
    ])

    # Right finger
    right_finger = np.array([
        [hw, -finger_width/2, 0],
        [hw, -finger_width/2, finger_length],
        [hw, finger_width/2, finger_length],
        [hw, finger_width/2, 0],
    ])

    # Palm base
    palm = np.array([
        [-hw, -finger_width/2, -palm_depth],
        [hw, -finger_width/2, -palm_depth],
        [hw, finger_width/2, -palm_depth],
        [-hw, finger_width/2, -palm_depth],
    ])

    # Transform to world frame
    R = grasp_pose[:3, :3]
    t = grasp_pose[:3, 3]

    def transform_points(pts):
        return (R @ pts.T).T + t

    left_finger_world = transform_points(left_finger)
    right_finger_world = transform_points(right_finger)
    palm_world = transform_points(palm)

    # Draw edges
    def draw_box(pts, color, tube_radius):
        edges = [(0,1), (1,2), (2,3), (3,0)]  # Rectangle edges
        for i, j in edges:
            mlab.plot3d([pts[i,0], pts[j,0]],
                       [pts[i,1], pts[j,1]],
                       [pts[i,2], pts[j,2]],
                       color=color, tube_radius=tube_radius)

    # Draw connecting lines between palm and fingers
    for i in range(4):
        idx = i if i < 2 else 3 - (i - 2)  # Map palm corners to finger bases
        mlab.plot3d([palm_world[i,0], left_finger_world[idx,0]],
                   [palm_world[i,1], left_finger_world[idx,1]],
                   [palm_world[i,2], left_finger_world[idx,2]],
                   color=color, tube_radius=tube_radius)
        mlab.plot3d([palm_world[i,0], right_finger_world[idx,0]],
                   [palm_world[i,1], right_finger_world[idx,1]],
                   [palm_world[i,2], right_finger_world[idx,2]],
                   color=color, tube_radius=tube_radius)

    draw_box(left_finger_world, color, tube_radius)
    draw_box(right_finger_world, color, tube_radius)
    draw_box(palm_world, color, tube_radius)


def main():
    debug_dir = "/home/maggie/research/robosuite/debug"

    # Load point cloud
    pcd = np.load(f"{debug_dir}/xyz.npy", allow_pickle=True)
    rgb = np.load(f"{debug_dir}/rgb.npy", allow_pickle=True)

    # Convert from torch tensor if needed
    if isinstance(pcd, torch.Tensor):
        pcd = pcd.cpu().numpy()
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().numpy()

    # Denormalize RGB (ImageNet normalization)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    rgb_vis = rgb * std + mean
    rgb_vis = np.clip(rgb_vis, 0, 1)

    # Load grasp
    grasp_pose = np.load(f"{debug_dir}/best_grasp.npy")

    print(f"Point cloud shape: {pcd.shape}")
    print(f"RGB shape: {rgb_vis.shape}")
    print(f"Grasp pose:\n{grasp_pose}")

    # Create mayavi figure
    mlab.figure(bgcolor=(0.1, 0.1, 0.1), size=(1024, 768))

    # Draw point cloud colored by height (z-coordinate)
    pts = mlab.points3d(pcd[:, 0], pcd[:, 1], pcd[:, 2], pcd[:, 2],
                        scale_factor=0.003,
                        scale_mode='none',
                        colormap='viridis',
                        mode='sphere')
    pts.actor.property.opacity = 0.8

    # Draw gripper wireframe
    draw_gripper_wireframe(grasp_pose, color=(0, 1, 0), tube_radius=0.002)

    # Add coordinate frame at grasp
    axis_length = 0.05
    origin = grasp_pose[:3, 3]
    for i, color in enumerate([(1,0,0), (0,1,0), (0,0,1)]):
        axis = grasp_pose[:3, i] * axis_length
        mlab.plot3d([origin[0], origin[0] + axis[0]],
                   [origin[1], origin[1] + axis[1]],
                   [origin[2], origin[2] + axis[2]],
                   color=color, tube_radius=0.002)

    mlab.show()


if __name__ == '__main__':
    main()
