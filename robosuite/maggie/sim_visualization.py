"""
Grasp visualization utilities for Robosuite simulation viewer.
"""

import numpy as np


def visualize_grasp_in_viewer(env, grasp_pose, axis_length=0.08):
    """
    Visualize the predicted grasp pose in the Robosuite viewer.

    Note: Currently MuJoCo viewer in Robosuite doesn't support persistent markers.
    This function prints the grasp pose for reference.

    Args:
        env: Robosuite environment with viewer
        grasp_pose: 4x4 grasp transformation matrix in world frame
        axis_length: Length of coordinate frame axes in meters
    """
    # Extract grasp position and orientation
    grasp_pos = grasp_pose[:3, 3]

    # Print grasp information instead of trying to visualize
    # (MuJoCo viewer API in Robosuite doesn't support add_marker)
    print(f"✓ Grasp visualization coordinates:")
    print(f"  Position: {grasp_pos}")
    print(f"  (To visualize, check saved images in --save-dir)")


def visualize_cube_position(env, cube_pos):
    """
    Visualize the cube position with a marker.

    Args:
        env: Robosuite environment with viewer
        cube_pos: (3,) array with cube position
    """
    # Print cube position for reference
    print(f"✓ Cube position: {cube_pos}")
