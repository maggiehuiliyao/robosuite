"""
DP3 Policy Wrapper for Robosuite Evaluation

Wraps the trained DP3 policy to work with Robosuite's action space.
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import sys
sys.path.insert(0, '/home/maggie/research/object_centric_diffusion')

from utils.pose_utils import get_rel_pose, calculate_goal_pose, relative_to_target_to_world
import robosuite.utils.transform_utils as T

# Import utility functions from maggie.py to avoid duplication
from robosuite.maggie.maggie import (
    get_ee_position,
    get_ee_orientation,
    get_cube_pose_7d,
    get_marker_pose,
    mujoco_quat_to_xyzw
)


def pose_7d_to_matrix(pose_7d):
    """Convert [x, y, z, qx, qy, qz, qw] to 4x4 matrix."""
    pos = pose_7d[:3]
    quat = pose_7d[3:]  # [qx, qy, qz, qw]
    rot_mat = R.from_quat(quat).as_matrix()

    mat = np.eye(4)
    mat[:3, :3] = rot_mat
    mat[:3, 3] = pos
    return mat


def matrix_to_pose_7d(mat):
    """Convert 4x4 matrix to [x, y, z, qx, qy, qz, qw]."""
    pos = mat[:3, 3]
    rot_mat = mat[:3, :3]
    quat = R.from_matrix(rot_mat).as_quat()  # [qx, qy, qz, qw]
    return np.concatenate([pos, quat])


class DP3PolicyWrapper:
    """
    Wrapper for DP3 policy to work with Robosuite environment.

    Converts between:
    - Robosuite observations -> Policy input (object-centric poses)
    - Policy output (relative pose deltas) -> Robosuite actions (delta control)
    """

    def __init__(self, policy, env, horizon=4, n_obs_steps=1):

        self.policy = policy
        self.env = env
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        
        self.history = []  # List of agent_pos observations

        # State tracking
        self.T_obj_to_gripper = None
        self.step_count = 0

    def reset(self, T_obj_to_gripper):
        """
        Reset policy state. Called after grasping.

        Args:
            T_obj_to_gripper: Transformation from object to gripper frame (7D pose)
        """
        self.T_obj_to_gripper = T_obj_to_gripper
        self.history = []
        self.step_count = 0
        print(f"Policy wrapper reset. T_obj_to_gripper: {T_obj_to_gripper}")

    def get_action(self, obs):
        """
        Get action from policy given current observation.

        Args:
            obs: Robosuite observation dict

        Returns:
            action: Robosuite action [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        # Get current poses in world frame
        cube_pose = get_cube_pose_7d(self.env)  # [x, y, z, qx, qy, qz, qw]
        target_pose = get_marker_pose()          # [x, y, z, qx, qy, qz, qw]

        # Compute agent_pos (cube relative to target) - OBJECT-CENTRIC!
        # This is the observation the policy was trained on
        agent_pos = get_rel_pose(target_pose, cube_pose)

        # Update history buffer with CURRENT observation
        self._update_history(agent_pos)

        # Build observation dict for policy
        obs_dict = {
            'agent_pos': torch.from_numpy(
                np.array(self.history)
            ).float().unsqueeze(0).cuda()  # (1, To, 7)
        }

        # Predict action
        with torch.no_grad():
            action_dict = self.policy.predict_action(obs_dict)

        # Extract predicted action
        action_pred = action_dict['action'].cpu().numpy()  # (1, Ta, 8)
        predicted_action = action_pred[0, 0, :]  # First action step (8,)

        # Split into pose delta and progress
        relative_pose_delta = predicted_action[:7]  # [dx, dy, dz, dqx, dqy, dqz, dqw]
        progress = predicted_action[7]

        # SIMPLIFIED: Policy predicts where object should be (relative to target)
        # Step 1: Apply the delta to get next object pose (relative to target)
        next_obj_relative = calculate_goal_pose(agent_pos, relative_pose_delta)

        # Step 2: Convert from target-relative frame to world frame
        next_obj_world = relative_to_target_to_world(next_obj_relative, target_pose)

        # Step 3: Compute object delta in world frame
        # Key insight: Since gripper is rigidly attached to object,
        # object delta in world frame = gripper delta in world frame
        # No need for T_obj_to_gripper transform!
        current_obj_world = cube_pose
        pos_delta = next_obj_world[:3] - current_obj_world[:3]

        # Compute orientation delta (as euler angles for Robosuite)
        current_rot = R.from_quat(current_obj_world[3:])
        next_rot = R.from_quat(next_obj_world[3:])
        delta_rot = next_rot * current_rot.inv()
        ori_delta = delta_rot.as_euler('xyz')

        # Gripper stays closed (1 = closed)
        gripper_action = 1.0

        # Create Robosuite action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        # Use gentle scaling for manipulation (not aggressive like grasping)
        # The policy already learned appropriate delta magnitudes, so we scale minimally
        action = np.concatenate([
            pos_delta * 1,  # Gentle position control for manipulation
            ori_delta * 1,  # Gentle orientation control
            [gripper_action]
        ])

        if self.step_count % 10 == 0:
            print(f"Step {self.step_count}: progress={progress:.3f}, "
                  f"pos_delta_norm={np.linalg.norm(pos_delta):.4f}, "
                  f"cube_to_target={np.linalg.norm(cube_pose[:3] - target_pose[:3]):.4f}")

        self.step_count += 1
        return action

    def _update_history(self, agent_pos):
        """Update history buffer with new observation."""
        if len(self.history) >= self.n_obs_steps:
            self.history.pop(0)
        self.history.append(agent_pos)
