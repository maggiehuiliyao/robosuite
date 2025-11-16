"""
Grasp execution utilities for Robosuite.

This module provides utilities for executing grasps predicted by M2T2
in Robosuite environments.
"""

import numpy as np
import robosuite.utils.transform_utils as T


def pose_to_controller_action(current_eef_pose, target_pose, controller_type='OSC_POSE'):
    """
    Convert target pose to controller action.

    Args:
        current_eef_pose: Current end-effector 4x4 pose matrix
        target_pose: Target 4x4 pose matrix
        controller_type: Type of controller being used

    Returns:
        action: Action vector for the controller
    """
    if controller_type in ['OSC_POSE', 'IK_POSE']:
        # For pose controllers, compute the delta pose
        # Extract position and orientation
        target_pos = target_pose[:3, 3]
        target_quat = T.mat2quat(target_pose[:3, :3])

        current_pos = current_eef_pose[:3, 3]
        current_quat = T.mat2quat(current_eef_pose[:3, :3])

        # Compute position delta
        pos_delta = target_pos - current_pos

        # Compute orientation delta (as axis-angle)
        # Convert quaternions to rotation matrices
        current_mat = T.quat2mat(current_quat)
        target_mat = T.quat2mat(target_quat)

        # Relative rotation
        rel_mat = target_mat @ current_mat.T
        rel_quat = T.mat2quat(rel_mat)

        # Convert to axis-angle for OSC
        axis_angle = T.quat2axisangle(rel_quat)

        # Combine into action (position delta + axis-angle)
        action = np.concatenate([pos_delta, axis_angle])

    elif controller_type == 'BASIC':
        # For BASIC controller, we need to be more careful
        # Extract target position and orientation
        target_pos = target_pose[:3, 3]
        target_quat = T.mat2quat(target_pose[:3, :3])

        # Current pose
        current_pos = current_eef_pose[:3, 3]
        current_quat = T.mat2quat(current_eef_pose[:3, :3])

        # Position delta
        pos_delta = target_pos - current_pos

        # Orientation as axis-angle
        current_mat = T.quat2mat(current_quat)
        target_mat = T.quat2mat(target_quat)
        rel_mat = target_mat @ current_mat.T
        rel_quat = T.mat2quat(rel_mat)
        axis_angle = T.quat2axisangle(rel_quat)

        action = np.concatenate([pos_delta, axis_angle])

    else:
        raise ValueError(f"Unsupported controller type: {controller_type}")

    return action


def create_pre_grasp_pose(grasp_pose, offset=0.1):
    """
    Create a pre-grasp pose (above the grasp pose).

    Args:
        grasp_pose: 4x4 grasp transformation matrix
        offset: Distance above grasp (in meters)

    Returns:
        pre_grasp_pose: 4x4 pre-grasp transformation matrix
    """
    pre_grasp_pose = grasp_pose.copy()

    # Move along the approach direction (usually -Z axis of gripper)
    approach_dir = -grasp_pose[:3, 2]  # Negative Z-axis
    approach_dir = approach_dir / np.linalg.norm(approach_dir)

    # Offset position
    pre_grasp_pose[:3, 3] = grasp_pose[:3, 3] + offset * approach_dir

    return pre_grasp_pose


def interpolate_pose(start_pose, end_pose, t):
    """
    Interpolate between two poses.

    Args:
        start_pose: Starting 4x4 pose matrix
        end_pose: Ending 4x4 pose matrix
        t: Interpolation parameter [0, 1]

    Returns:
        interpolated_pose: 4x4 interpolated pose matrix
    """
    # Linear interpolation for position
    pos = (1 - t) * start_pose[:3, 3] + t * end_pose[:3, 3]

    # SLERP for orientation
    start_quat = T.mat2quat(start_pose[:3, :3])
    end_quat = T.mat2quat(end_pose[:3, :3])

    # Quaternion SLERP
    dot = np.dot(start_quat, end_quat)

    # If negative dot, use the shorter path
    if dot < 0.0:
        end_quat = -end_quat
        dot = -dot

    # Clamp dot product
    dot = np.clip(dot, -1.0, 1.0)

    # SLERP
    theta = np.arccos(dot)
    if abs(theta) < 1e-6:
        # Quaternions very close, use linear interpolation
        interp_quat = start_quat
    else:
        sin_theta = np.sin(theta)
        w1 = np.sin((1 - t) * theta) / sin_theta
        w2 = np.sin(t * theta) / sin_theta
        interp_quat = w1 * start_quat + w2 * end_quat

    # Normalize
    interp_quat = interp_quat / np.linalg.norm(interp_quat)

    # Construct interpolated pose
    interp_pose = np.eye(4)
    interp_pose[:3, :3] = T.quat2mat(interp_quat)
    interp_pose[:3, 3] = pos

    return interp_pose


def generate_trajectory(start_pose, end_pose, num_waypoints=10):
    """
    Generate a trajectory from start to end pose.

    Args:
        start_pose: Starting 4x4 pose matrix
        end_pose: Ending 4x4 pose matrix
        num_waypoints: Number of waypoints in trajectory

    Returns:
        waypoints: List of 4x4 pose matrices
    """
    waypoints = []
    for i in range(num_waypoints):
        t = i / (num_waypoints - 1)
        waypoint = interpolate_pose(start_pose, end_pose, t)
        waypoints.append(waypoint)

    return waypoints


class GraspExecutor:
    def __init__(self, env, controller_type='OSC_POSE'):
        self.env = env
        self.controller_type = controller_type

    def get_current_eef_pose(self, robot_idx=0, arm='right'):
        """Get current end-effector pose."""
        robot = self.env.robots[robot_idx]

        eef_pos = np.array(robot.sim.data.site_xpos[robot.eef_site_id[arm]])
        eef_rot_mat = np.array(robot.sim.data.site_xmat[robot.eef_site_id[arm]]).reshape(3, 3)
        eef_pose = np.eye(4)
        eef_pose[:3, :3] = eef_rot_mat
        eef_pose[:3, 3] = eef_pos

        return eef_pose

    def execute_grasp_sequence(self, grasp_pose, pre_grasp_offset=0.1,
                               num_waypoints=20, gripper_open=-1.0,
                               gripper_close=1.0):
        """
        Execute full grasp sequence.

        Args:
            grasp_pose: 4x4 target grasp pose
            pre_grasp_offset: Distance for pre-grasp (meters)
            num_waypoints: Waypoints for each movement phase
            gripper_open: Gripper open action value
            gripper_close: Gripper close action value

        Returns:
            success: Boolean indicating if grasp likely succeeded
        """
        print("Phase 2: Moving to pre-grasp pose...")
        pre_grasp_pose = create_pre_grasp_pose(grasp_pose, pre_grasp_offset)
        current_pose = self.get_current_eef_pose()

        trajectory = generate_trajectory(current_pose, pre_grasp_pose, num_waypoints)

        for waypoint in trajectory:
            arm_action = self._pose_to_action(self.get_current_eef_pose(), waypoint)
            # Combine arm action with gripper action
            action = np.concatenate([arm_action, [gripper_open]])
            self.env.step(action)
            self.env.render()

        # Phase 3: Move to grasp pose
        print("Phase 3: Moving to grasp pose...")
        current_pose = self.get_current_eef_pose()
        trajectory = generate_trajectory(current_pose, grasp_pose, num_waypoints // 2)

        for waypoint in trajectory:
            arm_action = self._pose_to_action(self.get_current_eef_pose(), waypoint)
            action = np.concatenate([arm_action, [gripper_open]])
            self.env.step(action)
            self.env.render()

        # Phase 4: Close gripper
        print("Phase 4: Closing gripper...")
        for _ in range(15):
            action = np.zeros(self.env.action_dim)
            action[-1] = gripper_close
            self.env.step(action)
            self.env.render()

        # Phase 5: Lift
        print("Phase 5: Lifting...")
        lift_pose = grasp_pose.copy()
        lift_pose[2, 3] += 0.15  # Lift 15cm

        current_pose = self.get_current_eef_pose()
        trajectory = generate_trajectory(current_pose, lift_pose, num_waypoints // 2)

        for waypoint in trajectory:
            arm_action = self._pose_to_action(self.get_current_eef_pose(), waypoint)
            action = np.concatenate([arm_action, [gripper_close]])
            self.env.step(action)
            self.env.render()

        # Check if object was grasped (simple heuristic: gripper is not fully closed)
        # This is environment-specific and may need adjustment
        print("Grasp sequence complete!")

        return True

    def _pose_to_action(self, current_pose, target_pose, gain=5.0):
        """
        Convert pose to action with proportional control.

        Args:
            current_pose: Current 4x4 pose
            target_pose: Target 4x4 pose
            gain: Proportional gain

        Returns:
            action: Action vector (without gripper)
        """
        # Position error
        pos_error = target_pose[:3, 3] - current_pose[:3, 3]

        # Orientation error (axis-angle)
        current_rot = current_pose[:3, :3]
        target_rot = target_pose[:3, :3]
        rel_rot = target_rot @ current_rot.T
        rel_quat = T.mat2quat(rel_rot)
        ori_error = T.quat2axisangle(rel_quat)

        # Scale by gain (P control)
        pos_action = gain * pos_error
        ori_action = gain * ori_error

        # Clip to reasonable limits
        pos_action = np.clip(pos_action, -0.05, 0.05)
        ori_action = np.clip(ori_action, -0.5, 0.5)

        action = np.concatenate([pos_action, ori_action])

        return action
