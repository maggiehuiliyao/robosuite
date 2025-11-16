import argparse
import numpy as np
import sys
from pathlib import Path

from robosuite.maggie.grasp_utils import GraspExecutor

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import robosuite as suite
from robosuite.wrappers import VisualizationWrapper 
from robosuite.maggie.point_cloud_utils import (
    extract_point_cloud_from_obs,
    prepare_point_cloud_for_m2t2,
)
from robosuite.maggie.m2t2_wrapper import M2T2GraspPredictor
import robosuite.utils.transform_utils as T


def get_ee_pose(env):
    """Get end-effector pose as 4x4 matrix."""
    name = "gripper0_right_grip_site"
    site_id = env.sim.model.site_name2id(name)
    pos = env.sim.data.site_xpos[site_id].copy()
    mat = env.sim.data.site_xmat[site_id].reshape(3, 3).copy()
    pose = np.eye(4)
    pose[:3, :3] = mat
    pose[:3, 3] = pos
    return pose


def create_pre_grasp_pose(grasp_pose, offset=0.05):
    """Create pre-grasp pose offset along approach direction."""
    pre_grasp = grasp_pose.copy()
    approach_dir = -grasp_pose[:3, 2]  # -Z axis
    approach_dir = approach_dir / np.linalg.norm(approach_dir)
    pre_grasp[:3, 3] = grasp_pose[:3, 3] + offset * approach_dir
    return pre_grasp


def pose_error(current_pose, target_pose):
    """Calculate position and orientation error."""
    pos_err = target_pose[:3, 3] - current_pose[:3, 3]
    rel_rot = target_pose[:3, :3] @ current_pose[:3, :3].T
    rel_quat = T.mat2quat(rel_rot)
    ori_err = T.quat2axisangle(rel_quat)
    return pos_err, ori_err


def create_action(pos_err, ori_err, gripper, gain=5.0):
    """Create OSC_POSE action."""
    pos_action = np.clip(gain * pos_err, -0.05, 0.05)
    ori_action = np.clip(gain * ori_err, -0.5, 0.5)
    return np.concatenate([pos_action, ori_action, [gripper]])


def move_to_pose(env, target_pose, gripper, max_steps=200, threshold=0.01):
    """Move to target pose using OSC."""
    for step in range(max_steps):
        current_pose = get_ee_pose(env)
        pos_err, ori_err = pose_error(current_pose, target_pose)
        print(pos_err)
        action = create_action(pos_err, ori_err, gripper)
        print(action)
        env.step(action)
        print("flop")
        env.render()

        if np.linalg.norm(pos_err) < threshold:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--robot', type=str, default='Panda')
    parser.add_argument('--camera', type=str, default='agentview')
    args = parser.parse_args()

    env = suite.make(
        env_name="Lift",
        robots=args.robot,
        controller_configs=None,
        has_renderer=True,  # Disable to avoid segfault
        has_offscreen_renderer=True,
        render_camera=args.camera,
        use_camera_obs=True,
        camera_names=[args.camera],
        camera_heights=512,
        camera_widths=512,
        camera_depths=True,
        reward_shaping=True,
        control_freq=20,
    )

    # VisualizationWrapper causes segfault with has_renderer=True
    indicator_config = {
        "name": "target",
        "type": None,
        "size": [0.02],
        "rgba": [0, 1, 0, 0.7]
    }
    env = VisualizationWrapper(env, indicator_configs=indicator_config)
    m2t2 = M2T2GraspPredictor(
        checkpoint_path=args.checkpoint,
        device='cuda',
        use_language=False
    )

    obs = env.reset()
    env.render()

    # Extract point cloud
    print("Extracting point cloud...")
    xyz, rgb = extract_point_cloud_from_obs(obs, camera_name=args.camera, env=env)
    # print(f"Point cloud: {xyz.shape[0]} points")
    # Save debug data
    import os
    debug_dir = "/home/maggie/research/robosuite/debug"
    os.makedirs(debug_dir, exist_ok=True)
    np.save(f"{debug_dir}/xyz.npy", xyz)
    np.save(f"{debug_dir}/rgb.npy", rgb)
    np.save(f"{debug_dir}/depth.npy", obs[f"{args.camera}_depth"])
    np.save(f"{debug_dir}/rgb_image.npy", obs[f"{args.camera}_image"])
    print(f"Saved debug data to {debug_dir}")

    # Prepare for M2T2
    inputs, points, center_offset = prepare_point_cloud_for_m2t2(
        xyz, rgb, num_points=16384, center=True
    )
    np.save(f"{debug_dir}/inputs.npy", inputs)
    np.save(f"{debug_dir}/points.npy", points)

    # Get camera extrinsics
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix
    cam_extrinsics = get_camera_extrinsic_matrix(env.sim, args.camera)

    print("Running M2T2 inference...")
    grasp_pose, confidence = m2t2.get_best_grasp(
        point_cloud_inputs=inputs,
        point_cloud_xyz=points,
        language_tokens=None,
        mask_thresh=0.01,
        num_runs=3,
        cam_pose=cam_extrinsics,
        center_offset=center_offset
    )

    if grasp_pose is None:
        print("No grasp found!")
        return

    # Transform to world frame
    if cam_extrinsics is not None:
        grasp_pose = cam_extrinsics @ grasp_pose

    print(f"Grasp found with confidence: {confidence:.3f}")
    print(f"Grasp position: {grasp_pose[:3, 3]}")

    # Visualize target (disabled - VisualizationWrapper removed)
    # env.set_indicator_pos("target", grasp_pose[:3, 3])

    # Execute grasp sequence
    grasp_executor = GraspExecutor(env, controller_type='OSC_POSE')
    success = grasp_executor.execute_grasp_sequence(
                grasp_pose=grasp_pose,
                pre_grasp_offset=0.05,
                num_waypoints=30
            )
    # pre_grasp = create_pre_grasp_pose(grasp_pose, offset=0.05)

    # print("Phase 1: Moving to pre-grasp...")
    # move_to_pose(env, pre_grasp, gripper=-1)

    # print("Phase 2: Moving to grasp...")
    # move_to_pose(env, grasp_pose, gripper=-1)

    # print("Phase 3: Closing gripper...")
    # for _ in range(20):
    #     action = create_action(np.zeros(3), np.zeros(3), gripper=1)
    #     env.step(action)
    #     env.render()

    # print("Done!")
    env.render()
    import time
    while True:
        env.render()
        time.sleep(0.01)


if __name__ == '__main__':
    main()
