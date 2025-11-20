"""
Evaluation script for trained DP3 policy in Robosuite.

Loads a trained policy from object_centric_diffusion and evaluates it
on the Lift task in Robosuite, using M2T2 for grasping.
"""

import argparse
import os
import sys
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torchvision import transforms
import xml.etree.ElementTree as ET

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, '/home/maggie/research/object_centric_diffusion')

import robosuite as suite
from robosuite.wrappers.visualization_wrapper import VisualizationWrapper
from robosuite.utils.mjcf_utils import new_body, new_site
import robosuite.utils.transform_utils as T
from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix, get_real_depth_map

# Import from maggie.py and dp3_policy_wrapper
from robosuite.maggie.dp3_policy_wrapper import DP3PolicyWrapper
from robosuite.maggie.maggie import (
    get_ee_position,
    get_ee_orientation,
    get_cube_pose_7d,
    get_marker_pose,
    mujoco_quat_to_xyzw,
    add_marker_to_model,
    create_osc_action,
    depth_to_xyz,
    sample_points
)
from robosuite.maggie.video_utils import save_episode_video, add_text_to_frame

# Import from object_centric_diffusion
from tools.train_dp3 import TrainDP3Workspace
from utils.pose_utils import get_rel_pose
import hydra
from omegaconf import OmegaConf


normalize_rgb = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def setup_environment(args):
    """Setup Robosuite environment with marker and visualization."""
    env = suite.make(
        env_name="Lift",
        robots=args.robot,
        controller_configs=None,
        has_renderer=True,
        has_offscreen_renderer=True,
        render_camera=args.camera,
        use_camera_obs=True,
        camera_names=[args.camera],
        camera_heights=512,
        camera_widths=512,
        camera_depths=True,
        reward_shaping=True,
        control_freq=15,
        ignore_done=True,
    )

    env.set_xml_processor(processor=add_marker_to_model)
    return env


def load_trained_policy(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    cfg = checkpoint['cfg']

    print(f"Loaded config from checkpoint")
    print(f"  Task: {cfg.task.task_name}")
    print(f"  Use EMA: {cfg.training.use_ema}")

    workspace = TrainDP3Workspace(cfg)
    workspace.load_checkpoint(path=checkpoint_path)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.eval()
    policy.cuda()
    return policy, cfg


def reset_and_randomize_cube(env):
    """Reset environment and randomize cube position/orientation."""
    obs = env.reset()
    cube_joint_id = env.sim.model.joint_name2id('cube_joint0')
    qpos_addr = env.sim.model.jnt_qposadr[cube_joint_id]
    env.sim.data.qpos[qpos_addr + 0] -= 0.1   # x
    env.sim.data.qpos[qpos_addr + 1] -= 0.18   # y
    angle_deg = np.random.uniform(0, 360)
    quat = R.from_euler('z', angle_deg, degrees=True).as_quat()  # [x,y,z,w]
    quat_mj = np.array([quat[3], quat[0], quat[1], quat[2]])
    env.sim.data.qpos[qpos_addr + 3 : qpos_addr + 7] = quat_mj
    env.sim.forward()

    # Force camera observations to update
    for cam_name in env.camera_names:
        env.sim.render(
            camera_name=cam_name,
            width=env.camera_widths[0],
            height=env.camera_heights[0],
            depth=True
        )

    obs = env._get_observations(force_update=True)
    return obs


def predict_grasp_with_m2t2(env, m2t2, obs, camera_name):
    """Predict grasp pose using M2T2 model."""
    # Extract point cloud
    rgb_raw = obs[f"{camera_name}_image"]  # (H, W, 3)
    depth = get_real_depth_map(env.sim, obs[f"{camera_name}_depth"])  # (H, W)
    intrinsics = get_camera_intrinsic_matrix(env.sim, camera_name, depth.shape[0], depth.shape[1])

    pcd_raw = depth_to_xyz(depth, intrinsics)  # Shape: (H, W, 3)
    cam_extrinsics = get_camera_extrinsic_matrix(env.sim, camera_name)
    pcd_raw = pcd_raw @ cam_extrinsics[:3, :3].T + cam_extrinsics[:3, 3]  # Shape: (H, W, 3)

    # Flatten to (N, 3) for filtering
    pcd_raw = pcd_raw.reshape(-1, 3)  # (H*W, 3)
    rgb_raw = rgb_raw.reshape(-1, 3)  # (H*W, 3)

    # Filter workspace (z > 0.6)
    mask = pcd_raw[:, 2] > 0.6
    pcd_raw = pcd_raw[mask]
    rgb_raw = rgb_raw[mask]

    grasp_pose, confidence = None, 0.0
    max_attempts = 100

    for attempt in range(max_attempts):
        # Prepare data for M2T2
        rgb = normalize_rgb(rgb_raw[:, None]).squeeze(2).T
        pcd = torch.from_numpy(pcd_raw).float()
        pt_idx = sample_points(pcd, 16384)
        pcd, rgb = pcd[pt_idx], rgb[pt_idx]

        data = {
            'inputs': torch.cat([pcd - pcd.mean(dim=0), rgb], dim=1),
            'points': pcd,
            'task': 'pick',
            'cam_pose': torch.from_numpy(cam_extrinsics).float(),
            # Dummy fields required by M2T2
            'object_inputs': torch.rand(1024, 6),
            'ee_pose': torch.eye(4),
            'bottom_center': torch.zeros(3),
            'object_center': torch.zeros(3)
        }

        # Batch the data
        data_batch = m2t2.collate([data])
        m2t2.to_gpu(data_batch)

        eval_config = OmegaConf.create({
            'mask_thresh': 0.01,
            'world_coord': True,
            'placement_height': 0.02,
            'object_thresh': 0.1
        })

        # Run inference
        with torch.no_grad():
            outputs = m2t2.model.infer(data_batch, eval_config)

        m2t2.to_cpu(outputs)

        # Extract grasps
        all_grasps = []
        all_confidences = []

        if 'grasps' in outputs and 'grasp_confidence' in outputs:
            grasps_list = outputs['grasps']
            confs_list = outputs['grasp_confidence']

            for batch_idx in range(len(grasps_list)):
                batch_grasps = grasps_list[batch_idx]
                batch_confs = confs_list[batch_idx]

                for obj_idx in range(len(batch_grasps)):
                    obj_grasps = batch_grasps[obj_idx]
                    obj_confs = batch_confs[obj_idx]

                    if torch.is_tensor(obj_grasps) and obj_grasps.shape[0] > 0:
                        for i in range(obj_grasps.shape[0]):
                            grasp = obj_grasps[i].detach().cpu().numpy()
                            conf = obj_confs[i].item() if torch.is_tensor(obj_confs[i]) else obj_confs[i]

                            if conf >= 0.01:
                                all_grasps.append(grasp)
                                all_confidences.append(conf)

        # Select best grasp
        if len(all_grasps) > 0:
            best_idx = np.argmax(all_confidences)
            confidence = all_confidences[best_idx]
            if confidence > 0.93:
                grasp_pose = all_grasps[best_idx].copy()
                print(f"Found grasp with confidence: {confidence:.4f}")
                break

    if grasp_pose is None:
        print(f"Failed to find grasp after {max_attempts} attempts")

    return grasp_pose, confidence


def execute_grasp_sequence(env, grasp_pose):
    """Execute reaching, lowering, and grasping. Returns T_obj_to_gripper."""
    target_pos = grasp_pose[:3, 3]
    target_ori = grasp_pose[:3, :3]

    print("Phase 1: Reaching to pre-grasp pose...")
    # Phase 1: Reach to pre-grasp
    for step in range(500):
        ee_pos = get_ee_position(env)
        ee_ori = get_ee_orientation(env)

        pos_error = target_pos - ee_pos
        ori_error_mat = target_ori @ ee_ori.T
        ori_error_quat = T.mat2quat(ori_error_mat)
        ori_error = T.quat2axisangle(ori_error_quat)

        action = create_osc_action(pos_error * 5.0, ori_error * 2.0, gripper=-1)
        obs, _, _, _ = env.step(action)
        env.render()

        if np.linalg.norm(pos_error) < 0.01 and np.linalg.norm(ori_error) < 0.1:
            print(f"Reached pre-grasp at step {step}")
            break

    print("Phase 2: Lowering to contact...")
    # Phase 2: Lower to contact
    target_pos_lower = target_pos.copy()
    target_pos_lower[2] -= 0.135

    for step in range(200):
        ee_pos = get_ee_position(env)
        ee_ori = get_ee_orientation(env)

        pos_error = target_pos_lower - ee_pos
        ori_error_mat = target_ori @ ee_ori.T
        ori_error_quat = T.mat2quat(ori_error_mat)
        ori_error = T.quat2axisangle(ori_error_quat)

        action = create_osc_action(pos_error * 5.0, ori_error * 2.0, gripper=-1)
        obs, _, _, _ = env.step(action)
        env.render()

        if np.linalg.norm(pos_error) < 0.01:
            print(f"Reached lower position at step {step}")
            break

    print("Phase 3: Closing gripper...")
    # Phase 3: Close gripper
    for step in range(100):
        action = create_osc_action(np.zeros(3), np.zeros(3), gripper=1)
        obs, _, _, _ = env.step(action)
        env.render()

    # Compute T_obj_to_gripper
    ee_pos = get_ee_position(env)
    ee_ori = get_ee_orientation(env)
    ee_quat = T.mat2quat(ee_ori)  # [w, x, y, z]
    ee_quat_xyzw = mujoco_quat_to_xyzw(ee_quat)
    ee_pose_7d = np.concatenate([ee_pos, ee_quat_xyzw])

    cube_pose = get_cube_pose_7d(env)
    T_obj_to_gripper = get_rel_pose(cube_pose, ee_pose_7d)

    print("Grasp sequence complete!")
    return T_obj_to_gripper


def execute_policy(env, policy_wrapper, T_obj_to_gripper, max_steps=500, save_video=True):
    """Execute policy until success or timeout."""
    policy_wrapper.reset(T_obj_to_gripper)
    frames = []

    target_pose = get_marker_pose()  # Static target
    marker_pos = target_pose[:3]

    success = False
    final_distance = None

    print(f"Starting policy execution (max {max_steps} steps)...")

    for step in range(max_steps):
        # Get current state
        cube_pose = get_cube_pose_7d(env)
        ee_pos = get_ee_position(env)

        # Check success: cube within 5cm of marker
        cube_pos = cube_pose[:3]
        distance_to_target = np.linalg.norm(cube_pos - marker_pos)
        final_distance = distance_to_target

        if distance_to_target < 0.05:  # 5cm threshold
            print(f"Success! Cube reached target at step {step}")
            success = True
            # Continue for a few more steps to record success
            if step > max_steps - 20:
                break

        # Get action from policy wrapper
        obs = env._get_observations()
        action = policy_wrapper.get_action(obs)

        # Execute action
        obs, reward, done, info = env.step(action)
        env.render()
        # Record frame
        # if save_video:
        #     frame = obs[f'{env.camera_names[0]}_image'].copy()
        #     # Add text overlay
        #     text = f"Step: {step} | Dist: {distance_to_target:.3f}m"
        #     if success:
        #         text += " | SUCCESS"
        #     frame = add_text_to_frame(frame, text)
        #     frames.append(frame)

    if not success:
        print(f"Failed: Timeout after {max_steps} steps. Final distance: {final_distance:.4f}m")

    return success, frames, final_distance


def run_episode(env, m2t2, policy, episode_idx, args):
    obs = reset_and_randomize_cube(env)
    env.render()
    grasp_pose, confidence = predict_grasp_with_m2t2(env, m2t2, obs, args.camera)

    if grasp_pose is None:
        print("Failed to find grasp. Skipping episode.")
        return {
            'episode': episode_idx,
            'success': False,
            'grasp_confidence': 0.0,
            'failure_reason': 'grasp_prediction_failed',
            'num_steps': 0,
            'final_distance': None
        }

    # Phase 3: Execute grasp
    T_obj_to_gripper = execute_grasp_sequence(env, grasp_pose)

    # Phase 4: Policy execution
    policy_wrapper = DP3PolicyWrapper(policy, env, horizon=args.horizon, n_obs_steps=args.n_obs_steps)
    success, frames, final_distance = execute_policy(env, policy_wrapper, T_obj_to_gripper,
                                                      max_steps=args.max_steps,
                                                      save_video=args.save_video)

    # Phase 5: Save video
    # video_path = None
    # if args.save_video and len(frames) > 0:
    #     videos_dir = os.path.join(args.output_dir, 'videos')
    #     video_path = save_episode_video(frames, episode_idx, videos_dir)

    return {
        'episode': episode_idx,
        'success': success,
        'grasp_confidence': confidence,
        # 'video_path': video_path,
        'num_steps': len(frames),
        'final_distance': final_distance,
        'failure_reason': None if success else 'timeout'
    }


def save_evaluation_results(results, output_dir):
    """Save evaluation results to JSON and pickle."""
    os.makedirs(output_dir, exist_ok=True)

    # Compute summary statistics
    total_episodes = len(results)
    successful_episodes = sum(1 for r in results if r['success'])
    success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0.0

    successful_results = [r for r in results if r['success']]
    avg_steps_to_success = np.mean([r['num_steps'] for r in successful_results]) if successful_results else 0.0
    avg_grasp_confidence = np.mean([r['grasp_confidence'] for r in results if r['grasp_confidence'] > 0])

    # Count failure reasons
    failures = {}
    for r in results:
        if not r['success'] and r.get('failure_reason'):
            reason = r['failure_reason']
            failures[reason] = failures.get(reason, 0) + 1

    summary = {
        'success_rate': success_rate,
        'total_episodes': total_episodes,
        'successful_episodes': successful_episodes,
        'avg_steps_to_success': float(avg_steps_to_success),
        'avg_grasp_confidence': float(avg_grasp_confidence),
        'failures': failures
    }

    # Save summary as JSON
    summary_path = os.path.join(output_dir, 'results.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to: {summary_path}")

    # Save detailed results as pickle
    detailed_path = os.path.join(output_dir, 'detailed_results.pkl')
    with open(detailed_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved detailed results to: {detailed_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    if failures:
        print(f"\nFailure Breakdown:")
        for reason, count in failures.items():
            print(f"  {reason}: {count}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate DP3 policy in Robosuite')
    parser.add_argument('--policy_checkpoint', type=str, required=True,
                        help='Path to trained DP3 checkpoint')
    parser.add_argument('--m2t2_checkpoint', type=str, required=True,
                        help='Path to M2T2 grasp predictor checkpoint')
    parser.add_argument('--robot', type=str, default='Panda',
                        help='Robot type')
    parser.add_argument('--camera', type=str, default='agentview',
                        help='Camera name')
    parser.add_argument('--n_episodes', type=int, default=25,
                        help='Number of evaluation episodes')
    parser.add_argument('--output_dir', type=str, default='eval_results/maggie_block_robosuite',
                        help='Output directory for results and videos')
    parser.add_argument('--max_steps', type=int, default=500,
                        help='Maximum steps per episode')
    parser.add_argument('--horizon', type=int, default=4,
                        help='Policy horizon')
    parser.add_argument('--n_obs_steps', type=int, default=1,
                        help='Number of observation steps')
    parser.add_argument('--save_video', action='store_true', default=True,
                        help='Save videos of episodes')
    parser.add_argument('--no_video', dest='save_video', action='store_false',
                        help='Do not save videos')
    args = parser.parse_args()

    print(f"Starting evaluation with {args.n_episodes} episodes")
    print(f"Output directory: {args.output_dir}")
    env = setup_environment(args)

    # Load M2T2 grasp predictor
    print("Loading M2T2 grasp predictor...")
    from robosuite.maggie.m2t2_wrapper import M2T2GraspPredictor
    m2t2 = M2T2GraspPredictor(
        checkpoint_path=args.m2t2_checkpoint,
        device='cuda',
        use_language=False
    )

    # Load DP3 policy
    print("Loading DP3 policy...")
    policy, cfg = load_trained_policy(args.policy_checkpoint)

    # Run evaluation
    results = []
    for episode_idx in range(args.n_episodes):
        try:
            result = run_episode(env, m2t2, policy, episode_idx, args)
            results.append(result)
        except Exception as e:
            print(f"Error in episode {episode_idx}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'episode': episode_idx,
                'success': False,
                'grasp_confidence': 0.0,
                'failure_reason': 'exception',
                'error': str(e)
            })

    # Save results
    # save_evaluation_results(results, args.output_dir)

    # Cleanup
    env.close()
    print("Evaluation complete!")


if __name__ == '__main__':
    main()
