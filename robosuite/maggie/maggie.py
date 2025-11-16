import argparse
from torchvision import transforms
import numpy as np
import sys
from pathlib import Path

from omegaconf import OmegaConf
import torch

from robosuite.maggie.point_cloud_utils import extract_point_cloud_from_obs, filter_point_cloud_workspace, get_table_bounds, prepare_point_cloud_for_m2t2
from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix, get_real_depth_map

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import robosuite as suite
from robosuite.wrappers import VisualizationWrapper

normalize_rgb = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_cube_position(env):
    obj_name = 'cube_main'
    if obj_name in env.sim.model.body_names:
        body_id = env.sim.model.body_name2id(obj_name)
        return env.sim.data.body_xpos[body_id].copy()
    return None


def get_ee_position(env):
    """Get end-effector position."""
    possible_names = ['gripper0_grip_site', 'gripper0_right_grip_site', 'grip_site', 'ee_site']
    for name in possible_names:
        if name in env.sim.model.site_names:
            site_id = env.sim.model.site_name2id(name)
            return env.sim.data.site_xpos[site_id].copy()
    return None

def sample_points(xyz, num_points):
    num_replica = num_points // xyz.shape[0]
    num_remain = num_points % xyz.shape[0]
    pt_idx = torch.randperm(xyz.shape[0])
    pt_idx = torch.cat(
        [pt_idx for _ in range(num_replica)] + [pt_idx[:num_remain]]
    )
    return pt_idx

def create_osc_action(target_pos, target_ori=None, gripper=1):
    """Create OSC_POSE action: [dx, dy, dz, dax, day, daz, gripper]"""
    if target_ori is None:
        target_ori = np.zeros(3)
    return np.concatenate([target_pos, target_ori, [gripper]])

def depth_to_xyz(depth, intrinsics):
        # Ensure depth is 2D (H, W)
        if len(depth.shape) == 3:
            depth = depth.squeeze(-1)

        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        u, v = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        Z = depth
        X = (u - cx) * (Z / fx)
        Y = (v - cy) * (Z / fy)
        xyz = np.stack((X, Y, Z), axis=-1)
        return xyz

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='Panda')
    parser.add_argument('--camera', type=str, default='agentview')
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

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
        control_freq=20,
    )

    # Add visualization wrapper with green indicator
    indicator_config = {
        "name": "target",
        "type": "sphere",
        "size": [0.02],
        "rgba": [0, 1, 0, 0.7]  # Green
    }


    # yaoyao
    from robosuite.maggie.m2t2_wrapper import M2T2GraspPredictor
    m2t2 = M2T2GraspPredictor(
        checkpoint_path=args.checkpoint,
        device='cuda',
        use_language=False
    )
    env = VisualizationWrapper(env, indicator_configs=indicator_config)

    obs = env.reset()
    env.render()

    # i'm doing this shit myself wtf

    # loading in all the data values
    data = {}
    rgb_raw = obs["agentview_image"]  # (H, W, 3)
    depth = get_real_depth_map(env.sim, obs["agentview_depth"])  # (H, W) or (H, W, 1)
    intrinsics = get_camera_intrinsic_matrix(env.sim, args.camera, depth.shape[0], depth.shape[1])

    pcd_raw = depth_to_xyz(depth, intrinsics)  # Shape: (H, W, 3)
    cam_extrinsics = get_camera_extrinsic_matrix(env.sim, args.camera)
    pcd_raw = pcd_raw @ cam_extrinsics[:3, :3].T + cam_extrinsics[:3, 3]  # Shape: (H, W, 3)

    # Flatten to (N, 3) for filtering
    pcd_raw = pcd_raw.reshape(-1, 3)  # (H*W, 3)
    rgb_raw = rgb_raw.reshape(-1, 3)  # (H*W, 3)

    bounds = get_table_bounds(env)
    pcd_raw, rgb_raw = filter_point_cloud_workspace(pcd_raw, rgb_raw, bounds)

    # copied directly from rlbench
    rgb = normalize_rgb(rgb_raw[:, None]).squeeze(2).T
    pcd = torch.from_numpy(pcd_raw).float()
    pt_idx = sample_points(pcd_raw, 16384)
    pcd, rgb = pcd[pt_idx], rgb[pt_idx]
    data = {
            'inputs': torch.cat([pcd - pcd.mean(dim=0), rgb], dim=1),
            'points': pcd,
            'task': 'pick'
        }
    
    data['cam_pose'] = torch.from_numpy(cam_extrinsics).float()
    # model requires these fields even for pick task (uses dummy data like M2T2 dataset.py:100-106)
    data['object_inputs'] = torch.rand(1024, 6)
    data['ee_pose'] = torch.eye(4)
    data['bottom_center'] = torch.zeros(3)
    data['object_center'] = torch.zeros(3)


    ############### ONLY OPERATE HERE ################

    # Calculate center offset (used to center the point cloud)
    center_offset = pcd.mean(dim=0)

    # Batch the data
    data_batch = m2t2.collate([data])
    m2t2.to_gpu(data_batch)

    # Create eval config with world_coord=True (since data is in world frame)
    eval_config = OmegaConf.create({
        'mask_thresh': 0.01,
        'world_coord': True,  # Data is already in world frame
        'placement_height': 0.02,
        'object_thresh': 0.1
    })

    # Run inference
    with torch.no_grad():
        outputs = m2t2.model.infer(data_batch, eval_config)

    # Move outputs to CPU
    m2t2.to_cpu(outputs)

    # Extract grasps and confidences from nested structure
    # Structure: outputs['grasps'][batch_idx][object_idx] -> (N_grasps, 4, 4)
    #           outputs['grasp_confidence'][batch_idx][object_idx] -> (N_grasps,)
    all_grasps = []
    all_confidences = []

    if 'grasps' in outputs and 'grasp_confidence' in outputs:
        grasps_list = outputs['grasps']
        confs_list = outputs['grasp_confidence']

        # Iterate through batch (should be 1)
        for batch_idx in range(len(grasps_list)):
            batch_grasps = grasps_list[batch_idx]
            batch_confs = confs_list[batch_idx]

            # Iterate through objects
            for obj_idx in range(len(batch_grasps)):
                obj_grasps = batch_grasps[obj_idx]
                obj_confs = batch_confs[obj_idx]

                # Extract individual grasps
                if torch.is_tensor(obj_grasps) and obj_grasps.shape[0] > 0:
                    for i in range(obj_grasps.shape[0]):
                        grasp = obj_grasps[i].detach().cpu().numpy()
                        conf = obj_confs[i].item() if torch.is_tensor(obj_confs[i]) else obj_confs[i]

                        # Filter by confidence threshold
                        if conf >= 0.01:
                            all_grasps.append(grasp)
                            all_confidences.append(conf)

    # Select best grasp (highest confidence)
    if len(all_grasps) > 0:
        best_idx = np.argmax(all_confidences)
        grasp_pose = all_grasps[best_idx].copy()
        confidence = all_confidences[best_idx]

        # Transform from centered world frame to actual world frame
        # Add center_offset to translation component
        grasp_pose[:3, 3] += center_offset.cpu().numpy()

        print(f"\n✓ Found {len(all_grasps)} grasps, selected best with confidence: {confidence:.4f}")
    else:
        print("\n✗ No grasps found!")
        grasp_pose = np.eye(4)
        confidence = 0.0

    ############### ONLY OPERATE HERE ################

    target_pos = grasp_pose[:3, 3]

    print(f"Predicted grasp pose:\n{grasp_pose}")
    print(f"Grasp confidence: {confidence:.4f}")

    # target_pos[0] = 0.0  # x center
    # target_pos[1] = 0.0  # y center
    target_pos[2] -= 1.1  # 9cm above cube

    print(f"Moving to: {target_pos}")

    # Set indicator to target position
    env.set_indicator_pos("target", target_pos)

    # Control loop - move to target position
    for step in range(500):
        ee_pos = get_ee_position(env)

        # Simple proportional control
        pos_error = target_pos - ee_pos
        action = create_osc_action(pos_error * 5.0, gripper=1)  # Open gripper

        obs, reward, done, info = env.step(action)
        env.render()

        if np.linalg.norm(pos_error) < 0.01:
            print(f"Reached target at step {step}")
            break

    # Hold position
    import time
    while True:
        action = create_osc_action(np.zeros(3), gripper=1)
        env.step(action)
        env.render()
        time.sleep(0.01)

    env.close()


if __name__ == '__main__':
    main()
