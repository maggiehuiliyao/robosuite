import argparse
from torchvision import transforms
import numpy as np
import sys
from pathlib import Path
import pickle
import shutil
import os

from omegaconf import OmegaConf
import torch
from pynput import keyboard

from robosuite.maggie.point_cloud_utils import extract_point_cloud_from_obs, filter_point_cloud_workspace, get_table_bounds, prepare_point_cloud_for_m2t2
from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix, get_real_depth_map
import robosuite.utils.transform_utils as T
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import robosuite as suite
from robosuite.devices import SpaceMouse
import xml.etree.ElementTree as ET
from robosuite.utils.mjcf_utils import new_body, new_site
from robosuite.wrappers.visualization_wrapper import VisualizationWrapper

# Import from object_centric_diffusion for data collection
import sys
sys.path.insert(0, '/home/maggie/research/object_centric_diffusion')
from utils.collect_utils import collect_narr_function, save_zarr
from utils.pose_utils import get_rel_pose

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

def get_ee_orientation(env):
    possible_names = ['gripper0_grip_site', 'gripper0_right_grip_site', 'grip_site', 'ee_site']
    for name in possible_names:
        if name in env.sim.model.site_names:
            site_id = env.sim.model.site_name2id(name)
            return env.sim.data.site_xmat[site_id].reshape(3, 3).copy()
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

def add_marker_to_model(xml):
    root = ET.fromstring(xml)
    worldbody = root.find("worldbody")
    marker_body = new_body(name="marker_body", pos=(-0.03, 0.2, 0.80))
    marker_body.append(new_site(name="marker", type="cylinder", size=[0.01, 0.0003], rgba=[0, 0, 1, 0.4]))
    worldbody.append(marker_body)
    return ET.tostring(root, encoding="utf8").decode("utf8")

def get_next_experiment_number():
    data_dir = "/home/maggie/research/object_centric_diffusion/data/maggie_block/episodes"
    if not os.path.exists(data_dir):
        return 0
    episode_folders = [f for f in os.listdir(data_dir) if f.startswith("episode_")]
    if not episode_folders:
        return 0
    numbers = [int(f.split("_")[1]) for f in episode_folders]
    return max(numbers) + 1

def get_cube_orientation(env):
    obj_name = 'cube_main'
    if obj_name in env.sim.model.body_names:
        body_id = env.sim.model.body_name2id(obj_name)
        return env.sim.data.body_xquat[body_id].copy()
    return None

def get_marker_pose():
    """Get marker pose as 7D vector [x, y, z, qx, qy, qz, qw] with identity quaternion."""
    # Marker is at fixed position (-0.03, 0.2, 0.80) with identity orientation
    position = np.array([-0.03, 0.2, 0.80])
    quaternion_xyzw = np.array([0.0, 0.0, 0.0, 1.0])  # identity quaternion in (x,y,z,w) format
    return np.concatenate([position, quaternion_xyzw])

def mujoco_quat_to_xyzw(quat):
    """Convert MuJoCo quaternion (w,x,y,z) to scipy/OCD format (x,y,z,w)."""
    return np.array([quat[1], quat[2], quat[3], quat[0]])

def get_cube_pose_7d(env):
    """Get cube pose as 7D vector [x, y, z, qx, qy, qz, qw] in OCD format."""
    position = get_cube_position(env)
    quat_mujoco = get_cube_orientation(env)  # (w, x, y, z)
    quat_xyzw = mujoco_quat_to_xyzw(quat_mujoco)  # (x, y, z, w)
    return np.concatenate([position, quat_xyzw])

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
        control_freq=15,
        ignore_done=True,
    )

    env.set_xml_processor(processor=add_marker_to_model)

    # Add gripper indicator (hidden initially, shown after grasping)
    indicator_config = {
        "name": "gripper_indicator",
        "type": "cylinder",
        "size": [0.002, 0.3],  # [radius, half-height] - 0.6m tall line
        "rgba": [1, 0, 0, 0.5],
        "pos": [0, 0, -10]  # Start hidden underground
    }
    env = VisualizationWrapper(env, indicator_configs=indicator_config)

    # Disable default gripper visualization (the green line)
    for setting in env.get_visualization_settings():
        if 'gripper' in setting.lower():
            env.set_visualization_setting(setting, False)

    from robosuite.maggie.m2t2_wrapper import M2T2GraspPredictor
    m2t2 = M2T2GraspPredictor(
        checkpoint_path=args.checkpoint,
        device='cuda',
        use_language=False
    )

    obs = env.reset()

    # Get joint ID for cube
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

    # Get fresh observations
    obs = env._get_observations(force_update=True)
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

    # bounds = get_table_bounds(env)
    # pcd_raw, rgb_raw = filter_point_cloud_workspace(pcd_raw, rgb_raw, bounds)
    pcd_raw = pcd_raw.reshape(-1, 3)   # (H*W, 3)
    rgb_raw = rgb_raw.reshape(-1, 3)   # (H*W, 3)

    # Create mask for points with z ≤ 1
    mask = pcd_raw[:, 2] > 0.6

    # Apply mask
    pcd_raw = pcd_raw[mask]
    rgb_raw = rgb_raw[mask]


    grasp_pose, notFound = None, True
    while notFound:
        rgb = normalize_rgb(rgb_raw[:, None]).squeeze(2).T
        pcd = torch.from_numpy(pcd_raw).float()
        pt_idx = sample_points(pcd_raw, 16384)
        pcd, rgb = pcd[pt_idx], rgb[pt_idx]
        data = {
                'inputs': torch.cat([pcd - pcd.mean(dim=0), rgb], dim=1),
                'points': pcd,
                'task': 'pick'
            }
        print("pointcloud mean: ", pcd.mean(dim=0))
        data['cam_pose'] = torch.from_numpy(cam_extrinsics).float()
        # model requires these fields even for pick task (uses dummy data like M2T2 dataset.py:100-106)
        data['object_inputs'] = torch.rand(1024, 6)
        data['ee_pose'] = torch.eye(4)
        data['bottom_center'] = torch.zeros(3)
        data['object_center'] = torch.zeros(3)


        ############### ONLY OPERATE HERE ################
        import os
        debug_dir = "/home/maggie/research/robosuite/debug"
        os.makedirs(debug_dir, exist_ok=True)
        np.save(f"{debug_dir}/xyz.npy", pcd)
        np.save(f"{debug_dir}/rgb.npy", rgb)
        np.save(f"{debug_dir}/depth.npy", obs[f"{args.camera}_depth"])
        np.save(f"{debug_dir}/rgb_image.npy", obs[f"{args.camera}_image"])

        # Batch the data
        data_batch = m2t2.collate([data])
        m2t2.to_gpu(data_batch)

        eval_config = OmegaConf.create({
            'mask_thresh': 0.01,
            'world_coord': True,  # Data is already in world frame
            'placement_height': 0.02,
            'object_thresh': 0.1
        })

        # running in
    
        with torch.no_grad():
            outputs = m2t2.model.infer(data_batch, eval_config)

        m2t2.to_cpu(outputs)
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
            confidence = all_confidences[best_idx]
            print(confidence)
            if confidence > 0.93:
                notFound = False
                grasp_pose = all_grasps[best_idx].copy()
                print("for maggie: ", grasp_pose[:3, 3])

                print(f"\n✓ Found {len(all_grasps)} grasps, selected best with confidence: {confidence:.4f}")


    target_pos = grasp_pose[:3, 3]
    target_ori = grasp_pose[:3, :3]

    print(f"Predicted grasp pose:\n{grasp_pose}")
    print(f"Grasp confidence: {confidence:.4f}")
    print(f"Moving to: {target_pos}")

    for step in range(500):
        ee_pos = get_ee_position(env)
        ee_ori = get_ee_orientation(env)

        pos_error = target_pos - ee_pos
        ori_error_mat = target_ori @ ee_ori.T
        ori_error_quat = T.mat2quat(ori_error_mat)
        ori_error = T.quat2axisangle(ori_error_quat)

        action = create_osc_action(pos_error * 5.0, ori_error * 2.0, gripper=-1)

        obs, reward, done, info = env.step(action)
        env.render()

        if np.linalg.norm(pos_error) < 0.01 and np.linalg.norm(ori_error) < 0.1:
            print(f"Reached target at step {step}")
            break

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

        obs, reward, done, info = env.step(action)
        env.render()

        if np.linalg.norm(pos_error) < 0.01:
            print(f"Reached lower position at step {step}")
            break

    print("Closing gripper...")
    for step in range(100):
        action = create_osc_action(np.zeros(3), np.zeros(3), gripper=1)
        obs, reward, done, info = env.step(action)
        env.render()

    # Show the gripper indicator now that we've grasped the block
    ee_pos = get_ee_position(env)
    if ee_pos is not None:
        env.set_indicator_pos("gripper_indicator", [ee_pos[0], ee_pos[1], 1.11])

    # Set up data collection directory in OCD format
    data_collection_dir = "/home/maggie/research/object_centric_diffusion/data/maggie_block/episodes"
    os.makedirs(data_collection_dir, exist_ok=True)
    experiment_num = get_next_experiment_number()
    episode_folder = os.path.join(data_collection_dir, f"episode_{experiment_num}")
    os.makedirs(episode_folder, exist_ok=True)

    # Initialize poses_dict for OCD format
    poses_dict = {
        "task_stage": [],
        "grasp_obj_pose": [],
        "grasp_obj_pose_relative_to_target": [],
        "target_obj_pose": [],
        "keypoint": [],
    }

    # Get marker (target) pose once - it's static
    target_pose = get_marker_pose()

    save_flag = [False]
    discard_flag = [False]

    def on_press(key):
        try:
            if hasattr(key, 'char') and key.char == 's':
                save_flag[0] = True
            elif hasattr(key, 'char') and key.char == 'q':
                discard_flag[0] = True
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print(f"Data collection started for episode_{experiment_num}")
    print("Press 's' to save and exit, 'q' to discard and exit")

    device = SpaceMouse(env=env, pos_sensitivity = 0.35, rot_sensitivity=0.35)
    device.start_control()

    all_prev_gripper_actions = [{f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
                                 for robot_arm in robot.arms if robot.gripper[robot_arm].dof > 0}
                                for robot in env.robots]

    while True:
        # Update gripper indicator position
        ee_pos = get_ee_position(env)
        if ee_pos is not None:
            env.set_indicator_pos("gripper_indicator", [ee_pos[0], ee_pos[1], 1.11])

        # Get cube pose in OCD format (7D with xyzw quaternion)
        grasp_pose = get_cube_pose_7d(env)

        # Compute relative pose (grasp object relative to target)
        relative_pose = get_rel_pose(target_pose, grasp_pose)

        # Append to poses_dict
        poses_dict["task_stage"].append(0)  # single-stage task
        poses_dict["grasp_obj_pose"].append(grasp_pose)
        poses_dict["grasp_obj_pose_relative_to_target"].append(relative_pose)
        poses_dict["target_obj_pose"].append(target_pose)

        if save_flag[0]:
            # Process the episode using OCD's collect_narr_function
            if len(poses_dict["grasp_obj_pose"]) < 3:
                print("Not enough frames collected (need at least 3). Discarding.")
                shutil.rmtree(episode_folder)
            else:
                arrays_dict_list, total_count_list = collect_narr_function([poses_dict], None)

                # Extract arrays from the result (single episode)
                arrays_dict = arrays_dict_list[0]
                total_count = total_count_list[0]

                # Prepare arrays for save_zarr
                state_arrays = arrays_dict["state"]
                state_in_world_arrays = arrays_dict["state_in_world"]
                state_next_arrays = arrays_dict["state_next"]
                state_next_in_world_arrays = arrays_dict["state_next_in_world"]
                goal_arrays = arrays_dict["goal"]
                action_arrays = arrays_dict["action"]
                progress_arrays = arrays_dict["progress"]
                progress_binary_arrays = arrays_dict["progress_binary"]
                task_stage_arrays = [np.array([s]) for s in arrays_dict["task_stage"]]
                variation_arrays = [np.array([0]) for _ in range(len(action_arrays))]  # variation = 0
                episode_ends_arrays = [total_count]  # single episode

                # Save to zarr format
                save_dir = os.path.join(episode_folder, "zarr")
                save_zarr(
                    save_dir,
                    state_arrays,
                    state_in_world_arrays,
                    state_next_arrays,
                    state_next_in_world_arrays,
                    goal_arrays,
                    action_arrays,
                    progress_arrays,
                    progress_binary_arrays,
                    task_stage_arrays,
                    variation_arrays,
                    episode_ends_arrays,
                )
                print(f"Saved episode_{experiment_num} with {total_count} keyframes (from {len(poses_dict['grasp_obj_pose'])} raw frames)")

            listener.stop()
            break

        if discard_flag[0]:
            shutil.rmtree(episode_folder)
            print(f"Discarded data for episode_{experiment_num}")
            listener.stop()
            break

        input_ac_dict = device.input2action()
        if input_ac_dict is None:
            break

        from copy import deepcopy
        action_dict = deepcopy(input_ac_dict)
        active_robot = env.robots[device.active_robot]
        if input_ac_dict is None:
            break

        from copy import deepcopy

        action_dict = deepcopy(input_ac_dict)
        for arm in active_robot.arms:
            controller_input_type = active_robot.part_controllers[arm].input_type

            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]
            else:
                raise ValueError

        env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
        env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
        env_action = np.concatenate(env_action)
        for gripper_ac in all_prev_gripper_actions[device.active_robot]:
            all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

        obs, reward, done, info = env.step(env_action)
        env.render()

    env.close()


if __name__ == '__main__':
    main()
