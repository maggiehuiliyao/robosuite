import argparse
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import robosuite as suite
from robosuite.wrappers import VisualizationWrapper


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


def create_osc_action(target_pos, target_ori=None, gripper=1):
    """Create OSC_POSE action: [dx, dy, dz, dax, day, daz, gripper]"""
    if target_ori is None:
        target_ori = np.zeros(3)
    return np.concatenate([target_pos, target_ori, [gripper]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='Panda')
    parser.add_argument('--camera', type=str, default='agentview')
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
    env = VisualizationWrapper(env, indicator_configs=indicator_config)

    obs = env.reset()
    env.render()

    # Get cube position
    cube_pos = get_cube_position(env)
    print(f"Cube position: {cube_pos}")

    # Move to above cube
    target_pos = cube_pos.copy()
    target_pos[2] += 0.01  # 1cm above cube

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
