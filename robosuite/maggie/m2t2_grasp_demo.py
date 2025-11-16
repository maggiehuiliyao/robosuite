import argparse
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import robosuite as suite
from robosuite.wrappers import VisualizationWrapper
from robosuite.maggie.point_cloud_utils import (
    extract_point_cloud_from_obs,
    filter_point_cloud_workspace,
    prepare_point_cloud_for_m2t2
)
from robosuite.maggie.m2t2_wrapper import M2T2GraspPredictor, create_simple_language_embedding
from robosuite.maggie.grasp_utils import GraspExecutor
from robosuite.maggie.visualization_utils import save_all_visualizations, save_grasp_visualization
from robosuite.maggie.sim_visualization import visualize_grasp_in_viewer, visualize_cube_position


class M2T2RobosuiteDemo:
    """Main demo class for M2T2 + Robosuite integration."""

    def __init__(self, checkpoint_path, robot='Panda', camera='agentview',
                 use_language=True, device='cuda', save_dir=None):
        """
        Initialize demo.

        Args:
            checkpoint_path: Path to M2T2 checkpoint (.pth file)
            robot: Robot type (e.g., 'Panda', 'Sawyer')
            camera: Camera name for observations
            use_language: Whether to use language-conditioned model
            device: 'cuda' or 'cpu'
            save_dir: Directory to save visualizations (None to disable)
        """
        self.checkpoint_path = checkpoint_path
        self.robot = robot
        self.camera = camera
        self.use_language = use_language
        self.device = device
        self.save_dir = save_dir

        self._setup_environment()
        self._setup_m2t2()
        self.grasp_executor = GraspExecutor(self.env, controller_type='OSC_POSE')

    def _setup_environment(self):
        """Set up Robosuite environment."""
        print("Setting up Robosuite environment...")

        # Create environment
        self.env = suite.make(
            env_name="Lift",
            robots=self.robot,
            controller_configs=None,  # Use robot default
            has_renderer=True,
            has_offscreen_renderer=True,
            render_camera=self.camera,
            use_camera_obs=True,
            camera_names=[self.camera],
            camera_heights=512,
            camera_widths=512,
            camera_depths=True,  # Enable depth
            reward_shaping=True,
            control_freq=20,
        )

        self.env = VisualizationWrapper(self.env)
        print(f"Camera: {self.camera}")
        print(f"Action space: {self.env.action_spec}")

    def _setup_m2t2(self):
        self.m2t2 = M2T2GraspPredictor(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            use_language=self.use_language
        )

    def _get_cube_position(self):
        # doesn't do the rotation currently
        obj_name = 'cube_main'
        if obj_name in self.env.sim.model.body_names:
            body_id = self.env.sim.model.body_name2id(obj_name)
            cube_pos = self.env.sim.data.body_xpos[body_id].copy()
            return cube_pos

    def _get_ee_pose(self):
        """Get current end-effector pose as 4x4 matrix."""
        # possible_names = ['gripper0_grip_site', 'gripper0_right_grip_site', 'grip_site', 'ee_site']
        ee_site_name = "gripper0_right_grip_site"
        ee_pos = np.array(self.env.sim.data.site_xpos[self.env.sim.model.site_name2id(ee_site_name)])
        ee_mat = np.array(self.env.sim.data.site_xmat[self.env.sim.model.site_name2id(ee_site_name)]).reshape(3, 3)

        # Create 4x4 pose matrix
        ee_pose = np.eye(4)
        ee_pose[:3, :3] = ee_mat
        ee_pose[:3, 3] = ee_pos
        return ee_pose

    def _extract_object_point_cloud(self, xyz, rgb, object_center, radius=0.1, num_points=1024):
        """
        Extract object point cloud by spatial filtering around object center.

        Args:
            xyz: (N, 3) full scene point cloud
            rgb: (N, 3) RGB colors
            object_center: (3,) center position of object
            radius: Radius around center to extract points
            num_points: Target number of points

        Returns:
            object_inputs: (num_points, 6) object point cloud [xyz, rgb]
        """
        # Filter points within radius of object center
        distances = np.linalg.norm(xyz - object_center, axis=1)
        mask = distances < radius

        obj_xyz = xyz[mask]
        obj_rgb = rgb[mask]

        if len(obj_xyz) == 0:
            print(f"Warning: No points found within {radius}m of object. Using dummy inputs.")
            return np.random.rand(num_points, 6).astype(np.float32)

        # Downsample or upsample to target number of points
        if len(obj_xyz) >= num_points:
            indices = np.random.choice(len(obj_xyz), num_points, replace=False)
        else:
            indices = np.random.choice(len(obj_xyz), num_points, replace=True)

        obj_xyz_sampled = obj_xyz[indices]
        obj_rgb_sampled = obj_rgb[indices]

        # Center the object point cloud
        obj_xyz_centered = obj_xyz_sampled - object_center

        # Normalize RGB (same as scene point cloud)
        from robosuite.maggie.point_cloud_utils import normalize_point_cloud_rgb
        obj_rgb_norm = normalize_point_cloud_rgb(obj_rgb_sampled)

        # Concatenate
        object_inputs = np.concatenate([obj_xyz_centered, obj_rgb_norm], axis=1)

        print(f"Extracted object point cloud: {len(obj_xyz)} points found, sampled to {num_points}")
        return object_inputs.astype(np.float32)

    def capture_and_predict_grasp(self, obs, language_prompt=None):
        """
        Capture point cloud and predict grasp.

        Args:
            obs: Observation from environment
            language_prompt: Optional language instruction

        Returns:
            grasp_pose: 4x4 grasp transformation matrix (or None)
            confidence: Confidence score
        """
        print("\n" + "="*60)
        print("STEP 1: Extracting point cloud from observation")
        print("="*60)

        # Extract point cloud from observation (pass env for accurate camera parameters)
        # Point cloud will be in camera frame (no extrinsics transform applied during extraction)
        xyz, rgb = extract_point_cloud_from_obs(obs, camera_name=self.camera, env=self.env)
        print(f"Raw point cloud: {xyz.shape[0]} points (camera frame)")

        # Get camera extrinsics for grasp transformation
        from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix
        camera_extrinsics = get_camera_extrinsic_matrix(self.env.sim, self.camera)

        # Save visualizations if requested
        if self.save_dir is not None:
            print(f"Saving visualizations to {self.save_dir}...")
            save_all_visualizations(obs, xyz, rgb, self.camera, self.save_dir)

        # Skip workspace filtering for now
        # xyz, rgb = filter_point_cloud_workspace(xyz, rgb, workspace_bounds)

        # if xyz.shape[0] < 100:
        #     print("ERROR: Too few points in point cloud!")
        #     return None, 0.0

        # Prepare for M2T2
        inputs, points, center_offset = prepare_point_cloud_for_m2t2(
            xyz, rgb, num_points=16384, center=True
        )
        print("\n" + "="*60)
        print("STEP 2: Running M2T2 inference")
        print("="*60)

        # Prepare language tokens if using language model
        lang_tokens = None
        if self.use_language and language_prompt is not None:
            lang_tokens = create_simple_language_embedding(language_prompt)

        # Get real object information for M2T2
        cube_pos = self._get_cube_position()
        ee_pose = self._get_ee_pose()

        # Extract object point cloud and calculate centers
        object_inputs = None
        object_center = None
        bottom_center = None

        if cube_pos is not None:
            # Transform cube position to camera frame for point cloud extraction
            if camera_extrinsics is not None:
                # Transform world position to camera frame
                cube_pos_homog = np.append(cube_pos, 1.0)
                cube_pos_cam = (np.linalg.inv(camera_extrinsics) @ cube_pos_homog)[:3]
            else:
                cube_pos_cam = cube_pos

            # Extract object point cloud (in camera frame, centered at object)
            object_inputs = self._extract_object_point_cloud(
                xyz, rgb, cube_pos_cam, radius=0.1, num_points=1024
            )

            # Object center and bottom_center in world frame for M2T2
            object_center = cube_pos
            # Assume cube is 0.02m tall (standard Robosuite cube)
            cube_half_size = 0.01
            bottom_center = cube_pos.copy()
            bottom_center[2] -= cube_half_size  # Bottom is half-size below center

            print(f"Object center (world): {object_center}")
            print(f"Bottom center (world): {bottom_center}")
            print(f"EE pose (world): {ee_pose[:3, 3]}")

        # Predict grasp
        grasp_pose, confidence = self.m2t2.get_best_grasp(
            point_cloud_inputs=inputs,
            point_cloud_xyz=points,
            language_tokens=lang_tokens,
            mask_thresh=0.01,  # Very low threshold to see any predictions
            num_runs=3,  # Multiple runs for better coverage
            cam_pose=camera_extrinsics,  # Required for world_coord=False
            center_offset=center_offset  # Transform from centered frame to camera frame
        )

        if grasp_pose is not None:
            print(f"\n✓ Grasp found with confidence: {confidence:.3f}")
            print(f"Grasp position (camera frame): {grasp_pose[:3, 3]}")

            # M2T2 with world_coord=False returns grasp in CAMERA frame
            # We must manually transform to world frame (see demo.py:112-113)
            if camera_extrinsics is not None:
                grasp_pose = camera_extrinsics @ grasp_pose
                print(f"Grasp position (world frame): {grasp_pose[:3, 3]}")
            else:
                print("WARNING: No camera extrinsics - grasp is still in camera frame!")

            # Get cube position for comparison
            cube_pos = self._get_cube_position()
            if cube_pos is not None:
                print(f"Cube actual position: {cube_pos}")
                distance = np.linalg.norm(grasp_pose[:3, 3] - cube_pos)
                print(f"Distance from grasp to cube: {distance:.3f}m")

            # Visualize grasp and cube in simulation viewer
            print("\nVisualizing grasp in simulation viewer...")
            visualize_grasp_in_viewer(self.env, grasp_pose, axis_length=0.08)
            if cube_pos is not None:
                visualize_cube_position(self.env, cube_pos)

            # Save grasp visualization if requested
            if self.save_dir is not None:
                # grasp_pose is now in world frame, convert back to camera frame for visualization
                # Point cloud xyz is in camera frame, so grasp needs to be too
                if camera_extrinsics is not None:
                    grasp_camera = np.linalg.inv(camera_extrinsics) @ grasp_pose
                else:
                    grasp_camera = grasp_pose  # If no extrinsics, assume it's still camera frame
                save_grasp_visualization(
                    xyz, rgb, grasp_camera,
                    Path(self.save_dir) / f"{self.camera}_grasp_visualization.png"
                )
        else:
            print("\n✗ No grasp found!")

        return grasp_pose, confidence

    def run_demo(self, language_prompt=None, execute_grasp=True,
                 enable_keyboard_control=False, test_mode=False):
        """
        Run the full demo.

        Args:
            language_prompt: Language instruction for M2T2 (None for non-language model)
            execute_grasp: Whether to execute the predicted grasp
            enable_keyboard_control: Enable keyboard control after grasp
            test_mode: If True, use hardcoded grasp at cube position
        """
        print("\n" + "="*60)
        print("M2T2 + ROBOSUITE GRASP DEMO" + (" [TEST MODE]" if test_mode else ""))
        print("="*60)
        print(f"Task: Grasp the cube in Lift environment")
        print(f"Robot: {self.robot}")
        if test_mode:
            print(f"Mode: Test mode - using hardcoded grasp at cube position")
        else:
            print(f"Model type: {'Language-conditioned' if self.use_language else 'Non-language'}")
            if language_prompt is not None:
                print(f"Language prompt: '{language_prompt}'")
        print("="*60 + "\n")

        # Reset environment
        print("Resetting environment...")
        obs = self.env.reset()
        self.env.render()

        # Wait a bit for visualization
        for _ in range(10):
            self.env.step(np.zeros(self.env.action_dim))
            self.env.render()

        if test_mode:
            cube_pos = self._get_cube_position()

            print("\n" + "="*60)
            print("TEST MODE: Creating grasp at cube position")
            print("="*60)
            print(f"Cube position: {cube_pos}")

            # Create top-down grasp (gripper approaching from above)
            grasp_pose = np.eye(4)
            grasp_pose[:3, 3] = cube_pos.copy()

            # Orientation: Z-axis pointing down (approach from top)
            # X-axis forward, Y-axis to the right
            grasp_pose[:3, :3] = np.array([
                [1, 0, 0],   # X-axis (forward)
                [0, 1, 0],   # Y-axis (right)
                [0, 0, -1]   # Z-axis (down - approach direction)
            ])

            confidence = 1.0
            print(f"Test grasp pose:\n{grasp_pose}")
            print(f"Test grasp position: {grasp_pose[:3, 3]}")

            # Visualize
            visualize_grasp_in_viewer(self.env, grasp_pose)
            visualize_cube_position(self.env, cube_pos)
        else:
            # Normal mode: use M2T2
            grasp_pose, confidence = self.capture_and_predict_grasp(obs, language_prompt)

        if grasp_pose is None:
            print("\nDemo failed: No grasp found")
            return False

        # Execute grasp if requested
        if execute_grasp:
            print("\n" + "="*60)
            print("STEP 3: Executing grasp")
            print("="*60)

            success = self.grasp_executor.execute_grasp_sequence(
                grasp_pose=grasp_pose,
                pre_grasp_offset=0.05,
                num_waypoints=30
            )

            if success:
                print("\n✓ Grasp execution complete!")
            else:
                print("\n✗ Grasp execution failed!")

            import time
            while True:
                self.env.render()
                time.sleep(0.01)

        # # Keyboard control mode
        # if enable_keyboard_control:
        #     print("\n" + "="*60)
        #     print("STEP 4: Switching to keyboard control")
        #     print("="*60)
        #     print("You can now control the robot with keyboard")
        #     print("Press ESC to exit")

        #     from robosuite.devices import Keyboard

        #     device = Keyboard(env=self.env)
        #     self.env.viewer.add_keypress_callback(device.on_press)

        #     while True:
        #         device.start_control()
        #         action_dict = device.input2action()

        #         if action_dict is None:
        #             break

        #         # Convert to action vector
        #         # (This is simplified - adapt based on your controller)
        #         action = np.zeros(self.env.action_dim)
        #         # Fill in action based on device input
        #         # This depends on your controller configuration

        #         self.env.step(action)
        #         self.env.render()

        return True

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'env'):
            self.env.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='M2T2 + Robosuite Grasp Demo')

    parser.add_argument(
        '--checkpoint',
        type=str, required=False, default=None,
        help='Path to M2T2 checkpoint (.pth file) - not needed for --test-grasp'
    )

    parser.add_argument(
        '--robot',
        type=str, required=False, default='Panda',
        choices=['Panda', 'Sawyer', 'IIWA', 'Jaco', 'Kinova3', 'UR5e'],
        help='Robot type'
    )

    parser.add_argument(
        '--camera',
        type=str, required=False, default='agentview',
        help='Camera name for observations'
    )

    parser.add_argument(
        '--language',
        type=str, required=False, default=None,
        help='Language instruction for M2T2 (enables language-conditioned model if provided)'
    )

    parser.add_argument(
        '--device',
        type=str, required=False, default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for M2T2 inference'
    )

    parser.add_argument(
        '--no-execute',
        action='store_true',
        help='Only predict grasp, do not execute'
    )

    parser.add_argument(
        '--keyboard',
        action='store_true',
        help='Enable keyboard control after grasp'
    )

    parser.add_argument(
        '--save-dir',
        type=str, required=False, default=None,
        help='Directory to save visualizations (RGB, depth, point cloud)'
    )

    parser.add_argument(
        '--test-grasp',
        action='store_true',
        help='Test mode: use hardcoded grasp at cube position instead of M2T2'
    )

    args = parser.parse_args()

    if not args.test_grasp:
        if args.checkpoint is None:
            print(f"ERROR: --checkpoint is required unless using --test-grasp")
            sys.exit(1)
        if not Path(args.checkpoint).exists():
            print(f"ERROR: Checkpoint not found: {args.checkpoint}")
            sys.exit(1)

    use_language = args.language is not None

    try:
        if args.test_grasp:
            from robosuite.maggie.m2t2_grasp_demo import M2T2RobosuiteDemo
            demo = object.__new__(M2T2RobosuiteDemo)
            demo.robot = args.robot
            demo.camera = args.camera
            demo.save_dir = args.save_dir
            demo.use_language = False
            demo._setup_environment()
            from robosuite.maggie.grasp_utils import GraspExecutor
            demo.grasp_executor = GraspExecutor(demo.env, controller_type='OSC_POSE')
        else:
            demo = M2T2RobosuiteDemo(
                checkpoint_path=args.checkpoint,
                robot=args.robot,
                camera=args.camera,
                use_language=use_language,
                device=args.device,
                save_dir=args.save_dir
            )

        demo.run_demo(
            language_prompt=args.language,
            execute_grasp=not args.no_execute,
            enable_keyboard_control=args.keyboard,
            test_mode=args.test_grasp
        )

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'demo' in locals():
            demo.close()


if __name__ == '__main__':
    main()
