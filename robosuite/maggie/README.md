# M2T2 + Robosuite Integration

This package integrates NVIDIA's M2T2 (Multi-Task Masked Transformer) grasp prediction model with Robosuite simulation environments.

## Overview

The integration allows you to:
1. Extract point clouds from Robosuite RGB-D camera observations
2. Use M2T2 to predict 6-DoF grasp poses for objects
3. Execute predicted grasps using Robosuite's controllers
4. (Optional) Switch to keyboard control after grasping

## Files

- `point_cloud_utils.py`: Point cloud extraction and processing utilities
- `m2t2_wrapper.py`: M2T2 model wrapper for grasp prediction
- `grasp_utils.py`: Grasp execution and trajectory generation utilities
- `m2t2_grasp_demo.py`: Main executable demo script
- `__init__.py`: Package initialization
- `README.md`: This file

## Requirements

### Software Dependencies

1. **Robosuite** (already installed)
   ```bash
   pip install robosuite
   ```

2. **M2T2**
   ```bash
   cd ~/research/M2T2
   pip install pointnet2_ops/ --no-build-isolation
   pip install -r requirements.txt
   pip install .
   ```

3. **PyTorch with CUDA** (for M2T2)
   ```bash
   pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
   ```

### Model Checkpoint

Download the M2T2 model checkpoint:
- **Language model**: `m2t2_language.pth` from [Hugging Face](https://huggingface.co/wentao-yuan/m2t2)
- Place it somewhere accessible (e.g., `~/models/m2t2_language.pth`)

## Usage

### Basic Usage

Run the demo with default settings:

```bash
python robosuite/maggie/m2t2_grasp_demo.py --checkpoint /path/to/m2t2_language.pth
```

### Command Line Options

```bash
python robosuite/maggie/m2t2_grasp_demo.py \
    --checkpoint /path/to/m2t2_language.pth \
    --robot Panda \
    --camera agentview \
    --language "grasp the red cube" \
    --device cuda \
    [--no-execute]  # Only predict, don't execute \
    [--keyboard]    # Enable keyboard control after grasp
```

**Arguments:**
- `--checkpoint`: Path to M2T2 .pth checkpoint file (required)
- `--robot`: Robot type (default: Panda)
  - Options: Panda, Sawyer, IIWA, Jaco, Kinova3, UR5e
- `--camera`: Camera name for observations (default: agentview)
- `--language`: Language instruction for M2T2 (default: "grasp the red cube")
- `--device`: Device for inference (default: cuda)
- `--no-execute`: Only predict grasp, don't execute it
- `--keyboard`: Enable keyboard control after grasping

### Example Commands

**Predict and execute grasp:**
```bash
python robosuite/maggie/m2t2_grasp_demo.py \
    --checkpoint ~/models/m2t2_language.pth \
    --language "pick up the cube"
```

**Only predict, don't execute:**
```bash
python robosuite/maggie/m2t2_grasp_demo.py \
    --checkpoint ~/models/m2t2_language.pth \
    --no-execute
```

**Execute grasp, then enable keyboard control:**
```bash
python robosuite/maggie/m2t2_grasp_demo.py \
    --checkpoint ~/models/m2t2_language.pth \
    --keyboard
```

**Use different robot:**
```bash
python robosuite/maggie/m2t2_grasp_demo.py \
    --checkpoint ~/models/m2t2_language.pth \
    --robot Sawyer
```

## How It Works

### Pipeline Overview

1. **Environment Setup**
   - Creates Robosuite Lift environment
   - Enables RGB-D camera observations
   - Initializes OSC_POSE controller for end-effector control

2. **Point Cloud Extraction**
   - Captures RGB and depth images from camera
   - Converts depth to 3D points using camera intrinsics
   - Filters points to workspace bounds
   - Downsamples to 16,384 points (M2T2 requirement)

3. **M2T2 Inference**
   - Prepares point cloud (centering, RGB normalization)
   - Runs M2T2 forward pass
   - Returns grasp poses as 4x4 transformation matrices
   - Selects highest confidence grasp

4. **Grasp Execution**
   - Generates pre-grasp pose (above target)
   - Plans trajectory from current pose to pre-grasp
   - Moves to grasp pose
   - Closes gripper
   - Lifts object

5. **Keyboard Control (Optional)**
   - Switches to manual keyboard control
   - Maintains gripper state
   - Allows user to manipulate robot

### Coordinate Frames

The implementation handles several coordinate frames:

- **Camera Frame**: Where depth is measured
- **World Frame**: Robosuite world coordinates
- **Robot Base Frame**: Robot's coordinate system
- **End-Effector Frame**: Gripper coordinate system

Point clouds are extracted in world frame, M2T2 predicts in world frame, and the executor converts to end-effector control actions.

## Customization

### Adjust Workspace Bounds

Edit `m2t2_grasp_demo.py`, line ~190:

```python
workspace_bounds = {
    'x': [-0.3, 0.3],   # Left-right
    'y': [-0.3, 0.3],   # Front-back
    'z': [0.8, 1.2]     # Height (table is ~0.8)
}
```

### Change Grasp Parameters

In `m2t2_grasp_demo.py`, modify the `execute_grasp_sequence` call:

```python
success = self.grasp_executor.execute_grasp_sequence(
    grasp_pose=grasp_pose,
    pre_grasp_offset=0.1,  # Distance above grasp (meters)
    num_waypoints=30,       # Trajectory smoothness
)
```

### Adjust Controller Gains

Edit `grasp_utils.py`, `GraspExecutor._pose_to_action()`:

```python
def _pose_to_action(self, current_pose, target_pose, gain=5.0):
    # Increase gain for faster movement
    # Decrease gain for smoother movement
```

### Use Different Camera

```bash
python robosuite/maggie/m2t2_grasp_demo.py \
    --checkpoint ~/models/m2t2_language.pth \
    --camera frontview  # or robot0_eye_in_hand
```

## Troubleshooting

### Issue: "No module named 'm2t2'"

**Solution**: Install M2T2:
```bash
cd ~/research/M2T2
pip install .
```

### Issue: "CUDA out of memory"

**Solution**: Use CPU mode:
```bash
python robosuite/maggie/m2t2_grasp_demo.py \
    --checkpoint ~/models/m2t2_language.pth \
    --device cpu
```

### Issue: "No grasp found"

**Solutions**:
1. Lower confidence threshold in code (`mask_thresh=0.05`)
2. Increase number of inference runs (`num_runs=5`)
3. Check point cloud quality (visualize with meshcat)
4. Adjust workspace bounds to include object

### Issue: "Grasp execution fails"

**Solutions**:
1. Increase `num_waypoints` for smoother trajectory
2. Decrease controller `gain` for more stable movement
3. Check that predicted grasp is reachable
4. Adjust pre-grasp offset

### Issue: "Using dummy language embedding"

This is a **warning**, not an error. The language model requires CLIP embeddings, which are not yet implemented. For now, the script uses random embeddings as a placeholder.

**To fix properly**:
```python
from transformers import CLIPTokenizer, CLIPTextModel

def create_clip_embedding(text):
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.detach().numpy()
```

## Known Limitations

1. **Language Embeddings**: Currently using dummy embeddings. Real CLIP embeddings needed for language model.
2. **Camera Intrinsics**: Using estimated intrinsics. May need calibration for accuracy.
3. **Grasp Success Detection**: No explicit grasp success verification.
4. **Single Camera**: Only uses one camera view. Multi-view fusion could improve.
5. **Controller Compatibility**: Tested with OSC_POSE. Other controllers may need adjustment.

## Future Improvements

- [ ] Implement real CLIP language embeddings
- [ ] Add multi-camera point cloud fusion
- [ ] Implement grasp success verification
- [ ] Add collision checking for motion planning
- [ ] Support for placement task (not just picking)
- [ ] Integrate with other Robosuite environments
- [ ] Add visualization of predicted grasps in Robosuite viewer

## Code Structure

### Point Cloud Processing (`point_cloud_utils.py`)

- `depth_to_point_cloud()`: Convert depth image to 3D points
- `extract_point_cloud_from_obs()`: Extract from Robosuite obs dict
- `prepare_point_cloud_for_m2t2()`: Format for M2T2 input

### M2T2 Wrapper (`m2t2_wrapper.py`)

- `M2T2GraspPredictor`: Main model wrapper class
  - `predict_grasps()`: Get all grasps with confidence
  - `get_best_grasp()`: Get single best grasp

### Grasp Execution (`grasp_utils.py`)

- `GraspExecutor`: Grasp execution controller
  - `execute_grasp_sequence()`: Full pick-and-lift sequence
  - `get_current_eef_pose()`: Get robot state
- Helper functions:
  - `create_pre_grasp_pose()`: Generate pre-grasp
  - `interpolate_pose()`: SLERP interpolation
  - `generate_trajectory()`: Trajectory generation

### Main Demo (`m2t2_grasp_demo.py`)

- `M2T2RobosuiteDemo`: Main demo class
  - `capture_and_predict_grasp()`: Point cloud → grasp prediction
  - `run_demo()`: Full demo pipeline

## Contact

For questions or issues, contact Maggie or refer to:
- M2T2: https://github.com/NVlabs/M2T2
- Robosuite: https://github.com/ARISE-Initiative/robosuite

## License

This integration code follows the licenses of the constituent projects:
- M2T2: NVIDIA License
- Robosuite: MIT License
