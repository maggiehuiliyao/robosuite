"""
M2T2 Model Wrapper for Robosuite Integration.
This module provides a simple interface to load and run M2T2 grasp prediction.
"""

import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf


class M2T2GraspPredictor:
    """Wrapper for M2T2 model to predict grasps from point clouds."""

    def __init__(self, checkpoint_path, device='cuda', use_language=False):
        self.device = device
        self.use_language = use_language
        self.checkpoint_path = Path(checkpoint_path)

        from m2t2.m2t2 import M2T2
        from m2t2.dataset import collate
        from m2t2.train_utils import to_cpu, to_gpu
        self.M2T2 = M2T2
        self.collate = collate
        self.to_cpu = to_cpu
        self.to_gpu = to_gpu
        self._load_model()

    def _load_model(self):
        import yaml
        from pathlib import Path

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Use the M2T2 source directory for config files
        m2t2_dir = Path('/home/maggie/research/M2T2')

        # Auto-detect checkpoint type by checking for language-specific keys
        checkpoint_keys = set(checkpoint['model'].keys())
        has_lang_keys = 'transformer.lang_token_proj.weight' in checkpoint_keys
        has_task_embed = 'transformer.task_embed.weight' in checkpoint_keys

        # Determine actual model type from checkpoint architecture
        is_language_checkpoint = has_lang_keys and not has_task_embed

        # Load the appropriate config file based on checkpoint type
        if is_language_checkpoint:
            config_file = m2t2_dir / 'rlbench.yaml'
            print(f"Detected language-conditioned checkpoint")
        else:
            config_file = m2t2_dir / 'config.yaml'
            print(f"Detected non-language checkpoint")

        # Update use_language flag to match checkpoint
        if self.use_language and not is_language_checkpoint:
            print(f"Warning: Language prompt provided but checkpoint is non-language model. Ignoring language.")
            self.use_language = False
        elif not self.use_language and is_language_checkpoint:
            print(f"Note: Using language-conditioned checkpoint without language prompt.")
            self.use_language = True

        # Load YAML config
        with open(config_file, 'r') as f:
            full_config = yaml.safe_load(f)

        config_dict = full_config.get('m2t2', full_config)
        config = OmegaConf.create(config_dict)

        print(f"Loaded config from {config_file}")

        self.model = self.M2T2.from_config(config)
        self.model.load_state_dict(checkpoint['model'])
        self.model = self.model.to(self.device).eval()

        print(f"Loaded M2T2 model from {self.checkpoint_path}")

    def predict_grasps(self, point_cloud_inputs, point_cloud_xyz,
                       language_tokens=None, mask_thresh=0.3,
                       num_runs=1, cam_pose=None, center_offset=None):
        """
        Predict grasp poses from point cloud (pick task only).

        Args:
            point_cloud_inputs: (N, 6) array with centered xyz + normalized rgb
            point_cloud_xyz: (N, 3) array with original xyz coordinates
            language_tokens: Language embedding tokens (for language model)
            mask_thresh: Confidence threshold for filtering grasps
            num_runs: Number of forward passes to aggregate
            cam_pose: 4x4 camera pose matrix (camera-to-world transform)
                     Required when world_coord=False
            center_offset: (3,) array with the offset used to center the point cloud
                          If provided, grasps will be transformed from centered frame to camera frame

        Returns:
            grasps: List of 4x4 grasp transformation matrices
            confidences: List of confidence scores
        """
        # Convert to torch tensors
        inputs_tensor = torch.from_numpy(point_cloud_inputs).float()
        points_tensor = torch.from_numpy(point_cloud_xyz).float()

        # Prepare model input for pick task
        data = {
            'inputs': inputs_tensor,
            'points': points_tensor,
            'task': 'pick'
        }

        # Add camera pose if provided (needed for world_coord=False)
        if cam_pose is not None:
            data['cam_pose'] = torch.from_numpy(cam_pose).float()

        # Add language tokens if provided
        if language_tokens is not None and self.use_language:
            data['lang_tokens'] = torch.from_numpy(language_tokens).float()

        # Model requires these fields even for pick task (uses dummy data like M2T2 dataset.py:100-106)
        data['object_inputs'] = torch.rand(1024, 6)
        data['ee_pose'] = torch.eye(4)
        data['bottom_center'] = torch.zeros(3)
        data['object_center'] = torch.zeros(3)

        # Aggregate multiple runs
        all_grasps = []
        all_confidences = []

        for _ in range(num_runs):
            # Batch the data
            data_batch = self.collate([data])
            self.to_gpu(data_batch)

            # Run inference with complete eval config (from rlbench.yaml)
            eval_config = OmegaConf.create({
                'mask_thresh': mask_thresh,
                'world_coord': False,
                'placement_height': 0.02,  # Required even for pick tasks
                'object_thresh': 0.1  # Threshold for object detection
            })

            with torch.no_grad():
                outputs = self.model.infer(data_batch, eval_config)

            # Move outputs to CPU
            self.to_cpu(outputs)

            # Optional debug output (uncomment for debugging)
            # if _ == 0:  # Only print on first run
            #     print(f"\nDEBUG: M2T2 output keys: {list(outputs.keys())}")
            #     for key in ['grasps', 'grasp_confidence']:
            #         if key in outputs:
            #             val = outputs[key]
            #             if isinstance(val, list) and len(val) > 0:
            #                 print(f"  {key}: {len(val[0][0])} predictions")

            # Extract grasps for pick task
            if 'grasps' in outputs and 'grasp_confidence' in outputs:
                # Handle nested lists (batch x objects x grasps)
                grasps_list = outputs['grasps']
                confs_list = outputs['grasp_confidence']

                # Iterate through batch and objects
                for batch_idx in range(len(grasps_list)):
                    batch_grasps = grasps_list[batch_idx]
                    batch_confs = confs_list[batch_idx]

                    # Iterate through objects
                    for obj_idx in range(len(batch_grasps)):
                        obj_grasps = batch_grasps[obj_idx]  # Shape: (N, 4, 4) or empty
                        obj_confs = batch_confs[obj_idx]    # Shape: (N,) or empty

                        # Convert tensors to lists of individual grasps
                        if torch.is_tensor(obj_grasps) and obj_grasps.shape[0] > 0:
                            for i in range(obj_grasps.shape[0]):
                                all_grasps.append(obj_grasps[i])  # Single (4, 4) grasp
                                all_confidences.append(obj_confs[i])  # Single confidence
                        elif isinstance(obj_grasps, (list, tuple)):
                            all_grasps.extend(obj_grasps)
                            all_confidences.extend(obj_confs)

        # Filter by confidence threshold
        filtered_grasps = []
        filtered_confidences = []

        for grasp, conf in zip(all_grasps, all_confidences):
            # Extract confidence value
            if torch.is_tensor(conf):
                # Handle different tensor shapes
                if conf.numel() == 0:
                    continue  # Skip empty tensors
                elif conf.dim() > 0:
                    conf_val = conf.max().item()
                else:
                    conf_val = conf.item()
            elif isinstance(conf, (list, tuple)):
                if len(conf) == 0:
                    continue
                conf_val = max(conf)
            else:
                conf_val = float(conf)

            if conf_val >= mask_thresh:
                # Convert grasp to numpy
                if torch.is_tensor(grasp):
                    grasp_np = grasp.detach().cpu().numpy()
                elif isinstance(grasp, np.ndarray):
                    grasp_np = grasp
                else:
                    grasp_np = np.array(grasp)

                # Ensure grasp is 4x4 matrix
                if grasp_np.shape == (4, 4):
                    # Transform grasp from centered frame to camera frame if center_offset provided
                    if center_offset is not None:
                        grasp_np = grasp_np.copy()
                        grasp_np[:3, 3] += center_offset  # Add center offset to translation

                    filtered_grasps.append(grasp_np)
                    filtered_confidences.append(conf_val)

        # Sort by confidence (highest first)
        if len(filtered_grasps) > 0:
            sorted_indices = np.argsort(filtered_confidences)[::-1]
            filtered_grasps = [filtered_grasps[i] for i in sorted_indices]
            filtered_confidences = [filtered_confidences[i] for i in sorted_indices]

        return filtered_grasps, filtered_confidences

    def get_best_grasp(self, point_cloud_inputs, point_cloud_xyz,
                       language_tokens=None, mask_thresh=0.3,
                       num_runs=1, auto_reorient=True, cam_pose=None, center_offset=None):
        """
        Get the single best grasp prediction (pick task only).

        Args:
            point_cloud_inputs: (N, 6) array with centered xyz + normalized rgb
            point_cloud_xyz: (N, 3) array with original xyz coordinates
            language_tokens: Language embedding tokens (for language model)
            mask_thresh: Confidence threshold for filtering grasps
            num_runs: Number of forward passes to aggregate
            auto_reorient: Whether to auto-reorient grasp for top-down grasping
            cam_pose: 4x4 camera pose matrix (camera-to-world transform)
                     Required when world_coord=False
            center_offset: (3,) array with the offset used to center the point cloud
                          If provided, grasps will be transformed from centered frame to camera frame

        Returns:
            grasp: 4x4 transformation matrix (or None if no grasps found)
            confidence: Confidence score (or 0.0 if no grasps found)
        """
        grasps, confidences = self.predict_grasps(
            point_cloud_inputs, point_cloud_xyz,
            language_tokens, mask_thresh, num_runs, cam_pose, center_offset
        )

        if len(grasps) > 0:
            original = grasps[0]
            confidence = confidences[0]
            corrected, tag = _auto_reorient_grasp(original, enable=auto_reorient)
            return corrected, confidence
        else:
            return None, 0.0


def _auto_reorient_grasp(grasp_4x4, enable=True):
    """Detect simple frame mismatches and optionally rotate grasp.

    Strategy: evaluate identity plus +/-90 deg yaw; choose the one whose Z-axis
    best aligns with global -Z (top-down grasp). If tie, keep identity.
    """
    import numpy as np

    if not enable or grasp_4x4 is None or getattr(grasp_4x4, 'shape', None) != (4, 4):
        return grasp_4x4, 'identity'

    R = grasp_4x4[:3, :3]
    p = grasp_4x4[:3, 3]

    Rz_p90 = np.array([[0, -1, 0],
                       [1,  0, 0],
                       [0,  0, 1]], dtype=float)
    Rz_m90 = np.array([[0,  1, 0],
                       [-1, 0, 0],
                       [0,  0, 1]], dtype=float)

    candidates = {
        'identity': R,
        'rz+90': Rz_p90 @ R,
        'rz-90': Rz_m90 @ R,
    }

    target_dir = np.array([0, 0, -1.0])
    best_key = None
    best_score = -1
    for k, Rc in candidates.items():
        approach = Rc[:, 2]
        approach_norm = approach / (np.linalg.norm(approach) + 1e-8)
        score = abs(np.dot(approach_norm, target_dir))
        if score > best_score:
            best_score = score
            best_key = k

    if best_key != 'identity':
        corrected = np.eye(4)
        corrected[:3, :3] = candidates[best_key]
        corrected[:3, 3] = p
        return corrected, best_key
    else:
        return grasp_4x4, 'identity'


def create_simple_language_embedding(text, dim=512):
    """
    Create CLIP language embedding for M2T2.

    Args:
        text: Text description (e.g., "grasp the red cube")
        dim: Embedding dimension (should be 512 for CLIP)

    Returns:
        embedding: (77, 512) numpy array
    """
    try:
        from transformers import CLIPTokenizer, CLIPTextModel
        import torch

        # Load CLIP text encoder (cached after first load)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

        # Move to CPU to avoid GPU memory issues (text encoding is fast)
        text_encoder = text_encoder.cpu()
        text_encoder.eval()

        # Tokenize text
        tokens = tokenizer(
            [text],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )

        # Generate embeddings
        with torch.no_grad():
            embeddings = text_encoder(**tokens).last_hidden_state  # Shape: (1, 77, 512)

        # Convert to numpy and remove batch dimension
        embedding = embeddings[0].cpu().numpy()  # Shape: (77, 512)

        print(f"Generated CLIP embedding for: '{text}'")
        return embedding.astype(np.float32)

    except ImportError as e:
        print(f"ERROR: transformers library not installed!")
        print(f"Install with: pip install transformers")
        raise ImportError("transformers library required for CLIP embeddings") from e
    except Exception as e:
        print(f"ERROR generating CLIP embeddings: {e}")
        raise
