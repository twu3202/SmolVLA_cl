"""
LIBERO-specific SmolVLA configuration.

LIBERO's robot arm has a 7-DOF action space (6 joints + gripper) and
provides two camera views. This config adapts SmolVLA's feature specification
to match LIBERO, suitable for either fine-tuning or from-scratch training.

State vector (14-dim):
    - eef_pos  (3): end-effector XYZ position
    - eef_quat (4): end-effector orientation quaternion
    - joint_pos (7): joint positions

Action vector (7-dim):
    - delta_xyz (3): end-effector position delta
    - delta_rpy (3): end-effector rotation delta
    - gripper   (1): gripper open/close
"""

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

# Observation image keys produced by LiberoEnv
# (mapped from agentview_image → image, robot0_eye_in_hand_image → image2)
LIBERO_AGENTVIEW_KEY = "observation.images.image"
LIBERO_WRIST_KEY = "observation.images.image2"
LIBERO_STATE_KEY = "observation.state"
LIBERO_ACTION_KEY = "action"

STATE_DIM = 14   # eef_pos(3) + eef_quat(4) + joint_pos(7)
ACTION_DIM = 7   # delta_xyz(3) + delta_rpy(3) + gripper(1)
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256


def make_libero_smolvla_config(device: str = "mps") -> SmolVLAConfig:
    """Return a SmolVLAConfig adapted for LIBERO."""
    return SmolVLAConfig(
        # --- Feature specification ---
        input_features={
            LIBERO_STATE_KEY: PolicyFeature(
                type=FeatureType.STATE,
                shape=(STATE_DIM,),
            ),
            LIBERO_AGENTVIEW_KEY: PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, IMAGE_HEIGHT, IMAGE_WIDTH),
            ),
            LIBERO_WRIST_KEY: PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, IMAGE_HEIGHT, IMAGE_WIDTH),
            ),
        },
        output_features={
            LIBERO_ACTION_KEY: PolicyFeature(
                type=FeatureType.ACTION,
                shape=(ACTION_DIM,),
            ),
        },
        # --- Normalization ---
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
        # --- Architecture ---
        vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        load_vlm_weights=True,        # load VLM backbone from HuggingFace
        freeze_vision_encoder=True,   # freeze visual encoder for cheap demo
        train_expert_only=True,
        max_state_dim=32,
        max_action_dim=32,
        # --- Inference ---
        n_obs_steps=1,
        chunk_size=50,
        n_action_steps=50,
        num_steps=10,                 # flow matching denoising steps
        resize_imgs_with_padding=(512, 512),
        tokenizer_max_length=48,
        # --- Device ---
        device=device,
    )
