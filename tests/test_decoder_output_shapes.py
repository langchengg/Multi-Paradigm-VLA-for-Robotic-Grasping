import torch

from src.models.policy import VLAPolicy


def _batch(batch_size=2, horizon=4):
    return {
        "image": torch.zeros(batch_size, 3, 32, 32),
        "instruction": ["pick up the red cube"] * batch_size,
        "robot_state": torch.zeros(batch_size, 36),
        "action_chunk": torch.zeros(batch_size, horizon, 7),
    }


def test_decoder_output_shapes_for_all_decoders():
    for decoder in ["autoregressive", "diffusion", "flow_matching"]:
        policy = VLAPolicy(
            decoder_type=decoder,
            robot_state_dim=36,
            horizon=4,
            pretrained_clip=False,
            tiny_random_clip=True,
            decoder_hidden_dim=64,
            decoder_num_layers=2,
            diffusion_train_steps=8,
            inference_steps=2,
            num_action_bins=32,
        )
        batch = _batch(horizon=4)
        loss, info = policy.training_loss(batch)
        assert loss.shape == ()
        assert isinstance(info, dict)
        actions = policy.predict_action_chunk(batch)
        assert actions.shape == (2, 4, 7)
