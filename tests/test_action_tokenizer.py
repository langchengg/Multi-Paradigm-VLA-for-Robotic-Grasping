import torch

from src.models.action_tokenizer import ActionTokenizer


def test_action_tokenizer_round_trip_shape_and_range():
    tokenizer = ActionTokenizer(action_dim=7, num_bins=32)
    actions = torch.linspace(-1, 1, steps=2 * 4 * 7).reshape(2, 4, 7)
    tokens = tokenizer.encode(actions)
    decoded = tokenizer.decode(tokens)
    assert tokens.shape == (2, 4, 7)
    assert decoded.shape == actions.shape
    assert torch.all(decoded <= 1.0)
    assert torch.all(decoded >= -1.0)


def test_action_tokenizer_flatten_unflatten():
    tokenizer = ActionTokenizer(action_dim=7, num_bins=16)
    tokens = torch.randint(0, 16, (3, 5, 7))
    flat = tokenizer.flatten(tokens)
    restored = tokenizer.unflatten(flat, horizon=5)
    assert flat.shape == (3, 35)
    assert torch.equal(tokens, restored)

