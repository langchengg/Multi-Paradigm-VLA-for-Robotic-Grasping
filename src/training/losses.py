from __future__ import annotations


def scalar_loss_dict(info: dict) -> dict:
    return {key: float(value) for key, value in info.items() if isinstance(value, (int, float))}

