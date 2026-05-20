from __future__ import annotations

import numpy as np


def action_mse(pred_actions, expert_actions) -> float:
    pred = np.asarray(pred_actions, dtype=np.float32)
    expert = np.asarray(expert_actions, dtype=np.float32)
    if pred.size == 0 or expert.size == 0:
        return float("nan")
    n = min(len(pred), len(expert))
    return float(np.mean((pred[:n] - expert[:n]) ** 2))


def trajectory_smoothness(actions) -> float:
    actions = np.asarray(actions, dtype=np.float32)
    if len(actions) < 2:
        return 0.0
    return float(np.mean(np.linalg.norm(np.diff(actions, axis=0), axis=-1)))


def gripper_timing_error(pred_actions, expert_actions) -> float:
    pred = np.asarray(pred_actions, dtype=np.float32)
    expert = np.asarray(expert_actions, dtype=np.float32)
    if pred.size == 0 or expert.size == 0:
        return float("nan")

    def first_close(actions):
        close = np.where(actions[:, -1] > 0)[0]
        return int(close[0]) if len(close) else len(actions)

    return float(abs(first_close(pred) - first_close(expert)))


def classify_failure(success: bool, final_lift_height: float, average_xy_error: float) -> str:
    if success:
        return "success"
    if average_xy_error > 0.08:
        return "missed_object"
    if final_lift_height < 0.31:
        return "failed_lift"
    return "unstable_grasp"


def summarize_rollouts(rollouts: list[dict]) -> dict:
    if not rollouts:
        return {}
    success = np.asarray([r["success"] for r in rollouts], dtype=np.float32)
    returns = np.asarray([r["return"] for r in rollouts], dtype=np.float32)
    heights = np.asarray([r["final_object_lift_height"] for r in rollouts], dtype=np.float32)
    latencies = np.asarray([r["inference_latency_ms"] for r in rollouts], dtype=np.float32)
    return {
        "grasp_success_rate": float(success.mean()),
        "average_return": float(returns.mean()),
        "final_object_lift_height": float(heights.mean()),
        "action_mse": float(np.nanmean([r["action_mse"] for r in rollouts])),
        "trajectory_smoothness": float(np.nanmean([r["trajectory_smoothness"] for r in rollouts])),
        "inference_latency_ms": float(latencies.mean()),
        "number_of_inference_steps": float(np.mean([r["number_of_inference_steps"] for r in rollouts])),
        "gripper_timing_error": float(np.nanmean([r["gripper_timing_error"] for r in rollouts])),
        "failure_type_counts": {
            name: int(sum(r["failure_type"] == name for r in rollouts))
            for name in sorted({r["failure_type"] for r in rollouts})
        },
    }

