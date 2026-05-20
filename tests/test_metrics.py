import numpy as np

from src.training.metrics import action_mse, gripper_timing_error, summarize_rollouts, trajectory_smoothness


def test_action_mse_and_smoothness():
    pred = np.array([[0, 0, 0, 0, 0, 0, -1], [1, 0, 0, 0, 0, 0, 1]], dtype=np.float32)
    expert = np.zeros((2, 7), dtype=np.float32)
    assert action_mse(pred, expert) > 0
    assert trajectory_smoothness(pred) > 0


def test_gripper_timing_error():
    pred = np.array([[-1], [-1], [1]], dtype=np.float32)
    expert = np.array([[-1], [1], [1]], dtype=np.float32)
    assert gripper_timing_error(pred, expert) == 1.0


def test_summarize_rollouts():
    summary = summarize_rollouts([
        {
            "success": True,
            "return": 1.0,
            "final_object_lift_height": 0.4,
            "action_mse": 0.1,
            "trajectory_smoothness": 0.2,
            "inference_latency_ms": 3.0,
            "number_of_inference_steps": 2,
            "gripper_timing_error": 1.0,
            "failure_type": "success",
        }
    ])
    assert summary["grasp_success_rate"] == 1.0
    assert summary["failure_type_counts"]["success"] == 1

