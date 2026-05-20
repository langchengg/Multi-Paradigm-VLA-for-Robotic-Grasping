import numpy as np
import pytest


def test_parse_axis_scoped_franka_action_tokens():
    from models.openvla_policy import parse_franka_action

    action = parse_franka_action("dxp06 dyn03 dzz00 axp02 ayn11 azz00 gc")

    np.testing.assert_allclose(
        action,
        np.array([0.30, -0.15, 0.0, 0.10, -0.55, 0.0, 1.0], dtype=np.float32),
        atol=1e-6,
    )


def test_find_latest_adapter_prefers_final_then_latest_checkpoint(tmp_path):
    from models.openvla_policy import find_latest_adapter_dir

    root = tmp_path / "openvla-finetuned"
    checkpoint_2 = root / "checkpoint-2"
    checkpoint_10 = root / "checkpoint-10"
    final = root / "final"
    for path in (checkpoint_2, checkpoint_10, final):
        path.mkdir(parents=True)
        (path / "adapter_config.json").write_text("{}", encoding="utf-8")

    assert find_latest_adapter_dir(root) == final

    (final / "adapter_config.json").unlink()
    assert find_latest_adapter_dir(root) == checkpoint_10


def test_realtime_script_rejects_missing_adapter_before_env_setup():
    from scripts.run_realtime_openvla_mujoco import build_parser, run

    parser = build_parser()
    args = parser.parse_args(["--episodes", "1", "--no-viewer"])

    with pytest.raises(SystemExit, match="Pass --adapter-dir"):
        run(args)
