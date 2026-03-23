from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_prioritizes_run_order_and_real_outputs():
    source = (ROOT / "README.md").read_text()

    assert "## ✅ Recommended Run Order" in source
    assert "python scripts/run_demo.py --quick" in source
    assert "notebooks/01_env_setup_and_demo.py" in source
    assert "notebooks/02_openvla_qlora_finetune.py" in source
    assert "notebooks/03_flow_matching_eval.py" in source
    assert "data/droid_utils.py" in source
    assert "## 📦 Outputs After Running" in source
    assert "assets_quick.zip" in source
    assert "franka_action_config.json" in source
    assert "results/autoregressive/offline_eval.json" in source
    assert "results/diffusion/offline_eval.json" in source
    assert "results/flow_matching/offline_eval.json" in source
    assert "results/real_offline_summary.json" in source
    assert "results/real_offline_metrics.png" in source
    assert "results/technical_report.md" in source


def test_readme_no_longer_leads_with_embedded_demo_showcase_or_speculative_metrics():
    source = (ROOT / "README.md").read_text()

    assert "## 🎬 Demo" not in source
    assert "期望数据" not in source
    assert "| Expert Policy | Autoregressive | Diffusion | Flow-Matching |" not in source
