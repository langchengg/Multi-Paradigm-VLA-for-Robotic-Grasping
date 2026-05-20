from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_documents_clean_clip_benchmark_entrypoints():
    source = (ROOT / "README.md").read_text()

    assert "CLIP is frozen" in source
    assert "python -m src.data.generate_synthetic_demos" in source
    assert "python -m src.training.train" in source
    assert "--decoder autoregressive" in source
    assert "--decoder diffusion" in source
    assert "--decoder flow_matching" in source
    assert "python -m src.visualization.live_watch" in source
    assert "mjpython -m src.visualization.live_watch" in source
    assert "bash scripts/smoke_test_mac.sh" in source
    assert "src/data/libero_adapter.py" in source


def test_readme_no_longer_leads_with_embedded_demo_showcase_or_speculative_metrics():
    source = (ROOT / "README.md").read_text()

    assert "## 🎬 Demo" not in source
    assert "期望数据" not in source
    assert "| Expert Policy | Autoregressive | Diffusion | Flow-Matching |" not in source
    assert "python scripts/run_kaggle_pipeline.py" not in source
