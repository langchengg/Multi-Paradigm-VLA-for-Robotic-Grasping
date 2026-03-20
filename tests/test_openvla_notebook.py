from pathlib import Path


def test_openvla_notebook_pins_compatible_transformers_stack():
    notebook_path = (
        Path(__file__).resolve().parents[1] / "notebooks" / "02_openvla_qlora_finetune.py"
    )
    source = notebook_path.read_text()

    assert '"transformers==4.40.1"' in source
    assert '"tokenizers==0.19.1"' in source
    assert '"timm==0.9.10"' in source
    assert '"--upgrade"' in source


def test_openvla_notebook_uses_vision_model_fallback_and_real_demo_key():
    notebook_path = (
        Path(__file__).resolve().parents[1] / "notebooks" / "02_openvla_qlora_finetune.py"
    )
    source = notebook_path.read_text()

    assert "AutoModelForVision2Seq as OpenVLAModelClass" in source
    assert "AutoModelForImageTextToText as OpenVLAModelClass" in source
    assert 'model = OpenVLAModelClass.from_pretrained(' in source
    assert 'instructions = data["instructions"]' in source
    assert 'data["instruction"]' not in source
