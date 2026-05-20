import types
import importlib.util
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parent.parent))


MODULE_PATH = Path(__file__).resolve().parents[1] / "models" / "flow_matching_head.py"
SPEC = importlib.util.spec_from_file_location("test_flow_matching_head_module", MODULE_PATH)
FLOW_MATCHING_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(FLOW_MATCHING_MODULE)
FlowMatchingVLA = FLOW_MATCHING_MODULE.FlowMatchingVLA


def test_flow_matching_vla_uses_explicit_bert_loader(monkeypatch):
    calls = []

    class FakeTokenBatch(dict):
        def to(self, _device):
            return self

    class FakeViTModel(torch.nn.Module):
        config = types.SimpleNamespace(hidden_size=768)

        @classmethod
        def from_pretrained(cls, name):
            calls.append(("vit", name))
            return cls()

        def forward(self, pixel_values):
            batch = pixel_values.shape[0]
            hidden = torch.zeros(batch, 1, self.config.hidden_size, device=pixel_values.device)
            return types.SimpleNamespace(last_hidden_state=hidden)

    class FakeBertModel(torch.nn.Module):
        config = types.SimpleNamespace(hidden_size=512)

        @classmethod
        def from_pretrained(cls, name):
            calls.append(("bert_model", name))
            return cls()

        def forward(self, input_ids=None, attention_mask=None):
            batch = input_ids.shape[0]
            hidden = torch.zeros(batch, 1, self.config.hidden_size, device=input_ids.device)
            return types.SimpleNamespace(last_hidden_state=hidden)

    class FakeBertTokenizer:
        @classmethod
        def from_pretrained(cls, name):
            calls.append(("bert_tokenizer", name))
            return cls()

        def __call__(self, instructions, **_kwargs):
            batch = len(instructions)
            return FakeTokenBatch(
                {
                    "input_ids": torch.ones(batch, 4, dtype=torch.long),
                    "attention_mask": torch.ones(batch, 4, dtype=torch.long),
                }
            )

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.ViTModel = FakeViTModel
    fake_transformers.BertModel = FakeBertModel
    fake_transformers.BertTokenizer = FakeBertTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    model = FlowMatchingVLA(action_dim=7, action_horizon=4, flow_hidden_dim=64, flow_num_layers=2)

    images = torch.randn(2, 3, 224, 224)
    actions_gt = torch.randn(2, 4, 7)
    loss, info = model(images, ["pick up red cube", "lift blue block"], actions_gt)
    pred = model.predict(images, ["pick up red cube", "lift blue block"], num_steps=2)

    assert calls[:3] == [
        ("vit", "google/vit-base-patch16-224"),
        ("bert_model", "prajjwal1/bert-small"),
        ("bert_tokenizer", "prajjwal1/bert-small"),
    ]
    assert torch.is_tensor(loss)
    assert "flow_loss" in info
    assert pred.shape == (2, 4, 7)


def test_flow_matching_notebook_uses_explicit_bert_loader():
    notebook_path = (
        Path(__file__).resolve().parents[1] / "notebooks" / "03_flow_matching_eval.py"
    )
    source = notebook_path.read_text()

    assert "BertModel.from_pretrained(\"prajjwal1/bert-small\")" in source
    assert "BertTokenizer.from_pretrained(\"prajjwal1/bert-small\")" in source
    assert "AutoModel.from_pretrained(\"prajjwal1/bert-small\")" not in source
