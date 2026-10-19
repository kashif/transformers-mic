"""Regression coverage for non-language-model fixture inputs."""

import hashlib
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
safetensors = pytest.importorskip("safetensors.torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "example"))
import build_fixtures  # noqa: E402


def test_weight_named_processor_asset_is_copied(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "attention_mask.safetensors").write_bytes(b"processor tensor")
    (tmp_path / "model.safetensors").write_bytes(b"model weights")
    recipe = {"copy_assets": ["attention_mask.safetensors"]}

    assert "attention_mask.safetensors" in build_fixtures.copied_files(tmp_path, recipe)
    assert "model.safetensors" not in build_fixtures.copied_files(tmp_path, recipe)


@pytest.mark.parametrize("as_buffer", [False, True])
def test_selected_frozen_tensors_survive_random_initialization(
    tmp_path, monkeypatch, as_buffer
):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if as_buffer:
                self.register_buffer("geometry", torch.zeros(2))
            else:
                self.geometry = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
            self.register_buffer("geometry_scratch", torch.zeros(2), persistent=False)
            self.weights = torch.nn.Parameter(torch.zeros(2))

    source = tmp_path / "source"
    source.mkdir()
    safetensors.save_file(
        {"geometry": torch.tensor([2.0, 3.0]), "weights": torch.tensor([8.0, 9.0])},
        source / "model.safetensors",
    )
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda repo, filename, **kwargs: source / filename,
    )
    model = Model()
    hashes = build_fixtures.restore_frozen(
        model,
        source,
        "org/checkpoint",
        "pinned-sha",
        {"preserve_frozen_prefixes": ["geometry"]},
    )
    assert model.geometry.tolist() == [2.0, 3.0]
    assert model.geometry_scratch.tolist() == [0.0, 0.0]
    assert model.weights.tolist() == [0.0, 0.0]
    assert (
        hashes["geometry"]
        == hashlib.sha256(
            torch.tensor([2.0, 3.0]).view(torch.uint8).numpy().tobytes()
        ).hexdigest()
    )
    with pytest.raises(ValueError, match="trainable"):
        build_fixtures.restore_frozen(
            model,
            source,
            "org/checkpoint",
            "pinned-sha",
            {"preserve_frozen_prefixes": ["weights"]},
        )
