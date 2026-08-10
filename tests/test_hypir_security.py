import hashlib

import pytest
import torch

from gradio_scripts.hypir_sources import resolve_hypir_weight
from gradio_scripts.runtime_config import (
    gradio_server_name,
    has_gpt_captioning_config,
)
from gradio_scripts.safe_checkpoint import load_tensor_state_dict


class UnsafePayload:
    def __reduce__(self):
        return (eval, ("40 + 2",))


def test_restricted_loader_rejects_pickled_object(tmp_path):
    path = tmp_path / "unsafe.pth"
    torch.save(UnsafePayload(), path)
    with pytest.raises(Exception):
        load_tensor_state_dict(path)


def test_restricted_loader_accepts_tensor_state_dict(tmp_path):
    path = tmp_path / "weights.pth"
    torch.save({"layer.weight": torch.ones(2)}, path)
    loaded = load_tensor_state_dict(path)
    assert torch.equal(loaded["layer.weight"], torch.ones(2))


def test_download_is_revision_pinned_and_checksum_verified(tmp_path):
    path = tmp_path / "weight.pth"
    path.write_bytes(b"known")
    expected = hashlib.sha256(b"known").hexdigest()
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(path)

    assert (
        resolve_hypir_weight(
            "org/model", "weight.pth", "abc123", expected, download
        )
        == str(path)
    )
    assert calls == [
        {
            "repo_id": "org/model",
            "filename": "weight.pth",
            "revision": "abc123",
        }
    ]


def test_download_rejects_checksum_mismatch(tmp_path):
    path = tmp_path / "weight.pth"
    path.write_bytes(b"tampered")

    def download(**kwargs):
        return str(path)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        resolve_hypir_weight(
            "org/model", "weight.pth", "abc123", "0" * 64, download
        )


def test_runtime_defaults_are_private_and_gpt_requires_all_values():
    assert gradio_server_name({}) == "127.0.0.1"
    assert not has_gpt_captioning_config({"GPT_API_KEY": "x"})
    assert has_gpt_captioning_config(
        {
            "GPT_API_KEY": "x",
            "GPT_BASE_URL": "u",
            "GPT_MODEL": "m",
        }
    )
