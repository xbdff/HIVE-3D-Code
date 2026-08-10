import json
from pathlib import Path

import pytest

from gradio_scripts.scene_config import (
    SceneConfigError,
    load_scene_config,
    validate_scene_assets,
)


def write_config(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_accepts_flat_generated_labels(tmp_path):
    path = write_config(
        tmp_path / "config.json",
        {"labels": ["0", "1"], "scene_tree": {"scene": ["0", "1"]}},
    )
    assert load_scene_config(path, expected_labels=["0", "1"])["labels"] == [
        "0",
        "1",
    ]


@pytest.mark.parametrize("label", ["../secret", "/tmp/secret", "1/../../x", "01"])
def test_rejects_noncanonical_labels(tmp_path, label):
    path = write_config(
        tmp_path / "config.json",
        {"labels": [label], "scene_tree": {"scene": [label]}},
    )
    with pytest.raises(SceneConfigError):
        load_scene_config(path)


def test_rejects_nested_tree_until_recursive_transforms_are_supported(tmp_path):
    path = write_config(
        tmp_path / "config.json",
        {
            "labels": ["0", "1"],
            "scene_tree": {"scene": ["0"], "0": ["1"]},
        },
    )
    with pytest.raises(SceneConfigError, match="flat"):
        load_scene_config(path)


def test_requires_every_asset_inside_run_directory(tmp_path):
    config = {"labels": ["0"], "scene_tree": {"scene": ["0"]}}
    (tmp_path / "scene.png").touch()
    (tmp_path / "0.png").touch()
    with pytest.raises(SceneConfigError, match="0_mask.png"):
        validate_scene_assets(tmp_path, config)
