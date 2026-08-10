"""Validation for scene configuration files and their referenced assets."""

import json
import re
from pathlib import Path
from typing import Iterable


class SceneConfigError(ValueError):
    """Raised when a scene configuration is unsafe or unsupported."""


def _canonical_labels(values: object) -> list[str]:
    if not isinstance(values, list) or not all(
        isinstance(value, str) for value in values
    ):
        raise SceneConfigError("labels must be a list of strings")
    if len(values) != len(set(values)):
        raise SceneConfigError("labels must be unique")
    if any(
        re.fullmatch(r"0|[1-9][0-9]*", value) is None for value in values
    ):
        raise SceneConfigError(
            "labels must be canonical non-negative integer IDs"
        )
    return values


def load_scene_config(
    path: str | Path,
    expected_labels: Iterable[str] | None = None,
) -> dict:
    """Load and normalize a supported flat scene configuration."""
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SceneConfigError(f"invalid scene config: {error}") from error

    if not isinstance(config, dict):
        raise SceneConfigError("scene config must be a JSON object")

    labels = _canonical_labels(config.get("labels"))
    tree = config.get("scene_tree")
    if not isinstance(tree, dict) or set(tree) != {"scene"}:
        raise SceneConfigError(
            "only a flat scene_tree with the 'scene' root is supported"
        )

    children = _canonical_labels(tree["scene"])
    if set(children) != set(labels) or len(children) != len(labels):
        raise SceneConfigError("scene children must match labels exactly")

    if expected_labels is not None:
        expected = list(expected_labels)
        if set(labels) != set(expected) or len(labels) != len(expected):
            raise SceneConfigError(
                "config labels do not match generated object IDs"
            )

    return {"labels": labels, "scene_tree": {"scene": children}}


def validate_scene_assets(run_dir: str | Path, config: dict) -> None:
    """Require all scene assets to exist below the run directory."""
    root = Path(run_dir).resolve()
    names = ["scene.png"]
    for label in config["labels"]:
        names.extend((f"{label}.png", f"{label}_mask.png"))

    for name in names:
        candidate = (root / name).resolve()
        if not candidate.is_relative_to(root):
            raise SceneConfigError(f"asset escapes run directory: {name}")
        if not candidate.is_file():
            raise SceneConfigError(f"missing scene asset: {name}")
