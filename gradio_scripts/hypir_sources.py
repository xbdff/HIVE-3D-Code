import hashlib
from pathlib import Path
from typing import Callable

from huggingface_hub import hf_hub_download


def resolve_hypir_weight(
    repo_id: str,
    filename: str,
    revision: str,
    expected_sha256: str,
    download_file: Callable[..., str] = hf_hub_download,
) -> str:
    """Download a pinned HYPIR weight file and verify its SHA-256."""
    path = download_file(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
    )
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "HYPIR checkpoint SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return path
