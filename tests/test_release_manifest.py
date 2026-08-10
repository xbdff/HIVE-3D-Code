import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def git(*args: str, git_dir: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(GIT_DIR=str(git_dir), GIT_WORK_TREE=str(ROOT))
    return subprocess.run(
        ["git", *args], check=False, text=True, capture_output=True, env=env
    )


def test_unreleased_and_generated_files_are_ignored(tmp_path: Path):
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    git_dir = repo / ".git"
    excluded = [
        "hive_3d/experiment_clean_amodal.py",
        "hive_3d/experiment_helpers_amodal.py",
        "outputs/764435e1/result_hive3d.mp4",
        "gradio_scripts/__pycache__/grounding_sam.cpython-310.pyc",
        "docs/superpowers/specs/2026-08-10-code-release-design.md",
        "assets/example_image/T.png",
        "assets/example_multi_image/character_1.png",
    ]
    for path in excluded:
        result = git("check-ignore", "-q", path, git_dir=git_dir)
        assert result.returncode == 0, path


def test_core_runtime_files_are_not_ignored(tmp_path: Path):
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    git_dir = repo / ".git"
    for path in ["hive_3d/helper.py", "gradio_scripts/gradio_app.py"]:
        result = git("check-ignore", "-q", path, git_dir=git_dir)
        assert result.returncode == 1, path
