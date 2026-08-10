"""Environment-derived runtime policy for the Gradio application."""

from collections.abc import Mapping


GPT_ENV_NAMES = ("GPT_API_KEY", "GPT_BASE_URL", "GPT_MODEL")


def has_gpt_captioning_config(env: Mapping[str, str]) -> bool:
    """Return whether every GPT captioning setting has a non-empty value."""
    return all(bool(env.get(name, "").strip()) for name in GPT_ENV_NAMES)


def gradio_server_name(env: Mapping[str, str]) -> str:
    """Bind privately unless the operator explicitly selects another address."""
    return env.get("GRADIO_SERVER_NAME", "127.0.0.1").strip() or "127.0.0.1"
