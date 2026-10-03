"""Lightweight inventory of local configuration and tokenizer inputs."""

from pathlib import Path

_CONFIG_TOKENIZER_SUFFIXES = {".json", ".txt", ".model", ".tiktoken", ".bpe", ".jinja"}


def config_and_tokenizer_files(directory: Path) -> list[Path]:
    """Include Transformers' default and named chat templates as content inputs.

    Inspect only the model root and dedicated named-template directories;
    checkpoints and unrelated descendants do not contribute to model identity.
    """
    files = {
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix in _CONFIG_TOKENIZER_SUFFIXES
    }
    for name in ("additional_chat_templates", "chat_templates"):
        files.update(path for path in (directory / name).glob("*.jinja") if path.is_file())
    return sorted(files)
