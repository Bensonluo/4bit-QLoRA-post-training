"""Deployment-only read-only controls; the normal local workbench is unchanged."""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

DISPLAY_ONLY = os.environ.get("TUNESMITH_DISPLAY_ONLY", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _action_widget(name: str, *args: Any, container: Any = None, **kwargs: Any) -> Any:
    target = st if container is None else container
    if DISPLAY_ONLY:
        kwargs["disabled"] = True
        # Disabled widgets must not retain executable callbacks.
        kwargs.pop("on_click", None)
        kwargs.pop("on_change", None)
    result = getattr(target, name)(*args, **kwargs)
    # Enforce the boundary in Python too, rather than trusting UI disabled state.
    return None if DISPLAY_ONLY else result


def action_button(*args: Any, container: Any = None, **kwargs: Any) -> bool:
    return bool(_action_widget("button", *args, container=container, **kwargs))


def action_submit(*args: Any, container: Any = None, **kwargs: Any) -> bool:
    return bool(_action_widget("form_submit_button", *args, container=container, **kwargs))


def action_upload(*args: Any, container: Any = None, **kwargs: Any) -> Any:
    return _action_widget("file_uploader", *args, container=container, **kwargs)


def action_chat_input(*args: Any, container: Any = None, **kwargs: Any) -> Any:
    return _action_widget("chat_input", *args, container=container, **kwargs)
