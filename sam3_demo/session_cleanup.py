"""Best-effort cleanup of resources owned by one server session.

This module intentionally has no Gradio or model-runtime dependencies.  It is
safe to call from a session registry cleanup callback and is deliberately
explicit about failures so that a failed cache cleanup cannot hide a failed
filesystem cleanup.
"""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping


_SESSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def validate_server_session_id(value: Any) -> str:
    """Return *value* when it is exactly a server-issued session id.

    Session ids are deliberately narrower than a generic UUID: this prevents
    separators, ``..`` components, uppercase aliases, and other path/key
    ambiguities from reaching cleanup code.
    """

    if not isinstance(value, str) or _SESSION_ID_RE.fullmatch(value) is None:
        raise ValueError("invalid server session id")
    return value


def _error_text(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _call_cache_clear(
    name: str,
    callback: Callable[[str], Any],
    session_id: str,
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    try:
        result = callback(session_id)
    except Exception as exc:  # cleanup is intentionally best effort
        error = {"resource": name, "error": _error_text(exc)}
        errors.append(error)
        return {"ok": False, "result": None, "error": error["error"]}
    return {"ok": True, "result": result}


def _remove_layout_keys(
    layout_cache: MutableMapping[Any, Any],
    layout_cache_lock: Any,
    session_id: str,
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    removed: list[Any] = []
    try:
        with layout_cache_lock:
            keys = list(layout_cache.keys())
            for key in keys:
                if key == session_id or (
                    isinstance(key, str) and key.startswith(session_id + ":")
                ):
                    del layout_cache[key]
                    removed.append(key)
    except Exception as exc:  # retain already removed keys in the report
        error = {"resource": "layout_cache", "error": _error_text(exc)}
        errors.append(error)
        return {"removed": removed, "count": len(removed), "error": error["error"]}
    return {"removed": removed, "count": len(removed)}


def _remove_prompt_epoch(
    prompt_epochs: MutableMapping[str, Any],
    prompt_epoch_lock: Any,
    session_id: str,
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    removed = False
    try:
        with prompt_epoch_lock:
            if session_id in prompt_epochs:
                del prompt_epochs[session_id]
                removed = True
    except Exception as exc:
        error = {"resource": "prompt_epochs", "error": _error_text(exc)}
        errors.append(error)
        return {"removed": removed, "count": int(removed), "error": error["error"]}
    return {"removed": removed, "count": int(removed)}


def _safe_remove_tree(path: Path, root: Path) -> tuple[list[str], list[str]]:
    """Remove one direct child of root using no-follow directory descriptors."""

    removed: list[str] = []
    errors: list[str] = []
    root_abs = root.absolute()
    path_abs = path.absolute()
    if path_abs.parent != root_abs:
        return removed, ["session path is outside persistent root"]

    try:
        root_resolved = root_abs.resolve(strict=True)
    except FileNotFoundError:
        return removed, errors
    except OSError as exc:
        return removed, [_error_text(exc)]
    if root_resolved != root_abs:
        return removed, ["persistent root or ancestor must not be a symlink"]

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        root_fd = os.open(root_abs, directory_flags)
    except FileNotFoundError:
        return removed, errors
    except OSError as exc:
        return removed, [_error_text(exc)]

    def remove_entry(parent_fd: int, name: str, display_path: Path) -> None:
        try:
            info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        except OSError as exc:
            errors.append(f"{display_path}: {_error_text(exc)}")
            return

        if not stat.S_ISDIR(info.st_mode):
            try:
                os.unlink(name, dir_fd=parent_fd)
                removed.append(str(display_path))
            except OSError as exc:
                errors.append(f"{display_path}: {_error_text(exc)}")
            return

        try:
            child_fd = os.open(name, directory_flags, dir_fd=parent_fd)
        except OSError as exc:
            errors.append(f"{display_path}: {_error_text(exc)}")
            return
        try:
            try:
                child_names = os.listdir(child_fd)
            except OSError as exc:
                errors.append(f"{display_path}: {_error_text(exc)}")
                return
            for child_name in child_names:
                remove_entry(child_fd, child_name, display_path / child_name)
        finally:
            os.close(child_fd)

        try:
            os.rmdir(name, dir_fd=parent_fd)
            removed.append(str(display_path))
        except FileNotFoundError:
            return
        except OSError as exc:
            errors.append(f"{display_path}: {_error_text(exc)}")

    try:
        remove_entry(root_fd, path_abs.name, path_abs)
    finally:
        os.close(root_fd)
    return removed, errors


def cleanup_session_resources(
    session_id: str,
    *,
    clear_workspace_cache: Callable[[str], Any],
    clear_source_image_cache: Callable[[str], Any],
    layout_cache: MutableMapping[Any, Any],
    layout_cache_lock: Any,
    prompt_epochs: MutableMapping[str, Any],
    prompt_epoch_lock: Any,
    persistent_roots: Iterable[Any] = (),
) -> dict[str, Any]:
    """Clean all resources belonging to one validated server session.

    Each resource is attempted independently.  The result always contains
    per-resource details and a flat ``errors`` list for callers/observability.
    """

    session_id = validate_server_session_id(session_id)
    errors: list[dict[str, str]] = []

    workspace = _call_cache_clear(
        "workspace_cache", clear_workspace_cache, session_id, errors
    )
    source_images = _call_cache_clear(
        "source_image_cache", clear_source_image_cache, session_id, errors
    )
    layout = _remove_layout_keys(layout_cache, layout_cache_lock, session_id, errors)
    epochs = _remove_prompt_epoch(prompt_epochs, prompt_epoch_lock, session_id, errors)

    persistent: list[dict[str, Any]] = []
    for raw_root in persistent_roots:
        try:
            root = Path(raw_root)
            session_path = root / session_id
            removed, path_errors = _safe_remove_tree(session_path, root)
            item: dict[str, Any] = {
                "root": str(root),
                "path": str(session_path),
                "removed": removed,
                "count": len(removed),
            }
            if path_errors:
                item["errors"] = path_errors
                for error in path_errors:
                    errors.append({
                        "resource": "persistent_root",
                        "path": str(session_path),
                        "error": error,
                    })
            persistent.append(item)
        except Exception as exc:
            error = {"resource": "persistent_root", "error": _error_text(exc)}
            errors.append(error)
            persistent.append({
                "root": str(raw_root),
                "path": None,
                "removed": [],
                "count": 0,
                "errors": [error["error"]],
            })

    return {
        "session_id": session_id,
        "workspace_cache": workspace,
        "source_image_cache": source_images,
        "layout_cache": layout,
        "prompt_epochs": epochs,
        "persistent_roots": persistent,
        "errors": errors,
    }
