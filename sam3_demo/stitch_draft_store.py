"""Small, session-independent persistence store for stitch drafts.

Only stitch business data is persisted.  Server session identity
(``session_id`` and ``owner_token``) is never written or returned by
``load``.  Input tiles are copied into a per-draft directory as PNG files;
the manifest is the commit point for a save, so a failed save leaves the
previous manifest readable.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import threading
import time
import uuid
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from PIL import Image


_RESUME_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_KEYS = frozenset({"session_id", "owner_token", "resume_id"})
_DERIVED_IMAGE_KEYS = frozenset({"mosaic", "mosaic_full"})
_MANIFEST_NAME = "manifest.json"
_STORE_SCHEMA_VERSION = 1


class StitchDraftStore:
    """Persist one independent stitch draft per validated resume id.

    ``ttl_seconds`` is measured from the latest successful save.  ``clock``
    exists so callers and tests can provide a wall-clock substitute; it must
    return a finite numeric value in seconds.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        ttl_seconds: float,
        clock: Callable[[], float] | None = None,
    ) -> None:
        try:
            ttl = float(ttl_seconds)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("ttl_seconds must be a finite non-negative number") from exc
        if not math.isfinite(ttl) or ttl < 0:
            raise ValueError("ttl_seconds must be a finite non-negative number")
        self.root = Path(root).expanduser()
        self.ttl_seconds = ttl
        self.clock = clock or time.time
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            raise ValueError("draft root must be a directory")

    @staticmethod
    def _validate_resume_id(resume_id: Any) -> str:
        if not isinstance(resume_id, str) or _RESUME_ID_RE.fullmatch(resume_id) is None:
            raise ValueError("invalid resume_id")
        return resume_id

    def _now(self) -> float:
        try:
            value = float(self.clock())
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("clock must return a finite number") from exc
        if not math.isfinite(value):
            raise ValueError("clock must return a finite number")
        return value

    def _root_resolved(self) -> Path:
        root = self.root.absolute()
        try:
            resolved = root.resolve(strict=True)
        except OSError as exc:
            raise ValueError("draft root is not accessible") from exc
        if resolved != root:
            raise ValueError("draft root must not be a symlink")
        return resolved

    def _draft_path(self, resume_id: Any) -> Path:
        resume_id = self._validate_resume_id(resume_id)
        root = self._root_resolved()
        path = root / resume_id
        if path.parent != root:
            raise ValueError("invalid resume_id path")
        if path.is_symlink():
            raise ValueError("draft path must not be a symlink")
        try:
            resolved = path.resolve(strict=False)
        except OSError as exc:
            raise ValueError("invalid draft path") from exc
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError("draft path is outside the store root") from exc
        return path

    @staticmethod
    def _json_value(value: Any) -> Any:
        """Convert ordinary containers to JSON and reject opaque data."""

        if value is None or isinstance(value, (str, bool, int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("draft state contains a non-finite number")
            return value
        if isinstance(value, (list, tuple)):
            return [StitchDraftStore._json_value(item) for item in value]
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError("draft state mapping keys must be strings")
                if key in _IDENTITY_KEYS:
                    continue
                result[key] = StitchDraftStore._json_value(item)
            return result
        raise ValueError("draft state contains unsupported non-JSON data")

    @staticmethod
    def _relative_image_path(value: Any) -> str | None:
        if not isinstance(value, str) or not value:
            return None
        # Reject Windows separators too; a backslash must never become a path
        # separator if the manifest is moved to another host.
        if "\\" in value or "\x00" in value:
            return None
        path = PurePosixPath(value)
        if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
            return None
        if not path.parts or path.parts[0] != "images":
            return None
        return path.as_posix()

    def _safe_manifest_path(self, draft: Path) -> Path | None:
        if draft.is_symlink() or not draft.is_dir():
            return None
        try:
            root = self._root_resolved()
            draft_resolved = draft.resolve(strict=True)
            draft_resolved.relative_to(root)
        except (OSError, ValueError):
            return None
        if draft_resolved.parent != root:
            return None
        manifest = draft / _MANIFEST_NAME
        if manifest.is_symlink() or not manifest.is_file():
            return None
        try:
            manifest.resolve(strict=True).relative_to(draft_resolved)
        except (OSError, ValueError):
            return None
        return manifest

    @staticmethod
    def _image_items(state: Mapping[str, Any]) -> tuple[str, list[Image.Image]]:
        key = "images" if "images" in state else "input_tiles" if "input_tiles" in state else "images"
        raw = state.get(key, [])
        if raw is None:
            raw = []
        if not isinstance(raw, (list, tuple)):
            raise ValueError("draft images must be a list")
        images: list[Image.Image] = []
        for image in raw:
            if not isinstance(image, Image.Image):
                raise ValueError("draft images must be PIL images")
            images.append(image.copy())
        return key, images

    @staticmethod
    def _strip_identity(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: StitchDraftStore._strip_identity(item)
                for key, item in value.items()
                if key not in _IDENTITY_KEYS
            }
        if isinstance(value, list):
            return [StitchDraftStore._strip_identity(item) for item in value]
        return value

    def _state_for_save(self, state: Any) -> tuple[str, list[Image.Image], dict[str, Any]]:
        if not isinstance(state, Mapping):
            raise ValueError("draft state must be a mapping")
        image_key, images = self._image_items(state)
        serializable: dict[str, Any] = {}
        for key, value in state.items():
            if key in _IDENTITY_KEYS or key in _DERIVED_IMAGE_KEYS:
                continue
            if key == image_key or key in {"generated_revision", "mosaic_view_revision"}:
                continue
            serializable[key] = self._json_value(value)
        serializable[image_key] = []
        return image_key, images, self._strip_identity(serializable)

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

    @staticmethod
    def _cleanup(path: Path) -> None:
        try:
            if path.is_symlink() or path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                shutil.rmtree(path)
        except OSError:
            pass

    def save(self, resume_id: str, state: Mapping[str, Any]) -> dict[str, Any]:
        """Atomically persist *state* and return a small save summary."""

        with self._lock:
            draft = self._draft_path(resume_id)
            image_key, images, serializable = self._state_for_save(state)
            now = self._now()
            draft.mkdir(parents=True, exist_ok=True)
            if draft.is_symlink() or not draft.is_dir():
                raise ValueError("draft path must be a directory")
            images_dir = draft / "images"
            if images_dir.exists() and (images_dir.is_symlink() or not images_dir.is_dir()):
                raise ValueError("draft images path must be a directory")
            images_dir.mkdir(exist_ok=True)

            token = uuid.uuid4().hex
            staging = draft / f".staging-{token}"
            staging_images = staging / "images"
            staging.mkdir()
            staging_images.mkdir()
            image_files: list[str] = []
            try:
                for index, image in enumerate(images):
                    filename = f"{token}-tile-{index:06d}.png"
                    staged_path = staging_images / filename
                    image.save(staged_path, format="PNG")
                    with staged_path.open("rb") as handle:
                        os.fsync(handle.fileno())
                    image_files.append(f"images/{filename}")

                serializable[image_key] = []
                manifest = {
                    "schema_version": _STORE_SCHEMA_VERSION,
                    "updated_at": now,
                    "state": serializable,
                    "images": image_files,
                }
                staged_manifest = staging / _MANIFEST_NAME
                self._write_json(staged_manifest, manifest)
                self._fsync_directory(staging_images)
                self._fsync_directory(staging)

                for relative in image_files:
                    filename = relative.split("/", 1)[1]
                    os.replace(staging_images / filename, images_dir / filename)
                # This replace is the save commit point.  If it fails, the
                # previous manifest still points at the previous image set.
                os.replace(staged_manifest, draft / _MANIFEST_NAME)
                self._fsync_directory(images_dir)
                self._fsync_directory(draft)

                # Old PNGs are unreachable after the manifest commit.  Their
                # deletion is best-effort and cannot invalidate the new draft.
                referenced = set(image_files)
                try:
                    candidates = list(images_dir.iterdir())
                except OSError:
                    candidates = []
                for candidate in candidates:
                    if candidate.is_symlink() or not candidate.is_file():
                        continue
                    if f"images/{candidate.name}" not in referenced:
                        self._cleanup(candidate)
            finally:
                self._cleanup(staging)

            return {
                "resume_id": resume_id,
                "revision": serializable.get("revision", 0),
                "updated_at": now,
            }

    def load(self, resume_id: str) -> dict[str, Any] | None:
        """Load a draft, returning ``None`` for missing or invalid contents."""

        with self._lock:
            draft = self._draft_path(resume_id)
            manifest_path = self._safe_manifest_path(draft)
            if manifest_path is None:
                return None
            try:
                with manifest_path.open("r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
                if not isinstance(manifest, Mapping) or manifest.get("schema_version") != _STORE_SCHEMA_VERSION:
                    return None
                updated_at = float(manifest["updated_at"])
                if not math.isfinite(updated_at) or self._now() - updated_at > self.ttl_seconds:
                    self._delete_path(draft)
                    return None
                state = manifest.get("state")
                image_files = manifest.get("images")
                if not isinstance(state, Mapping) or not isinstance(image_files, list):
                    return None
                loaded_images: list[Image.Image] = []
                draft_resolved = draft.resolve(strict=True)
                for raw_path in image_files:
                    relative = self._relative_image_path(raw_path)
                    if relative is None:
                        return None
                    image_path = draft / Path(*relative.split("/"))
                    if image_path.is_symlink() or not image_path.is_file():
                        return None
                    try:
                        image_path.resolve(strict=True).relative_to(draft_resolved)
                        with Image.open(image_path) as image:
                            image.verify()
                        with Image.open(image_path) as image:
                            loaded_images.append(image.copy())
                    except Exception:
                        return None
                result = self._strip_identity(dict(state))
                if not isinstance(result, dict):
                    return None
                result["images"] = loaded_images
                result.pop("input_tiles", None)
                return result
            except Exception:
                return None

    def _delete_path(self, draft: Path) -> bool:
        if draft.is_symlink() or not draft.exists():
            return False
        try:
            root = self._root_resolved()
            draft.resolve(strict=True).relative_to(root)
        except (OSError, ValueError):
            return False
        self._cleanup(draft)
        return not draft.exists()

    def delete(self, resume_id: str) -> bool:
        """Delete one draft without ever following a draft symlink."""

        with self._lock:
            return self._delete_path(self._draft_path(resume_id))

    def prune(self) -> list[str]:
        """Delete expired drafts and return the resume ids actually removed."""

        with self._lock:
            now = self._now()
            removed: list[str] = []
            try:
                entries = list(self.root.iterdir())
            except OSError:
                return removed
            for draft in entries:
                if draft.is_symlink() or not draft.is_dir() or _RESUME_ID_RE.fullmatch(draft.name) is None:
                    continue
                manifest_path = self._safe_manifest_path(draft)
                if manifest_path is None:
                    continue
                try:
                    with manifest_path.open("r", encoding="utf-8") as handle:
                        manifest = json.load(handle)
                    updated_at = float(manifest["updated_at"])
                except Exception:
                    continue
                if not math.isfinite(updated_at) or now - updated_at <= self.ttl_seconds:
                    continue
                if self._delete_path(draft):
                    removed.append(draft.name)
            return removed


__all__ = ["StitchDraftStore"]
