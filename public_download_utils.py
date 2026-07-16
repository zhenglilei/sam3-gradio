"""Publish explicit download artifacts without exposing internal runtime files."""

from __future__ import annotations

import os
import shutil
import stat
import time
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias


PUBLIC_DOWNLOAD_CATEGORIES = frozenset(
    {
        "pcs_pvs_exports",
        "layout_mask_exports",
        "region_annotation_exports",
    }
)
DEFAULT_PUBLIC_DOWNLOAD_TTL_SECONDS = 24 * 60 * 60

PathLike: TypeAlias = str | os.PathLike[str]
PublishSources: TypeAlias = Mapping[str, PathLike] | Sequence[PathLike]


class PublicDownloadError(ValueError):
    """Raised when an unsafe or invalid public export is requested."""


def _require_category(category: str) -> str:
    if category not in PUBLIC_DOWNLOAD_CATEGORIES:
        raise PublicDownloadError(f"Unsupported public download category: {category!r}")
    return category


def _require_directory(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise PublicDownloadError(f"{label} must not be a symlink: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise PublicDownloadError(f"{label} is not a directory: {path}")
    return path.resolve(strict=True)


def _public_root(root: PathLike) -> Path:
    return _require_directory(Path(root).expanduser(), "Public download root")


def _category_dir(root: PathLike, category: str) -> Path:
    public_root = _public_root(root)
    return _require_directory(public_root / _require_category(category), "Category directory")


def ensure_public_download_dirs(root: PathLike) -> Path:
    """Create the public root and its fixed category directories."""

    public_root = _public_root(root)
    for category in PUBLIC_DOWNLOAD_CATEGORIES:
        _require_directory(public_root / category, "Category directory")
    return public_root


def _new_export_dir(root: PathLike, category: str) -> Path:
    category_dir = _category_dir(root, category)
    while True:
        export_dir = category_dir / uuid.uuid4().hex
        try:
            export_dir.mkdir(mode=0o700)
        except FileExistsError:
            continue
        return export_dir


def _safe_output_name(name: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "\x00" in name
        or "/" in name
        or "\\" in name
    ):
        raise PublicDownloadError(f"Unsafe public output name: {name!r}")
    return name


def _normalized_sources(sources: PublishSources) -> list[tuple[str, Path]]:
    if isinstance(sources, Mapping):
        entries = [(_safe_output_name(name), Path(source)) for name, source in sources.items()]
    elif isinstance(sources, Sequence) and not isinstance(sources, (str, bytes, os.PathLike)):
        entries = [(_safe_output_name(Path(source).name), Path(source)) for source in sources]
    else:
        raise PublicDownloadError("sources must be a filename mapping or a path sequence")

    if not entries:
        raise PublicDownloadError("At least one source file is required")
    names = [name for name, _ in entries]
    if len(names) != len(set(names)):
        raise PublicDownloadError("Public output filenames must be unique")
    return entries


def _open_regular_file(path: Path):
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise PublicDownloadError(f"Cannot inspect source file: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise PublicDownloadError(f"Source must be a regular non-symlink file: {path}")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PublicDownloadError(f"Cannot open source file safely: {path}") from exc
    try:
        opened_metadata = os.fstat(descriptor)
        if not stat.S_ISREG(opened_metadata.st_mode):
            raise PublicDownloadError(f"Source must be a regular file: {path}")
        return os.fdopen(descriptor, "rb"), opened_metadata
    except Exception:
        os.close(descriptor)
        raise


def _copy_regular_file(source: Path, destination: Path) -> None:
    source_file, metadata = _open_regular_file(source)
    try:
        with source_file, destination.open("xb") as destination_file:
            shutil.copyfileobj(source_file, destination_file)
        os.chmod(destination, stat.S_IMODE(metadata.st_mode))
        os.utime(destination, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def publish_files(root: PathLike, category: str, sources: PublishSources) -> Path:
    """Copy approved source files into a new unguessable public export directory.

    A mapping maps public basenames to source paths. A sequence retains each
    source basename. The returned path is the newly created export directory.
    """

    entries = _normalized_sources(sources)
    export_dir = _new_export_dir(root, category)
    try:
        for output_name, source in entries:
            _copy_regular_file(source, export_dir / output_name)
        return export_dir
    except Exception:
        shutil.rmtree(export_dir, ignore_errors=True)
        raise


def _safe_archive_relative_path(relative_path: Path) -> str:
    parts = relative_path.parts
    if (
        not parts
        or relative_path.is_absolute()
        or any(part in {"", ".", ".."} or "\\" in part or ":" in part for part in parts)
    ):
        raise PublicDownloadError(f"Unsafe ZIP member path: {relative_path}")
    return relative_path.as_posix()


def _zip_sources(source_dir: PathLike) -> list[tuple[Path, str]]:
    lexical_root = Path(source_dir).expanduser()
    if lexical_root.is_symlink():
        raise PublicDownloadError(f"ZIP source directory must not be a symlink: {lexical_root}")
    try:
        resolved_root = lexical_root.resolve(strict=True)
    except OSError as exc:
        raise PublicDownloadError(f"Cannot resolve ZIP source directory: {lexical_root}") from exc
    if not resolved_root.is_dir():
        raise PublicDownloadError(f"ZIP source is not a directory: {resolved_root}")

    files: list[tuple[Path, str]] = []
    for candidate in sorted(resolved_root.rglob("*"), key=lambda item: item.as_posix()):
        try:
            metadata = candidate.lstat()
        except OSError as exc:
            raise PublicDownloadError(f"Cannot inspect ZIP source: {candidate}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise PublicDownloadError(f"ZIP source contains a symlink: {candidate}")

        try:
            resolved_candidate = candidate.resolve(strict=True)
            relative_path = resolved_candidate.relative_to(resolved_root)
        except (OSError, ValueError) as exc:
            raise PublicDownloadError(f"ZIP source escapes its root: {candidate}") from exc
        archive_name = _safe_archive_relative_path(relative_path)

        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise PublicDownloadError(f"ZIP source contains a non-regular file: {candidate}")
        files.append((resolved_candidate, archive_name))

    if not files:
        raise PublicDownloadError("ZIP source directory contains no regular files")
    return files


def publish_zip(
    root: PathLike,
    category: str,
    source_dir: PathLike,
    zip_name: str = "result.zip",
) -> Path:
    """Create a ZIP from safe regular files in a new public export directory."""

    output_name = _safe_output_name(zip_name)
    if not output_name.lower().endswith(".zip"):
        raise PublicDownloadError("Public archive name must end with .zip")
    files = _zip_sources(source_dir)
    export_dir = _new_export_dir(root, category)
    archive_path = export_dir / output_name
    try:
        with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for source, archive_name in files:
                source_file, _ = _open_regular_file(source)
                with source_file:
                    with archive.open(archive_name, "w") as archive_member:
                        shutil.copyfileobj(source_file, archive_member)
        return archive_path
    except Exception:
        shutil.rmtree(export_dir, ignore_errors=True)
        raise


def prune_public_downloads(
    root: PathLike,
    max_age_seconds: float = DEFAULT_PUBLIC_DOWNLOAD_TTL_SECONDS,
    *,
    now: float | None = None,
) -> list[Path]:
    """Remove expired export entries from the fixed public categories only."""

    if isinstance(max_age_seconds, bool) or max_age_seconds <= 0:
        raise PublicDownloadError("max_age_seconds must be positive")
    cutoff = (time.time() if now is None else float(now)) - float(max_age_seconds)
    public_root = ensure_public_download_dirs(root)
    removed: list[Path] = []

    for category in PUBLIC_DOWNLOAD_CATEGORIES:
        category_dir = public_root / category
        for entry in category_dir.iterdir():
            try:
                modified_at = entry.stat(follow_symlinks=False).st_mtime
            except FileNotFoundError:
                continue
            if modified_at >= cutoff:
                continue
            if entry.is_symlink() or not entry.is_dir():
                entry.unlink(missing_ok=True)
            else:
                shutil.rmtree(entry)
            removed.append(entry)
    return removed
