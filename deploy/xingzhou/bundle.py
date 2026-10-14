"""Package a reviewed Git commit into the flat /src production deployment ZIP."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import tempfile
import zipfile

COMPONENTS = ("image_gesture_overlay", "layout_transform_editor", "layout_region_annotator",
              "stitch_preview_canvas", "repair_mask_editor")
MODELS = ("models/sam3.pt", "models/big-lama.pt")
REQUIRED = ("run.sh", "sam3_demo/app.py", "sam3_demo/config.py",
            "assets/bpe_simple_vocab_16e6.txt.gz", "LICENSE")
ROOT_FILES = {"run.sh", "LICENSE", "layout_categories.json", "download_models.py",
              "image_crop_utils.py", "layout_region_utils.py", "layout_transform_utils.py",
              "periodic_template_matching.py", "public_download_utils.py", "template_match_workflow.py"}
# These modules are imported by inference; the rest of sam3/train is excluded.
TRAIN_RUNTIME = {"sam3/train/__init__.py", "sam3/train/masks_ops.py", "sam3/train/matcher.py",
                 "sam3/train/data/__init__.py", "sam3/train/data/collator.py",
                 "sam3/train/data/sam3_image_dataset.py", "sam3/train/data/coco_json_loaders.py"}
FILE_LIST = "deploy/xingzhou/runtime-files.txt"
MANIFEST = "bundle-manifest.json"


def selected(name):
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or "\\" in name:
        return False
    if any(part.startswith(".") or part in {"__pycache__", "node_modules", "tests", "test"}
           for part in path.parts):
        return False
    if path.name.startswith("test_") or path.suffix in {".pyc", ".pyo", ".pyi", ".pem", ".key", ".orig", ".pt"}:
        return False
    if len(path.parts) == 1:
        return name in ROOT_FILES
    if name == "assets/bpe_simple_vocab_16e6.txt.gz":
        return True
    if name.startswith("sam3/train/"):
        return name in TRAIN_RUNTIME
    if path.parts[0] == "sam3":
        return (len(path.parts) == 2 or path.parts[1] in {"model", "sam", "perflib"}) and path.suffix == ".py"
    if path.parts[0] == "sam3_demo":
        return path.suffix in {".py", ".json", ".js", ".css"}
    if path.parts[0] in COMPONENTS:
        return len(path.parts) > 2 and path.parts[1] == "backend" and path.suffix in {
            ".py", ".js", ".css", ".json", ".map", ".svg", ".woff", ".woff2",
        }
    return False


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root)


def digest(stream):
    result = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        result.update(chunk)
    return result.hexdigest()


def build(root, output, *, ref="HEAD", with_models=False):
    root, output = root.resolve(), Path(output)
    sha = git(root, "rev-parse", "--verify", ref + "^{commit}").decode().strip()
    tree = {}
    for row in git(root, "ls-tree", "-r", "-z", sha).split(b"\0"):
        if row:
            metadata, name = row.split(b"\t", 1)
            tree[name.decode()] = metadata.decode().split()
    entries = git(root, "show", sha + ":" + FILE_LIST).decode().splitlines()
    names = sorted({line.strip() for line in entries if line.strip() and not line.lstrip().startswith("#")})
    if not set(REQUIRED).issubset(names):
        raise ValueError("Production file list is missing required runtime files")
    for name in names:
        if not selected(name):
            raise ValueError("Non-production file in allowlist: " + name)
        if name not in tree or tree[name][0] not in {"100644", "100755"}:
            raise ValueError("Source must be a committed regular file: " + name)
    for component in COMPONENTS:
        if not any(name.startswith(component + "/backend/") and "/templates/component/" in name for name in names):
            raise ValueError("Missing compiled component: " + component)
    manifest = {"schema": 2, "git_sha": sha, "source": "committed Git blobs",
                "container_port": 8030, "entrypoint": "/src/run.sh",
                "models": "included" if with_models else "external mount", "files": {}}
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".sam3-bundle-", suffix=".zip", dir=output.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            for name in names + (list(MODELS) if with_models else []):
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | (0o755 if name == "run.sh" else 0o644)) << 16
                info.compress_type = zipfile.ZIP_STORED if name in MODELS else zipfile.ZIP_DEFLATED
                if name in MODELS:
                    path = root / name
                    before = path.stat()
                    if not path.is_file() or before.st_size == 0:
                        raise ValueError("Missing model: " + name)
                    with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
                        sha256 = hashlib.sha256()
                        while chunk := source.read(1024 * 1024):
                            sha256.update(chunk)
                            target.write(chunk)
                    after = path.stat()
                    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                        raise ValueError("Model changed during packaging: " + name)
                    entry = {"size": after.st_size, "sha256": sha256.hexdigest()}
                else:
                    data = git(root, "cat-file", "blob", tree[name][2])
                    if name in REQUIRED and not data:
                        raise ValueError("Empty required runtime file: " + name)
                    archive.writestr(info, data)
                    entry = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                manifest["files"][name] = entry
            archive.writestr(MANIFEST, json.dumps(manifest, indent=2))
        # Atomic publication with no overwrite, including under concurrent builds.
        os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return manifest


def verify(archive_path):
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP members")
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or str(path) != name:
                raise ValueError("Unsafe ZIP path: " + name)
            mode = archive.getinfo(name).external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError("ZIP must contain regular files: " + name)
            if name != MANIFEST and name not in MODELS and not selected(name):
                raise ValueError("Unexpected ZIP member: " + name)
        manifest = json.loads(archive.read(MANIFEST))
        if manifest["schema"] != 2 or set(names) != set(manifest["files"]) | {MANIFEST} or not set(REQUIRED).issubset(names):
            raise ValueError("ZIP/manifest entries mismatch or missing runtime files")
        if manifest["entrypoint"] != "/src/run.sh" or manifest["container_port"] != 8030:
            raise ValueError("Unexpected startup contract")
        expected_models = set(MODELS) if manifest["models"] == "included" else set()
        if set(names) & set(MODELS) != expected_models:
            raise ValueError("Model entries do not match the deployment contract")
        for name, entry in manifest["files"].items():
            if archive.getinfo(name).file_size != entry["size"]:
                raise ValueError("Size mismatch: " + name)
            with archive.open(name) as source:
                if digest(source) != entry["sha256"]:
                    raise ValueError("Checksum mismatch: " + name)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify", "extract"))
    parser.add_argument("zip", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--ref", default="HEAD", help="Committed revision to package (working files are ignored)")
    parser.add_argument("--with-models", action="store_true", help="Include local trusted model bytes instead of mounting them")
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.action == "build":
        build(args.root, args.zip, ref=args.ref, with_models=args.with_models)
    manifest = verify(args.zip)
    if args.action == "extract":
        if args.destination is None:
            parser.error("extract requires --destination")
        args.destination.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(args.zip) as archive:
            archive.extractall(args.destination)
        (args.destination / "run.sh").chmod(0o755)
    print(json.dumps({"verified": True, "files": len(manifest["files"]), "git_sha": manifest["git_sha"],
                      "models": manifest["models"], "bytes": args.zip.stat().st_size, "zip": str(args.zip)}))


if __name__ == "__main__":
    main()
