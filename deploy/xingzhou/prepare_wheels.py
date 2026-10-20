"""Build the pinned Linux application wheelhouse; never install into the host."""

import argparse
from email.parser import BytesParser
from pathlib import Path
import platform
import re
import subprocess
import sys
import zipfile


def normalized(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def gpu_package(name):
    name = normalized(name)
    return name in {"torch", "torchvision", "torchaudio", "triton"} or name.startswith("nvidia-")


def read_lock(text):
    pins = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)", line)
        if not match:
            raise ValueError("Offline requirements must use name==version: " + line)
        name, version = normalized(match[1]), match[2]
        if gpu_package(name):
            raise ValueError("GPU dependency belongs in the image: " + name)
        if name in pins:
            raise ValueError("Duplicate offline dependency: " + name)
        pins[name] = version
    if not pins:
        raise ValueError("Offline dependency lock is empty")
    return pins


def wheel_files(directory, lock_text):
    pins = read_lock(lock_text)
    found = {}
    for path in sorted(Path(directory).glob("*.whl")):
        with zipfile.ZipFile(path) as wheel:
            entries = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
            if len(entries) != 1:
                raise ValueError("Wheel has no unique metadata: " + path.name)
            metadata = BytesParser().parsebytes(wheel.read(entries[0]))
        name, version = normalized(metadata["Name"]), metadata["Version"]
        if gpu_package(name) or pins.get(name) != version or name in found:
            raise ValueError("Wheel does not match application lock: " + path.name)
        found[name] = path
    missing = pins.keys() - found.keys()
    if missing:
        raise ValueError("Missing offline wheels: " + ", ".join(sorted(missing)))
    return list(found.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if sys.platform != "linux" or platform.machine() != "x86_64" or sys.version_info[:2] != (3, 12):
        parser.error("Build these wheels with CPython 3.12 on Linux x86_64")
    lock = Path(__file__).resolve().parents[2] / "requirements-offline.txt"
    read_lock(lock.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    # iopath is published as a pure-Python sdist; build its wheel here, not at startup.
    subprocess.run([
        sys.executable, "-m", "pip", "--isolated", "wheel",
        "--index-url", "https://pypi.org/simple", "--disable-pip-version-check",
        "--no-deps", "--only-binary=:all:", "--no-binary=iopath",
        "--wheel-dir", str(args.output), "-r", str(lock),
    ], check=True)
    files = wheel_files(args.output, lock.read_text())
    print(f"Prepared {len(files)} application wheels; GPU packages excluded", flush=True)


if __name__ == "__main__":
    main()
