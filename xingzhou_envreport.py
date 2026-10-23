"""Print the platform image's Python, OS, GPU/CUDA and preinstalled packages.

run.sh calls this before the offline install, so the report shows the base
image as delivered. Every probe is best effort: failures are reported as
values and never stop startup. Environment values are limited to CUDA/NVIDIA
names and never include anything that looks like a credential.
"""

import faulthandler
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time


SENSITIVE = ("KEY", "TOKEN", "SECRET", "PASS", "CRED", "AUTH")


def emit(section, **details):
    print(json.dumps({"event": "xingzhou_env", "section": section, **details}, default=str),
          file=sys.stderr, flush=True)


def probe(function):
    try:
        return function()
    except Exception as exc:
        return {"error": type(exc).__name__ + ": " + str(exc)[:200]}


def run(*command):
    if shutil.which(command[0]) is None:
        return None
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    return (result.stdout or result.stderr).strip()[:2000]


def os_release():
    path = Path("/etc/os-release")
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[1].strip('"')
    return None


def mount_of(path):
    # Longest /proc/mounts entry containing path: device, mount point, fs type.
    best = None
    for line in Path("/proc/mounts").read_text().splitlines():
        fields = line.split()
        if len(fields) >= 3 and (str(path) + "/").startswith(fields[1].rstrip("/") + "/"):
            if best is None or len(fields[1]) > len(best[1]):
                best = fields[:3]
    return dict(zip(("device", "mount_point", "fs_type"), best)) if best else None


def memory_limit_mib():
    for name in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        path = Path(name)
        if path.is_file():
            value = path.read_text().strip()
            return value if not value.isdigit() else int(value) // (1024 * 1024)
    return None


def system():
    root = Path.cwd()
    usage = shutil.disk_usage(root)
    return {
        "python": sys.version.split()[0], "implementation": platform.python_implementation(),
        "executable": sys.executable, "prefix": sys.prefix,
        "platform": platform.platform(), "machine": platform.machine(),
        "libc": "-".join(platform.libc_ver()), "os": os_release(),
        "uid": os.getuid(), "gid": os.getgid(), "cwd": str(root),
        "cwd_writable": os.access(root, os.W_OK), "cwd_free_mib": usage.free // (1024 * 1024),
        "cpu_count": os.cpu_count(),
        "cwd_mount": probe(lambda: mount_of(root)), "memory_limit_mib": probe(memory_limit_mib),
        "pip": run(sys.executable, "-m", "pip", "--version"),
    }


def gpu():
    query = run("nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free,compute_cap",
                "--format=csv,noheader")
    header = run("nvidia-smi")
    cuda_line = next((line for line in (header or "").splitlines() if "CUDA Version" in line), None)
    environment = {
        name: value for name, value in sorted(os.environ.items())
        if name.startswith(("CUDA", "NV_", "NVIDIA", "LD_LIBRARY_PATH"))
        and not any(word in name.upper() for word in SENSITIVE)
    }
    return {"nvidia_smi": query, "driver_cuda": cuda_line.strip("| ") if cuda_line else None,
            "nvcc": run("nvcc", "--version"), "environment": environment}


def torch_stack():
    import torch

    details = {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
               "cudnn": probe(torch.backends.cudnn.version),
               "cuda_available": torch.cuda.is_available(),
               "arch_list": probe(torch.cuda.get_arch_list)}
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        details.update(device=torch.cuda.get_device_name(0), capability=".".join(map(str, capability)),
                       native_bf16=capability >= (8, 0))
    for name in ("torchvision", "triton"):
        details[name] = probe(lambda: importlib.metadata.version(name))
    return details


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def packages():
    installed = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        if name and canonical(name) not in installed:
            installed[canonical(name)] = distribution.version
    locked = {}
    lock = Path("requirements-offline.txt")
    if lock.is_file():
        for line in lock.read_text().splitlines():
            if "==" in line and not line.lstrip().startswith("#"):
                name, version = line.strip().split("==", 1)
                locked[canonical(name)] = version
    # Packages the bundle shadows through PYTHONPATH; a version gap here is
    # where an image-built extension (e.g. against NumPy) could break.
    overlap = {name: {"image": installed[name], "bundle": version}
               for name, version in sorted(locked.items()) if name in installed}
    return {"count": len(installed), "image_packages": dict(sorted(installed.items())),
            "bundle_overrides_image": overlap}


def main():
    for section, function in (("system", system), ("gpu", gpu),
                              ("torch", torch_stack), ("packages", packages)):
        started = time.monotonic()
        emit(section, status="begin")
        details = probe(function)
        emit(section, status="warning" if "error" in details else "ok",
             duration_s=round(time.monotonic() - started, 3), **details)


if __name__ == "__main__":
    # This optional child must not hold up the actual server startup.
    faulthandler.register(signal.SIGALRM, all_threads=True, chain=True)
    # Torch import from slow shared storage can take a while; 90 s keeps the
    # report useful without holding up startup indefinitely.
    signal.alarm(90)
    main()
