"""Run WebUI, offline evaluation, and custom-component tests independently."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def test_groups(root: Path, suite: str) -> list[tuple[str, Path]]:
    groups = []
    if suite in {"web", "all"}:
        groups.append(("web", root / "tests"))
    if suite in {"offline", "all"}:
        groups.append(("offline", root / "tests" / "offline_eval"))
    if suite in {"components", "all"}:
        groups.extend(
            (directory.parent.name, directory)
            for directory in sorted(root.glob("*/tests"))
            if (directory.parent / "frontend" / "gradio.config.js").is_file()
            and list(directory.glob("test_*.py"))
        )
    return groups


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=("web", "offline", "components", "all"), default="web"
    )
    parser.add_argument(
        "--list", action="store_true", help="List files without importing tests"
    )
    args = parser.parse_args(argv)
    groups = test_groups(ROOT, args.suite)
    if not groups:
        parser.error("no test groups found")
    failed = False
    for name, directory in groups:
        files = sorted(directory.glob("test_*.py"))
        if not files:
            parser.error(f"no tests found in {directory}")
        if args.list:
            for file in files:
                print(f"{name}: {file.relative_to(ROOT)}")
            continue
        env = os.environ.copy()
        paths = [str(directory), str(ROOT), env.get("PYTHONPATH", "")]
        env["PYTHONPATH"] = os.pathsep.join(path for path in paths if path)
        env.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
        print(f"\n[{name}] {len(files)} test files", flush=True)
        started = time.perf_counter()
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover",
             "-s", str(directory), "-p", "test_*.py", "-q"],
            cwd=ROOT,
            env=env,
            check=False,
        )
        failed |= result.returncode != 0
        print(
            f"[{name}] exit={result.returncode} "
            f"elapsed={time.perf_counter() - started:.2f}s", flush=True
        )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
