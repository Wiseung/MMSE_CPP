from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "build" / "precommit"
NATIVE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".cmake"}
NATIVE_FILES = {"CMakeLists.txt"}


def run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def staged_files() -> list[Path]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    files: list[Path] = []
    for line in result.stdout.splitlines():
        if line.strip():
            files.append(Path(line.strip()))
    return files


def requires_native_gate(files: list[Path]) -> bool:
    for path in files:
        if path.name in NATIVE_FILES:
            return True
        if path.suffix.lower() in NATIVE_SUFFIXES:
            return True
        if path.parts and path.parts[0] in {"src", "include", "tests", "bench"}:
            return True
    return False


def main() -> int:
    files = staged_files()
    if not files:
        print("precommit_smoke: no staged files, skipping")
        return 0

    if not requires_native_gate(files):
        print("precommit_smoke: no native build/test impact, skipping")
        return 0

    run(["cmake", "-S", ".", "-B", str(BUILD_DIR), "-G", "Visual Studio 17 2022", "-A", "x64"])
    run(["cmake", "--build", str(BUILD_DIR), "--config", "Release", "--target", "mmse_tests"])
    run(["ctest", "--test-dir", str(BUILD_DIR), "-C", "Release", "--output-on-failure", "-R", "mmse_tests"])
    print("precommit_smoke: build and tests passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"precommit_smoke: command failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode)
