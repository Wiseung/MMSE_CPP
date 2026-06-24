from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CPP_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp"}


def candidate_clang_format_paths() -> list[Path]:
    candidates: list[Path] = []

    env_value = None
    try:
        import os

        env_value = os.environ.get("CLANG_FORMAT")
    except Exception:
        env_value = None

    if env_value:
        candidates.append(Path(env_value))

    which_path = shutil.which("clang-format")
    if which_path:
        candidates.append(Path(which_path))

    for edition in ("BuildTools", "Community", "Professional", "Enterprise"):
        candidates.append(
            Path(
                f"C:/Program Files/Microsoft Visual Studio/2022/{edition}/VC/Tools/Llvm/x64/bin/clang-format.exe"
            )
        )
        candidates.append(
            Path(
                f"C:/Program Files (x86)/Microsoft Visual Studio/2022/{edition}/VC/Tools/Llvm/x64/bin/clang-format.exe"
            )
        )

    candidates.append(Path("C:/Program Files/LLVM/bin/clang-format.exe"))
    candidates.append(Path("C:/Program Files (x86)/LLVM/bin/clang-format.exe"))
    return candidates


def find_clang_format() -> Path:
    for candidate in candidate_clang_format_paths():
        if candidate and candidate.exists():
            return candidate
    raise FileNotFoundError(
        "clang-format not found. Install LLVM clang-format or set the CLANG_FORMAT environment variable."
    )


def collect_repo_cpp_files() -> list[Path]:
    files: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in CPP_SUFFIXES:
            if any(part in {"build", "node_modules", ".git"} for part in path.parts):
                continue
            files.append(path)
    return files


def normalize_files(raw_files: list[str], all_files: bool) -> list[Path]:
    if all_files:
        return collect_repo_cpp_files()

    files: list[Path] = []
    for raw in raw_files:
        path = (REPO_ROOT / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
        if not path.exists() or not path.is_file():
            continue
        if path.suffix.lower() not in CPP_SUFFIXES:
            continue
        files.append(path)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def restage(files: list[Path]) -> None:
    rel_files = [str(path.relative_to(REPO_ROOT)) for path in files]
    if rel_files:
        subprocess.run(["git", "add", "--", *rel_files], cwd=REPO_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Format every C/C++ file in the repo.")
    parser.add_argument("files", nargs="*", help="Files to format.")
    args = parser.parse_args()

    files = normalize_files(args.files, args.all)
    if not files:
        print("run_clang_format: no C/C++ files to format")
        return 0

    clang_format = find_clang_format()
    subprocess.run([str(clang_format), "-i", *[str(path) for path in files]], cwd=REPO_ROOT, check=True)
    restage(files)
    print(f"run_clang_format: formatted {len(files)} file(s) with {clang_format}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
