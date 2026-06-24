from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUILD_DIR = REPO_ROOT / "build" / "release-bundle"
DEFAULT_STAGE_DIR = REPO_ROOT / "dist" / "mmse_cpp"
DEFAULT_ARCHIVE = REPO_ROOT / "dist" / "mmse_cpp-release.zip"


def run(cmd: list[str], cwd: Path = REPO_ROOT) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def remove_if_exists(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    stage_dir = args.stage_dir.resolve()
    archive_path = args.archive.resolve()

    remove_if_exists(stage_dir)
    remove_if_exists(archive_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    run(["cmake", "-S", ".", "-B", str(build_dir), "-G", "Visual Studio 17 2022", "-A", "x64"])
    run(["cmake", "--build", str(build_dir), "--config", "Release", "--parallel"])
    run(["ctest", "--test-dir", str(build_dir), "-C", "Release", "--output-on-failure"])
    run(["cmake", "--install", str(build_dir), "--config", "Release", "--prefix", str(stage_dir)])

    base_name = archive_path.with_suffix("")
    archive_created = Path(shutil.make_archive(str(base_name), "zip", root_dir=stage_dir))
    print(f"create_release_bundle: wrote {archive_created}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"create_release_bundle: command failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode)
