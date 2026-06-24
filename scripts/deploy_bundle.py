from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARCHIVE = REPO_ROOT / "dist" / "mmse_cpp-release.zip"


def deploy_directory(environment: str) -> Path:
    key = f"MMSE_{environment.upper()}_DEPLOY_DIR"
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(
            f"{key} is not set. Configure it as a GitHub Actions variable or secret before enabling deployment."
        )
    return Path(value)


def deploy_bundle(archive: Path, environment: str) -> Path:
    target_root = deploy_directory(environment)
    target_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = target_root / f"mmse_cpp-{environment}-{timestamp}.zip"
    shutil.copy2(archive, destination)

    latest_link = target_root / f"mmse_cpp-{environment}-latest.zip"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    shutil.copy2(destination, latest_link)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", required=True, choices=("staging", "production"))
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()

    archive = args.archive.resolve()
    if not archive.exists():
        raise FileNotFoundError(f"archive does not exist: {archive}")

    deployed = deploy_bundle(archive, args.environment)
    print(f"deploy_bundle: copied archive to {deployed}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"deploy_bundle: {exc}", file=sys.stderr)
        raise SystemExit(1)
