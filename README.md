# MMSE_CPP

## Local Commit Gates

Run `npm install` once after cloning. The repo uses `husky + lint-staged` to block local commits when staged files fail formatting or when native changes break the lightweight `mmse_tests` build/test smoke check.

Native gate details:

- staged `*.cpp/*.h` files are formatted locally with `clang-format`
- staged `*.md/*.json/*.yml/*.yaml` files are formatted with `prettier`
- if staged changes touch native build/test surfaces such as `src/`, `include/`, `tests/`, `bench/`, or `CMakeLists.txt`, the pre-commit hook runs a local `cmake -> build mmse_tests -> ctest` smoke check and rejects the commit on failure

## CI / CD

The repository now uses:

- `ci`: builds and tests on Windows for `main`, `codex/**`, and pull requests
- `cd`: automatically creates a release bundle and deploys to `staging` after a successful `main` CI run; `production` is intentionally manual through `workflow_dispatch`

Required GitHub Environment / repo variables before enabling real deployment:

- `production` environment with `MMSE_PRODUCTION_DEPLOY_DIR`

Optional / recommended for staging:

- `staging` environment with `MMSE_STAGING_DEPLOY_DIR`

Current deployment behavior is file-copy based: the CD workflow creates `dist/mmse_cpp-release.zip` and copies it into the configured target directory with a timestamped filename plus a `latest` copy. If `MMSE_STAGING_DEPLOY_DIR` is not configured yet, staging falls back to a runner-local drop directory so the CD path can still execute end to end.
