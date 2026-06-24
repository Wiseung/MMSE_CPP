# MMSE_CPP

## Local Commit Gates

Run `npm install` once after cloning. The repo uses `husky + lint-staged` to block local commits when staged files fail formatting or when native changes break the lightweight `mmse_tests` build/test smoke check.

Native gate details:

- staged `*.cpp/*.h` files are formatted locally with `clang-format`
- staged `*.md/*.json/*.yml/*.yaml` files are formatted with `prettier`
- if staged changes touch native build/test surfaces such as `src/`, `include/`, `tests/`, `bench/`, or `CMakeLists.txt`, the pre-commit hook runs a local `cmake -> build mmse_tests -> ctest` smoke check and rejects the commit on failure
