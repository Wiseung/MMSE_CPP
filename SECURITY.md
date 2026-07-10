# Security Policy

`MMSE_CPP` processes raw LTE baseband data and exposes CPU/AVX2/CUDA paths that operate on
caller-provided buffers and metadata. Security-relevant defects in this repository can therefore
include, but are not limited to:

- out-of-bounds reads or writes
- buffer size validation bugs
- integer overflow or invalid index calculations
- malformed-input denial-of-service paths
- undefined behavior in SIMD or CUDA kernels

This document defines the supported security maintenance scope and the expected reporting path.

## Supported Versions

This repository does not currently publish a stable release line with long-term backport support.
Security fixes are applied to the active default branch.

| Branch / version                                            | Supported |
| ----------------------------------------------------------- | --------- |
| `main`                                                      | Yes       |
| historical commits, feature branches, unpublished snapshots | No        |

If formal release lines are introduced later, the latest maintained release line should be treated
as the supported production version unless the table above is updated.

## Reporting a Vulnerability

Do **not** report security vulnerabilities through public GitHub issues, discussions, or pull
requests.

Use GitHub Private Vulnerability Reporting / Security Advisories:

- <https://github.com/Wiseung/MMSE_CPP/security/advisories/new>

If the GitHub advisory flow is unavailable in your environment, contact the maintainers privately
before any public disclosure.

## What to Include

Include enough detail for the issue to be reproduced and triaged safely:

- affected commit, branch, or build artifact
- operating system, compiler, CUDA version, and hardware details
- exact API surface involved, for example `run_pdcch(...)`, `run(...)`, or a benchmark/demo target
- minimal input or call pattern that reproduces the problem
- observed impact, for example crash, memory corruption, invalid memory access, or denial of service
- whether the issue requires malformed radio data, malformed metadata, local file input, or another attacker-controlled source

If proof-of-concept inputs are large or sensitive, provide the smallest private reproducer that
still triggers the problem.

## Preferred Disclosure Process

The expected process is:

1. report privately
2. allow maintainers to reproduce and triage
3. agree on a fix and validation path
4. disclose publicly after a fix or mitigation is available

Please avoid publishing exploit details before maintainers have had a reasonable chance to
investigate and respond.

## Response Targets

These are response targets, not absolute guarantees:

- initial acknowledgement: within 3 business days
- triage update: within 7 business days
- follow-up status updates: as fixes or mitigations become available

Complex issues in low-level CPU/GPU code may require additional time to reproduce across hardware
and runtime configurations.

## Scope Notes

The following are generally in scope for responsible security reports:

- memory safety bugs in runtime code
- incorrect bounds handling in DTO/helper conversion code
- malformed-input crashes in tests, demos, or release tooling when they process untrusted input
- security-impacting undefined behavior in AVX2 or CUDA paths

The following are generally not treated as security vulnerabilities by themselves:

- purely numerical accuracy bugs without confidentiality, integrity, or availability impact
- performance regressions
- missing protocol features
- unsupported-configuration rejections that fail closed

## Security Updates

When a security-relevant fix is accepted, maintainers may:

- patch `main`
- add regression tests where practical
- document behavior changes in the repository history or release materials

Public disclosure details will depend on impact, exploitability, and whether affected downstream
users need coordinated notice.
