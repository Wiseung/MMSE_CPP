# Security Policy

Thank you for taking the time to review the security policy for **MMSE_CPP**. 

Security is a critical priority for this project. Since this engine processes raw, untrusted baseband IQ data and parses complex LTE PHY control plane structures (PDCCH/DCI), we take potential vulnerabilities—such as buffer overflows, out-of-bounds memory accesses in AVX2/CUDA kernels, and denial-of-service (DoS) vectors—very seriously.

## Supported Versions

Please note that only the latest major versions of the project are actively maintained with security updates. We recommend users always build from the latest stable release.

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

*(Note: Update the table above to reflect your actual release tags and versioning strategy.)*

## Reporting a Vulnerability

**Please do NOT report security vulnerabilities through public GitHub issues.**

If you discover a security vulnerability in this project, we kindly ask that you report it to us privately so we can address it before it is publicly disclosed.

### How to Report

1. **GitHub Private Vulnerability Reporting:** Please use the [Security Advisories](../../security/advisories/new) feature in this repository to privately report a vulnerability directly to the maintainers.
2. **Email (Alternative):** You can also reach out via email at `[Your Email Address Here]`. Please include "MMSE_CPP Security Report" in the subject line.

### What to Include in Your Report
To help us resolve the issue as quickly as possible, please include the following details in your report:
* **Description:** A clear summary of the vulnerability.
* **Environment:** The hardware/OS/Compiler versions (e.g., CUDA version, GCC/Clang version, CPU architecture) where the bug is reproducible.
* **Steps to Reproduce:** Detailed steps to trigger the vulnerability. If it involves a specific payload (e.g., a malformed IQ data grid or specific DCI bitstream), please provide a minimal reproducible example or hex dump.
* **Impact:** What could an attacker achieve? (e.g., kernel crash, remote code execution, memory leak).

## Response Timeline

* We will acknowledge receipt of your vulnerability report within **48 hours**.
* We will send you a status update regarding our triage process within **5 days**.
* If the vulnerability is confirmed, we will work with you to patch the issue and publish a coordinated security advisory.

We deeply appreciate the efforts of security researchers and the open-source community in keeping this project safe and robust!

