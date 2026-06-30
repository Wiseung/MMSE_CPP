# PDCCH Chain SDK Documentation

This page is the stable documentation entrypoint for the LTE PDCCH CE/MMSE SDK.

Primary include:

```cpp
#include "mmse/pdcch_chain_sdk.h"
```

Current interface version:

- `PDCCH Chain SDK v1`

Documentation set:

- [Quick Start](/G:/MMSE_CPP/docs/pdcch_chain_sdk_quick_start.md)
- [API Reference](/G:/MMSE_CPP/docs/pdcch_chain_sdk_api_reference.md)
- [Versioning Policy](/G:/MMSE_CPP/docs/pdcch_chain_sdk_versioning_policy.md)
- [Integration Example](/G:/MMSE_CPP/docs/pdcch_module_api_example.md)

## Recommended Reading Order

1. Read [Quick Start](/G:/MMSE_CPP/docs/pdcch_chain_sdk_quick_start.md) for first integration.
2. Use [API Reference](/G:/MMSE_CPP/docs/pdcch_chain_sdk_api_reference.md) for field-level lookup.
3. Use [Versioning Policy](/G:/MMSE_CPP/docs/pdcch_chain_sdk_versioning_policy.md) before making
   interface changes or depending on compatibility guarantees.

## Scope

The SDK covers:

- LTE PDCCH control-region RE extraction
- CRS-based channel estimation
- MMSE equalization
- per-RE soft-symbol and SINR handoff

The SDK does not cover:

- PCFICH decoding
- PHICH decoding
- REG/CCE regrouping
- blind detection
- channel decoding

## Page Summary

### Quick Start

Use this page when you need:

- the minimum include path
- the minimum DTO flow
- a compact integration sequence
- a validated demo build path

Link:

- [Quick Start](/G:/MMSE_CPP/docs/pdcch_chain_sdk_quick_start.md)

### API Reference

Use this page when you need:

- field definitions
- units
- helper semantics
- boundary conditions
- error-code meanings
- field index and status-code index

Link:

- [API Reference](/G:/MMSE_CPP/docs/pdcch_chain_sdk_api_reference.md)

### Versioning Policy

Use this page when you need:

- frozen v1 surface
- allowed additive changes
- disallowed incompatible changes
- rules for introducing `v2`

Link:

- [Versioning Policy](/G:/MMSE_CPP/docs/pdcch_chain_sdk_versioning_policy.md)
