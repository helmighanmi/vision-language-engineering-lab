<!--
Path: docs/decisions/003-trust-remote-code.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# ADR-003: `trust_remote_code` is opt-in

## Decision

Default `trust_remote_code=False` for all model adapters.

## Why

Enabling the flag permits execution of Python supplied by a remote model repository. Current Qwen3-VL integrates with Transformers natively, so the project avoids this trust expansion unless a user explicitly chooses it for another model.
