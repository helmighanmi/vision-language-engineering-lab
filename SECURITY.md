<!--
Path: SECURITY.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Security policy

- Never commit Hugging Face tokens, credentials, private model weights or proprietary documents.
- `trust_remote_code` is disabled by default.
- Pin or record model revisions for production deployments.
- Treat VLM output as untrusted data: validate JSON and never execute generated code.
- Remote images and documents should be downloaded through bounded, allowlisted ingestion components in production.
- Report suspected vulnerabilities through GitHub security advisories rather than a public issue.
