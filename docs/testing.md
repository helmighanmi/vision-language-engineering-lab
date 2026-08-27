<!--
Path: docs/testing.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Testing strategy

- Unit tests inject fake models and do not download multi-gigabyte weights.
- CI validates code quality, typing, notebook JSON integrity, tests, Docker build, dependency vulnerabilities and CodeQL.
- Real-model evaluation belongs in a controlled GPU environment and should measure structured-output validity, entity/relation accuracy, latency, memory, retrieval Recall@K and final grounded-answer quality.
