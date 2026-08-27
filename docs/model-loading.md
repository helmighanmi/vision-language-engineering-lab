<!--
Path: docs/model-loading.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Model loading modes

## Hub/cache mode

`QwenVLModel.from_hub()` resolves the model by Hugging Face model ID. The first execution may download weights; later executions reuse the Hugging Face cache.

## Explicit local/offline mode

`vlm-lab download-model` downloads a complete snapshot to `models/...`. `QwenVLModel.from_local()` sets `local_files_only=True`, so inference does not fall back to the network.

## Remote custom code

`trust_remote_code` is **False by default**. Current Qwen3-VL is integrated into Transformers and normal inference does not require executing arbitrary Python from the model repository. Keep the flag opt-in for other models that genuinely require it.
