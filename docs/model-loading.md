<!--
Path: docs/model-loading.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Qwen model loading modes

The package separates **which model you want** from **where it is loaded from**.

## Supported presets

```text
2b -> Qwen/Qwen3-VL-2B-Instruct  (default)
4b -> Qwen/Qwen3-VL-4B-Instruct
8b -> Qwen/Qwen3-VL-8B-Instruct
```

List them from the CLI:

```bash
vlm-lab models
```

## Python API

```python
from vlm_engineering import QwenVLModel

# Default 2B
model = QwenVLModel()

# Presets
model = QwenVLModel(model_size="4b")
model = QwenVLModel.from_preset("8b")

# Any compatible Hugging Face model ID
model = QwenVLModel.from_hub("Qwen/Qwen3-VL-4B-Instruct")

# Explicit local/offline model
model = QwenVLModel.from_local("models/Qwen3-VL-4B-Instruct")
```

## CLI model selection

```bash
# Default 2B
vlm-lab describe image.jpg

# Presets
vlm-lab describe image.jpg --model-size 4b
vlm-lab analyze diagram.png --model-size 8b

# Explicit compatible Hub model ID
vlm-lab describe image.jpg --model-id Qwen/Qwen3-VL-4B-Instruct

# Explicit local/offline model
vlm-lab describe image.jpg --model-path models/Qwen3-VL-4B-Instruct
```

The three model-source flags are mutually exclusive.

## Hub/cache mode

Hub-backed models download on first use if required and later reuse the Hugging Face cache.

## Explicit download

```bash
vlm-lab download-model --model-size 4b
```

Default output:

```text
models/Qwen3-VL-4B-Instruct/
```

Or choose the directory:

```bash
vlm-lab download-model --model-size 4b --output /models/qwen4b
```

## Explicit local/offline mode

`QwenVLModel.from_local()` sets `local_files_only=True`, so the package does not silently fall back to the network.

For an entirely disconnected Hugging Face environment you may additionally use:

```bash
export HF_HUB_OFFLINE=1
```

## Remote custom code

`trust_remote_code` is **False by default**. Current Qwen3-VL is integrated into Transformers and normal inference does not require executing arbitrary Python from the model repository.

Use `--trust-remote-code` only after reviewing and trusting a model repository that truly needs it.

## Choosing a size

| Size | Practical guidance |
|---|---|
| 2B | start here; cheapest iteration and local learning |
| 4B | balanced step up when quality is insufficient |
| 8B | stronger hardware; validate that quality gain justifies cost |

A rough BF16 weight-only floor is ~2 bytes/parameter. Actual inference memory is higher because of the vision encoder, activations, KV cache, image tokens and runtime overhead.

## Common problems

- **Out of memory:** move to 2B, reduce visual resolution/output length, or use more capable hardware.
- **Slow first run:** Hub/cache mode may be downloading model files.
- **No network allowed:** download once and use `from_local()` / `--model-path`.
- **Wrong model selected:** use only one of `--model-size`, `--model-id`, `--model-path`.
- **Private/gated model:** configure `HF_TOKEN` if the model license/access rules require it.
