<!--
Path: docs/multimodal-retrieval.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Multimodal retrieval engineering notes

## Upstream basis

Reviewed 2026-09-10 against current official documentation and the Sentence
Transformers v5.4.0 source tag (`fe9361218c10b2ee18f497d73863788a6b592210`).

- [Qwen 2B embedding model card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
  documents SentenceTransformer encoding, text/image combinations and MRL dimensions
  64–2048. [The 8B card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B)
  extends the maximum to 4096.
- [Qwen 2B reranker card](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B)
  documents CrossEncoder pair prediction, raw scores and sigmoid activation.
  [The 8B reranker](https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B) uses the same interface.
- [Sentence Transformers' official multimodal guide](https://huggingface.co/blog/multimodal-sentence-transformers)
  identifies v5.4 as the multimodal integration release and recommends `[image]`.
- [The tagged transformer module](https://github.com/huggingface/sentence-transformers/blob/v5.4.0/sentence_transformers/base/modules/transformer.py)
  imports `AutoModelForMultimodalLM` for `any-to-any` and explicitly requires
  Transformers v5+ when unavailable. Therefore the direct-Transformers model card
  requirement `>=4.57` is insufficient for the Sentence Transformers reranker route.
- [CrossEncoder API](https://sbert.net/docs/package_reference/cross_encoder/model.html)
  supports `cache_folder`, `device`, `revision`, `local_files_only`,
  `trust_remote_code`, batch size, prompt and activation control.
- [SentenceTransformer API](https://sbert.net/docs/package_reference/sentence_transformer/model.html)
  provides NumPy outputs and backend embedding dimension inspection. This wrapper
  validates native output shape then applies truncation and normalization itself.
- [Transformers 4.57 auto classes](https://huggingface.co/docs/transformers/v4.57.0/en/model_doc/auto#transformers.AutoModelForImageTextToText)
  include Qwen3-VL in `AutoModelForImageTextToText`; the generator now uses that
  class to match its declared minimum version.

These APIs support the package's existing architecture, so no upstream scripts
were vendored and no remote Python code is enabled by default.

## Resources

The official Sentence Transformers guide gives approximate GPU guidance of
8 GB VRAM for 2B and 20 GB for 8B. Treat these as workload-dependent orientation,
not measured guarantees for this package. Three simultaneously loaded models
require substantially more memory. Each extra evidence image increases vision
and context work. CPU inference is possible but can be extremely slow.
[Official guide](https://huggingface.co/blog/multimodal-sentence-transformers).

Weight-only estimates at 2 bytes per parameter are approximately 4 GB (2B) and
16 GB (8B) **per model**, before allocator state, activations and processor data.
FP32 storage roughly doubles those estimates. Reserve disk space for each
snapshot, download metadata and temporary files; cache mode and explicit local
downloads may duplicate storage. Inspect actual snapshot sizes before deployment.

Defaults: 2B, batch size 1, no automatic parallel model workers. `unload()` drops
the wrapper's model reference; application-held references and framework allocator
caches may remain. In a staged job, unload the embedder before loading the
reranker, then unload the reranker before generation. An interactive RAG pipeline
may retain all three models and must be provisioned accordingly.

[Accelerate big-model inference](https://huggingface.co/docs/accelerate/usage_guides/big_modeling)
can place model layers on GPU, CPU and disk. The retrieval wrapper exposes a
single `device`; it does not invent an automatic offloading policy. Advanced
applications can inject an appropriately configured SentenceTransformer or
CrossEncoder backend (e.g. `model_kwargs` with `device_map`, dtype and an
`offload_folder`). Such deployments need their own integration tests. The
existing generator exposes `device_map`, but CPU/disk offload remains slower and
requires adequate writable disk and RAM. OS/container OOM kills cannot be caught
by Python; inspect process exit status and host logs.

Install torch and torchvision from the same CPU/CUDA distribution channel.
[PyTorch's torchvision documentation](https://docs.pytorch.org/vision/stable/index.html)
is the compatibility reference. The lower bounds 2.8/0.23 represent a matching
release family; pip still resolves the exact torchvision torch requirement.
No new direct transitive package pins were added.

## Deployment and limitations

- Hub mode uses normal Hugging Face caching; an explicit revision helps reproducibility.
- Local mode forces offline loading. Root-layout snapshots are supported, not arbitrary
  nested Sentence Transformers module exports or adapter-only directories.
- Structural preflight does not hash multi-GB weights or parse safetensors headers.
  Corrupt weights are detected by the backend and mapped to repair guidance.
- Local single-frame PNG/JPEG/WEBP/BMP/TIFF only for retrieval. No video or remote
  retrieval image downloads; the legacy generator's HTTP input support remains.
- In-memory index, synchronous API, no concurrent index mutation guarantees.
- Inputs and generated evidence remain untrusted data. Prompt instructions request
  grounded answers and citations; they do not mathematically guarantee faithfulness.
- Raw scores are model-specific logits; sigmoid is monotonic, not calibration.
- Custom checkpoints must follow a compatible multimodal Sentence Transformers
  contract. This API is not a universal adapter for every Hugging Face architecture.

## Docker

```bash
docker compose build
docker compose run --rm vlm-lab validate-config configs/qwen_multimodal_rag.example.yaml
docker compose run --rm vlm-lab download-model --embedding-size 2b
HF_HUB_OFFLINE=1 docker compose run --rm vlm-lab embed \
  --text "cache architecture" --model-path models/Qwen3-VL-Embedding-2B
```

Models and data are mounted, not baked into the image. Configs are mounted read-only.
The `hf-cache` volume is persistent. The default Compose file does not reserve a
GPU: add the GPU configuration appropriate to your Docker host before GPU work.
The non-root container user needs write permission for `./models` on the host.
