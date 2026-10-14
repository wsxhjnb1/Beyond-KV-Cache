# Usage

[Project overview](README.md) · [Paper](paper/Beyond.pdf) · [Final poster](docs/poster/EMNLP2026_Beyond_Poster.pdf)

Run the commands below from the repository root. This repository contains the
maintained training and inference code, but does not distribute model weights,
datasets, checkpoints, training outputs, or raw measurement data.

## Measurement scope

The figures in the README reproduce the final conference poster's results.
Quality uses training-equivalent fake quantize/dequantize (fake-QDQ)
evaluation, retaining a floating-point cache. Perplexity is evaluated on 150
disjoint held-out The Pile chunks; LongBench Avg21 is the unweighted mean over
21 tasks. Reported PPL reductions follow the poster and are not recomputed from
its rounded displayed values. Aggregate gains do not imply gains on every task.

Packed-cache serving is a separate experiment on one NVIDIA B300 with
Llama-3.1-8B-Instruct:

- Kernel timing covers batches 1 and 16 and contexts from 1K to 128K, with
  50 warmups and 500 CUDA-graph replays. The peak speedup over
  FlashAttention-4 is 1.50×.
- End-to-end throughput uses vLLM's official offline benchmark: cold
  1024-token prompts, 128-token generation, BF16 model execution, an
  8192-token scheduling budget, GPU memory utilization 0.9, and prefix
  caching disabled. The peak matched-batch speedup is 1.17× at batch 1536.
- The FlashAttention-4 path succeeds at batch 1582 and fails its next tested
  KV-capacity check at 1583. Beyond succeeds at batch 4500 and fails its next
  tested check at 4800. This establishes at least 2.84× tested batch capacity,
  not an exact maximum or a universal out-of-memory boundary.

The 4-bit/G32 cache uses 128 code bits plus 32 bits of min/scale metadata
per group: 5 effective bits per scalar, or 3.2× KV-payload compression
relative to BF16 before page padding. Total GPU memory includes other costs.
These measurements are configuration-specific, not guarantees for arbitrary
hardware and workloads.

## Requirements

- Linux and Python 3.10 or newer
- CUDA 13 and a compatible NVIDIA driver
- NVIDIA Blackwell SM103 for the fused packed-cache decode path
- Access to the selected Hugging Face model and training dataset

Create an isolated environment and install both Python packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pip install --no-deps -e ./quant
```

## Training

The maintained trainer targets K/V projection quantizers and writes learned
configurations below the selected run directory. Launch it with `torchrun`,
including for a single GPU:

```bash
export DISCOVER_TRAIN_SEQLEN=8192
export DISCOVER_EVAL_EVERY_STEPS=10

torchrun --standalone --nproc_per_node=1 -m beyond.train \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --num_bits 4 --group_size 32 \
  --dataset pile --pile_train_mode streaming \
  --reference_tokens 50000000 --train_steps_per_pass 500 --epochs 1 \
  --batch_size 32 --per_gpu_batch_size 1 \
  --lr 0.0007 --lr_schedule cosine \
  --boundary_window 0.009 \
  --seed 42 --data_seed 42 --eval_seed 10000042 \
  --early_stop_patience 3 --attn_implementation sdpa \
  --logs_root runs --no_auto_resume
```

The final learned configuration is written as
`runs/<run-name>/quant_configs/quant_config_final.pt`. Generated run data is
ignored by Git.

For multiple GPUs, change `--nproc_per_node` and keep `--batch_size` equal to
the intended global sequence batch. Tensor and pipeline parallel sizes can be
set with the trainer's corresponding command-line options.

## Inference

Start the packed-cache OpenAI-compatible server with a trained configuration:

```bash
python -m beyond.inference \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --quant-config runs/REPLACE_ME/quant_configs/quant_config_final.pt \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --host 127.0.0.1 --port 8000
```

The server deliberately exposes only the packed 4-bit/G32 backend. Unsupported
bit widths, group sizes, missing configurations, or unsupported hardware fail
closed instead of selecting an uncompressed fallback.

A request can then be sent with any OpenAI-compatible client:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Explain KV-cache quantization."}],
    "temperature": 0
  }'
```

## Layout

```text
src/beyond/train/          training entrypoint and runtime
src/beyond/quantization/   learned quantizers and configuration I/O
src/beyond/runtime/vllm/   packed vLLM attention backend
src/beyond/inference/      OpenAI-compatible server entrypoint
quant/                     packed cache operators and SM103 decode kernel
tests/                     CPU and GPU contract tests
```

## Development

```bash
python -m pip install -r requirements-dev.txt
ruff check .
python -m pytest -q
```

GPU tests skip automatically when their required runtime or hardware is not
available.

## License

Original Beyond code is licensed under the Apache License 2.0. The
`quant/fa4_cute/` subtree retains its BSD 3-Clause license and upstream
attribution; see `quant/fa4_cute/LICENSE` and `quant/fa4_cute/AUTHORS`.
Model weights, datasets, and external software remain governed by their own
terms.
