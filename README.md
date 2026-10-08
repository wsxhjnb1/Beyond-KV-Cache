# Beyond KV Cache

**[Project page](https://wsxhjnb1.github.io/Beyond-KV-Cache/)** | **[Read the paper (PDF)](https://wsxhjnb1.github.io/Beyond-KV-Cache/paper/Beyond.pdf)** | EMNLP 2026 main conference (accepted)

*Beyond: Better-than-Full-Precision KV Caches via Learnable Non-Uniform Quantization as Implicit Regularization*

Yuhao Xie and Mingjie Lin, University of Central Florida.

Beyond learns non-uniform quantization levels and decision thresholds for
4-bit, group-size-32 K/V caches. The forward pass uses hard assignments while
the training path keeps FP32 master parameters and a straight-through gradient.
The inference path stores packed cache values and serves models through vLLM's
OpenAI-compatible API.

This repository contains the paper and the maintained training and inference
code. It does not distribute model weights, datasets, checkpoints, generated
outputs, or measurement data.

## Project website

The public project page and direct paper PDF are served by GitHub Pages from
`docs/` on `main`. This is a static site with no build dependencies. The original
paper is also available at [`paper/Beyond.pdf`](paper/Beyond.pdf).

[EMNLP 2026 conference poster (PDF)](docs/poster/EMNLP2026_Beyond_Poster.pdf).

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
