# Beyond KV Cache

**[Project page](https://wsxhjnb1.github.io/Beyond-KV-Cache/)** | **[Paper (PDF)](https://wsxhjnb1.github.io/Beyond-KV-Cache/paper/Beyond.pdf)** | **[Poster (PDF)](docs/poster/EMNLP2026_Beyond_Poster.pdf)** | EMNLP 2026 main conference

*Beyond: Better-than-Full-Precision KV Caches via Learnable Non-Uniform Quantization as Implicit Regularization*

Yuhao Xie and Mingjie Lin, University of Central Florida.

KV-cache compression need not be only a trade-off between memory and model
quality. Beyond learns how to quantize the cache from **next-token prediction
loss**, rather than optimizing local reconstruction alone. With the language
model frozen, its learned 4-bit caches improve held-out perplexity and average
LongBench scores across three instruction-tuned models. A fused decode kernel
brings the learned quantizer to packed-cache serving in vLLM.

This repository contains the paper and the maintained training and inference
code. It does not distribute model weights, datasets, checkpoints, training
outputs, or raw measurement data.

## Method

1. **Learn from the downstream loss.** Keep the backbone fixed and learn
   non-uniform reconstruction levels (R) and decision thresholds (T) for K/V
   quantization. The objective is next-token prediction loss, not simply how
   closely each cache entry is reconstructed.
2. **Train with hard assignments.** Use hard quantization in the forward pass,
   straight-through input gradients, and boundary-localized threshold
   pseudo-gradients. Learned parameters are maintained as FP32 master values
   during training.
3. **Serve the packed cache.** Store 4-bit codes in groups of 32 values, then
   fuse their lookup and rescaling into attention decoding. The maintained
   inference path integrates this representation with vLLM's
   OpenAI-compatible API.

Each group stores 128 code bits plus 32 bits of minimum/scale metadata:
**5 effective bits per scalar, or 3.2× KV-payload compression relative to BF16**.
This includes the per-group min/scale metadata but excludes page padding; it is not a claim
of 3.2× lower total GPU memory use.

## Results from the final conference poster

### Model quality

Perplexity is measured on 150 disjoint held-out The Pile chunks. LongBench
Avg21 is the unweighted mean over 21 tasks; higher aggregate scores do not
mean every task improves. Parenthesized differences are score points. Values
and relative reductions follow the final poster; percentages are not
recomputed from the rounded PPL values shown below.

| Backbone | Full-precision PPL ↓ | Beyond PPL ↓ | PPL reduction | Native LongBench Avg21 ↑ | Beyond Avg21 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama-3.1-8B-Instruct | 5.50 | **5.38** | 2.2% | 43.4 | **44.9** (+1.5) |
| Ministral-3-14B-Instruct-2512-BF16 | 6.77 | **6.58** | 2.8% | 45.4 | **47.4** (+2.0) |
| Qwen3-30B-A3B-Instruct-2507 | 5.31 | **4.94** | 6.9% | 48.5 | **50.2** (+1.7) |

Quality experiments use training-equivalent fake quantize/dequantize
(fake-QDQ) evaluation, which retains a floating-point cache. The packed-cache
serving measurements below are separate experiments, not a joint measurement
of every quality and speed gain in one execution path.

### Packed-cache serving

- **Attention decode kernel: up to 1.50× speedup over FlashAttention-4.**
  Measured on one NVIDIA B300 with Llama-3.1-8B-Instruct attention geometry,
  batches 1 and 16, and 1K–128K context lengths; timing uses 50 warmups and 500 CUDA-graph
  replays.
- **End-to-end throughput: up to 1.17× over the FlashAttention-4 path.**
  The peak occurs at batch 1536 in vLLM's official offline throughput
  benchmark on one B300 with Llama-3.1-8B-Instruct, using cold 1024-token
  prompts and 128-token generation,
  BF16 model execution, an 8192-token scheduling budget, GPU memory
  utilization 0.9, and prefix caching disabled.
- **Tested batch capacity: at least 2.84×.** The FlashAttention-4 path
  succeeds at batch 1582; Beyond succeeds through at least batch 4500.
  The first tested KV-capacity failures are 1583 and 4800, respectively.
  These tested points do not establish Beyond's exact maximum batch size
  or a universal out-of-memory boundary.

These are configuration-specific results reported in the poster, not speed,
capacity, or quality guarantees for arbitrary hardware and workloads. See the
[paper](paper/Beyond.pdf) for experimental details and limitations.

## Project website

The public project page and direct paper PDF are served by GitHub Pages from
`docs/` on `main`. This is a static site with no build dependencies. The original
paper is also available at [`paper/Beyond.pdf`](paper/Beyond.pdf).

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
