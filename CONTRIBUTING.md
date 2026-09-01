# Contributing to Beyond

Keep changes focused on the maintained training, quantization, or inference
path. Discuss large API, dependency, training-protocol, or kernel changes in an
issue before implementation.

Do not commit model weights, dataset caches, checkpoints, generated samples,
TensorBoard events, profiler captures, machine-specific paths, credentials, or
full training logs.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .
```

Run the checks relevant to the code you changed:

```bash
ruff check .
ruff format --check .
python -m pytest -q
```

GPU tests require the pinned CUDA/vLLM environment. Record the GPU, driver,
CUDA, PyTorch, and vLLM versions when reporting a GPU failure.

Before opening a pull request, confirm that no secrets, absolute machine paths,
generated run data, or large binaries are included and that third-party files
retain their original license and attribution.

Contributions are licensed under Apache License 2.0 unless a file explicitly
states otherwise.
