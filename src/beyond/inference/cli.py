"""Serve the packed Beyond KV cache through vLLM's OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from beyond.common.compile_cache import configure_compile_cache
from beyond.runtime.vllm.config import env_bool, validate_real_packed_layout

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def _visible_devices_need_standard_all_reduce() -> bool:
    """Detect device sets on which vLLM's custom all-reduce is unsupported."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except Exception:
        output = ""

    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        visible_tokens = None
    else:
        value = value.strip()
        if value.lower() in {"", "-1", "none", "void", "nodevfiles"}:
            visible_tokens = []
        else:
            visible_tokens = [part.strip() for part in value.split(",") if part.strip()]

    def is_visible(index: int, uuid: str) -> bool:
        if visible_tokens is None:
            return True
        for token in visible_tokens:
            if token.isdigit() and int(token) == index:
                return True
            if uuid and (token == uuid or uuid.startswith(token) or token.startswith(uuid)):
                return True
        return False

    capabilities: list[tuple[int, int]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        if not is_visible(index, parts[1]):
            continue
        match = re.search(r"(\d+)\D+(\d+)", parts[2])
        if match:
            capabilities.append((int(match.group(1)), int(match.group(2))))

    if capabilities:
        return any(major == 12 and minor in {0, 1} for major, minor in capabilities)

    arch = os.environ.get("CUTE_DSL_ARCH") or os.environ.get("FLASH_ATTENTION_ARCH", "")
    return arch.startswith(("sm_120", "sm_121"))


def _configure_environment(args: argparse.Namespace) -> None:
    quant_config = Path(args.quant_config).expanduser().resolve()
    if not quant_config.is_file():
        raise SystemExit(f"Quantization configuration does not exist: {quant_config}")

    validate_real_packed_layout(args.bits, args.group_size, args.v_group_size)
    os.environ.setdefault("CUTE_DSL_CACHE_DIR", str(REPO_ROOT / ".cute_dsl_cache"))
    os.environ.setdefault("CUDA_CACHE_PATH", str(REPO_ROOT / ".cuda_cache"))
    os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(2 * 1024 * 1024 * 1024))
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("VLLM_USE_V1", "1")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ["BEYOND_REGISTER_CUSTOM_BACKEND"] = "1"
    os.environ["BEYOND_BITS"] = str(args.bits)
    os.environ["BEYOND_K_GROUP_SIZE"] = str(args.group_size)
    os.environ["BEYOND_V_GROUP_SIZE"] = str(args.v_group_size)
    os.environ["BEYOND_QUANT_CONFIG"] = str(quant_config)
    os.environ["BEYOND_MODEL_ID"] = str(args.model)
    os.environ.setdefault("BEYOND_QUANT_CONFIG_STRICT", "1")
    os.environ.setdefault("BEYOND_VLLM_PAGE_TABLE_EXACT", "0")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local path")
    parser.add_argument("--quant-config", "--quant_config", required=True)
    parser.add_argument("--served-model-name", "--served_model_name", default=None)
    parser.add_argument("--bits", type=int, default=4, choices=[4])
    parser.add_argument("--group-size", "--group_size", type=int, default=32, choices=[32])
    parser.add_argument("--v-group-size", "--v_group_size", type=int, default=32, choices=[32])
    parser.add_argument("--block-size", "--block_size", type=int, default=128, choices=[64, 128])
    parser.add_argument("--tensor-parallel-size", "--tensor_parallel_size", "--tp", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", "--gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--max-model-len", "--max_model_len", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", "--max_num_batched_tokens", type=int, default=None)
    parser.add_argument("--max-num-seqs", "--max_num_seqs", type=int, default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", "--top_p", type=float, default=1.0)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--api-key", "--api_key", default=os.environ.get("BEYOND_API_KEY", "EMPTY"))
    parser.add_argument("--enforce-eager", "--enforce_eager", action="store_true")
    parser.add_argument("--disable-custom-all-reduce", "--disable_custom_all_reduce", action="store_true")
    parser.add_argument("--disable-prefix-caching", action="store_true")
    return parser


def _serve(args: argparse.Namespace) -> None:
    sys.modules.setdefault("flash_attn", None)
    _configure_environment(args)
    configure_compile_cache(REPO_ROOT)

    from beyond.runtime.vllm import packed_backend

    packed_backend.refresh_env_tunables()
    packed_backend.register_backend()

    from vllm.entrypoints.openai import api_server

    api_server.cli_env_setup()
    parser = api_server.FlexibleArgumentParser(description=__doc__)
    parser = api_server.make_arg_parser(parser)

    served_name = args.served_model_name or args.model
    server_argv = [
        "--model",
        args.model,
        "--served-model-name",
        served_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--api-key",
        args.api_key,
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed),
        "--override-generation-config",
        json.dumps(
            {"temperature": float(args.temperature), "top_p": float(args.top_p)},
            separators=(",", ":"),
        ),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--block-size",
        str(args.block_size),
        "--attention-backend",
        "CUSTOM",
        "--disable-log-stats",
    ]
    if not args.disable_prefix_caching:
        server_argv.append("--enable-prefix-caching")
    if args.gpu_memory_utilization is not None:
        server_argv += ["--gpu-memory-utilization", str(args.gpu_memory_utilization)]
    if args.max_model_len is not None:
        server_argv += ["--max-model-len", str(args.max_model_len)]
    if args.max_num_batched_tokens is not None:
        server_argv += ["--max-num-batched-tokens", str(args.max_num_batched_tokens)]
    if args.max_num_seqs is not None:
        server_argv += ["--max-num-seqs", str(args.max_num_seqs)]
    if args.disable_custom_all_reduce or (
        args.tensor_parallel_size > 1 and _visible_devices_need_standard_all_reduce()
    ):
        server_argv.append("--disable-custom-all-reduce")
    if args.enforce_eager or env_bool("BEYOND_VLLM_PAGE_TABLE_EXACT", False):
        server_argv.append("--enforce-eager")

    print(
        "[beyond-serve] "
        f"model={args.model} served={served_name} bits=4 group_size=32 "
        f"tensor_parallel_size={args.tensor_parallel_size}"
    )
    parsed = parser.parse_args(server_argv)
    api_server.validate_parsed_serve_args(parsed)
    api_server.uvloop.run(api_server.run_server(parsed))


def main() -> None:
    args = _build_parser().parse_args()
    _serve(args)


if __name__ == "__main__":
    main()
