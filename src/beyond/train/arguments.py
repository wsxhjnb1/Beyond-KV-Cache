"""Training command-line argument definitions."""

from __future__ import annotations

import argparse
import os

from beyond.models.model_utils import (
    DISCOVER_CHAT_WRAP_PILE_PROMPT,
    DISCOVER_DATA_SEED,
    DISCOVER_DATASET_CHOICES,
    DISCOVER_DATASET_NAME,
    DISCOVER_EVAL_SEED,
    DISCOVER_EXPERTS_IMPLEMENTATION_CHOICES,
    DISCOVER_REFERENCE_TOKENS,
    DISCOVER_TRAIN_SEQLEN,
    SUPPORTED_DISCOVER_MODELS,
)

DISCOVER_TRAIN_SEQLEN_LABEL = f"{DISCOVER_TRAIN_SEQLEN // 1024}K"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Beyond 4-bit/G32 quantizer")
    parser.set_defaults(
        experiment_control="beyond_nll",
        train_quant_points=True,
        train_thresholds=True,
        threshold_grad_mode="half_wave",
    )
    supported_models = ", ".join(SUPPORTED_DISCOVER_MODELS)
    parser.add_argument(
        "--base_model",
        type=str,
        default=SUPPORTED_DISCOVER_MODELS[0],
        help=f"Base model ID or short alias. Supported: {supported_models}.",
    )
    parser.add_argument(
        "--num_bits",
        type=int,
        default=4,
        choices=[4],
        help="Quantization bit width; the maintained path is 4-bit",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=32,
        choices=[32],
        help="Quantization group size; the maintained path uses 32 values per group",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DISCOVER_DATASET_NAME,
        choices=DISCOVER_DATASET_CHOICES,
        help=f"Dataset recipe to use for training. Supported: {', '.join(DISCOVER_DATASET_CHOICES)}.",
    )
    parser.add_argument(
        "--reference_tokens",
        type=int,
        default=DISCOVER_REFERENCE_TOKENS,
        help="Reference token budget used for train schedule sizing and run metadata.",
    )
    parser.set_defaults(chat_wrap_pile=None)
    parser.add_argument(
        "--chat_wrap_pile",
        dest="chat_wrap_pile",
        action="store_true",
        help="Wrap ordinary Pile documents in the model chat template before the target text.",
    )
    parser.add_argument(
        "--no_chat_wrap_pile",
        "--no-chat-wrap-pile",
        dest="chat_wrap_pile",
        action="store_false",
        help="Disable chat-template wrapping for ordinary Pile documents.",
    )
    parser.add_argument(
        "--chat_wrap_pile_prompt",
        type=str,
        default=DISCOVER_CHAT_WRAP_PILE_PROMPT,
        help="User prompt used when chat-wrapping Pile documents.",
    )
    parser.add_argument(
        "--train_steps_per_pass",
        type=int,
        default=None,
        help=(
            "Streaming optimizer/effective-batch steps per pass. Defaults to "
            "DISCOVER_STREAM_TRAIN_STEPS_PER_PASS or 500; <=0 uses that default."
        ),
    )
    parser.add_argument(
        "--pile_train_mode",
        "--pile-train-mode",
        type=str,
        default="streaming",
        choices=["streaming", "materialized"],
        help=(
            "How to feed the selected dataset recipe. 'streaming' streams during training; "
            "'materialized' prepares fixed local chunks before training starts."
        ),
    )

    # Learning rate for custom Adam
    parser.add_argument(
        "--lr",
        type=float,
        default=7e-4,
        help="Learning rate for both q_points and thresholds in adam_custom.",
    )
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="cosine",
        choices=["cosine", "constant"],
        help="Optimizer learning-rate schedule over discover training steps.",
    )
    parser.add_argument(
        "--lr_min_ratio",
        type=float,
        default=0.0,
        help="Minimum learning rate for cosine decay as a fraction of --lr.",
    )

    # Quantizer input gradients use identity STE. q_points/thresholds still
    # receive hard-assignment updates from the quantizer backward pass.
    parser.add_argument(
        "--boundary_window",
        "--boundary-window",
        type=float,
        default=0.009,
        help="Boundary window epsilon used for threshold q_points/thresholds gradients.",
    )
    parser.add_argument(
        "--ce_loss_weight",
        type=float,
        default=1.0,
        help="Weight for the standard next-token cross-entropy loss in the training objective.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Optimizer/model RNG seed. Dataset sampling is controlled independently.",
    )
    parser.add_argument(
        "--data_seed",
        "--data-seed",
        type=int,
        default=DISCOVER_DATA_SEED,
        help="Training-data shuffle/collection seed, independent of --seed.",
    )
    parser.add_argument(
        "--eval_seed",
        "--eval-seed",
        type=int,
        default=DISCOVER_EVAL_SEED,
        help="Validation/eval split seed, independent of --seed and --data_seed.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help=f"Number of passes over the {DISCOVER_TRAIN_SEQLEN_LABEL} token-budget stage.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help=("Target accumulated global sequence batch per optimizer update."),
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=2,
        help=(
            "Training local microbatch per GPU/data-parallel rank. "
            "Gradient accumulation is computed from --batch_size."
        ),
    )
    parser.add_argument(
        "--train_val_ratio",
        type=float,
        default=0.0,
        help="Optional validation split ratio carved out of each training stage (default: 0; set >0 to enable pass-level Val).",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=3,
        help="Stop if validation PPL is worse than best for N consecutive eval intervals (<=0 disables).",
    )
    parser.add_argument(
        "--skip_initial_eval",
        action="store_true",
        help="Skip native eval-set, quantized eval-set, and full validation baselines before training starts.",
    )
    parser.add_argument(
        "--skip_final_eval",
        action="store_true",
        default=os.getenv("DISCOVER_SKIP_FINAL_EVAL", "0").strip().lower()
        in {"1", "true", "yes", "on"},
        help=(
            "Skip final held-out eval after saving quant_config_final.pt. "
            "Can also be enabled with DISCOVER_SKIP_FINAL_EVAL=1."
        ),
    )
    parser.add_argument(
        "--final_eval_max_batches",
        type=int,
        default=int(os.getenv("DISCOVER_FINAL_EVAL_MAX_BATCHES", "0") or 0),
        help=(
            "Cap final held-out eval batches per stage; <=0 runs the full split. "
            "Useful when checkpointing is more important than final PPL."
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing_from_seqlen",
        type=int,
        default=DISCOVER_TRAIN_SEQLEN,
        help=(
            "Enable model gradient checkpointing for training stages with seqlen >= this value. "
            "Set <=0 to disable automatic gradient checkpointing."
        ),
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="auto",
        choices=["auto", "flash_attention_4", "flash_attention_2", "sdpa", "eager"],
        help=(
            "Attention implementation preference. For mistral3/llama/qwen the default auto order is "
            "flash_attention_4 -> flash_attention_2 -> sdpa -> eager."
        ),
    )
    parser.add_argument(
        "--experts_implementation",
        type=str,
        default="auto",
        choices=DISCOVER_EXPERTS_IMPLEMENTATION_CHOICES,
        help=(
            "MoE experts implementation. The auto default uses eager experts for Qwen3-MoE "
            "to avoid the current grouped_mm CUTLASS launch failure while keeping FA4 attention."
        ),
    )
    parser.add_argument(
        "--curriculum_cache_dir",
        type=str,
        default=os.getenv("DISCOVER_CURRICULUM_CACHE_DIR", "~/.cache/discover_curriculum"),
        help=(
            "Directory used for discover curriculum cache. "
            "Defaults to DISCOVER_CURRICULUM_CACHE_DIR or ~/.cache/discover_curriculum."
        ),
    )
    parser.add_argument(
        "--logs_root",
        type=str,
        default=os.getenv("DISCOVER_LOGS_ROOT", "logs"),
        help=(
            "Root directory for run checkpoints and TensorBoard data."
        ),
    )
    parser.add_argument(
        "--runtime_backend",
        type=str,
        default="megatron",
        choices=["megatron"],
        help=(
            "Training runtime abstraction backend. Megatron is currently the only supported backend."
        ),
    )
    parser.add_argument(
        "--launcher",
        type=str,
        default="torchrun",
        choices=["torchrun"],
        help="Distributed launcher mode (Megatron requires torchrun).",
    )
    parser.add_argument(
        "--tensor_model_parallel_size",
        type=int,
        default=1,
        help=(
            "Megatron tensor model parallel size. TP>1 rewrites supported HF "
            "attention/MLP projections to Megatron tensor-parallel linears."
        ),
    )
    parser.add_argument(
        "--allow_linear_only_tensor_parallel",
        action="store_true",
        help=("Deprecated compatibility flag; TP>1 now uses the Megatron tensor-parallel path."),
    )
    parser.add_argument(
        "--pipeline_model_parallel_size",
        type=int,
        default=1,
        help="Megatron pipeline model parallel size.",
    )

    # Logging/metrics
    parser.add_argument(
        "--run_name_suffix", type=str, default="", help="Optional suffix appended to the run name."
    )
    parser.add_argument(
        "--metrics_path", type=str, default="", help="Optional path to write final metrics JSON."
    )
    parser.add_argument(
        "--no_auto_resume",
        action="store_true",
        help="Disable automatic resume from the latest quant_config_step_*.pt checkpoint in the run directory.",
    )

    return parser.parse_args()


__all__ = ["parse_args"]
