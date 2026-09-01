"""
Quantization-Aware Training

This module trains learned K/V-cache quantizers for the supported models:
- mistralai/Ministral-3-14B-Instruct-2512-BF16
- meta-llama/Llama-3.1-8B-Instruct
- Qwen/Qwen3-30B-A3B-Instruct-2507

Usage Examples:
1. Ministral 3 14B Instruct:
   python discover.py --base_model mistralai/Ministral-3-14B-Instruct-2512-BF16 --num_bits 4 --epochs 1

2. Llama 3.1 8B Instruct:
   python discover.py --base_model meta-llama/Llama-3.1-8B-Instruct --num_bits 4 --epochs 1

3. Qwen3 30B A3B Instruct:
   python discover.py --base_model Qwen/Qwen3-30B-A3B-Instruct-2507 --num_bits 4 --epochs 1

Features:
- Minimal model and tokenizer loading for the supported target models
- Streaming The Pile text for discover QAT pilots
- Attention layer quantization (k_proj, v_proj)
- Fixed-size held-out validation/eval sets with validation-PPL early stopping
- TensorBoard logging
"""

import hashlib
import json
import math
import os
import re
import shutil
import time
from abc import ABC, abstractmethod
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

_PROCESS_START_TIME = time.perf_counter()
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None

# --- Performance: cuDNN autotuning (TF32 enabled after model load) ---
torch.backends.cudnn.benchmark = True

# Import common utilities
import sys

_REPO_ROOT = str(Path(__file__).resolve().parents[3])
sys.path.append(_REPO_ROOT)
from beyond.common.compile_cache import configure_compile_cache
from beyond.train.arguments import parse_args
from beyond.train.early_stopping import update_validation_early_stopping

configure_compile_cache(_REPO_ROOT)

from beyond.models.model_utils import (
    DISCOVER_CHAT_WRAP_PILE_PROMPT,
    DISCOVER_EVAL_CHUNKS,
    DISCOVER_LENGTH_SCHEDULE_NAME,
    DISCOVER_TRAIN_SEQLEN,
    DISCOVER_VALIDATION_CHUNKS,
    StreamingTokenChunkDataset,
    TextDataset,
    _sample_token_count,
    _samples_token_count,
    apply_megatron_parallel_wrappers,
    build_length_curriculum,
    canonical_model_id,
    collate_text_batch,
    detect_model_type,
    discover_default_chat_wrap_pile,
    get_discover_sources,
    get_model_config,
    get_training_curriculum_dataset,
    load_model,
    load_tokenizer,
    pack_samples_to_seqlen,
    set_random_seed,
    validate_discover_megatron_parallel_request,
    validate_model_setup,
)
from beyond.quantization.training import (
    apply_kv_quantization_targets,
    apply_quant_config,
    build_kv_quant_targets,
    extract_quant_config,
    find_kv_proj_layers,
    load_quant_config,
    project_quant_points,
    project_thresholds,
    resolve_experiment_control,
)

DISCOVER_TRAIN_SEQLEN_LABEL = f"{DISCOVER_TRAIN_SEQLEN // 1024}K"


@dataclass(frozen=True)
class RuntimeBackendInfo:
    name: str
    world_size: int
    local_rank: int = 0
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    tensor_model_parallel_rank: int = 0
    pipeline_model_parallel_rank: int = 0
    data_parallel_rank: int = 0
    data_parallel_world_size: int = 1
    data_parallel_group: Any = None
    tensor_model_parallel_group: Any = None
    pipeline_model_parallel_group: Any = None
    tensor_model_parallel_src_rank: int = 0
    pipeline_model_parallel_prev_rank: int = -1
    pipeline_model_parallel_next_rank: int = -1
    pipeline_model_parallel_first_rank: int = 0
    pipeline_model_parallel_last_rank: int = 0


@dataclass(frozen=True)
class TrainStepResult:
    loss: Any
    ce_weighted_loss: Any = None
    valid: bool = True


def _safe_int(value, minimum=1):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return minimum
    if parsed < minimum:
        return minimum
    return parsed


def _safe_call(fn, args, default):
    if not callable(fn):
        return default
    try:
        return fn(*args)
    except Exception:
        return default


def _safe_group(fn):
    return _safe_call(fn, (), None)


def _initialize_mcore_parallel(tp: int, pp: int):
    try:
        from megatron.core import parallel_state
    except Exception as exc:
        raise RuntimeError(
            "Megatron runtime requested but megatron.core import failed. "
            "Install it from requirements.txt and set --runtime_backend megatron."
        ) from exc

    init_fn = getattr(parallel_state, "initialize_model_parallel", None)
    if not callable(init_fn):
        raise RuntimeError(
            "Megatron runtime requested but megatron.core.parallel_state.initialize_model_parallel is unavailable."
        )

    already_initialized = bool(
        _safe_call(getattr(parallel_state, "is_initialized", None), (), False)
    )
    if not already_initialized:
        try:
            init_fn(
                tensor_model_parallel_size=int(tp),
                pipeline_model_parallel_size=int(pp),
            )
        except TypeError:
            init_fn(int(tp), int(pp))
    return parallel_state


class DataSubsystem(ABC):
    """Dataset / dataloader access abstraction."""

    @abstractmethod
    def train_loader(self, stage: Dict[str, Any]) -> Iterable[Any]:
        raise NotImplementedError

    @abstractmethod
    def val_loader(self, stage: Dict[str, Any]) -> Iterable[Any]:
        raise NotImplementedError

    @abstractmethod
    def eval_loader(self, stage: Dict[str, Any]) -> Iterable[Any]:
        raise NotImplementedError

    def get_loader(self, stage: Dict[str, Any], split: str) -> Optional[Iterable[Any]]:
        split_lower = str(split).lower()
        if split_lower in {"train", "train_dataloader"}:
            return self.train_loader(stage)
        if split_lower in {"val", "validation", "valid", "val_dataloader"}:
            return self.val_loader(stage)
        if split_lower in {"eval", "evaluation", "eval_dataloader"}:
            return self.eval_loader(stage)
        return stage.get(split)


class HFDataSubsystem(DataSubsystem):
    """Current discover dataloader layout is already dict-based."""

    def train_loader(self, stage: Dict[str, Any]) -> Iterable[Any]:
        return stage["train_dataloader"]

    def val_loader(self, stage: Dict[str, Any]) -> Iterable[Any]:
        return stage["val_dataloader"]

    def eval_loader(self, stage: Dict[str, Any]) -> Iterable[Any]:
        return stage["eval_dataloader"]


class TrainSubsystem(ABC):
    """Training-step abstraction."""

    @abstractmethod
    def forward_step(
        self,
        model: Any,
        batch: Dict[str, Any],
        accum_steps_val: int,
        stage_name: Optional[str] = None,
    ) -> Any:
        raise NotImplementedError

    def train_step(
        self,
        model: Any,
        batch: Dict[str, Any],
        accum_steps_val: int,
        stage_name: Optional[str] = None,
    ) -> TrainStepResult:
        result = self.forward_step(model, batch, accum_steps_val, stage_name=stage_name)
        if isinstance(result, TrainStepResult):
            return result
        if isinstance(result, dict):
            result = result.get("loss", float("nan"))
        elif isinstance(result, (tuple, list)):
            if not result:
                return TrainStepResult(float("nan"), valid=False)
            result = result[0]
        try:
            if isinstance(result, torch.Tensor):
                if result.ndim == 0 and torch.isfinite(result).all():
                    return TrainStepResult(result.detach())
                return TrainStepResult(float("nan"), valid=False)
            value = float(result)
            return TrainStepResult(value, valid=math.isfinite(value))
        except Exception:
            return TrainStepResult(float("nan"), valid=False)

    @abstractmethod
    def optimizer_step(
        self,
        grad_scale: float = 1.0,
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def zero_grad(self) -> None:
        raise NotImplementedError


class MegatronTrainSubsystem(TrainSubsystem):
    """Megatron-style training wrapper with explicit forward_step path."""

    def __init__(
        self,
        forward_step_fn,
        apply_optimizer_step_fn,
        backend_stepper: Any,
        optimizer_step_params,
    ):
        self._forward_step_fn = forward_step_fn
        self._apply_optimizer_step_fn = apply_optimizer_step_fn
        self._backend_stepper = backend_stepper
        self._optimizer_step_params = optimizer_step_params
        self._quantizer_modules = []

    def configure(
        self,
        *,
        train_micro_step_fn=None,
        forward_step_fn=None,
        apply_optimizer_step_fn=None,
        backend_stepper=None,
        optimizer_step_params=None,
        quantizer_modules=None,
    ) -> None:
        if forward_step_fn is None:
            forward_step_fn = train_micro_step_fn
        if forward_step_fn is not None:
            self._forward_step_fn = forward_step_fn
        if apply_optimizer_step_fn is not None:
            self._apply_optimizer_step_fn = apply_optimizer_step_fn
        if backend_stepper is not None:
            self._backend_stepper = backend_stepper
        if optimizer_step_params is not None:
            self._optimizer_step_params = optimizer_step_params
        if quantizer_modules is not None:
            self._quantizer_modules = list(quantizer_modules)

    def bind_model(self, model: Any) -> None:
        self._model = model

    def bind_quantizer_modules(self, quantizer_modules):
        self._quantizer_modules = list(quantizer_modules)

    def forward_step(
        self,
        model: Any,
        batch: Dict[str, Any],
        accum_steps_val: int,
        stage_name: Optional[str] = None,
    ) -> Any:
        if self._forward_step_fn is None:
            raise RuntimeError("Megatron forward_step function is not configured")
        return self._forward_step_fn(model, batch, accum_steps_val, stage_name=stage_name)

    def optimizer_step(
        self,
        grad_scale: float = 1.0,
    ) -> bool:
        if self._apply_optimizer_step_fn is None:
            raise RuntimeError("optimizer step function is not configured")
        effective_scale = float(grad_scale)
        return bool(
            self._apply_optimizer_step_fn(
                self._backend_stepper,
                self._optimizer_step_params,
                self._quantizer_modules,
                grad_scale=effective_scale,
            )
        )

    def zero_grad(self) -> None:
        _zero_stepper_grad(self._backend_stepper, self._quantizer_modules)


class EvalSubsystem(ABC):
    """Evaluation abstraction."""

    @abstractmethod
    def run_stages(
        self,
        model_eval: Any,
        stages: Iterable[Dict[str, Any]],
        dataloader_key: str,
        label: str,
        writer_obj: Any = None,
        loss_tag: Optional[str] = None,
        ppl_tag: Optional[str] = None,
        step: Optional[int] = None,
        max_batches_per_stage: int = 0,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class HfEvalSubsystem(EvalSubsystem):
    """Current discover eval path wrapper."""

    def __init__(self, run_stage_ppl_fn, data_subsystem: DataSubsystem):
        self._run_stage_ppl_fn = run_stage_ppl_fn
        self._data_subsystem = data_subsystem

    def configure(self, *, run_stage_ppl_fn=None):
        if run_stage_ppl_fn is not None:
            self._run_stage_ppl_fn = run_stage_ppl_fn

    def run_stages(
        self,
        model_eval: Any,
        stages: Iterable[Dict[str, Any]],
        dataloader_key: str,
        label: str,
        writer_obj: Any = None,
        loss_tag: Optional[str] = None,
        ppl_tag: Optional[str] = None,
        step: Optional[int] = None,
        max_batches_per_stage: int = 0,
    ) -> Dict[str, Any]:
        runtime_dataloader_key = "__runtime_dataloader__"
        prepared_stages = []
        for stage in stages:
            loader = self._data_subsystem.get_loader(stage, dataloader_key)
            if loader is None:
                continue
            stage_with_loader = dict(stage)
            stage_with_loader[runtime_dataloader_key] = loader
            prepared_stages.append(stage_with_loader)
        return self._run_stage_ppl_fn(
            model_eval,
            prepared_stages,
            runtime_dataloader_key,
            label,
            writer_obj=writer_obj,
            loss_tag=loss_tag,
            ppl_tag=ppl_tag,
            step=step,
            max_batches_per_stage=max_batches_per_stage,
        )


class CurriculumScheduler:
    """Curriculum unit math and boundaries."""

    def __init__(self, total_epochs: int, num_curriculum_stages: int, stage_batch_counts):
        if num_curriculum_stages <= 0:
            raise ValueError("num_curriculum_stages must be positive")
        self.total_epochs = max(1, int(total_epochs))
        self.num_curriculum_stages = int(num_curriculum_stages)
        self.stage_batch_counts = [max(0, int(v)) for v in stage_batch_counts]
        self.total_training_units = self.total_epochs * self.num_curriculum_stages
        self.total_scheduled_batches = max(0, sum(self.stage_batch_counts))

    @property
    def unit_ids(self):
        return list(range(self.total_training_units))

    def unit_to_epoch(self, unit_idx: int) -> int:
        return int(unit_idx) // self.num_curriculum_stages

    def unit_stage(self, unit_idx: int) -> int:
        return int(unit_idx) % self.num_curriculum_stages

    def is_pass_boundary(self, unit_idx: int) -> bool:
        return ((unit_idx + 1) % self.num_curriculum_stages == 0) or (
            (unit_idx + 1) == self.total_training_units
        )

    def pass_index(self, unit_idx: int) -> int:
        return (int(unit_idx) + self.num_curriculum_stages) // self.num_curriculum_stages

    def unit_batch_count(self, unit_idx: int) -> int:
        return self.stage_batch_counts[int(unit_idx) % self.num_curriculum_stages]

    def unit_start_offset(self, unit_idx: int) -> int:
        if unit_idx <= 0:
            return 0
        return sum(self.stage_batch_counts[u % self.num_curriculum_stages] for u in range(unit_idx))

    def progress_fraction(self, unit_idx: int, batch_idx: int) -> float:
        if self.total_scheduled_batches <= 1:
            return 0.0
        base = self.unit_start_offset(unit_idx)
        current_count = self.unit_batch_count(unit_idx)
        if current_count <= 0:
            return float(base) / float(self.total_scheduled_batches - 1)
        safe_batch_idx = max(0, min(int(batch_idx), current_count - 1))
        return float(base + safe_batch_idx) / float(self.total_scheduled_batches - 1)

    def unit_start_progress(self, unit_idx: int) -> float:
        if self.total_scheduled_batches <= 1:
            return 0.0
        return min(1.0, self.unit_start_offset(unit_idx) / float(self.total_scheduled_batches - 1))


class MegatronRuntimeBackend:
    """Container representing a Megatron runtime binding."""

    def __init__(
        self,
        data_subsystem: DataSubsystem,
        train_subsystem: TrainSubsystem,
        eval_subsystem: EvalSubsystem,
        backend_info: RuntimeBackendInfo,
    ):
        self.data = data_subsystem
        self.train = train_subsystem
        self.eval = eval_subsystem
        self.info = backend_info

    def __iter__(self):
        yield self.data
        yield self.train
        yield self.eval
        yield self.info


def _resolve_megatron_parallel(
    world_size: int,
    local_rank: int,
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
):
    tp = max(1, int(tensor_model_parallel_size))
    pp = max(1, int(pipeline_model_parallel_size))
    world_size = max(1, int(world_size))

    parallel_world = tp * pp
    if world_size < parallel_world:
        raise RuntimeError(
            f"world_size={world_size} is smaller than tensor_model_parallel_size*"
            f"pipeline_model_parallel_size={parallel_world}"
        )
    if world_size % parallel_world != 0:
        raise RuntimeError(
            f"world_size={world_size} must be divisible by tensor_model_parallel_size*"
            f"pipeline_model_parallel_size={parallel_world}"
        )
    os.environ["TENSOR_MODEL_PARALLEL_SIZE"] = str(tp)
    os.environ["PIPELINE_MODEL_PARALLEL_SIZE"] = str(pp)
    os.environ["MEGATRON_PARALLEL_WORLD_SIZE"] = str(parallel_world)
    os.environ["MEGATRON_DATA_PARALLEL_SIZE"] = str(max(1, world_size // parallel_world))

    if parallel_world > 1:
        os.environ["VIRTUAL_PIPELINE_MODEL_PARALLEL_SIZE"] = str(max(1, pp))

    parallel_state = _initialize_mcore_parallel(tp, pp)

    data_world_size = max(1, world_size // parallel_world)
    data_rank = max(0, local_rank // parallel_world)
    tensor_rank = local_rank % tp if tp > 0 else 0
    pipeline_rank = (local_rank // max(1, tp)) % pp if pp > 0 else 0

    if parallel_state is not None:
        data_world_size = _safe_int(
            _safe_call(
                getattr(parallel_state, "get_data_parallel_world_size", None), (), data_world_size
            ),
            minimum=1,
        )
        data_rank = _safe_call(
            getattr(parallel_state, "get_data_parallel_rank", None),
            (),
            data_rank,
        )
        tensor_rank = _safe_call(
            getattr(parallel_state, "get_tensor_model_parallel_rank", None),
            (),
            tensor_rank,
        )
        pipeline_rank = _safe_call(
            getattr(parallel_state, "get_pipeline_model_parallel_rank", None),
            (),
            pipeline_rank,
        )

        data_parallel_group = _safe_group(getattr(parallel_state, "get_data_parallel_group", None))
        tensor_model_parallel_group = _safe_group(
            getattr(parallel_state, "get_tensor_model_parallel_group", None)
        )
        pipeline_model_parallel_group = _safe_group(
            getattr(parallel_state, "get_pipeline_model_parallel_group", None)
        )
        tensor_model_parallel_src_rank = _safe_call(
            getattr(parallel_state, "get_tensor_model_parallel_src_rank", None),
            (),
            None,
        )
        pipeline_prev_rank = _safe_call(
            getattr(parallel_state, "get_pipeline_model_parallel_prev_rank", None),
            (),
            None,
        )
        pipeline_next_rank = _safe_call(
            getattr(parallel_state, "get_pipeline_model_parallel_next_rank", None),
            (),
            None,
        )
        pipeline_first_rank = _safe_call(
            getattr(parallel_state, "get_pipeline_model_parallel_first_rank", None),
            (),
            None,
        )
        pipeline_last_rank = _safe_call(
            getattr(parallel_state, "get_pipeline_model_parallel_last_rank", None),
            (),
            None,
        )
    else:
        data_parallel_group = None
        tensor_model_parallel_group = None
        pipeline_model_parallel_group = None
        tensor_model_parallel_src_rank = None
        pipeline_prev_rank = None
        pipeline_next_rank = None
        pipeline_first_rank = None
        pipeline_last_rank = None

    data_world_size = _safe_int(data_world_size, minimum=1)
    data_rank = max(0, int(data_rank))
    tensor_rank = max(0, int(tensor_rank))
    pipeline_rank = max(0, int(pipeline_rank))
    if tensor_model_parallel_src_rank is None:
        tensor_model_parallel_src_rank = local_rank - tensor_rank
    tensor_model_parallel_src_rank = max(0, int(tensor_model_parallel_src_rank))
    if pipeline_first_rank is None:
        pipeline_first_rank = local_rank - pipeline_rank * tp
    if pipeline_last_rank is None:
        pipeline_last_rank = int(pipeline_first_rank) + (pp - 1) * tp
    if pipeline_rank <= 0:
        pipeline_prev_rank = -1
    elif pipeline_prev_rank is None:
        pipeline_prev_rank = local_rank - tp
    if pipeline_rank >= pp - 1:
        pipeline_next_rank = -1
    elif pipeline_next_rank is None:
        pipeline_next_rank = local_rank + tp
    return RuntimeBackendInfo(
        name="megatron",
        world_size=world_size,
        local_rank=local_rank,
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        tensor_model_parallel_rank=tensor_rank,
        pipeline_model_parallel_rank=pipeline_rank,
        data_parallel_rank=data_rank,
        data_parallel_world_size=data_world_size,
        data_parallel_group=data_parallel_group,
        tensor_model_parallel_group=tensor_model_parallel_group,
        pipeline_model_parallel_group=pipeline_model_parallel_group,
        tensor_model_parallel_src_rank=tensor_model_parallel_src_rank,
        pipeline_model_parallel_prev_rank=int(pipeline_prev_rank),
        pipeline_model_parallel_next_rank=int(pipeline_next_rank),
        pipeline_model_parallel_first_rank=int(pipeline_first_rank),
        pipeline_model_parallel_last_rank=int(pipeline_last_rank),
    )


def initialize_megatron_runtime(
    world_size: int,
    local_rank: int,
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
):
    """Initialize Megatron model-parallel state and return runtime metadata."""
    return _resolve_megatron_parallel(
        world_size=world_size,
        local_rank=local_rank,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
    )


def build_runtime_backends(
    args,
    train_micro_step_fn=None,
    apply_optimizer_step_fn=None,
    run_stage_ppl_fn=None,
    runtime_backend_info=None,
):
    """Build runtime adapters for discover."""

    if getattr(args, "runtime_backend", "megatron") != "megatron":
        raise RuntimeError(
            f"Unsupported runtime backend: {getattr(args, 'runtime_backend', 'megatron')}"
        )

    data = HFDataSubsystem()
    backend_info = runtime_backend_info or _resolve_megatron_parallel(
        world_size=int(getattr(args, "world_size", 1)),
        local_rank=int(getattr(args, "local_rank", 0)),
        tensor_model_parallel_size=int(getattr(args, "tensor_model_parallel_size", 1)),
        pipeline_model_parallel_size=int(getattr(args, "pipeline_model_parallel_size", 1)),
    )
    train = MegatronTrainSubsystem(
        forward_step_fn=train_micro_step_fn,
        apply_optimizer_step_fn=apply_optimizer_step_fn,
        backend_stepper=getattr(args, "backend_stepper", None),
        optimizer_step_params=getattr(args, "optimizer_step_params", None),
    )
    eval_subsystem = HfEvalSubsystem(run_stage_ppl_fn, data)
    backend = MegatronRuntimeBackend(data, train, eval_subsystem, backend_info)
    return backend


# --- Define command line arguments ---

# --- Parse arguments ---
args = parse_args()
args.base_model = canonical_model_id(args.base_model)
experiment_control = resolve_experiment_control(args.experiment_control)
args.threshold_grad_mode = str(args.threshold_grad_mode or "half_wave").strip().lower()
os.environ["DISCOVER_THRESHOLD_GRAD_MODE"] = args.threshold_grad_mode
if not math.isfinite(float(args.boundary_window)) or float(args.boundary_window) <= 0.0:
    raise ValueError("--boundary_window must be a positive finite value.")
args.boundary_window = float(args.boundary_window)
os.environ["DISCOVER_BOUNDARY_WINDOW"] = str(args.boundary_window)
if args.chat_wrap_pile is None:
    args.chat_wrap_pile = discover_default_chat_wrap_pile(args.base_model)
args.chat_wrap_pile = bool(args.chat_wrap_pile)
args.chat_wrap_pile_prompt = str(
    args.chat_wrap_pile_prompt or DISCOVER_CHAT_WRAP_PILE_PROMPT
).strip()
if not args.chat_wrap_pile_prompt:
    args.chat_wrap_pile_prompt = DISCOVER_CHAT_WRAP_PILE_PROMPT
DISCOVER_ACTIVE_SOURCES = get_discover_sources(
    args.dataset,
    chat_wrap_pile=args.chat_wrap_pile,
    chat_user_prompt=args.chat_wrap_pile_prompt,
)
if str(args.pile_train_mode) != "materialized" and any(
    source.get("target_unit") == "supervised_tokens" for source in DISCOVER_ACTIVE_SOURCES
):
    raise ValueError(
        "Dataset recipes with equal supervised-token quotas require "
        "--pile_train_mode materialized; streaming allocates fixed chunks and cannot "
        "guarantee an exact supervised-token mixture."
    )
DISCOVER_LONG_DOC_TRUNCATION = any(
    str(source.get("doc_chunk_mode") or "").lower().startswith("truncate")
    for source in DISCOVER_ACTIVE_SOURCES
)
if not math.isfinite(float(args.ce_loss_weight)) or float(args.ce_loss_weight) < 0.0:
    raise ValueError("--ce_loss_weight must be a non-negative finite value.")
if (
    experiment_control["training_objective"] != "causal_lm_nll"
    and float(args.ce_loss_weight) != 1.0
):
    raise ValueError(
        "--ce_loss_weight is only defined for NLL controls; keep it at 1.0 for "
        f"--experiment_control {args.experiment_control}."
    )
if experiment_control["quantizer_parameterization"] == "uniform_affine_endpoints" and (
    not args.train_quant_points or not args.train_thresholds
):
    raise ValueError(
        "uniform_affine_nll always trains both low/high endpoints; the q-point and "
        "threshold freeze ablations are only defined for Beyond's non-uniform tables."
    )
if experiment_control["quantizer_parameterization"] in {
    "full_precision_chebyshev_residual",
    "full_precision_bucket_residual",
} and (not args.train_quant_points or not args.train_thresholds):
    raise ValueError(
        f"{args.experiment_control} always trains its complete parameter-matched "
        "adapter; q-point/threshold freeze flags do not apply."
    )
if not math.isfinite(float(args.lr)) or float(args.lr) < 0.0:
    raise ValueError("--lr must be a non-negative finite value.")
if not math.isfinite(float(args.lr_min_ratio)) or not (0.0 <= float(args.lr_min_ratio) <= 1.0):
    raise ValueError("--lr_min_ratio must be a finite value in [0, 1].")
if int(args.tensor_model_parallel_size) > 1:
    raise ValueError(
        "Per-group quant tables currently require --tensor_model_parallel_size 1; "
        "the per-group tables are local-head parameters and are not yet sharded "
        "through the TP resume/save path."
    )

# --- Distributed/backend initialization ---
_world_size = int(os.environ.get("WORLD_SIZE", "1"))
_rank = int(os.environ.get("RANK", "0"))
_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
_launcher = (args.launcher or "").strip().lower()
if args.runtime_backend != "megatron":
    raise ValueError("discover.py now uses Megatron runtime only. Set --runtime_backend megatron.")
if _launcher not in {"", "torchrun"}:
    raise ValueError("Megatron runtime requires --launcher torchrun.")
_launcher = "torchrun"
args.launcher = _launcher
_distributed = _world_size > 1
if args.runtime_backend == "megatron" and args.launcher != "torchrun":
    raise RuntimeError("Megatron backend requires launcher=torchrun.")
_torchrun_env_ready = all(
    key in os.environ for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
)
_requires_process_group = _distributed or args.runtime_backend == "megatron"
if _requires_process_group and not _torchrun_env_ready:
    raise RuntimeError(
        "Megatron backend must be launched with torchrun, even for a single GPU. "
        "Use: torchrun --standalone --nproc_per_node=1 analyze/discover.py ..."
    )

if args.runtime_backend == "megatron":
    tp_world = max(1, int(args.tensor_model_parallel_size))
    pp_world = max(1, int(args.pipeline_model_parallel_size))
    parallel_world = tp_world * pp_world
    if _world_size < parallel_world:
        raise RuntimeError(
            "WORLD_SIZE must be at least tensor_model_parallel_size * pipeline_model_parallel_size "
            f"({_world_size} < {tp_world} * {pp_world})."
        )
    if _world_size % parallel_world != 0:
        raise RuntimeError(
            f"WORLD_SIZE={_world_size} must be divisible by "
            f"tensor_model_parallel_size * pipeline_model_parallel_size={parallel_world}."
        )
    validate_discover_megatron_parallel_request(
        world_size=_world_size,
        tensor_model_parallel_size=tp_world,
        pipeline_model_parallel_size=pp_world,
        allow_linear_only_tensor_parallel=bool(args.allow_linear_only_tensor_parallel),
    )
    os.environ["TENSOR_MODEL_PARALLEL_SIZE"] = str(tp_world)
    os.environ["PIPELINE_MODEL_PARALLEL_SIZE"] = str(pp_world)

if _requires_process_group:
    if _world_size > 8:
        raise ValueError(f"discover.py supports 1-8 GPUs; got WORLD_SIZE={_world_size}.")
    if not torch.cuda.is_available():
        raise RuntimeError("Megatron runtime requires CUDA.")
    pg_timeout_seconds = int(os.getenv("DISCOVER_PG_TIMEOUT_SECONDS", "3600"))
    init_timeout = timedelta(seconds=max(60, pg_timeout_seconds))
    visible_devices = torch.cuda.device_count()
    if _local_rank >= visible_devices:
        raise RuntimeError(
            f"LOCAL_RANK={_local_rank} is out of range for {visible_devices} visible CUDA devices."
        )
    torch.cuda.set_device(_local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", timeout=init_timeout)
    if _distributed:
        print(
            f"[Rank {_rank}/{_world_size}] Distributed process group initialized on GPU {_local_rank} ({_launcher})."
        )
    elif _rank == 0:
        print(
            f"[Rank {_rank}/{_world_size}] Single-GPU process group initialized on GPU {_local_rank} ({_launcher})."
        )

# --- Megatron parallel initialization (runtime/backend-aware) ---
_data_parallel_world_size = _world_size
_data_parallel_rank = _rank
_tensor_model_parallel_rank = 0
_pipeline_model_parallel_rank = 0
_data_parallel_group = None
_tensor_model_parallel_group = None
_tensor_model_parallel_src_rank = 0
_pipeline_model_parallel_group = None
_pipeline_model_parallel_prev_rank = -1
_pipeline_model_parallel_next_rank = -1
_pipeline_model_parallel_first_rank = 0
_pipeline_model_parallel_last_rank = 0
if args.runtime_backend == "megatron" and _distributed:
    megatron_runtime_info = initialize_megatron_runtime(
        world_size=_world_size,
        local_rank=_rank,
        tensor_model_parallel_size=int(args.tensor_model_parallel_size),
        pipeline_model_parallel_size=int(args.pipeline_model_parallel_size),
    )
    _data_parallel_world_size = max(
        1, int(getattr(megatron_runtime_info, "data_parallel_world_size", _world_size))
    )
    _data_parallel_rank = max(0, int(getattr(megatron_runtime_info, "data_parallel_rank", _rank)))
    _tensor_model_parallel_rank = max(
        0, int(getattr(megatron_runtime_info, "tensor_model_parallel_rank", 0))
    )
    _pipeline_model_parallel_rank = max(
        0, int(getattr(megatron_runtime_info, "pipeline_model_parallel_rank", 0))
    )
    _data_parallel_group = getattr(megatron_runtime_info, "data_parallel_group", None)
    _tensor_model_parallel_group = getattr(
        megatron_runtime_info, "tensor_model_parallel_group", None
    )
    _pipeline_model_parallel_group = getattr(
        megatron_runtime_info, "pipeline_model_parallel_group", None
    )
    _tensor_model_parallel_src_rank = max(
        0,
        int(
            getattr(
                megatron_runtime_info,
                "tensor_model_parallel_src_rank",
                _rank - _tensor_model_parallel_rank,
            )
        ),
    )
    _pipeline_model_parallel_prev_rank = int(
        getattr(megatron_runtime_info, "pipeline_model_parallel_prev_rank", -1)
    )
    _pipeline_model_parallel_next_rank = int(
        getattr(megatron_runtime_info, "pipeline_model_parallel_next_rank", -1)
    )
    _pipeline_model_parallel_first_rank = int(
        getattr(megatron_runtime_info, "pipeline_model_parallel_first_rank", 0)
    )
    _pipeline_model_parallel_last_rank = int(
        getattr(megatron_runtime_info, "pipeline_model_parallel_last_rank", _world_size - 1)
    )
    if _rank == 0:
        print(
            f"[Rank {_rank}/{_world_size}] Megatron parallel initialized: "
            f"tp={int(args.tensor_model_parallel_size)}, pp={int(args.pipeline_model_parallel_size)}, "
            f"dp_rank={_data_parallel_rank}, dp_world={_data_parallel_world_size}, "
            f"tp_rank={_tensor_model_parallel_rank}, pp_rank={_pipeline_model_parallel_rank}",
            flush=True,
        )
else:
    megatron_runtime_info = None


def is_main():
    return _rank == 0


def dist_barrier(label="barrier"):
    if _distributed and torch.distributed.is_initialized():
        debug_barriers = os.getenv("DISCOVER_DEBUG_BARRIERS", "0").lower() not in {
            "",
            "0",
            "false",
            "no",
        }
        if debug_barriers:
            print(f"[Rank {_rank}/{_world_size}] entering barrier: {label}", flush=True)
        barrier_kwargs = {}
        if torch.distributed.get_backend() == "nccl":
            barrier_kwargs["device_ids"] = [_local_rank]
        try:
            torch.distributed.barrier(**barrier_kwargs)
        except TypeError as exc:
            # Older torch builds (including some 2.x series) do not expose a `timeout`
            # or `device_ids` argument for barrier(). Fallback for compatibility.
            msg = str(exc)
            if "unexpected keyword argument" in msg:
                torch.distributed.barrier()
            else:
                raise
        if debug_barriers:
            print(f"[Rank {_rank}/{_world_size}] leaving barrier: {label}", flush=True)


def _distributed_reduce_group():
    if args.runtime_backend == "megatron" and _distributed and _data_parallel_group is not None:
        return _data_parallel_group
    return None


def _tensor_parallel_reduce_group():
    if (
        args.runtime_backend == "megatron"
        and _distributed
        and int(args.tensor_model_parallel_size) > 1
        and _tensor_model_parallel_group is not None
    ):
        return _tensor_model_parallel_group
    return None


def _distributed_all_reduce(tensor, op):
    if not _distributed:
        return
    reduce_group = _distributed_reduce_group()
    if reduce_group is not None:
        torch.distributed.all_reduce(tensor, op=op, group=reduce_group)
    elif args.runtime_backend == "megatron" and (
        int(args.tensor_model_parallel_size) > 1 or int(args.pipeline_model_parallel_size) > 1
    ):
        if _data_parallel_world_size > 1:
            raise RuntimeError(
                "Megatron data-parallel group is unavailable for model-parallel reduction."
            )
        return
    else:
        torch.distributed.all_reduce(tensor, op=op)


def _all_reduce_group(tensor, op, group=None):
    if not _distributed:
        return
    if group is not None:
        torch.distributed.all_reduce(tensor, op=op, group=group)
    else:
        torch.distributed.all_reduce(tensor, op=op)


def rank0_print(*msg, **kwargs):
    if is_main():
        kwargs.setdefault("flush", True)
        print(*msg, **kwargs)


def _env_flag(name, default="0"):
    return os.getenv(name, default).lower() not in {"", "0", "false", "no"}


def _env_int(name, default, *, minimum=0):
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return int(default)
    if value < minimum:
        return int(default)
    return value


def _is_oom_exception(exc):
    msg = str(exc).lower()
    oom_type = getattr(torch, "OutOfMemoryError", MemoryError)
    return (
        "out of memory" in msg
        or "out-of-memory" in msg
        or isinstance(exc, oom_type)
        or isinstance(exc, MemoryError)
    )


def _default_stream_train_workers():
    cpu_count = os.cpu_count() or 1
    per_rank_cpus = max(1, cpu_count // max(1, _world_size))
    if per_rank_cpus <= 2:
        return 0
    return min(4, max(1, per_rank_cpus // 2))


def rank_debug_print(flag_name, *msg):
    if _env_flag(flag_name):
        print(f"[Rank {_rank}/{_world_size}]", *msg, flush=True)


PACK_TRAIN_TO_SEQLEN = _env_flag("DISCOVER_PACK_TRAIN_TO_SEQLEN", "1")
DROP_TRAIN_PACK_REMAINDER = _env_flag("DISCOVER_DROP_TRAIN_PACK_REMAINDER", "1")
EVAL_EVERY_STAGE = _env_flag("DISCOVER_EVAL_EVERY_STAGE", "0")
EVAL_EVERY_STEPS = _env_int("DISCOVER_EVAL_EVERY_STEPS", 10, minimum=0)
PERIODIC_EVAL_TARGET_BATCHES = _env_int("DISCOVER_PERIODIC_EVAL_TARGET_BATCHES", 0, minimum=0)
STREAM_TRAIN_NUM_WORKERS = _env_int(
    "DISCOVER_STREAM_TRAIN_NUM_WORKERS",
    _default_stream_train_workers(),
    minimum=0,
)
STREAM_TRAIN_PREFETCH_FACTOR = _env_int("DISCOVER_STREAM_TRAIN_PREFETCH_FACTOR", 4, minimum=1)
TQDM_POSTFIX_INTERVAL = _env_int("DISCOVER_TQDM_POSTFIX_INTERVAL", 5, minimum=1)
STREAM_TRAIN_STEPS_PER_PASS = _env_int("DISCOVER_STREAM_TRAIN_STEPS_PER_PASS", 500, minimum=1)
TRAIN_LOSS_SYNC_INTERVAL = _env_int("DISCOVER_TRAIN_LOSS_SYNC_INTERVAL", 0, minimum=0)
STRICT_TRAIN_LOSS_CHECKS = _env_flag("DISCOVER_STRICT_TRAIN_LOSS_CHECKS", "0")
CHECK_OPTIMIZER_GRADS = _env_flag("DISCOVER_CHECK_OPTIMIZER_GRADS", "0")
ASSUME_QUANTIZER_GRADS = _env_flag("DISCOVER_ASSUME_QUANTIZER_GRADS", "1")
if args.train_steps_per_pass is None or int(args.train_steps_per_pass) <= 0:
    args.train_steps_per_pass = STREAM_TRAIN_STEPS_PER_PASS
else:
    args.train_steps_per_pass = int(args.train_steps_per_pass)


def build_doc_attn_mask(position_ids, dtype=torch.bfloat16):
    """Build a causal block mask for packed samples whose position_ids reset per document."""

    if position_ids is None:
        return None
    if position_ids.dim() != 2:
        raise ValueError(
            f"position_ids must be 2D [batch, seq], got shape {tuple(position_ids.shape)}"
        )

    device_pos = position_ids.device
    batch_size, seq_len = position_ids.shape
    doc_starts = position_ids.eq(0)
    if seq_len > 0:
        doc_starts[:, 0] = True
    doc_ids = doc_starts.to(torch.long).cumsum(dim=1)
    same_doc = doc_ids[:, :, None].eq(doc_ids[:, None, :])
    causal = torch.arange(seq_len, device=device_pos)
    causal = causal[None, :, None] >= causal[None, None, :]
    allowed = same_doc & causal

    mask = torch.zeros((batch_size, 1, seq_len, seq_len), device=device_pos, dtype=dtype)
    mask_value = torch.finfo(dtype).min if torch.is_floating_point(mask) else -1
    return mask.masked_fill(~allowed[:, None, :, :], mask_value)


def _get_causal_lm_parts(model_obj):
    model_for_parts = unwrap_model(model_obj)
    backbone = getattr(model_for_parts, "model", None)
    if backbone is None:
        backbone = getattr(model_for_parts, "transformer", None)
    lm_head = getattr(model_for_parts, "lm_head", None)
    if lm_head is None and hasattr(model_for_parts, "get_output_embeddings"):
        lm_head = model_for_parts.get_output_embeddings()
    if backbone is None or lm_head is None:
        raise RuntimeError(
            "Streaming LM loss requires a causal LM with a backbone `.model`/`.transformer` "
            "and an output embedding/lm_head."
        )
    return model_for_parts, backbone, lm_head


def _backbone_forward(backbone, *, input_ids, attention_mask, position_ids, use_cache):
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": use_cache,
        "return_dict": True,
    }
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    try:
        return backbone(**kwargs)
    except TypeError:
        kwargs.pop("return_dict", None)
        return backbone(**kwargs)


def _pipeline_enabled():
    return _distributed and int(args.pipeline_model_parallel_size) > 1


def _pipeline_is_first_stage():
    return not _pipeline_enabled() or _pipeline_model_parallel_rank == 0


def _pipeline_is_last_stage():
    return (
        not _pipeline_enabled()
        or _pipeline_model_parallel_rank == int(args.pipeline_model_parallel_size) - 1
    )


def _pipeline_layer_bounds(num_layers):
    pp = max(1, int(args.pipeline_model_parallel_size))
    rank = max(0, int(_pipeline_model_parallel_rank))
    start = (num_layers * rank) // pp
    end = (num_layers * (rank + 1)) // pp
    return start, end


def _pipeline_layer_index_from_name(name):
    match = re.search(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)(?:\.|$)", str(name))
    if match:
        return int(match.group(1))
    return None


def _pipeline_target_is_local(layer_name):
    if not _pipeline_enabled():
        return True
    model_for_parts, backbone, _lm_head = _get_causal_lm_parts(model)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        layers = getattr(backbone, "h", None)
    if layers is None:
        return True
    layer_idx = _pipeline_layer_index_from_name(layer_name)
    if layer_idx is None:
        return _pipeline_is_first_stage()
    start, end = _pipeline_layer_bounds(len(layers))
    return start <= layer_idx < end


def _normalize_attention_mask_for_fast_path(attention_mask):
    if attention_mask is None:
        return None
    if torch.is_tensor(attention_mask) and attention_mask.dim() == 2:
        try:
            if bool(attention_mask.all().item()):
                return None
        except Exception:
            return attention_mask
    return attention_mask


def _send_tensor(tensor, dst_rank, tag_base):
    if dst_rank < 0:
        return
    tensor = tensor.contiguous()
    shape = torch.tensor(list(tensor.shape), device=tensor.device, dtype=torch.long)
    ndim = torch.tensor([shape.numel()], device=tensor.device, dtype=torch.long)
    torch.distributed.send(ndim, dst=dst_rank, tag=tag_base)
    torch.distributed.send(shape, dst=dst_rank, tag=tag_base + 1)
    torch.distributed.send(tensor, dst=dst_rank, tag=tag_base + 2)


def _recv_tensor(src_rank, dtype, device_recv, tag_base):
    if src_rank < 0:
        return None
    ndim = torch.empty(1, device=device_recv, dtype=torch.long)
    torch.distributed.recv(ndim, src=src_rank, tag=tag_base)
    shape = torch.empty(int(ndim.item()), device=device_recv, dtype=torch.long)
    torch.distributed.recv(shape, src=src_rank, tag=tag_base + 1)
    tensor = torch.empty(tuple(int(v.item()) for v in shape), device=device_recv, dtype=dtype)
    torch.distributed.recv(tensor, src=src_rank, tag=tag_base + 2)
    return tensor


def _pipeline_prepare_masks_and_positions(
    backbone, hidden_states, attention_mask, position_ids, use_cache=False
):
    if position_ids is None:
        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(
            0
        )
    attention_mask = _normalize_attention_mask_for_fast_path(attention_mask)

    config = getattr(backbone, "config", getattr(unwrap_model(model), "config", None))
    if config is None:
        return attention_mask, position_ids, None

    try:
        from transformers.masking_utils import create_causal_mask
    except Exception:
        return attention_mask, position_ids, None

    try:
        if hasattr(config, "layer_types"):
            from transformers.masking_utils import create_sliding_window_causal_mask

            mask_kwargs = {
                "config": config,
                "inputs_embeds": hidden_states,
                "attention_mask": attention_mask,
                "past_key_values": None,
                "position_ids": position_ids,
            }
            sliding_mask_kwargs = dict(mask_kwargs)
            if getattr(config, "use_bidirectional_attention", False):
                try:
                    from transformers.models.gemma3.modeling_gemma3 import (
                        _bidirectional_window_overlay,
                    )

                    mask_kwargs["or_mask_function"] = lambda *args: torch.tensor(
                        True, dtype=torch.bool
                    )
                    sliding_mask_kwargs["or_mask_function"] = _bidirectional_window_overlay(
                        config.sliding_window
                    )
                except Exception:
                    pass
            masks = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**sliding_mask_kwargs),
            }
            position_embeddings = {}
            for layer_type in set(config.layer_types):
                try:
                    position_embeddings[layer_type] = backbone.rotary_emb(
                        hidden_states, position_ids, layer_type
                    )
                except TypeError:
                    position_embeddings[layer_type] = backbone.rotary_emb(
                        hidden_states,
                        position_ids=position_ids,
                        layer_type=layer_type,
                    )
            return masks, position_ids, position_embeddings

        causal_mask = create_causal_mask(
            config=config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )
        try:
            position_embeddings = backbone.rotary_emb(hidden_states, position_ids=position_ids)
        except TypeError:
            position_embeddings = backbone.rotary_emb(hidden_states, position_ids)
        return causal_mask, position_ids, position_embeddings
    except Exception:
        return attention_mask, position_ids, None


def _pipeline_local_forward(
    model_obj, input_ids, attention_mask, position_ids, use_cache=False, enable_backward=False
):
    model_for_parts, backbone, _lm_head = _get_causal_lm_parts(model_obj)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        layers = getattr(backbone, "h", None)
    if layers is None:
        raise RuntimeError(
            "Pipeline parallelism requires a backbone with `.layers` or `.h` decoder layers."
        )

    model_dtype = next(model_for_parts.parameters()).dtype
    if _pipeline_is_first_stage():
        hidden_states = backbone.embed_tokens(input_ids)
    else:
        hidden_states = _recv_tensor(
            _pipeline_model_parallel_prev_rank,
            dtype=model_dtype,
            device_recv=input_ids.device,
            tag_base=10_000,
        )
        if enable_backward:
            hidden_states = hidden_states.detach().requires_grad_(True)

    mask_payload, position_ids, position_embeddings = _pipeline_prepare_masks_and_positions(
        backbone,
        hidden_states,
        attention_mask,
        position_ids,
        use_cache=use_cache,
    )

    start, end = _pipeline_layer_bounds(len(layers))
    local_input = hidden_states
    for layer_idx in range(start, end):
        decoder_layer = layers[layer_idx]
        if isinstance(mask_payload, dict):
            layer_type = getattr(
                getattr(backbone, "config", None), "layer_types", [None] * len(layers)
            )[layer_idx]
            layer_mask = mask_payload.get(layer_type)
            layer_pos = (
                position_embeddings.get(layer_type)
                if isinstance(position_embeddings, dict)
                else position_embeddings
            )
        else:
            layer_mask = mask_payload
            layer_pos = position_embeddings

        kwargs = {
            "attention_mask": layer_mask,
            "position_ids": position_ids,
            "past_key_values": None,
            "use_cache": use_cache,
        }
        if layer_pos is not None:
            kwargs["position_embeddings"] = layer_pos
        try:
            hidden_states = decoder_layer(hidden_states, **kwargs)
        except TypeError:
            kwargs.pop("use_cache", None)
            hidden_states = decoder_layer(hidden_states, **kwargs)
        if isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]

    if _pipeline_is_last_stage():
        hidden_states = backbone.norm(hidden_states)
    else:
        _send_tensor(hidden_states.detach(), _pipeline_model_parallel_next_rank, tag_base=10_000)

    return hidden_states, local_input


def _maybe_gather_lm_head_params(lm_head):
    return nullcontext()


def _final_logit_softcap_value(model_for_parts):
    config = getattr(model_for_parts, "config", None)
    configs = []
    if config is not None:
        get_text_config = getattr(config, "get_text_config", None)
        if callable(get_text_config):
            try:
                configs.append(get_text_config())
            except Exception:
                pass
        configs.append(config)

    for candidate in configs:
        value = getattr(candidate, "final_logit_softcapping", None)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            return value
    return None


def _apply_final_logit_softcap(logits, model_for_parts):
    softcap = _final_logit_softcap_value(model_for_parts)
    if softcap is None:
        return logits
    return torch.tanh(logits / softcap) * softcap


def _lm_head_loss_chunk_tokens():
    raw_value = os.getenv("DISCOVER_LM_HEAD_CHUNK_TOKENS", "2048")
    try:
        chunk_tokens = int(raw_value)
    except (TypeError, ValueError):
        chunk_tokens = 2048
    return max(1, chunk_tokens)


def _streaming_lm_head_loss_from_hidden(
    model_for_parts,
    lm_head,
    hidden_states,
    labels,
    *,
    backward_scale=1.0,
    loss_normalizer=None,
    return_hidden_grad=False,
):
    _batch_size, seq_len, _ = hidden_states.shape
    autocast_enabled = hidden_states.is_cuda
    ignore_index = -100
    vocab_size = int(getattr(getattr(model_for_parts, "config", None), "vocab_size", 0) or 0)
    valid_width = max(0, min(labels.shape[1], seq_len) - 1)
    if valid_width <= 0:
        loss = torch.zeros((), device=hidden_states.device, dtype=torch.float32)
        target_tokens = torch.zeros((), device=hidden_states.device, dtype=torch.float32)
        hidden_grad = torch.zeros_like(hidden_states) if return_hidden_grad else None
        if return_hidden_grad:
            return loss, target_tokens, hidden_states, hidden_grad
        return loss, target_tokens

    targets = labels[:, 1 : valid_width + 1].contiguous().view(-1).to(hidden_states.device)
    target_tokens = (targets != ignore_index).sum(dtype=torch.float32)
    normalizer = (
        target_tokens if loss_normalizer is None else loss_normalizer.to(hidden_states.device)
    )
    normalizer = normalizer.to(device=hidden_states.device, dtype=torch.float32).clamp_min(1.0)
    hidden_grad = torch.zeros_like(hidden_states) if return_hidden_grad else None
    batch_size = max(1, int(hidden_states.shape[0]))
    chunk_width = max(1, min(valid_width, _lm_head_loss_chunk_tokens() // batch_size))
    ce_loss_sum_total = torch.zeros((), device=hidden_states.device, dtype=torch.float32)
    with _maybe_gather_lm_head_params(lm_head):
        weight = lm_head.weight.detach() if return_hidden_grad else lm_head.weight
        bias = getattr(lm_head, "bias", None)
        if bias is not None and return_hidden_grad:
            bias = bias.detach()

        for start in range(0, valid_width, chunk_width):
            end = min(valid_width, start + chunk_width)
            targets_chunk = (
                labels[:, start + 1 : end + 1].contiguous().view(-1).to(hidden_states.device)
            )

            if return_hidden_grad:
                hidden_for_loss = hidden_states[:, start:end, :].detach().requires_grad_(True)
                with (
                    torch.enable_grad(),
                    torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled),
                ):
                    logits = F.linear(hidden_for_loss, weight, bias)
                    logits = _apply_final_logit_softcap(logits, model_for_parts)
                    logit_vocab = int(logits.shape[-1])
                    if vocab_size <= 0 or vocab_size != logit_vocab:
                        vocab_size = logit_vocab
                    ce_loss_sum = F.cross_entropy(
                        logits.float().reshape(-1, vocab_size),
                        targets_chunk,
                        ignore_index=ignore_index,
                        reduction="sum",
                    )
                    scaled_loss = ce_loss_sum * (float(backward_scale) / normalizer)
                (grad_hidden_trim,) = torch.autograd.grad(scaled_loss, hidden_for_loss)
                hidden_grad[:, start:end, :].copy_(grad_hidden_trim.to(hidden_grad.dtype))
                ce_loss_sum_total = ce_loss_sum_total + ce_loss_sum.detach()
                del hidden_for_loss, grad_hidden_trim, scaled_loss, logits
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                    logits = F.linear(hidden_states[:, start:end, :], weight, bias)
                    logits = _apply_final_logit_softcap(logits, model_for_parts)
                logit_vocab = int(logits.shape[-1])
                if vocab_size <= 0 or vocab_size != logit_vocab:
                    vocab_size = logit_vocab
                ce_loss_sum = F.cross_entropy(
                    logits.float().reshape(-1, vocab_size),
                    targets_chunk,
                    ignore_index=ignore_index,
                    reduction="sum",
                )
                ce_loss_sum_total = ce_loss_sum_total + ce_loss_sum.detach()
                del logits

    total_loss = ce_loss_sum_total.detach()
    loss_denominator = target_tokens.to(total_loss.dtype).clamp_min(1.0)
    loss = total_loss / loss_denominator
    if return_hidden_grad:
        return loss, target_tokens, hidden_states, hidden_grad
    return loss, target_tokens


def _pipeline_streaming_causal_lm_loss(
    model_obj,
    input_ids,
    attention_mask,
    position_ids,
    labels,
    use_cache=False,
    backward_scale=1.0,
    loss_normalizer=None,
    return_hidden_grad=False,
):
    model_for_parts, _backbone, lm_head = _get_causal_lm_parts(model_obj)
    autocast_enabled = input_ids.is_cuda
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        hidden_states, local_input = _pipeline_local_forward(
            model_obj,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            enable_backward=return_hidden_grad,
        )

    if _pipeline_is_last_stage():
        if return_hidden_grad:
            loss, target_tokens, final_hidden, hidden_grad = _streaming_lm_head_loss_from_hidden(
                model_for_parts,
                lm_head,
                hidden_states,
                labels,
                backward_scale=backward_scale,
                loss_normalizer=loss_normalizer,
                return_hidden_grad=True,
            )
            surrogate_loss = torch.sum(final_hidden * hidden_grad.detach())
            surrogate_loss.backward()
            if (
                not _pipeline_is_first_stage()
                and _pipeline_model_parallel_prev_rank >= 0
                and local_input is not None
                and local_input.grad is not None
            ):
                _send_tensor(local_input.grad, _pipeline_model_parallel_prev_rank, tag_base=20_000)
        else:
            loss, target_tokens = _streaming_lm_head_loss_from_hidden(
                model_for_parts,
                lm_head,
                hidden_states,
                labels,
                backward_scale=backward_scale,
                loss_normalizer=loss_normalizer,
                return_hidden_grad=False,
            )
    else:
        if return_hidden_grad:
            grad_output = _recv_tensor(
                _pipeline_model_parallel_next_rank,
                dtype=hidden_states.dtype,
                device_recv=input_ids.device,
                tag_base=20_000,
            )
            hidden_states.backward(grad_output)
            if (
                not _pipeline_is_first_stage()
                and _pipeline_model_parallel_prev_rank >= 0
                and local_input is not None
                and local_input.grad is not None
            ):
                _send_tensor(local_input.grad, _pipeline_model_parallel_prev_rank, tag_base=20_000)
        loss = torch.zeros((), device=input_ids.device, dtype=torch.float32)
        target_tokens = torch.zeros((), device=input_ids.device, dtype=torch.float32)

    if return_hidden_grad:
        payload = torch.zeros(2, device=input_ids.device, dtype=torch.float32)
        if _pipeline_is_last_stage():
            payload[0] = loss.detach().float()
            payload[1] = target_tokens.detach().float()
        torch.distributed.broadcast(
            payload,
            src=_pipeline_model_parallel_last_rank,
            group=_pipeline_model_parallel_group,
        )
        return payload[0], payload[1], None, None

    return loss, target_tokens


def _streaming_causal_lm_loss(
    model_obj,
    input_ids,
    attention_mask,
    position_ids,
    labels,
    use_cache=False,
    backward_scale=1.0,
    loss_normalizer=None,
    return_hidden_grad=False,
):
    if _pipeline_enabled():
        return _pipeline_streaming_causal_lm_loss(
            model_obj,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            use_cache=use_cache,
            backward_scale=backward_scale,
            loss_normalizer=loss_normalizer,
            return_hidden_grad=return_hidden_grad,
        )

    model_for_parts, backbone, lm_head = _get_causal_lm_parts(model_obj)
    autocast_enabled = input_ids.is_cuda
    attention_mask = _normalize_attention_mask_for_fast_path(attention_mask)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        outputs = _backbone_forward(
            backbone,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
        )
    hidden_states = getattr(outputs, "last_hidden_state", None)
    if hidden_states is None:
        hidden_states = outputs[0]
    return _streaming_lm_head_loss_from_hidden(
        model_for_parts,
        lm_head,
        hidden_states,
        labels,
        backward_scale=backward_scale,
        loss_normalizer=loss_normalizer,
        return_hidden_grad=return_hidden_grad,
    )


def unwrap_model(model_obj):
    if hasattr(model_obj, "module"):
        return model_obj.module
    return model_obj


_gradient_checkpointing_enabled = False


def set_model_gradient_checkpointing(model_obj, enabled):
    """Toggle HF model gradient checkpointing when available."""
    global _gradient_checkpointing_enabled
    enabled = bool(enabled)
    if _gradient_checkpointing_enabled == enabled:
        return

    base_model_obj = unwrap_model(model_obj)
    if hasattr(base_model_obj, "config"):
        base_model_obj.config.use_cache = False

    method_name = "gradient_checkpointing_enable" if enabled else "gradient_checkpointing_disable"
    method = getattr(base_model_obj, method_name, None)
    if method is None:
        if enabled:
            rank0_print(
                "Warning: model does not expose gradient checkpointing; continuing without it."
            )
        _gradient_checkpointing_enabled = False
        return

    try:
        if enabled:
            use_reentrant = False
            if use_reentrant:
                enable_inputs = getattr(base_model_obj, "enable_input_require_grads", None)
                if enable_inputs is not None:
                    enable_inputs()
            try:
                method(gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})
            except TypeError:
                # Older Transformers versions may not accept kwargs. Keep the
                # fallback compatible with mostly-frozen models.
                enable_inputs = getattr(base_model_obj, "enable_input_require_grads", None)
                if enable_inputs is not None:
                    enable_inputs()
                method()
        else:
            method()
            disable_inputs = getattr(base_model_obj, "disable_input_require_grads", None)
            if disable_inputs is not None:
                disable_inputs()
    except Exception as exc:
        if enabled:
            rank0_print(f"Warning: failed to enable gradient checkpointing: {exc}")
        else:
            rank0_print(f"Warning: failed to disable gradient checkpointing: {exc}")
        return

    _gradient_checkpointing_enabled = enabled
    if enabled:
        mode = "non-reentrant"
        rank0_print(f"Gradient checkpointing enabled ({mode}).")
    else:
        rank0_print("Gradient checkpointing disabled.")


def wrap_model_distributed(model_obj, ignored_modules=None):
    return model_obj


_training_parallel_world_size = _data_parallel_world_size

# --- Quantization layer import ---
from beyond.quantization.layers import (
    THRESHOLD_GRAD_AGGREGATION,
    FullPrecisionBucketResidualAdapter,
    FullPrecisionChebyshevAdapter,
    MegatronQuantizedLinear,
    MegatronQuantizedQKVLinear,
    UnifiedQuantLayer,
    UniformAffineQuantLayer,
    clear_threshold_side_sums_,
    finalize_deferred_threshold_grads,
    has_pending_threshold_side_sums,
    install_post_rope_k_quantization,
    threshold_side_activity_roster,
)

QuantizedLinearCls = MegatronQuantizedLinear
QuantizedQKVLinearCls = MegatronQuantizedQKVLinear

if experiment_control["quantizer_parameterization"] == "full_precision_chebyshev_residual":
    rank0_print(
        "Continuous full-precision adapter mode: exact-identity initialization; no QDQ or STE."
    )
elif experiment_control["quantizer_parameterization"] == "full_precision_bucket_residual":
    rank0_print(
        "Hard-gated full-precision residual mode: exact-identity initialization; "
        "bucket routing with x + a_j, no cache-value discretization or input STE."
    )
else:
    rank0_print(
        "STE mode: quantizer input gradient uses identity passthrough; "
        + (
            "uniform low/high endpoints update."
            if experiment_control["quantizer_parameterization"] == "uniform_affine_endpoints"
            else "q_points/thresholds update."
        )
    )

# --- Model and tokenizer configuration ---
base_model = args.base_model
model_type = detect_model_type(base_model)
model_config = get_model_config(
    model_type,
    base_model,
    attn_implementation=args.attn_implementation,
    experts_implementation=args.experts_implementation,
)

# For multi-process training: load a full copy on the local GPU first.
if _distributed and isinstance(model_config, dict):
    model_config = dict(model_config)
    model_config["device_map"] = {"": _local_rank}

# Extract model name from base_model
model_name = base_model.split("/")[-1]

set_random_seed(int(args.seed))

# --- Load model ---
model = load_model(base_model, model_config)
_loaded_model_config = getattr(model, "config", None)
if args.runtime_backend == "megatron":
    model = apply_megatron_parallel_wrappers(
        model,
        tensor_model_parallel_size=max(1, int(args.tensor_model_parallel_size)),
        tensor_model_parallel_rank=_tensor_model_parallel_rank,
        pipeline_model_parallel_size=max(1, int(args.pipeline_model_parallel_size)),
        allow_linear_only_tensor_parallel=bool(args.allow_linear_only_tensor_parallel),
        log_fn=rank0_print,
    )

# Note: TF32 (torch.set_float32_matmul_precision('high')) is NOT used because
# it causes cross-entropy device-side asserts during evaluation on some GPUs.

tokenizer = load_tokenizer(base_model, model_type)
resolved_attn_implementation = None
for _attn_attr in (
    "_attn_implementation",
    "_attn_implementation_internal",
    "attn_implementation",
):
    _attn_value = getattr(_loaded_model_config, _attn_attr, None)
    if _attn_value:
        resolved_attn_implementation = str(_attn_value)
        break
if resolved_attn_implementation is None and isinstance(model_config, dict):
    _attn_value = model_config.get("attn_implementation")
    if _attn_value:
        resolved_attn_implementation = str(_attn_value)
model_revision = getattr(_loaded_model_config, "_commit_hash", None)
if model_revision is not None:
    model_revision = str(model_revision)
_tokenizer_init_kwargs = getattr(tokenizer, "init_kwargs", None)
tokenizer_revision = (
    _tokenizer_init_kwargs.get("_commit_hash") if isinstance(_tokenizer_init_kwargs, dict) else None
)
if tokenizer_revision is not None:
    tokenizer_revision = str(tokenizer_revision)
PAD_TOKEN_ID = int(
    tokenizer.pad_token_id
    if getattr(tokenizer, "pad_token_id", None) is not None
    else (tokenizer.eos_token_id if getattr(tokenizer, "eos_token_id", None) is not None else 0)
)

# --- Validate model setup ---
if not validate_model_setup(model, tokenizer, model_type):
    print("Model setup validation failed. Exiting...")
    exit(1)

# --- Prepare quantization modules (FP init before wrapping) ---
quantizer_params = []
quantized_layer_names = []
quant_targets = []


def _expected_kv_targets(model, model_type):
    config = getattr(model, "config", None)
    if config is None:
        return None

    for attr in (
        "num_hidden_layers",
        "n_layers",
        "num_layers",
        "num_layer",
        "n_layer",
    ):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value * 2

    for key, value in getattr(config, "to_dict", lambda: {})().items():
        if (
            key in {"num_hidden_layers", "n_layers", "num_layers", "num_layer", "n_layer"}
            and isinstance(value, int)
            and value > 0
        ):
            return value * 2
    return None


# Select number of bits for quantization.
num_bits = args.num_bits
if experiment_control["quantizer_parameterization"] == "uniform_affine_endpoints":
    QuantLayerCls = UniformAffineQuantLayer
elif experiment_control["quantizer_parameterization"] == "full_precision_chebyshev_residual":
    QuantLayerCls = FullPrecisionChebyshevAdapter
elif experiment_control["quantizer_parameterization"] == "full_precision_bucket_residual":
    QuantLayerCls = FullPrecisionBucketResidualAdapter
else:
    QuantLayerCls = UnifiedQuantLayer

print(f"Searching for attention layers in {model_type} model...")
attention_layers = find_kv_proj_layers(model)
rank0_print("K/V quantization grouping_dim: token")
rank0_print("K/V quantization qtables: per token group")
rank0_print("K quantization position: post_rope")
rank0_print(
    "Experiment control: "
    f"{args.experiment_control}, objective={experiment_control['training_objective']}, "
    f"parameterization={experiment_control['quantizer_parameterization']}"
)
print(f"Found {len(attention_layers)} attention layers to quantize")
expected_targets = _expected_kv_targets(model, model_type)
if expected_targets is not None and len(attention_layers) != expected_targets:
    rank0_print(
        f"Warning: expected {expected_targets} {model_type} KV projection targets, got {len(attention_layers)}. "
        "Check duplicate nested k_proj/v_proj matching."
    )


def _discover_quant_layer_factory(spec, _proj_module):
    return QuantLayerCls(
        num_bits=num_bits,
        group_size=args.group_size,
        grouping_dim=spec["grouping_dim"],
        one_group=False,
        table_axis="group",
        quant_width=spec.get("quant_width"),
    )


quant_targets = build_kv_quant_targets(
    model,
    _discover_quant_layer_factory,
    layer_specs=attention_layers,
)


def _aligned_group_size_candidates(head_dim, quant_width, limit=12):
    candidates = []
    for value in range(1, int(head_dim) + 1):
        if int(head_dim) % value == 0 and int(quant_width) % value == 0:
            candidates.append(value)
    preferred = [value for value in candidates if value <= 128]
    if not preferred:
        preferred = candidates
    preferred.sort(key=lambda value: (abs(value - int(args.group_size)), value))
    return preferred[:limit]


misaligned_targets = []
for target in quant_targets:
    head_dim = target.get("head_dim")
    quant_width = target.get("quant_width")
    if (
        isinstance(head_dim, int)
        and head_dim > 0
        and isinstance(quant_width, int)
        and quant_width > 0
        and int(args.group_size) > 0
        and head_dim % int(args.group_size) != 0
    ):
        misaligned_targets.append((target.get("name"), head_dim, quant_width))

if misaligned_targets:
    _, example_head_dim, example_width = misaligned_targets[0]
    candidates = _aligned_group_size_candidates(example_head_dim, example_width)
    candidate_text = ", ".join(str(value) for value in candidates) if candidates else "none"
    raise ValueError(
        f"--group_size {args.group_size} does not divide attention head_dim={example_head_dim} "
        f"for {len(misaligned_targets)} quantization targets. This crosses attention-head "
        "boundaries and gives misleading PPL for per-token KV quantization. "
        f"Use a head-aligned group size; nearest valid choices include: {candidate_text}."
    )

if not quant_targets:
    print(
        "Warning: No layers were selected for quantization. The model may not be compatible with the current quantization approach."
    )
    print("Available layer names:")
    for name, module in model.named_modules():
        if hasattr(module, "weight") and len(list(module.parameters())) > 0:
            print(f"  - {name}: {type(module).__name__}")
else:
    print(f"Prepared {len(quant_targets)} layers for quantization")

# Get the target device from the model's parameters (should be set by device_map="auto")
device = next(model.parameters()).device


def _clamp_ratio(value):
    return max(0.0, min(float(value), 1.0))


train_val_ratio = _clamp_ratio(args.train_val_ratio)
args.train_val_ratio = train_val_ratio

rank0_print(
    "Held-out dataset sizes: "
    f"validation={DISCOVER_VALIDATION_CHUNKS} chunks, "
    f"eval={DISCOVER_EVAL_CHUNKS} chunks; "
    f"train_val_ratio={train_val_ratio:g}"
)
rank0_print(
    "Ordinary-LM chat wrapping: "
    f"{'enabled' if args.chat_wrap_pile else 'disabled'}"
    + (f", prompt={args.chat_wrap_pile_prompt!r}" if args.chat_wrap_pile else "")
)
rank0_print(
    "Independent RNG seeds: "
    f"optimizer={int(args.seed)}, data={int(args.data_seed)}, eval={int(args.eval_seed)}"
)

CURRICULUM_CACHE_VERSION = "discover-curriculum-v23-recipes-independent-seeds"
CURRICULUM_CACHE_SYNC_TIMEOUT_SECONDS = int(
    os.getenv("DISCOVER_CURRICULUM_CACHE_TIMEOUT_SECONDS", "3600")
)


def _normalize_source_for_signature(source):
    if source is None:
        return {}
    return {
        "category": source.get("category"),
        "target_tokens": int(source.get("target_tokens", 0)),
        "path": source.get("path"),
        "split": source.get("split"),
        "name": source.get("name"),
        "revision": source.get("revision"),
        "data_dir": source.get("data_dir"),
        "fields": list(source.get("fields", ())),
        "trust_remote_code": bool(source.get("trust_remote_code", False)),
        "requires_auth": bool(source.get("requires_auth", False)),
        "label": source.get("label"),
        "format": source.get("format"),
        "separator": source.get("separator"),
        "doc_chunk_mode": source.get("doc_chunk_mode"),
        "min_doc_tokens": source.get("min_doc_tokens"),
        "require_doc_tokens_gt_seqlen": bool(source.get("require_doc_tokens_gt_seqlen", False)),
        "min_supervised_tokens": source.get("min_supervised_tokens"),
        "target_unit": source.get("target_unit", "tokens"),
        "allow_chat_wrap": bool(source.get("allow_chat_wrap", True)),
        "chat_wrap": bool(source.get("chat_wrap", False)),
        "chat_user_prompt": source.get("chat_user_prompt"),
    }


def _fingerprint_manifest(stages):
    fingerprints = sorted(
        {
            str(fingerprint)
            for stage in stages
            for fingerprint in stage.get("fingerprints", [])
            if fingerprint
        }
    )
    digest_body = "\n".join(fingerprints).encode("utf-8")
    return {
        "count": len(fingerprints),
        "sha256": hashlib.sha256(digest_body).hexdigest(),
        "values": fingerprints,
    }


def _curriculum_cache_key():
    curriculum = build_length_curriculum(base_model, int(args.reference_tokens))
    source_signature = [
        _normalize_source_for_signature(source) for source in DISCOVER_ACTIVE_SOURCES
    ]
    key_payload = {
        "version": CURRICULUM_CACHE_VERSION,
        "dataset": args.dataset,
        "model": base_model,
        "model_type": model_type,
        "reference_tokens": int(args.reference_tokens),
        "validation_chunks": int(DISCOVER_VALIDATION_CHUNKS),
        "eval_chunks": int(DISCOVER_EVAL_CHUNKS),
        "chat_wrap_pile": bool(args.chat_wrap_pile),
        "chat_wrap_pile_prompt": args.chat_wrap_pile_prompt,
        "data_seed": int(args.data_seed),
        "eval_seed": int(args.eval_seed),
        "pile_train_mode": str(args.pile_train_mode),
        "materialized_train_chunks_per_stage": (
            int(args.train_steps_per_pass) * int(args.batch_size)
            if str(args.pile_train_mode) == "materialized"
            else 0
        ),
        "curriculum": [
            {
                "name": stage["name"],
                "seqlen": int(stage["seqlen"]),
                "token_budget": int(stage["token_budget"]),
            }
            for stage in curriculum
        ],
        "sources": source_signature,
    }
    key_body = json.dumps(key_payload, sort_keys=True, ensure_ascii=False)
    cache_hash = hashlib.sha256(key_body.encode("utf-8")).hexdigest()[:16]
    return f"cur_{cache_hash}"


def _curriculum_cache_dir():
    return os.path.expanduser(args.curriculum_cache_dir)


def _curriculum_cache_path():
    cache_dir = _curriculum_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{_curriculum_cache_key()}.pt")


def _load_cached_curriculum(cache_path):
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(cache_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid curriculum cache payload type.")
    if payload.get("version") != CURRICULUM_CACHE_VERSION:
        raise RuntimeError("Curriculum cache version mismatch.")
    metadata = payload.get("metadata", {})
    if metadata.get("cache_key") != _curriculum_cache_key():
        raise RuntimeError("Curriculum cache key mismatch.")
    training_stages = payload.get("training_stages")
    eval_stages = payload.get("eval_stages")
    test_stages = payload.get("test_stages")
    if (
        not isinstance(training_stages, list)
        or not isinstance(eval_stages, list)
        or not isinstance(test_stages, list)
    ):
        raise RuntimeError("Curriculum cache payload missing stages.")
    return training_stages, eval_stages, test_stages


def _save_curriculum_cache(cache_path, payload):
    tmp_path = f"{cache_path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, cache_path)


def _wait_for_curriculum_cache(cache_path, timeout_seconds):
    deadline = time.time() + max(0, int(timeout_seconds))
    last_error = None
    while time.time() < deadline:
        if os.path.isfile(cache_path):
            try:
                return _load_cached_curriculum(cache_path)
            except Exception as exc:
                last_error = exc
                time.sleep(1.0)
                continue
        time.sleep(1.0)

    if last_error is not None:
        raise RuntimeError(f"Curriculum cache load failed after {timeout_seconds}s: {last_error}")
    raise RuntimeError(f"Curriculum cache was not ready at {cache_path} within {timeout_seconds}s.")


def _build_or_load_curriculum():
    cache_path = _curriculum_cache_path()
    cache_key = _curriculum_cache_key()
    rank0_print(f"Curriculum cache path: {cache_path}")
    training_stages = None
    eval_stages = None
    test_stages = None

    if is_main():
        if os.path.isfile(cache_path):
            try:
                training_stages, eval_stages, test_stages = _load_cached_curriculum(cache_path)
                rank0_print(f"Curriculum cache hit: {cache_path}")
            except Exception as exc:
                rank0_print(f"Curriculum cache invalid ({exc}); rebuilding.")
        else:
            rank0_print(f"Curriculum cache miss; building and saving to {cache_path}")

        if training_stages is None:
            training_stages, eval_stages, test_stages = get_training_curriculum_dataset(
                seed=int(args.seed),
                model_path=base_model,
                dataset_name=args.dataset,
                train_tokens=args.reference_tokens,
                validation_chunks=DISCOVER_VALIDATION_CHUNKS,
                eval_chunks=DISCOVER_EVAL_CHUNKS,
                chat_wrap_pile=args.chat_wrap_pile,
                chat_user_prompt=args.chat_wrap_pile_prompt,
                stream_train=str(args.pile_train_mode) == "streaming",
                materialized_train_chunks_per_stage=(
                    int(args.train_steps_per_pass) * int(args.batch_size)
                    if str(args.pile_train_mode) == "materialized"
                    else None
                ),
                data_seed=int(args.data_seed),
                eval_seed=int(args.eval_seed),
            )
            _save_curriculum_cache(
                cache_path,
                {
                    "version": CURRICULUM_CACHE_VERSION,
                    "metadata": {
                        "cache_key": cache_key,
                        "dataset": args.dataset,
                        "model": base_model,
                        "model_type": model_type,
                        "reference_tokens": int(args.reference_tokens),
                        "validation_chunks": int(DISCOVER_VALIDATION_CHUNKS),
                        "eval_chunks": int(DISCOVER_EVAL_CHUNKS),
                        "heldout_sizing": "fixed_chunks",
                        "chat_wrap_pile": bool(args.chat_wrap_pile),
                        "chat_wrap_pile_prompt": args.chat_wrap_pile_prompt,
                        "data_seed": int(args.data_seed),
                        "eval_seed": int(args.eval_seed),
                        "pile_train_mode": str(args.pile_train_mode),
                        "materialized_train_chunks_per_stage": (
                            int(args.train_steps_per_pass) * int(args.batch_size)
                            if str(args.pile_train_mode) == "materialized"
                            else 0
                        ),
                    },
                    "training_stages": training_stages,
                    "eval_stages": eval_stages,
                    "test_stages": test_stages,
                },
            )
            rank0_print(f"Saved curriculum cache to {cache_path}")

    # Rank zero builds or validates the shared cache. Other ranks wait at the
    # process-group barrier and then deserialize the completed payload.
    if _distributed:
        dist_barrier("curriculum_cache_ready")

    if not (training_stages is not None and eval_stages is not None and test_stages is not None):
        training_stages, eval_stages, test_stages = _wait_for_curriculum_cache(
            cache_path,
            CURRICULUM_CACHE_SYNC_TIMEOUT_SECONDS,
        )

    return training_stages, eval_stages, test_stages


raw_training_stages, raw_eval_stages, raw_test_stages = _build_or_load_curriculum()

if not raw_training_stages:
    raise RuntimeError(
        f"No raw training stages were built. Increase --reference_tokens to at least one {DISCOVER_TRAIN_SEQLEN_LABEL} sequence."
    )
model.seqlen = int(raw_training_stages[0]["seqlen"])
max_training_seqlen = max(int(stage["seqlen"]) for stage in raw_training_stages)


def _parse_positive_int(value, flag_name):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{flag_name} must be a positive integer.") from exc
    if parsed <= 0:
        raise ValueError(f"{flag_name} must be a positive integer.")
    return parsed


target_batch_size = _parse_positive_int(args.batch_size, "--batch_size")
per_gpu_batch_size = _parse_positive_int(args.per_gpu_batch_size, "--per_gpu_batch_size")
training_world = max(1, int(_training_parallel_world_size if _distributed else 1))
train_global_microbatch = per_gpu_batch_size * training_world
if target_batch_size < train_global_microbatch:
    raise ValueError(
        "--batch_size must be at least --per_gpu_batch_size * data_parallel_world_size. "
        f"Got batch_size={target_batch_size}, per_gpu_batch_size={per_gpu_batch_size}, "
        f"data_parallel_world_size={training_world}."
    )
if target_batch_size % train_global_microbatch != 0:
    raise ValueError(
        "--batch_size must be divisible by --per_gpu_batch_size * data_parallel_world_size "
        "so each optimizer step has the requested effective batch. "
        f"Got batch_size={target_batch_size}, per_gpu_batch_size={per_gpu_batch_size}, "
        f"data_parallel_world_size={training_world}, global_microbatch={train_global_microbatch}."
    )
train_accum_steps = target_batch_size // train_global_microbatch
rank0_print(
    "Batch plan: "
    f"target_effective_batch={target_batch_size}, "
    f"per_gpu_batch={per_gpu_batch_size}, "
    f"data_parallel_world={training_world}, "
    f"global_microbatch={train_global_microbatch}, "
    f"accum_steps={train_accum_steps}"
)


def _stage_batch_plan(seqlen, is_eval=False):
    global_microbatch = train_global_microbatch
    local_microbatch = per_gpu_batch_size
    accum_steps = 1 if is_eval else train_accum_steps
    return {
        "global_microbatch": global_microbatch,
        "local_microbatch": local_microbatch,
        "accum_steps": accum_steps,
        "effective_batch_size": global_microbatch * accum_steps,
    }


def _make_ignored_sample_like(sample):
    if len(sample) == 4:
        input_ids, labels, attention_mask, position_ids = sample
        ignored_labels = torch.full_like(labels, -100)
        return input_ids.clone(), ignored_labels, attention_mask.clone(), position_ids.clone()
    if len(sample) == 3:
        input_ids, labels, attention_mask = sample
        ignored_labels = torch.full_like(labels, -100)
        return input_ids.clone(), ignored_labels, attention_mask.clone()
    input_ids, labels = sample
    ignored_labels = torch.full_like(labels, -100)
    return input_ids.clone(), ignored_labels


def _shard_samples_for_rank(samples: List[Any]) -> List[Any]:
    """Shard samples, padding eval shards so distributed forwards stay aligned."""
    if not _distributed or _training_parallel_world_size <= 1:
        return samples
    if not samples:
        return []

    if args.runtime_backend == "megatron":
        shard_rank = _data_parallel_rank
        shard_world = max(1, _data_parallel_world_size)
    else:
        shard_rank = _rank
        shard_world = _world_size

    ranked_samples = list(samples[shard_rank::shard_world])
    target_count = (len(samples) + shard_world - 1) // shard_world
    if len(ranked_samples) < target_count:
        pad_template = ranked_samples[0] if ranked_samples else samples[0]
        ranked_samples.extend(
            _make_ignored_sample_like(pad_template)
            for _ in range(target_count - len(ranked_samples))
        )
    return ranked_samples


def _split_stage_samples(stage, stage_idx):
    samples = stage["samples"]
    total_samples_stage = len(samples)
    if total_samples_stage > 1 and train_val_ratio > 0:
        val_token_budget = int(round(int(stage["token_budget"]) * train_val_ratio))
        val_token_budget = max(int(stage["seqlen"]), val_token_budget)
        split_g = torch.Generator()
        split_g.manual_seed(int(args.data_seed) + stage_idx)
        indices = torch.randperm(total_samples_stage, generator=split_g).tolist()
        val_indices = []
        val_tokens = 0
        for idx in indices:
            if len(val_indices) >= total_samples_stage - 1:
                break
            val_indices.append(idx)
            val_tokens += _sample_token_count(samples[idx])
            if val_tokens >= val_token_budget:
                break
        val_set = set(val_indices)
        train_indices = [idx for idx in range(total_samples_stage) if idx not in val_set]
        return [samples[i] for i in train_indices], [samples[i] for i in val_indices]
    return samples, []


def _dataset_sequence_lengths(dataset):
    lengths = []
    input_ids = getattr(dataset, "input_ids", None)
    if input_ids is not None:
        for item in input_ids:
            try:
                lengths.append(max(1, int(item.numel())))
            except Exception:
                lengths.append(1)
        return lengths
    for idx in range(len(dataset)):
        try:
            lengths.append(max(1, int(dataset[idx]["input_ids"].numel())))
        except Exception:
            lengths.append(1)
    return lengths


def _sample_length_summary(samples, seqlen):
    lengths = [_sample_token_count(sample) for sample in samples if sample]
    if not lengths:
        return "empty"
    half_seqlen = max(1, int(seqlen) // 2)
    avg_len = sum(lengths) / len(lengths)
    return (
        f"avg={avg_len:,.0f}, max={max(lengths):,}, "
        f">={half_seqlen:,}={sum(1 for length in lengths if length >= half_seqlen):,}, "
        f">={int(seqlen):,}={sum(1 for length in lengths if length >= int(seqlen)):,}"
    )


def _ceil_div(numer, denom):
    return (max(0, int(numer)) + max(1, int(denom)) - 1) // max(1, int(denom))


def _make_token_budget_batches(dataset, token_budget, max_batch_size):
    """Pack eval/validation batches by real sequence length, not the training stage max."""

    dataset_len = len(dataset)
    if dataset_len <= 0:
        return []
    token_budget = max(1, int(token_budget))
    max_batch_size = max(1, int(max_batch_size))
    lengths = _dataset_sequence_lengths(dataset)
    sorted_indices = sorted(range(dataset_len), key=lambda idx: lengths[idx], reverse=True)

    batches = []
    current = []
    current_max = 0
    for idx in sorted_indices:
        length = max(1, int(lengths[idx]))
        next_max = max(current_max, length)
        would_exceed_tokens = current and next_max * (len(current) + 1) > token_budget
        would_exceed_count = current and len(current) >= max_batch_size
        if would_exceed_tokens or would_exceed_count:
            batches.append(current)
            current = []
            current_max = 0

        current.append(idx)
        current_max = max(current_max, length)

    if current:
        batches.append(current)
    return batches


def _make_eval_like_dataloader(dataset, *, num_workers=2):
    max_batch_size = max(
        1,
        min(
            int(per_gpu_batch_size),
            len(dataset) if len(dataset) > 0 else 1,
        ),
    )
    batch_sampler = _make_token_budget_batches(
        dataset,
        token_budget=max(1, int(per_gpu_batch_size) * int(DISCOVER_TRAIN_SEQLEN)),
        max_batch_size=max_batch_size,
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=collate_text_batch,
    )


def _build_stage_runtime(stage, stage_idx):
    if stage.get("streaming_train"):
        stage_seqlen = int(stage["seqlen"])
        stage_batch_plan = _stage_batch_plan(stage_seqlen, is_eval=False)
        if args.runtime_backend == "megatron":
            stream_replicas = max(1, _data_parallel_world_size)
            stream_rank = _data_parallel_rank
        else:
            stream_replicas = _world_size
            stream_rank = _rank

        stream_steps_per_pass = max(1, int(args.train_steps_per_pass))
        reference_global_sequence_budget = max(
            1, _ceil_div(int(stage["token_budget"]), stage_seqlen)
        )
        stream_microbatches_per_pass = stream_steps_per_pass * int(stage_batch_plan["accum_steps"])
        local_sequence_budget = stream_microbatches_per_pass * int(
            stage_batch_plan["local_microbatch"]
        )
        effective_global_sequences = local_sequence_budget * stream_replicas
        effective_tokens = effective_global_sequences * stage_seqlen
        rank0_print(
            f"Streaming Train stage {stage['name']}: "
            f"optimizer_steps/pass={stream_steps_per_pass:,}, "
            f"microbatches/pass={stream_microbatches_per_pass:,}, "
            f"global_sequences={effective_global_sequences:,}, "
            f"local_sequences/rank={local_sequence_budget:,}, "
            f"seqlen={stage_seqlen:,}, "
            f"reference_budget_sequences={reference_global_sequence_budget:,}, "
            f"sources={len(stage.get('sources', []))}, "
            f"heldout_skip_fingerprints={len(stage.get('exclude_fingerprints', [])):,}"
        )

        stage_train_dataset = StreamingTokenChunkDataset(
            tokenizer,
            stage_seqlen,
            stage.get("sources", DISCOVER_ACTIVE_SOURCES),
            int(stage.get("collection_seed", int(args.data_seed) + stage_idx)),
            num_samples=local_sequence_budget,
            pad_token_id=PAD_TOKEN_ID,
            data_rank=stream_rank,
            data_world_size=stream_replicas,
            exclude_fingerprints=stage.get("exclude_fingerprints", []),
        )
        train_loader_kwargs = {
            "batch_size": stage_batch_plan["local_microbatch"],
            "shuffle": False,
            "sampler": None,
            "num_workers": STREAM_TRAIN_NUM_WORKERS,
            "pin_memory": True,
            "persistent_workers": STREAM_TRAIN_NUM_WORKERS > 0,
            "collate_fn": collate_text_batch,
        }
        if STREAM_TRAIN_NUM_WORKERS > 0:
            train_loader_kwargs["prefetch_factor"] = STREAM_TRAIN_PREFETCH_FACTOR
        stage_train_dataloader = DataLoader(stage_train_dataset, **train_loader_kwargs)
        stage_val_dataset = TextDataset([], pad_token_id=PAD_TOKEN_ID)
        stage_val_dataloader = _make_eval_like_dataloader(stage_val_dataset, num_workers=0)
        return {
            **stage,
            "stage_idx": stage_idx,
            "train_samples": [],
            "val_samples": [],
            "val_samples_ranked": [],
            "train_dataloader": stage_train_dataloader,
            "val_dataloader": stage_val_dataloader,
            "train_sampler": None,
            "val_split_seed": None,
            "effective_tokens": int(effective_tokens),
            "train_effective_tokens": int(effective_tokens),
            "val_effective_tokens": 0,
            "global_sequence_budget": int(effective_global_sequences),
            "local_sequence_budget": int(local_sequence_budget),
            "reference_global_sequence_budget": int(reference_global_sequence_budget),
            "stream_steps_per_pass": int(stream_steps_per_pass),
            "stream_microbatches_per_pass": int(stream_microbatches_per_pass),
            **stage_batch_plan,
        }

    stage_train_samples, stage_val_samples = _split_stage_samples(stage, stage_idx)
    if stage_train_samples:
        rank0_print(
            f"Raw Train stage {stage['name']} samples before pack: "
            f"{_sample_length_summary(stage_train_samples, int(stage['seqlen']))}"
        )
    if PACK_TRAIN_TO_SEQLEN and stage_train_samples:
        before_train_count = len(stage_train_samples)
        before_train_tokens = _samples_token_count(stage_train_samples)
        packed_train_samples = pack_samples_to_seqlen(
            stage_train_samples,
            int(stage["seqlen"]),
            pad_token_id=PAD_TOKEN_ID,
            drop_last=DROP_TRAIN_PACK_REMAINDER,
        )
        if not packed_train_samples and DROP_TRAIN_PACK_REMAINDER:
            rank0_print(
                f"Packed Train stage {stage['name']} produced no full chunks; "
                "falling back to a padded final chunk."
            )
            packed_train_samples = pack_samples_to_seqlen(
                stage_train_samples,
                int(stage["seqlen"]),
                pad_token_id=PAD_TOKEN_ID,
                drop_last=False,
            )
        stage_train_samples = packed_train_samples
        after_train_tokens = _samples_token_count(stage_train_samples)
        rank0_print(
            f"Packed Train stage {stage['name']} samples "
            f"{before_train_count} -> {len(stage_train_samples)} at seqlen={int(stage['seqlen']):,} "
            f"(tokens {before_train_tokens:,} -> {after_train_tokens:,}, "
            f"drop_remainder={DROP_TRAIN_PACK_REMAINDER})"
        )
    if stage_val_samples:
        before_val_count = len(stage_val_samples)
        stage_val_samples = pack_samples_to_seqlen(
            stage_val_samples,
            int(stage["seqlen"]),
            pad_token_id=PAD_TOKEN_ID,
            drop_last=False,
        )
        rank0_print(
            f"Packed Val stage {stage['name']} samples "
            f"{before_val_count} -> {len(stage_val_samples)} at seqlen={int(stage['seqlen']):,}"
        )
    stage_train_dataset = TextDataset(stage_train_samples, pad_token_id=PAD_TOKEN_ID)
    stage_val_samples_ranked = _shard_samples_for_rank(stage_val_samples)
    stage_val_dataset = TextDataset(stage_val_samples_ranked, pad_token_id=PAD_TOKEN_ID)
    stage_batch_plan = _stage_batch_plan(stage["seqlen"], is_eval=False)
    if args.runtime_backend == "megatron":
        stage_sampler_replicas = max(1, _data_parallel_world_size)
        stage_sampler_rank = _data_parallel_rank
    else:
        stage_sampler_replicas = _world_size
        stage_sampler_rank = _rank
    stage_sampler = (
        DistributedSampler(
            stage_train_dataset,
            num_replicas=stage_sampler_replicas,
            rank=stage_sampler_rank,
            shuffle=True,
            seed=int(args.data_seed) + stage_idx,
            drop_last=False,
        )
        if _distributed
        else None
    )
    stage_generator = torch.Generator()
    stage_generator.manual_seed(int(args.data_seed) + stage_idx)
    stage_train_dataloader = DataLoader(
        stage_train_dataset,
        batch_size=stage_batch_plan["local_microbatch"],
        shuffle=(stage_sampler is None),
        sampler=stage_sampler,
        generator=None if stage_sampler is not None else stage_generator,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=collate_text_batch,
    )
    stage_val_dataloader = _make_eval_like_dataloader(stage_val_dataset, num_workers=2)
    return {
        **stage,
        "stage_idx": stage_idx,
        "train_samples": stage_train_samples,
        "val_samples": stage_val_samples,
        "val_samples_ranked": stage_val_samples_ranked,
        "train_dataloader": stage_train_dataloader,
        "val_dataloader": stage_val_dataloader,
        "train_sampler": stage_sampler,
        "val_split_seed": int(args.data_seed) + stage_idx,
        **stage_batch_plan,
    }


def _build_eval_stage_runtime(stage, stage_idx):
    eval_samples_ranked = _shard_samples_for_rank(stage["samples"])
    eval_dataset = TextDataset(eval_samples_ranked, pad_token_id=PAD_TOKEN_ID)
    stage_batch_plan = _stage_batch_plan(stage["seqlen"], is_eval=True)
    eval_dataloader = _make_eval_like_dataloader(eval_dataset, num_workers=2)
    return {
        **stage,
        "stage_idx": stage_idx,
        "eval_samples": stage["samples"],
        "eval_samples_ranked": eval_samples_ranked,
        "eval_dataloader": eval_dataloader,
        "global_microbatch": stage_batch_plan["global_microbatch"],
        "local_microbatch": stage_batch_plan["local_microbatch"],
    }


training_stages = [
    _build_stage_runtime(stage, stage_idx) for stage_idx, stage in enumerate(raw_training_stages)
]
if not training_stages:
    raise RuntimeError(
        f"No training stages were built. Increase --reference_tokens to at least one {DISCOVER_TRAIN_SEQLEN_LABEL} sequence."
    )
rank0_print("Per-stage batch sizes:")
for stage in training_stages:
    rank0_print(
        f"  train {stage['name']}: seqlen={int(stage['seqlen'])}, "
        f"global_microbatch={stage['global_microbatch']}, local_microbatch/GPU={stage['local_microbatch']}, "
        f"accum_steps={stage['accum_steps']}, effective_batch={stage['effective_batch_size']}"
    )

eval_stages = [
    _build_eval_stage_runtime(stage, stage_idx) for stage_idx, stage in enumerate(raw_eval_stages)
]
if not eval_stages:
    raise RuntimeError(
        "No validation stages were built. Increase the eval token budget or check dataset loading."
    )

# Internal naming note: eval_stages are the full validation set used for
# early-stopping; test_stages are the final held-out eval set reported before
# and after training.
test_stages = [
    _build_eval_stage_runtime(stage, stage_idx) for stage_idx, stage in enumerate(raw_test_stages)
]
if not test_stages:
    raise RuntimeError(
        "No test stages were built. Increase the eval token budget or check dataset loading."
    )

rank0_print(f"{DISCOVER_TRAIN_SEQLEN_LABEL} length schedule:")
for stage in training_stages:
    rank0_print(
        f"  {stage['stage_idx'] + 1}. {stage['name']}: seqlen={stage['seqlen']:,}, "
        f"budget={stage['token_budget']:,}, effective={stage['effective_tokens']:,}, "
        f"train_samples={len(stage['train_samples'])}, val_samples={len(stage['val_samples'])}"
    )
rank0_print("Validation schedule (training monitor):")
for stage in eval_stages:
    rank0_print(
        f"  {stage['stage_idx'] + 1}. {stage['name']}: seqlen={stage['seqlen']:,}, "
        f"chunks={stage.get('sample_budget', len(stage['eval_samples'])):,}, "
        f"budget={stage['token_budget']:,}, effective={stage['effective_tokens']:,}, "
        f"eval_samples={len(stage['eval_samples'])}"
    )
rank0_print("Eval schedule (final held-out):")
for stage in test_stages:
    rank0_print(
        f"  {stage['stage_idx'] + 1}. {stage['name']}: seqlen={stage['seqlen']:,}, "
        f"chunks={stage.get('sample_budget', len(stage['eval_samples'])):,}, "
        f"budget={stage['token_budget']:,}, effective={stage['effective_tokens']:,}, "
        f"eval_samples={len(stage['eval_samples'])}"
    )

run_train_validation = train_val_ratio > 0 and any(
    len(stage.get("val_samples", [])) > 0 for stage in training_stages
)
if run_train_validation:
    rank0_print(
        "Training-stage validation enabled; held-out validation remains the early-stopping signal."
    )
else:
    if train_val_ratio <= 0:
        rank0_print(
            "Training-stage validation disabled (--train_val_ratio=0); held-out validation is used for PPL checks."
        )
    else:
        rank0_print(
            "Training-stage validation disabled because the split produced no validation samples."
        )

# --- Training parameters ---
total_epochs = max(1, args.epochs)
LOGGING_INTERVAL = 100
GRADIENT_CHECKPOINTING_FROM_SEQLEN = int(args.gradient_checkpointing_from_seqlen)
if args.epochs <= 0:
    print("Warning: args.epochs <= 0. Using one training pass.")
elif total_epochs > 1:
    effective_stream_tokens = sum(
        int(stage.get("train_effective_tokens", stage["effective_tokens"]))
        for stage in training_stages
    )
    print(
        f"Training passes enabled: epochs={total_epochs}; "
        f"streamed token budget is approximately {effective_stream_tokens * total_epochs:,}."
    )
max_accum_steps = max(int(stage["accum_steps"]) for stage in training_stages)
if max_accum_steps > 1:
    print(f"Automatic per-stage gradient accumulation enabled: max_accum_steps={max_accum_steps}")
if GRADIENT_CHECKPOINTING_FROM_SEQLEN > 0:
    rank0_print(
        "Gradient checkpointing will be enabled for "
        f"seqlen >= {GRADIENT_CHECKPOINTING_FROM_SEQLEN:,}."
    )
else:
    rank0_print("Gradient checkpointing disabled for all training stages.")
rank0_print(
    "Streaming train loader: "
    f"workers={STREAM_TRAIN_NUM_WORKERS}, "
    f"prefetch_factor={STREAM_TRAIN_PREFETCH_FACTOR if STREAM_TRAIN_NUM_WORKERS > 0 else 0}, "
    f"loss_sync_interval={TRAIN_LOSS_SYNC_INTERVAL}, "
    f"strict_loss_checks={bool(STRICT_TRAIN_LOSS_CHECKS)}, "
    f"optimizer_grad_check={bool(CHECK_OPTIMIZER_GRADS)}, "
    f"assume_quantizer_grads={bool(ASSUME_QUANTIZER_GRADS)}."
)
if EVAL_EVERY_STEPS > 0:
    rank0_print(
        f"Periodic held-out validation enabled every {EVAL_EVERY_STEPS} training steps "
        "(full held-out validation set)."
    )
    if PERIODIC_EVAL_TARGET_BATCHES > 0:
        rank0_print(
            "Warning: DISCOVER_PERIODIC_EVAL_TARGET_BATCHES is deprecated and ignored; "
            "periodic validation now always uses the full validation set."
        )
else:
    rank0_print("Periodic held-out validation disabled; using curriculum boundary eval.")

# --- Set up TensorBoard ---
# Construct a more descriptive run name for TensorBoard
run_name_parts = [
    f"{model_name.replace('/', '-')}",
    "opt-adam_custom",  # Optimizer changed to Adam
    f"{args.num_bits}bit",
    f"g{args.group_size}",
    (
        "per_group_fp_adapter"
        if not experiment_control["compressed_kv_cache"]
        else "per_group_qtables"
    ),
    f"val{DISCOVER_VALIDATION_CHUNKS}",
    f"eval{DISCOVER_EVAL_CHUNKS}",
    f"effbs{target_batch_size}",
    f"ep{args.epochs}",
    f"data-{args.dataset}",
    (
        "continuous"
        if experiment_control["quantizer_parameterization"] == "full_precision_chebyshev_residual"
        else (
            "bucket_residual"
            if experiment_control["quantizer_parameterization"] == "full_precision_bucket_residual"
            else "ste"
        )
    ),
    f"control-{args.experiment_control}",
    f"seqlen{DISCOVER_TRAIN_SEQLEN // 1024}k",
]
if args.dataset == "pile":
    run_name_parts.append("pile-dedup")
if args.chat_wrap_pile:
    run_name_parts.append("chatwrap")
if DISCOVER_LONG_DOC_TRUNCATION:
    run_name_parts.append(f"docgt{DISCOVER_TRAIN_SEQLEN // 1024}ktrunc")
if PACK_TRAIN_TO_SEQLEN and not DISCOVER_LONG_DOC_TRUNCATION:
    run_name_parts.append("packtrain")
    if DROP_TRAIN_PACK_REMAINDER:
        run_name_parts.append("droprem")
run_name_parts.append(
    "materializedtrain" if str(args.pile_train_mode) == "materialized" else "streamtrain"
)
run_name_parts.append(f"step{args.train_steps_per_pass}")
run_name_parts.append(f"bw{args.boundary_window:g}")
run_name_parts.append(f"seed{int(args.seed)}")
run_name_parts.append(f"ds{int(args.data_seed)}")
run_name_parts.append(f"es{int(args.eval_seed)}")
run_name_parts.append(f"pat{int(args.early_stop_patience)}")
if EVAL_EVERY_STEPS > 0:
    run_name_parts.append(f"evalstep{EVAL_EVERY_STEPS}")
    run_name_parts.append("fullval")
else:
    run_name_parts.append("evalstage" if EVAL_EVERY_STAGE else "evalpass")
if train_val_ratio > 0:
    run_name_parts.append(f"trainval{train_val_ratio:g}")
if args.ce_loss_weight != 1.0:
    run_name_parts.append(f"cew{args.ce_loss_weight:g}")
if not args.train_quant_points:
    run_name_parts.append("noqtrain")
if args.threshold_grad_mode != "half_wave":
    run_name_parts.append(f"tgrad{args.threshold_grad_mode}")
run_name_parts.append("tgradaggv3")
if not args.train_thresholds:
    run_name_parts.append("notrainthr")

if _distributed:
    run_name_parts.extend(
        [
            f"mp{args.tensor_model_parallel_size}",
            f"pp{args.pipeline_model_parallel_size}",
            f"g{_world_size}",
        ]
    )

# Since optimizer is always adam_custom:
run_name_parts.append(f"lr{args.lr}")
run_name_parts.append(f"lrs{args.lr_schedule}")
if args.lr_schedule == "cosine" and float(args.lr_min_ratio) != 0.0:
    run_name_parts.append(f"lrmin{args.lr_min_ratio:g}")

if GRADIENT_CHECKPOINTING_FROM_SEQLEN > 0:
    run_name_parts.append(f"gcfrom{GRADIENT_CHECKPOINTING_FROM_SEQLEN // 1024}k")
else:
    run_name_parts.append("nogc")


if args.run_name_suffix:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_name_suffix):
        raise ValueError(
            "--run_name_suffix must be one safe path component containing only "
            "letters, digits, dot, underscore, and hyphen."
        )
    run_name_parts.append(args.run_name_suffix)


def find_latest_quant_config(quant_dir):
    if not os.path.isdir(quant_dir):
        return None, None

    step_re = re.compile(r"^quant_config_step_(\d+)\.(pt|json)$")
    last_step = None
    last_path = None
    last_ext = None
    try:
        entries = os.listdir(quant_dir)
    except Exception as e:
        print(f"Warning: Failed to list quant config dir {quant_dir}: {e}")
        return None, None

    for name in entries:
        match = step_re.match(name)
        if not match:
            continue
        step = int(match.group(1))
        ext = match.group(2)
        if (
            last_step is None
            or step > last_step
            or (step == last_step and last_ext == "json" and ext == "pt")
        ):
            last_step = step
            last_ext = ext
            last_path = os.path.join(quant_dir, name)

    if last_path:
        return last_path, last_step

    for final_name in ("quant_config_final.pt", "quant_config_final.json"):
        final_path = os.path.join(quant_dir, final_name)
        if os.path.isfile(final_path):
            return final_path, None

    return None, None


def _resume_neutral_run_tokens(name):
    tokens = []
    for token in str(name or "").split("__"):
        if not token:
            continue
        if re.match(r"^pgbs\d+$", token):
            continue
        if re.match(r"^maxaccum\d+$", token):
            continue
        tokens.append(token)
    return Counter(tokens)


def _run_names_resume_compatible(candidate_name, current_name):
    return _resume_neutral_run_tokens(candidate_name) == _resume_neutral_run_tokens(current_name)


def find_latest_resume_quant_config(
    logs_root, current_run_name, current_config_dir_name, current_quant_dir
):
    best_path = None
    best_step = None
    best_key = None

    def consider(quant_dir, *, is_current=False):
        nonlocal best_path, best_step, best_key
        path, step = find_latest_quant_config(quant_dir)
        if not path:
            return
        step_value = int(step) if step is not None else -1
        key = (
            step_value,
            1 if path.endswith(".pt") else 0,
            1 if is_current else 0,
        )
        if best_key is None or key > best_key:
            best_path = path
            best_step = step
            best_key = key

    consider(current_quant_dir, is_current=True)

    if not os.path.isdir(logs_root):
        return best_path, best_step

    try:
        config_dir_names = os.listdir(logs_root)
    except Exception as exc:
        rank0_print(f"Warning: Failed to list logs dir {logs_root}: {exc}")
        return best_path, best_step

    for candidate_dir_name in config_dir_names:
        if candidate_dir_name == current_config_dir_name:
            continue
        candidate_quant_dir = os.path.join(logs_root, candidate_dir_name, "quant_configs")
        if not os.path.isdir(candidate_quant_dir):
            continue
        if not _run_names_resume_compatible(candidate_dir_name, current_run_name):
            continue
        consider(candidate_quant_dir, is_current=False)

    return best_path, best_step


def _training_state_path_for_step(quant_dir, step, rank):
    return os.path.join(quant_dir, f"training_state_step_{int(step)}_rank{int(rank)}.pt")


def find_training_state_for_step(quant_dir, step, rank):
    if step is None or not os.path.isdir(quant_dir):
        return None
    candidates = [
        _training_state_path_for_step(quant_dir, step, rank),
        os.path.join(quant_dir, f"training_state_step_{int(step)}.pt"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _load_torch_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _iter_quant_config_leaf_entries(config):
    if not isinstance(config, dict):
        return
    for key, entry in config.items():
        if not isinstance(entry, dict):
            continue
        if "quant_points" in entry or "adapter_coefficients" in entry or "bucket_offsets" in entry:
            yield key, entry
        for subkey in ("k", "v"):
            nested = entry.get(subkey)
            if isinstance(nested, dict) and (
                "quant_points" in nested
                or "adapter_coefficients" in nested
                or "bucket_offsets" in nested
            ):
                yield f"{key}.{subkey}", nested


def _resume_quant_config_is_compatible(config):
    leaf_count = 0
    rank2_count = 0
    rank1_count = 0
    bit_mismatches = []
    bad_shapes = []
    parameterization_mismatches = []
    expected_parameterization = experiment_control["quantizer_parameterization"]

    for key, entry in _iter_quant_config_leaf_entries(config):
        leaf_count += 1
        found_parameterization = entry.get("quantizer_parameterization")
        if found_parameterization is None:
            if expected_parameterization != "nonuniform_points_thresholds":
                parameterization_mismatches.append((key, "<missing>"))
        elif str(found_parameterization) != str(expected_parameterization):
            parameterization_mismatches.append((key, found_parameterization))

        if "bucket_offsets" in entry:
            try:
                offset_tensor = torch.as_tensor(entry["bucket_offsets"])
                threshold_tensor = torch.as_tensor(entry.get("bucket_thresholds"))
            except Exception:
                bad_shapes.append((key, "<unreadable bucket residual state>"))
                continue
            expected_offsets = 1 << int(args.num_bits)
            expected_thresholds = expected_offsets - 1
            if offset_tensor.dim() == 2:
                rank2_count += 1
            else:
                bad_shapes.append((key, tuple(offset_tensor.shape)))
            if (
                offset_tensor.dim() != 2
                or int(offset_tensor.shape[-1]) != expected_offsets
                or threshold_tensor.dim() != 2
                or tuple(threshold_tensor.shape)
                != (int(offset_tensor.shape[0]), expected_thresholds)
            ):
                bad_shapes.append(
                    (key, (tuple(offset_tensor.shape), tuple(threshold_tensor.shape)))
                )
            if (
                not torch.isfinite(offset_tensor).all()
                or not torch.isfinite(threshold_tensor).all()
            ):
                parameterization_mismatches.append((key, "nonfinite_bucket_state"))
            if bool(entry.get("compressed_kv_cache", True)):
                parameterization_mismatches.append((key, "adapter_marked_compressed"))
            if bool(entry.get("hard_forward", True)):
                parameterization_mismatches.append((key, "adapter_marked_hard_qdq"))
            if not bool(entry.get("hard_bucket_assignment", False)):
                parameterization_mismatches.append((key, "missing_hard_bucket_assignment"))
            continue

        if "adapter_coefficients" in entry:
            try:
                coefficient_tensor = torch.as_tensor(entry["adapter_coefficients"])
            except Exception:
                bad_shapes.append((key, "<unreadable adapter coefficients>"))
                continue
            if coefficient_tensor.dim() == 2:
                rank2_count += 1
            else:
                bad_shapes.append((key, tuple(coefficient_tensor.shape)))
            expected_coefficients = (2 * (1 << int(args.num_bits))) - 1
            if (
                coefficient_tensor.dim() == 2
                and int(coefficient_tensor.shape[-1]) != expected_coefficients
            ):
                bad_shapes.append((key, tuple(coefficient_tensor.shape)))
            if not torch.isfinite(coefficient_tensor).all():
                parameterization_mismatches.append((key, "nonfinite_coefficients"))
            if bool(entry.get("compressed_kv_cache", True)):
                parameterization_mismatches.append((key, "adapter_marked_compressed"))
            if bool(entry.get("hard_forward", True)):
                parameterization_mismatches.append((key, "adapter_marked_hard_qdq"))
            continue

        try:
            q_tensor = torch.as_tensor(entry.get("quant_points"))
        except Exception:
            bad_shapes.append((key, "<unreadable>"))
            continue
        if q_tensor.dim() == 2:
            rank2_count += 1
        elif q_tensor.dim() == 1:
            rank1_count += 1
        else:
            bad_shapes.append((key, tuple(q_tensor.shape)))
        if "num_bits" in entry:
            try:
                if int(entry["num_bits"]) != int(args.num_bits):
                    bit_mismatches.append((key, int(entry["num_bits"])))
            except (TypeError, ValueError):
                bit_mismatches.append((key, entry.get("num_bits")))
        if expected_parameterization == "uniform_affine_endpoints" and q_tensor.dim() == 2:
            expected_uniform = torch.linspace(
                0.0,
                1.0,
                int(q_tensor.shape[-1]),
                dtype=q_tensor.dtype,
            )
            expected_uniform = (
                q_tensor[..., :1] + (q_tensor[..., -1:] - q_tensor[..., :1]) * expected_uniform
            )
            if not torch.allclose(q_tensor, expected_uniform, rtol=1e-5, atol=1e-6):
                parameterization_mismatches.append((key, "nonuniform_values"))
            if (
                not torch.isfinite(q_tensor).all()
                or bool((q_tensor[..., 0] < 0.0).any())
                or bool((q_tensor[..., -1] > 1.0).any())
                or bool((q_tensor[..., -1] <= q_tensor[..., 0]).any())
            ):
                parameterization_mismatches.append((key, "invalid_endpoints"))
            thresholds = entry.get("thresholds")
            if thresholds is None:
                parameterization_mismatches.append((key, "missing_midpoint_thresholds"))
            else:
                threshold_tensor = torch.as_tensor(thresholds, dtype=q_tensor.dtype)
                expected_thresholds = (q_tensor[..., :-1] + q_tensor[..., 1:]) * 0.5
                if tuple(threshold_tensor.shape) != tuple(
                    expected_thresholds.shape
                ) or not torch.allclose(
                    threshold_tensor,
                    expected_thresholds,
                    rtol=1e-5,
                    atol=1e-6,
                ):
                    parameterization_mismatches.append((key, "non_midpoint_thresholds"))

    if leaf_count <= 0:
        return False, "no K/V adaptation entries found"
    if bad_shapes:
        key, shape = bad_shapes[0]
        return False, f"{key} has unsupported quant_points shape {shape}"
    if bit_mismatches:
        key, found_bits = bit_mismatches[0]
        return False, (f"{key} has num_bits={found_bits}, requested num_bits={args.num_bits}")
    if parameterization_mismatches:
        key, found = parameterization_mismatches[0]
        return False, (
            f"{key} has quantizer_parameterization={found}, requested {expected_parameterization}"
        )
    if rank2_count != leaf_count:
        return False, (
            "per-group quant tables are required, but resume config contains "
            f"{rank1_count} shared-table entries out of {leaf_count}"
        )
    return True, ""


run_name = "__".join(run_name_parts)  # Using double underscore for better readability


def _safe_run_dir_name(name: str, max_len: int = 240) -> str:
    """Return a deterministic filesystem-safe component for long run names."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise ValueError(f"Run name is not a safe filesystem component: {name!r}")
    if len(name) <= max_len:
        return name
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
    keep = max(1, max_len - len(digest) - 2)
    return f"{name[:keep]}__{digest}"


# Create a config-specific directory under the selected run root so unrelated
# checkpoints cannot be discovered or overwritten.
config_dir_name = _safe_run_dir_name(run_name)
logs_root_dir = os.path.normpath(os.path.expanduser(str(args.logs_root).strip()))
if not logs_root_dir or logs_root_dir == ".":
    raise ValueError("--logs_root must name an explicit non-empty directory.")
logs_root_dir = os.path.abspath(logs_root_dir)
config_dir_path = os.path.abspath(os.path.join(logs_root_dir, config_dir_name))
if os.path.dirname(config_dir_path) != logs_root_dir:
    raise ValueError("Run artifact directory escaped --logs_root.")
QUANT_CONFIG_DIR = os.path.join(config_dir_path, "quant_configs")
runs_dir = os.path.join(config_dir_path, "runs")
if config_dir_name != run_name and is_main():
    print(f"Run name shortened for filesystem path: {config_dir_name}")

resume_quant_config_path = None
resume_config_step = None
resume_quant_config = None
resume_training_state_path = None
resume_training_state = None
if args.no_auto_resume:
    rank0_print("Auto resume disabled (--no_auto_resume).")
else:
    resume_quant_config_path, resume_config_step = find_latest_resume_quant_config(
        logs_root_dir,
        run_name,
        config_dir_name,
        QUANT_CONFIG_DIR,
    )
if resume_quant_config_path:
    try:
        resume_quant_config = load_quant_config(resume_quant_config_path)
        resume_ok, resume_reason = _resume_quant_config_is_compatible(resume_quant_config)
        if not resume_ok:
            rank0_print(
                "Warning: ignoring incompatible resume config "
                f"{resume_quant_config_path}: {resume_reason}"
            )
            resume_quant_config_path = None
            resume_config_step = None
            resume_quant_config = None
        if resume_config_step is not None:
            rank0_print(
                f"Resume: found {resume_quant_config_path} (last optimizer step {resume_config_step})."
            )
        elif resume_quant_config_path:
            rank0_print(f"Resume: found {resume_quant_config_path}.")
    except Exception as e:
        rank0_print(f"Warning: Failed to load resume config {resume_quant_config_path}: {e}")
        resume_quant_config_path = None
        resume_config_step = None
        resume_quant_config = None
    if resume_quant_config is not None and resume_config_step is not None:
        resume_quant_config_dir = os.path.dirname(resume_quant_config_path)
        resume_training_state_path = find_training_state_for_step(
            resume_quant_config_dir, resume_config_step, _rank
        )
        if resume_training_state_path:
            try:
                resume_training_state = _load_torch_checkpoint(resume_training_state_path)
                rank0_print(f"Resume: found training state {resume_training_state_path}.")
            except Exception as e:
                rank0_print(
                    f"Warning: Failed to load resume training state {resume_training_state_path}: {e}"
                )
                resume_training_state_path = None
                resume_training_state = None

# Create subdirectories for quant_configs and runs within the config directory.
if is_main():
    os.makedirs(QUANT_CONFIG_DIR, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)
dist_barrier("log_dirs_ready")

resume_enabled = resume_quant_config is not None
metrics_path = args.metrics_path or os.path.join(config_dir_path, "metrics.json")


def _empty_eval_metrics():
    return {
        "loss": float("nan"),
        "ppl": float("nan"),
        "target_tokens": 0,
        "stages": [],
    }


def _shifted_label_token_count(labels):
    if labels.numel() <= labels.shape[0]:
        return torch.zeros((), device=labels.device, dtype=torch.float32)
    return (labels[:, 1:] != -100).sum(dtype=torch.float32)


def _run_loss_forward(
    model_obj,
    input_ids,
    attention_mask,
    position_ids,
    labels,
    *,
    use_cache=False,
    backward_scale=1.0,
    return_hidden_grad=False,
):
    if return_hidden_grad:
        loss, target_tokens, hidden_states, hidden_grad = _streaming_causal_lm_loss(
            model_obj,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            use_cache=use_cache,
            backward_scale=backward_scale,
            return_hidden_grad=True,
        )
        surrogate_grad = None
        if hidden_states is not None and hidden_grad is not None:
            surrogate_grad = torch.sum(hidden_states * hidden_grad.detach())
        return loss, target_tokens, surrogate_grad

    return _streaming_causal_lm_loss(
        model_obj,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        labels=labels,
        use_cache=use_cache,
        backward_scale=backward_scale,
        return_hidden_grad=False,
    )


def _eval_batch_slice(batch_eval, start, end):
    return {
        key: value[start:end]
        if torch.is_tensor(value) and value.shape[:1] == batch_eval["input_ids"].shape[:1]
        else value
        for key, value in batch_eval.items()
    }


@torch.no_grad()
def _evaluate_batch_stats(model_eval, batch_eval, device_eval, progress_label=None, split_depth=0):
    batch_size = int(batch_eval["input_ids"].shape[0])
    stats = torch.zeros(2, device=device_eval, dtype=torch.float64)  # nll_sum, target_tokens
    try:
        input_ids = batch_eval["input_ids"].to(device_eval, non_blocking=True)
        attention_mask_all_ones = batch_eval.get("attention_mask_all_ones", False)
        if torch.is_tensor(attention_mask_all_ones):
            attention_mask_all_ones = bool(attention_mask_all_ones.item())
        else:
            attention_mask_all_ones = bool(attention_mask_all_ones)
        attention_mask = (
            None
            if attention_mask_all_ones and "position_ids" not in batch_eval
            else (
                batch_eval["attention_mask"].to(device_eval, non_blocking=True)
                if "attention_mask" in batch_eval
                else None
            )
        )
        labels = batch_eval["labels"].to(device_eval, non_blocking=True)
        position_ids = (
            batch_eval["position_ids"].to(device_eval, non_blocking=True)
            if "position_ids" in batch_eval
            else None
        )
        if position_ids is not None:
            attention_mask = build_doc_attn_mask(position_ids, dtype=torch.bfloat16)

        loss, target_tokens = _run_loss_forward(
            model_eval,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            use_cache=False,
        )
        target_tokens = target_tokens.to(torch.float64)
        if target_tokens.item() <= 0 or loss is None:
            return stats
        if torch.isnan(loss) or torch.isinf(loss):
            return stats
        stats[0] += loss.detach().float().to(torch.float64) * target_tokens.to(torch.float64)
        stats[1] += target_tokens.to(torch.float64)
        return stats
    except Exception as exc:
        if _is_oom_exception(exc):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if batch_size > 1:
                mid = batch_size // 2
                if split_depth == 0:
                    label = f"{progress_label}: " if progress_label else ""
                    rank0_print(
                        f"{label}eval OOM at batch_size={batch_size}; "
                        f"splitting into {mid} + {batch_size - mid}."
                    )
                left = _evaluate_batch_stats(
                    model_eval,
                    _eval_batch_slice(batch_eval, 0, mid),
                    device_eval,
                    progress_label=progress_label,
                    split_depth=split_depth + 1,
                )
                right = _evaluate_batch_stats(
                    model_eval,
                    _eval_batch_slice(batch_eval, mid, batch_size),
                    device_eval,
                    progress_label=progress_label,
                    split_depth=split_depth + 1,
                )
                return left + right
        print(f"Error evaluating batch on device {device_eval}: {exc}")
        return stats


@torch.no_grad()
def _evaluate_dataloader_stats(
    model_eval, dataloader_eval, device_eval, progress_label=None, max_batches=None
):
    model_eval.eval()
    stats = torch.zeros(2, device=device_eval, dtype=torch.float64)  # nll_sum, target_tokens
    is_final_eval = str(progress_label or "").startswith("Final eval")
    progress_interval = max(0, int(os.getenv("DISCOVER_EVAL_PROGRESS_INTERVAL", "10")))
    eval_batch_status_logging = _env_flag("DISCOVER_EVAL_BATCH_STATUS")
    eval_batch_timing_logging = _env_flag("DISCOVER_EVAL_BATCH_TIMING")
    if is_final_eval:
        eval_batch_status_logging = _env_flag("DISCOVER_FINAL_EVAL_BATCH_STATUS", "1")
        eval_batch_timing_logging = _env_flag("DISCOVER_FINAL_EVAL_BATCH_TIMING", "1")
        progress_interval = max(
            0,
            int(os.getenv("DISCOVER_FINAL_EVAL_PROGRESS_INTERVAL", "1")),
        )
    try:
        total_batches = len(dataloader_eval)
    except Exception:
        total_batches = None
    if max_batches is not None and int(max_batches) > 0:
        max_batches = int(max_batches)
        if total_batches is not None:
            total_batches = min(total_batches, max_batches)
    else:
        max_batches = None

    for batch_idx, batch_eval in enumerate(dataloader_eval, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break
        if (
            progress_label is not None
            and eval_batch_status_logging
            and (
                batch_idx == 1
                or (progress_interval > 0 and batch_idx % progress_interval == 0)
                or (total_batches is not None and batch_idx == total_batches)
            )
        ):
            if total_batches is None:
                rank0_print(f"{progress_label}: eval batch {batch_idx}")
            else:
                rank0_print(f"{progress_label}: eval batch {batch_idx}/{total_batches}")
        batch_start_time = time.perf_counter() if eval_batch_timing_logging else None
        batch_stats = _evaluate_batch_stats(
            model_eval,
            batch_eval,
            device_eval,
            progress_label=progress_label,
        )
        if batch_start_time is not None:
            if getattr(device_eval, "type", str(device_eval)) == "cuda":
                try:
                    torch.cuda.synchronize(device_eval)
                except Exception:
                    pass
            elapsed = time.perf_counter() - batch_start_time
            tokens = int(batch_stats[1].detach().item())
            if total_batches is None:
                rank0_print(
                    f"{progress_label}: eval batch {batch_idx} done "
                    f"in {elapsed:.2f}s, target_tokens={tokens:,}"
                )
            else:
                rank0_print(
                    f"{progress_label}: eval batch {batch_idx}/{total_batches} done "
                    f"in {elapsed:.2f}s, target_tokens={tokens:,}"
                )
        stats += batch_stats

    return stats


def _loss_ppl_from_stats(stats):
    if stats[1].item() <= 0:
        return float("inf"), float("inf")
    loss = (stats[0] / stats[1]).item()
    try:
        ppl = math.exp(loss)
    except OverflowError:
        ppl = float("inf")
    return loss, ppl


def _reduce_eval_stage_stats(stage_stats):
    if not _distributed:
        return
    if _pipeline_enabled():
        if not (_pipeline_is_last_stage() and _tensor_model_parallel_rank == 0):
            stage_stats.zero_()
        torch.distributed.all_reduce(stage_stats, op=torch.distributed.ReduceOp.SUM)
        return
    _distributed_all_reduce(stage_stats, op=torch.distributed.ReduceOp.SUM)


def run_stage_ppl(
    model_eval,
    stages,
    dataloader_key,
    label,
    writer_obj=None,
    loss_tag=None,
    ppl_tag=None,
    step=None,
    max_batches_per_stage=0,
):
    total_stats = torch.zeros(2, device=device, dtype=torch.float64)
    stage_results = []

    for stage in stages:
        dataloader = stage.get(dataloader_key)
        if dataloader is None:
            continue

        model_eval.seqlen = int(stage["seqlen"])
        stage_batches = len(dataloader)
        if max_batches_per_stage is not None and int(max_batches_per_stage) > 0:
            stage_batches = min(stage_batches, int(max_batches_per_stage))
        rank0_print(
            f"{label} stage {stage['name']} starting "
            f"(seqlen={int(stage['seqlen']):,}, batches/rank={stage_batches}, "
            f"global_samples={len(stage.get('eval_samples', stage.get('val_samples', [])))})"
        )
        try:
            stage_stats = _evaluate_dataloader_stats(
                model_eval,
                dataloader,
                device,
                progress_label=f"{label} stage {stage['name']}",
                max_batches=max_batches_per_stage,
            )
        except Exception as e:
            if is_main():
                print(f"Error running {label} stage {stage['name']}: {e}", flush=True)
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            continue

        if _distributed:
            _reduce_eval_stage_stats(stage_stats)

        stage_loss, stage_ppl = _loss_ppl_from_stats(stage_stats)
        total_stats += stage_stats
        stage_result = {
            "name": stage["name"],
            "seqlen": int(stage["seqlen"]),
            "loss": stage_loss,
            "ppl": stage_ppl,
            "target_tokens": int(stage_stats[1].item()),
        }
        stage_results.append(stage_result)

        if is_main() and math.isfinite(stage_loss):
            print(
                f"{label} stage {stage['name']} "
                f"(seqlen={int(stage['seqlen']):,}): loss={stage_loss:.6f}, "
                f"ppl={stage_ppl:.6f}, target_tokens={int(stage_stats[1].item()):,}"
            )
        if writer_obj is not None and step is not None and math.isfinite(stage_loss):
            writer_obj.add_scalar(f"Loss/{label}_stage/{stage['name']}", stage_loss, step)
            if math.isfinite(stage_ppl):
                writer_obj.add_scalar(f"PPL/{label}_stage/{stage['name']}", stage_ppl, step)

    loss, ppl = _loss_ppl_from_stats(total_stats)
    if is_main():
        if math.isfinite(loss):
            print(
                f"\n{label} aggregate: loss={loss:.6f}, ppl={ppl:.6f}, "
                f"target_tokens={int(total_stats[1].item()):,}"
            )
        else:
            print(f"\n{label} aggregate skipped: no valid target tokens.")
    if writer_obj is not None and step is not None and math.isfinite(loss):
        if loss_tag is not None:
            writer_obj.add_scalar(loss_tag, loss, step)
        if ppl_tag is not None and math.isfinite(ppl):
            writer_obj.add_scalar(ppl_tag, ppl, step)

    return {
        "loss": loss,
        "ppl": ppl,
        "target_tokens": int(total_stats[1].item()),
        "stages": stage_results,
    }


if args.skip_initial_eval:
    rank0_print("Skipping native eval/validation runs before quantization (--skip_initial_eval).")
    native_init_eval_metrics = _empty_eval_metrics()
    native_initial_periodic_val_metrics = _empty_eval_metrics()
else:
    rank0_print("Starting native model eval set before KV quantization.")
    native_init_eval_metrics = run_stage_ppl(
        model,
        test_stages,
        "eval_dataloader",
        "Native_eval_init",
        writer_obj=None,
        loss_tag=None,
        ppl_tag=None,
        step=0,
    )
    rank0_print("Starting native model validation set before KV quantization.")
    native_initial_periodic_val_metrics = run_stage_ppl(
        model,
        eval_stages,
        "eval_dataloader",
        "Native_val_init",
        writer_obj=None,
        loss_tag=None,
        ppl_tag=None,
        step=0,
    )
native_init_eval_ppl = native_init_eval_metrics["ppl"]
native_initial_val_ppl = native_initial_periodic_val_metrics["ppl"]
if torch.cuda.is_available():
    torch.cuda.empty_cache()
dist_barrier("native_initial_eval_done")

# --- Apply quantization modules ---
quantizer_params = []
quantized_layer_names = []


def _discover_quant_type(target):
    grouping_dim = target.get("grouping_dim")
    return f"{QuantLayerCls.__name__}_{num_bits}bit_{grouping_dim}_group_tables"


quantized_layer_names = apply_kv_quantization_targets(
    model,
    quant_targets,
    QuantizedLinearCls,
    type_fn=_discover_quant_type,
    qkv_quantized_linear_cls=QuantizedQKVLinearCls,
)

# --- Post-RoPE K quantization ---
install_post_rope_k_quantization(
    model,
    quantized_linear_cls=QuantizedLinearCls,
    quantized_qkv_linear_cls=QuantizedQKVLinearCls,
    log_fn=print,
)

quantized_layer_name_set = {
    entry[0] if isinstance(entry, tuple) else entry for entry in quantized_layer_names
}

# Track processed quant_modules by id to avoid duplicate parameter collection for fused QKV
processed_quant_module_ids = set()
local_quantizer_module_ids = set()

for target in quant_targets:
    layer_name = target.get("name")
    quant_module = target.get("quant_module")
    if quant_module is None or layer_name is None:
        continue
    if layer_name not in quantized_layer_name_set:
        continue
    if not _pipeline_target_is_local(layer_name):
        continue

    # Skip if this quant_module was already processed (avoid duplicates in fused QKV)
    quant_module_id = id(quant_module)
    if quant_module_id in processed_quant_module_ids:
        continue
    processed_quant_module_ids.add(quant_module_id)
    local_quantizer_module_ids.add(quant_module_id)

    for param_name, param in quant_module.named_parameters():
        quantizer_params.append(param)
    print(f"Successfully quantized: {layer_name}")

if not quantized_layer_names:
    print(
        "Warning: No layers were quantized. The model may not be compatible with the current quantization approach."
    )
    print("Available layer names:")
    for name, module in model.named_modules():
        if hasattr(module, "weight") and len(list(module.parameters())) > 0:
            print(f"  - {name}: {type(module).__name__}")
else:
    print(f"Successfully quantized {len(quantized_layer_names)} layers")

# --- Freeze non-quantized parameters ---
quantizer_param_ids = {id(param) for param in quantizer_params}
for name, param in model.named_parameters():
    is_quant_param = id(param) in quantizer_param_ids
    if not is_quant_param and quantizer_params:
        param.requires_grad = False
    elif is_quant_param:
        param.requires_grad = True

# --- Cache quantizer modules to avoid per-step scanning ---
quantizer_modules = [
    module
    for module in model.modules()
    if isinstance(module, UnifiedQuantLayer)
    and (not _pipeline_enabled() or id(module) in local_quantizer_module_ids)
]
print(f"Cached {len(quantizer_modules)} quantizer modules for threshold updates")


if resume_quant_config is not None:
    applied_layers = apply_quant_config(model, resume_quant_config, QuantizedLinearCls)
    if applied_layers == 0:
        print("Warning: Resume config did not match any quantized layers.")


def _set_unique_param_requires_grad(modules, attr_name, enabled):
    count = 0
    seen = set()
    for module_iter in modules:
        param = getattr(module_iter, attr_name, None)
        if param is None or id(param) in seen:
            continue
        param.requires_grad_(bool(enabled))
        count += 1
        seen.add(id(param))
    return count


q_point_tensor_count = _set_unique_param_requires_grad(
    quantizer_modules,
    "q_points",
    args.train_quant_points,
)
threshold_tensor_count = _set_unique_param_requires_grad(
    quantizer_modules,
    "thresholds",
    args.train_thresholds,
)
affine_low_tensor_count = _set_unique_param_requires_grad(
    quantizer_modules,
    "affine_low",
    True,
)
affine_high_tensor_count = _set_unique_param_requires_grad(
    quantizer_modules,
    "affine_high",
    True,
)
adapter_coefficient_tensor_count = _set_unique_param_requires_grad(
    quantizer_modules,
    "coefficients",
    True,
)
bucket_offset_tensor_count = _set_unique_param_requires_grad(
    quantizer_modules,
    "bucket_offsets",
    True,
)
rank0_print(
    "Quantizer ablation controls: "
    f"train_quant_points={bool(args.train_quant_points)} ({q_point_tensor_count} tensors), "
    f"train_thresholds={bool(args.train_thresholds)} ({threshold_tensor_count} tensors), "
    f"train_affine_endpoints={affine_low_tensor_count + affine_high_tensor_count} tensors, "
    f"train_adapter_coefficients={adapter_coefficient_tensor_count} tensors, "
    f"train_bucket_offsets={bucket_offset_tensor_count} tensors, "
    f"threshold_grad_mode={args.threshold_grad_mode}, "
    f"threshold_grad_aggregation={THRESHOLD_GRAD_AGGREGATION}"
)

# --- Parameter statistics ---
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params}")
print(f"Total parameters: {all_params}")
if all_params > 0:
    print(f"Percentage of trainable parameters: {100 * trainable_params / all_params:.6f}%")

# --- Optional distributed wrapping ---
model = wrap_model_distributed(model, ignored_modules=quantizer_modules)
model.seqlen = int(training_stages[0]["seqlen"])
if _distributed:
    device = torch.device("cuda", _local_rank)
else:
    device = next(model.parameters()).device

if _distributed:
    rank0_print(f"Megatron runtime distributed across {_world_size} GPUs.")

# --- Optimizer setup ---
# Since only adam_custom is supported, the logic simplifies significantly.
param_groups = []
q_points_params = []
threshold_params = []
affine_endpoint_params = []
adapter_coefficient_params = []
bucket_offset_params = []
collected_param_ids = set()

# Keep the stable parameter handles, but avoid hundreds of one-tensor Adam
# groups. Grouping these tiny tensors cuts Python and optimizer overhead.
for module in quantizer_modules:
    if not isinstance(module, UnifiedQuantLayer):
        continue
    if (
        args.train_quant_points
        and hasattr(module, "q_points")
        and module.q_points.requires_grad
        and id(module.q_points) not in collected_param_ids
    ):
        q_points_params.append(module.q_points)
        collected_param_ids.add(id(module.q_points))
    if (
        args.train_thresholds
        and hasattr(module, "thresholds")
        and module.thresholds.requires_grad
        and id(module.thresholds) not in collected_param_ids
    ):
        threshold_params.append(module.thresholds)
        collected_param_ids.add(id(module.thresholds))
    for endpoint_name in ("affine_low", "affine_high"):
        endpoint = getattr(module, endpoint_name, None)
        if (
            endpoint is not None
            and endpoint.requires_grad
            and id(endpoint) not in collected_param_ids
        ):
            affine_endpoint_params.append(endpoint)
            collected_param_ids.add(id(endpoint))
    coefficients = getattr(module, "coefficients", None)
    if (
        coefficients is not None
        and coefficients.requires_grad
        and id(coefficients) not in collected_param_ids
    ):
        adapter_coefficient_params.append(coefficients)
        collected_param_ids.add(id(coefficients))
    bucket_offsets = getattr(module, "bucket_offsets", None)
    if (
        bucket_offsets is not None
        and bucket_offsets.requires_grad
        and id(bucket_offsets) not in collected_param_ids
    ):
        bucket_offset_params.append(bucket_offsets)
        collected_param_ids.add(id(bucket_offsets))

if q_points_params:
    param_groups.append(
        {
            "params": q_points_params,
            "lr": args.lr,
            "weight_decay": 0.0,
        }
    )
if threshold_params:
    param_groups.append(
        {
            "params": threshold_params,
            "lr": args.lr,
            "weight_decay": 0.0,
        }
    )
if affine_endpoint_params:
    param_groups.append(
        {
            "params": affine_endpoint_params,
            "lr": args.lr,
            "weight_decay": 0.0,
        }
    )
if adapter_coefficient_params:
    param_groups.append(
        {
            "params": adapter_coefficient_params,
            "lr": args.lr,
            "weight_decay": 0.0,
        }
    )
if bucket_offset_params:
    param_groups.append(
        {
            "params": bucket_offset_params,
            "lr": args.lr,
            "weight_decay": 0.0,
        }
    )


class _NoOpOptimizer:
    def __init__(self):
        self.param_groups = []

    def zero_grad(self, set_to_none=True):
        return None

    def step(self):
        return None

    def state_dict(self):
        return {"state": {}, "param_groups": []}

    def load_state_dict(self, _state_dict):
        return None


def _build_quant_optimizer(param_groups_for_optimizer, lr):
    if not param_groups_for_optimizer:
        return _NoOpOptimizer()

    adam_impl = os.getenv("DISCOVER_ADAM_IMPL", "fused").strip().lower()
    if adam_impl in {"fused", "auto"}:
        try:
            return optim.Adam(param_groups_for_optimizer, lr=lr, fused=True)
        except (RuntimeError, TypeError, ValueError) as exc:
            rank0_print(
                f"Warning: fused Adam unavailable for quantizer params ({exc}); falling back."
            )
            if adam_impl == "fused":
                adam_impl = "foreach"
    if adam_impl in {"foreach", "auto"}:
        try:
            return optim.Adam(param_groups_for_optimizer, lr=lr, foreach=True)
        except (RuntimeError, TypeError, ValueError) as exc:
            rank0_print(
                f"Warning: foreach Adam unavailable for quantizer params ({exc}); falling back."
            )
    return optim.Adam(param_groups_for_optimizer, lr=lr)


optimizer = _build_quant_optimizer(param_groups, args.lr)
rank0_print(
    "Quantizer optimizer: "
    f"{optimizer.__class__.__name__}, param_groups={len(getattr(optimizer, 'param_groups', []))}, "
    f"q_tensors={len(q_points_params)}, threshold_tensors={len(threshold_params)}, "
    f"affine_endpoint_tensors={len(affine_endpoint_params)}, "
    f"adapter_coefficient_tensors={len(adapter_coefficient_params)}, "
    f"bucket_offset_tensors={len(bucket_offset_params)}"
)

resume_state_global_step = None
resume_state_global_micro_step = None
if isinstance(resume_training_state, dict):
    resume_control_name = resume_training_state.get("experiment_control")
    if resume_control_name is None:
        resume_control_name = "beyond_nll"
    if str(resume_control_name) != str(args.experiment_control):
        raise RuntimeError(
            "Resume training state control mismatch: "
            f"checkpoint={resume_control_name}, requested={args.experiment_control}"
        )
    resume_threshold_aggregation = resume_training_state.get("threshold_grad_aggregation")
    if resume_threshold_aggregation != THRESHOLD_GRAD_AGGREGATION:
        raise RuntimeError(
            "Resume training state threshold-gradient semantics mismatch: "
            f"checkpoint={resume_threshold_aggregation!r}, "
            f"requested={THRESHOLD_GRAD_AGGREGATION!r}. "
            "Start a fresh optimizer trajectory for the deferred global-side update."
        )
    resume_threshold_mode = resume_training_state.get("threshold_grad_mode")
    if resume_threshold_mode != args.threshold_grad_mode:
        raise RuntimeError(
            "Resume training state threshold-gradient mode mismatch: "
            f"checkpoint={resume_threshold_mode!r}, requested={args.threshold_grad_mode!r}"
        )
    try:
        resume_state_global_step = int(resume_training_state.get("global_step"))
    except (TypeError, ValueError):
        resume_state_global_step = None
    try:
        resume_state_global_micro_step = int(resume_training_state.get("global_micro_step"))
    except (TypeError, ValueError):
        resume_state_global_micro_step = None
    optimizer_state = resume_training_state.get("optimizer_state_dict")
    if optimizer_state is not None and hasattr(optimizer, "load_state_dict"):
        try:
            optimizer.load_state_dict(optimizer_state)
            rank0_print("Resume: restored optimizer state.")
        except Exception as exc:
            rank0_print(
                f"Warning: failed to restore optimizer state; continuing from quant config only: {exc}"
            )
    torch_rng_state = resume_training_state.get("torch_rng_state")
    if torch.is_tensor(torch_rng_state):
        try:
            torch.set_rng_state(torch_rng_state)
        except Exception as exc:
            rank0_print(f"Warning: failed to restore torch RNG state: {exc}")
    cuda_rng_state_all = resume_training_state.get("cuda_rng_state_all")
    if torch.cuda.is_available() and cuda_rng_state_all:
        try:
            torch.cuda.set_rng_state_all(cuda_rng_state_all)
        except Exception as exc:
            rank0_print(f"Warning: failed to restore CUDA RNG state: {exc}")

# For adam_custom, gradients are on individual tensors within param_groups.
optimizer_step_params = []
for group in optimizer.param_groups:
    optimizer_step_params.extend(group["params"])
optimizer_step_params = list(dict.fromkeys(optimizer_step_params))  # Ensure uniqueness

_backend_stepper = optimizer
lr_schedule_total_steps = None
last_optimizer_lr = float(args.lr)


def _zero_stepper_grad(stepper, quantizer_modules_for_step=None):
    if stepper is not None and hasattr(stepper, "zero_grad"):
        try:
            stepper.zero_grad(set_to_none=True)
        except TypeError:
            stepper.zero_grad()
    if quantizer_modules_for_step is not None:
        clear_threshold_side_sums_(quantizer_modules_for_step)


# --- Quantization configuration save settings ---
global_step = int(
    resume_state_global_step if resume_state_global_step is not None else (resume_config_step or 0)
)
if (
    resume_config_step is not None
    and resume_state_global_step is not None
    and int(resume_state_global_step) != int(resume_config_step)
):
    rank0_print(
        "Warning: resume training state step does not match quant config step; "
        f"using quant config step {resume_config_step}."
    )
    global_step = int(resume_config_step)
global_micro_step = int(resume_state_global_micro_step or 0)


# --- Quantization configuration extraction and saving function ---
def save_metrics(metrics_path, metrics_payload):
    if not metrics_path:
        return
    metrics_dir = os.path.dirname(metrics_path)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    try:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving metrics: {str(e)}")


def _local_quantized_layer_entries(entries):
    if not _pipeline_enabled():
        return list(entries)
    local_entries = []
    for entry in entries:
        layer_name = entry[0] if isinstance(entry, tuple) else entry
        if _pipeline_target_is_local(layer_name):
            local_entries.append(entry)
    return local_entries


def _merge_tensor_parallel_quant_configs(config_parts):
    valid_parts = [part for part in config_parts if isinstance(part, dict)]
    if not valid_parts:
        return {}
    merged = {}

    def _merge_entry_for_key(key):
        entries = [part.get(key) for part in valid_parts if isinstance(part.get(key), dict)]
        if not entries:
            return None
        base_entry = dict(entries[0])
        for subkey in ("k", "v"):
            if isinstance(base_entry.get(subkey), dict):
                merged_sub = _merge_quant_leaf([entry.get(subkey) for entry in entries])
                if merged_sub is not None:
                    base_entry[subkey] = merged_sub
        if (
            "quant_points" in base_entry
            or "adapter_coefficients" in base_entry
            or "bucket_offsets" in base_entry
        ):
            merged_leaf = _merge_quant_leaf(entries)
            if merged_leaf is not None:
                base_entry = merged_leaf
        return base_entry

    def _merge_quant_leaf(entries):
        leaves = [
            entry
            for entry in entries
            if isinstance(entry, dict)
            and (
                "quant_points" in entry
                or "adapter_coefficients" in entry
                or "bucket_offsets" in entry
            )
        ]
        if not leaves:
            return None
        base = dict(leaves[0])
        if "adapter_coefficients" in base:
            coefficient_tensors = [
                torch.as_tensor(entry["adapter_coefficients"]).cpu().contiguous()
                for entry in leaves
            ]
            base["adapter_coefficients"] = coefficient_tensors[0]
            return base
        if "bucket_offsets" in base:
            offset_tensors = [
                torch.as_tensor(entry["bucket_offsets"]).cpu().contiguous() for entry in leaves
            ]
            threshold_tensors = [
                torch.as_tensor(entry["bucket_thresholds"]).cpu().contiguous() for entry in leaves
            ]
            base["bucket_offsets"] = offset_tensors[0]
            base["bucket_thresholds"] = threshold_tensors[0]
            return base
        q_tensors = [torch.as_tensor(entry["quant_points"]).cpu().contiguous() for entry in leaves]
        t_tensors = [torch.as_tensor(entry["thresholds"]).cpu().contiguous() for entry in leaves]
        base["quant_points"] = q_tensors[0]
        base["thresholds"] = t_tensors[0]
        return base

    all_keys = set()
    for part in valid_parts:
        all_keys.update(part.keys())
    for key in sorted(all_keys):
        merged_entry = _merge_entry_for_key(key)
        if merged_entry is not None:
            merged[key] = merged_entry
    return merged


def save_distributed_quant_config(step_or_epoch_label):
    checkpoint_metadata = {
        "schema_version": 1,
        "artifact_type": (
            "beyond_kv_quant_config"
            if experiment_control["compressed_kv_cache"]
            else "beyond_uncompressed_kv_adapter_config"
        ),
        "base_model": args.base_model,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "num_bits": int(args.num_bits),
        "cache_bits": (int(args.num_bits) if experiment_control["compressed_kv_cache"] else 16),
        "nominal_bits_for_parameter_matching": (
            int(args.num_bits) if not experiment_control["compressed_kv_cache"] else None
        ),
        "group_size": int(args.group_size),
        "table_axis": "physical_token_group",
        "k_activation_location": "post_rope",
        "v_activation_location": "post_v_projection",
        "hard_forward": bool(experiment_control.get("hard_forward_qdq", True)),
        "hard_bucket_assignment": bool(experiment_control.get("hard_bucket_assignment", False)),
        "compressed_kv_cache": bool(experiment_control["compressed_kv_cache"]),
        "dataset": args.dataset,
        "experiment_control": args.experiment_control,
        "threshold_grad_mode": args.threshold_grad_mode,
        "threshold_grad_aggregation": THRESHOLD_GRAD_AGGREGATION,
        "quantizer_parameterization": experiment_control["quantizer_parameterization"],
        "checkpoint_label": str(step_or_epoch_label),
        "optimizer_step": int(global_step),
        "run_name": run_name,
        "optimizer_seed": int(args.seed),
        "data_seed": int(args.data_seed),
        "eval_seed": int(args.eval_seed),
        "chat_wrap_pile": bool(args.chat_wrap_pile),
        "reference_tokens": int(args.reference_tokens),
        "epochs": int(args.epochs),
        "train_steps_per_pass": int(args.train_steps_per_pass),
        "early_stop_patience": int(args.early_stop_patience),
        "eval_every_steps": int(EVAL_EVERY_STEPS),
        "eval_every_stage": bool(EVAL_EVERY_STAGE),
        "periodic_validation_full": True,
        "lr": float(args.lr),
        "lr_schedule": args.lr_schedule,
        "lr_min_ratio": float(args.lr_min_ratio),
    }
    if not _pipeline_enabled():
        local_config = extract_quant_config(
            unwrap_model(model),
            quantized_layer_names,
            num_bits,
            QuantizedLinearCls,
            metadata=checkpoint_metadata,
        )
        if _distributed and int(args.tensor_model_parallel_size) > 1:
            gathered = [None for _ in range(max(1, int(args.tensor_model_parallel_size)))]
            torch.distributed.all_gather_object(
                gathered, local_config, group=_tensor_model_parallel_group
            )
            if _data_parallel_rank == 0 and _tensor_model_parallel_rank == 0:
                merged_config = _merge_tensor_parallel_quant_configs(gathered)
                final_path = os.path.join(
                    QUANT_CONFIG_DIR, f"quant_config_{step_or_epoch_label}.pt"
                )
                os.makedirs(QUANT_CONFIG_DIR, exist_ok=True)
                try:
                    torch.save(merged_config, final_path)
                    return final_path
                except Exception as exc:
                    print(f"Error saving tensor-parallel quantization config: {exc}")
                    return None
            return None
        if is_main():
            final_path = os.path.join(QUANT_CONFIG_DIR, f"quant_config_{step_or_epoch_label}.pt")
            os.makedirs(QUANT_CONFIG_DIR, exist_ok=True)
            try:
                torch.save(local_config, final_path)
                return final_path
            except Exception as exc:
                print(f"Error saving quantization config: {exc}")
                return None
        return None

    local_entries = _local_quantized_layer_entries(quantized_layer_names)
    local_config = extract_quant_config(
        unwrap_model(model),
        local_entries,
        num_bits,
        QuantizedLinearCls,
        metadata=checkpoint_metadata,
    )
    final_path = None
    should_gather_pipeline = (
        _data_parallel_rank == 0
        and _tensor_model_parallel_rank == 0
        and _pipeline_model_parallel_group is not None
    )
    if should_gather_pipeline:
        gathered = [None for _ in range(max(1, int(args.pipeline_model_parallel_size)))]
        torch.distributed.all_gather_object(
            gathered, local_config, group=_pipeline_model_parallel_group
        )
        if _pipeline_is_first_stage():
            merged_config = {}
            for part in gathered:
                if isinstance(part, dict):
                    merged_config.update(part)
            final_path = os.path.join(QUANT_CONFIG_DIR, f"quant_config_{step_or_epoch_label}.pt")
            os.makedirs(QUANT_CONFIG_DIR, exist_ok=True)
            try:
                torch.save(merged_config, final_path)
            except Exception as exc:
                print(f"Error saving distributed quantization config: {exc}")
                final_path = None
    return final_path


def broadcast_main_path(path, label):
    """Broadcast a filesystem path produced by rank 0 to all workers."""
    if not _distributed:
        return path
    payload = [path if is_main() else None]
    torch.distributed.broadcast_object_list(payload, src=0)
    resolved = payload[0]
    if resolved in ("", None, False):
        return None
    return resolved


def _cumulative_training_wall_seconds():
    start = globals().get("training_wall_start_time")
    if start is None:
        return float(prior_training_wall_seconds)
    end = globals().get("training_wall_end_time")
    if end is None:
        end = time.perf_counter()
    return float(prior_training_wall_seconds) + max(0.0, float(end) - float(start))


def save_training_state(step, quant_config_path=None):
    if step is None:
        return None
    step = int(step)
    state_path = _training_state_path_for_step(QUANT_CONFIG_DIR, step, _rank)
    tmp_path = f"{state_path}.tmp"
    try:
        optimizer_state_dict = optimizer.state_dict() if hasattr(optimizer, "state_dict") else None
    except Exception as exc:
        rank0_print(f"Warning: failed to capture optimizer state: {exc}")
        optimizer_state_dict = None

    payload = {
        "version": 1,
        "global_step": step,
        "global_micro_step": int(global_micro_step),
        "optimizer_update_count": int(optimizer_update_count),
        "epochs_completed": int(epochs_completed),
        "next_periodic_eval_step": (
            int(next_periodic_eval_step) if next_periodic_eval_step is not None else None
        ),
        "val_worse_streak": int(val_worse_streak),
        "validation_history": list(validation_history),
        "early_stop_patience": int(args.early_stop_patience),
        "early_stop_triggered": bool(early_stop_triggered),
        "training_stop_reason": training_stop_reason,
        "best_val_ppl": best_val_ppl,
        "best_val_loss": best_val_loss,
        "best_val_epoch": best_val_epoch,
        "best_val_pass": best_val_pass,
        "best_quant_config_path": best_quant_config_path,
        "final_val_ppl": final_val_ppl,
        "final_val_loss": final_val_loss,
        "quant_config_path": quant_config_path,
        "run_name": run_name,
        "config_dir_name": config_dir_name,
        "rank": int(_rank),
        "world_size": int(_world_size),
        "data_parallel_world_size": int(_data_parallel_world_size),
        "tensor_model_parallel_size": int(args.tensor_model_parallel_size),
        "pipeline_model_parallel_size": int(args.pipeline_model_parallel_size),
        "base_model": args.base_model,
        "num_bits": int(args.num_bits),
        "group_size": int(args.group_size),
        "dataset": args.dataset,
        "experiment_control": args.experiment_control,
        "training_objective": experiment_control["training_objective"],
        "quantizer_parameterization": experiment_control["quantizer_parameterization"],
        "compressed_kv_cache": bool(experiment_control["compressed_kv_cache"]),
        "local_training_input_tokens_processed": int(local_training_input_tokens_processed),
        "local_training_supervised_tokens_processed": int(
            local_training_supervised_tokens_processed
        ),
        "cumulative_training_wall_seconds": _cumulative_training_wall_seconds(),
        "train_quant_points": bool(args.train_quant_points),
        "train_thresholds": bool(args.train_thresholds),
        "threshold_grad_mode": args.threshold_grad_mode,
        "threshold_grad_aggregation": THRESHOLD_GRAD_AGGREGATION,
        "train_steps_per_pass": int(args.train_steps_per_pass),
        "lr": float(args.lr),
        "lr_schedule": args.lr_schedule,
        "lr_min_ratio": float(args.lr_min_ratio),
        "lr_min": float(args.lr) * float(args.lr_min_ratio),
        "lr_schedule_total_steps": int(lr_schedule_total_steps or 0),
        "last_optimizer_lr": float(last_optimizer_lr),
        "optimizer_state_dict": optimizer_state_dict,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    try:
        os.makedirs(QUANT_CONFIG_DIR, exist_ok=True)
        torch.save(payload, tmp_path)
        os.replace(tmp_path, state_path)
        return state_path
    except Exception as exc:
        rank0_print(f"Warning: failed to save training state {state_path}: {exc}")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return None


if SummaryWriter is None:
    rank0_print("Warning: tensorboard is not installed; TensorBoard logging is disabled.")
writer = (
    SummaryWriter(os.path.join(runs_dir, config_dir_name), purge_step=global_step)
    if is_main() and SummaryWriter is not None
    else None
)

# --- Runtime abstraction layer (Megatron-only) ---
args.world_size = _world_size
args.local_rank = _local_rank
args.data_parallel_world_size = _data_parallel_world_size
args.data_parallel_rank = _data_parallel_rank
args.backend_stepper = _backend_stepper
args.optimizer_step_params = optimizer_step_params
_runtime_backend = build_runtime_backends(
    args,
    train_micro_step_fn=None,
    apply_optimizer_step_fn=None,
    run_stage_ppl_fn=None,
    runtime_backend_info=megatron_runtime_info,
)
data_subsystem = _runtime_backend.data
train_subsystem = _runtime_backend.train
eval_subsystem = _runtime_backend.eval
runtime_backend_info = _runtime_backend.info
rank0_print(
    f"Runtime backend selected: {runtime_backend_info.name} "
    f"(world_size={runtime_backend_info.world_size}, local_rank={runtime_backend_info.local_rank})"
)
eval_subsystem.configure(run_stage_ppl_fn=run_stage_ppl)
train_subsystem.configure(
    quantizer_modules=quantizer_modules,
    backend_stepper=_backend_stepper,
    optimizer_step_params=optimizer_step_params,
)

# Persist the actual pre-training hard-QDQ state. Never manufacture an "init"
# configuration from a resumed mid-training state: retain the original file
# when present, or leave the field unset.
initial_quant_config_path = os.path.join(
    QUANT_CONFIG_DIR,
    "quant_config_init.pt",
)
if resume_enabled and int(global_step) > 0:
    if not os.path.isfile(initial_quant_config_path):
        initial_quant_config_path = None
        rank0_print(
            "Resume state has no preserved quant_config_init.pt; refusing to "
            "label the resumed quantizer as the initialization baseline."
        )
else:
    initial_quant_config_path = save_distributed_quant_config("init")
    initial_quant_config_path = broadcast_main_path(
        initial_quant_config_path,
        "initial quantization config",
    )
dist_barrier("initial_quant_config_ready")
if initial_quant_config_path and is_main():
    rank0_print(f"Saved pre-training quantization config to: {initial_quant_config_path}")

# --- Initial evaluation ---
initial_eval_step = global_step if resume_enabled and global_step > 0 else 0
if args.skip_initial_eval:
    rank0_print("Skipping initial eval-set and validation baselines before training.")
    init_eval_metrics = _empty_eval_metrics()
    initial_periodic_val_metrics = _empty_eval_metrics()
else:
    rank0_print("Starting initial eval set before training.")
    init_eval_metrics = eval_subsystem.run_stages(
        model,
        test_stages,
        "eval",
        "Eval_init",
        writer_obj=writer,
        loss_tag="Loss/Eval_init",
        ppl_tag="PPL/Eval_init",
        step=initial_eval_step,
    )
    rank0_print("Starting full validation baseline before training.")
    initial_periodic_val_metrics = eval_subsystem.run_stages(
        model,
        eval_stages,
        "eval",
        "Val_init",
        writer_obj=writer,
        loss_tag="Loss/Val_step",
        ppl_tag="PPL/Val_step",
        step=initial_eval_step,
    )
init_eval_ppl = init_eval_metrics["ppl"]
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# --- Training step function ---
@torch.no_grad()
def _foreach_copy_slices_(params, stacked):
    slices = list(stacked.unbind(0))
    if hasattr(torch, "_foreach_copy_"):
        try:
            torch._foreach_copy_(params, slices)
            return
        except (RuntimeError, TypeError):
            pass
    for param, value in zip(params, slices):
        param.copy_(value)


@torch.no_grad()
def _build_param_buckets(params):
    buckets = {}
    for param in params:
        if param is None:
            continue
        key = (param.device, param.dtype, tuple(param.shape))
        buckets.setdefault(key, []).append(param)
    return list(buckets.values())


@torch.no_grad()
def _project_prebuilt_param_buckets(buckets, project_fn):
    for bucket in buckets:
        if len(bucket) == 1:
            project_fn(bucket[0])
            continue
        stacked = torch.stack(bucket, dim=0)
        project_fn(stacked)
        _foreach_copy_slices_(bucket, stacked)


@torch.no_grad()
def _project_param_buckets(params, project_fn):
    _project_prebuilt_param_buckets(_build_param_buckets(params), project_fn)


_PROJECT_Q_POINT_BUCKETS = _build_param_buckets(q_points_params)
_PROJECT_THRESHOLD_BUCKETS = _build_param_buckets(threshold_params)


@torch.no_grad()
def project_quantizer_modules(quantizer_modules_for_step):
    if quantizer_modules_for_step is quantizer_modules:
        _project_prebuilt_param_buckets(_PROJECT_Q_POINT_BUCKETS, project_quant_points)
        _project_prebuilt_param_buckets(_PROJECT_THRESHOLD_BUCKETS, project_thresholds)
        for module_iter in quantizer_modules:
            project_parameters = getattr(module_iter, "project_parameters_", None)
            if callable(project_parameters):
                project_parameters()
        return

    q_params = []
    threshold_params = []
    seen_q = set()
    seen_t = set()
    for module_iter in quantizer_modules_for_step:
        q_points = getattr(module_iter, "q_points", None)
        if (
            args.train_quant_points
            and q_points is not None
            and q_points.requires_grad
            and id(q_points) not in seen_q
        ):
            q_params.append(q_points)
            seen_q.add(id(q_points))
        thresholds = getattr(module_iter, "thresholds", None)
        if (
            args.train_thresholds
            and thresholds is not None
            and thresholds.requires_grad
            and id(thresholds) not in seen_t
        ):
            threshold_params.append(thresholds)
            seen_t.add(id(thresholds))

    _project_param_buckets(q_params, project_quant_points)
    _project_param_buckets(threshold_params, project_thresholds)
    for module_iter in quantizer_modules_for_step:
        project_parameters = getattr(module_iter, "project_parameters_", None)
        if callable(project_parameters):
            project_parameters()


def _bucketed_reduce_quantizer_grads(
    params,
    tensor_group=None,
    globally_active=None,
):
    params = list(params)
    if globally_active is None:
        globally_active = [param.grad is not None for param in params]
        activity_buckets = {}
        for index, param in enumerate(params):
            activity_buckets.setdefault(param.device, []).append(index)
        for activity_device, indices in activity_buckets.items():
            activity = torch.tensor(
                [1 if globally_active[index] else 0 for index in indices],
                device=activity_device,
                dtype=torch.int32,
            )
            if tensor_group is not None:
                _all_reduce_group(
                    activity,
                    op=torch.distributed.ReduceOp.SUM,
                    group=tensor_group,
                )
            _distributed_all_reduce(activity, op=torch.distributed.ReduceOp.SUM)
            reduced_flags = activity.gt(0).detach().cpu().tolist()
            for index, flag in zip(indices, reduced_flags):
                globally_active[index] = bool(flag)
    else:
        globally_active = [bool(flag) for flag in globally_active]
        if len(globally_active) != len(params):
            raise ValueError(
                "Pre-reduced quantizer activity must match the parameter roster: "
                f"got {len(globally_active)} flags for {len(params)} parameters."
            )

    buckets = {}
    for param in params:
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        key = (param.grad.device, param.grad.dtype, tuple(param.grad.shape))
        buckets.setdefault(key, []).append(param)

    for bucket in buckets.values():
        if len(bucket) == 1:
            grad = bucket[0].grad
            if tensor_group is not None:
                _all_reduce_group(grad, op=torch.distributed.ReduceOp.SUM, group=tensor_group)
            _distributed_all_reduce(grad, op=torch.distributed.ReduceOp.AVG)
            continue

        stacked = torch.stack([param.grad for param in bucket], dim=0)
        if tensor_group is not None:
            _all_reduce_group(stacked, op=torch.distributed.ReduceOp.SUM, group=tensor_group)
        _distributed_all_reduce(stacked, op=torch.distributed.ReduceOp.AVG)
        _foreach_copy_slices_([param.grad for param in bucket], stacked)

    # A fixed all-parameter roster keeps DP collectives aligned, but a
    # parameter unused on every replica must retain autograd's grad=None
    # semantics so Adam cannot advance it through stale momentum.
    for param, is_active in zip(params, globally_active):
        if not is_active:
            param.grad = None


def _reduce_optimizer_activity(params, side_entries, tensor_group=None):
    """Reduce the optimizer-boundary control plane in one compact pass.

    Ordinary/deferred parameter presence and raw threshold-side participation
    share one fixed activity vector.  This avoids separate status, ordinary
    presence, and threshold presence collectives (and their host syncs) while
    leaving the full gradient and raw-side reductions unchanged.
    """
    params = list(params)
    side_entries = list(side_entries)
    records = [
        ("parameter", index, param.device, param.grad is not None)
        for index, param in enumerate(params)
    ]
    records.extend(
        ("side", id(module), left.device, bool(is_pending))
        for module, left, _right, is_pending in side_entries
    )

    reduced_param_activity = [False] * len(params)
    reduced_side_activity = {}
    activity_buckets = {}
    for record_index, record in enumerate(records):
        activity_buckets.setdefault(record[2], []).append(record_index)

    for activity_device, indices in activity_buckets.items():
        activity = torch.tensor(
            [1 if records[index][3] else 0 for index in indices],
            device=activity_device,
            dtype=torch.int32,
        )
        if tensor_group is not None:
            _all_reduce_group(
                activity,
                op=torch.distributed.ReduceOp.SUM,
                group=tensor_group,
            )
        _distributed_all_reduce(activity, op=torch.distributed.ReduceOp.SUM)
        # One D2H synchronization per device publishes every control decision.
        reduced_flags = activity.gt(0).detach().cpu().tolist()
        for record_index, flag in zip(indices, reduced_flags):
            kind, key, _record_device, _local_flag = records[record_index]
            if kind == "parameter":
                reduced_param_activity[key] = bool(flag)
            else:
                reduced_side_activity[key] = bool(flag)

    return reduced_param_activity, reduced_side_activity


last_ce_weighted_loss_value = None
last_reconstruction_element_count_value = None
local_training_input_tokens_processed = int(
    resume_training_state.get("local_training_input_tokens_processed", 0)
    if isinstance(resume_training_state, dict)
    else 0
)
local_training_supervised_tokens_processed = int(
    resume_training_state.get("local_training_supervised_tokens_processed", 0)
    if isinstance(resume_training_state, dict)
    else 0
)
prior_training_wall_seconds = float(
    resume_training_state.get("cumulative_training_wall_seconds", 0.0)
    if isinstance(resume_training_state, dict)
    else 0.0
)


def _run_reconstruction_training_forward(
    model_obj,
    *,
    input_ids,
    attention_mask,
    position_ids,
    token_mask,
    accum_steps_val,
):
    """Run hard-QDQ K/V reconstruction without an LM-head objective.

    Quantizer capture occurs at the deployed locations: K after RoPE and V
    immediately after v_proj.  Each layer detaches its activation before local
    QDQ and detaches the returned value, so gradients are strictly local table
    reconstruction gradients rather than a hidden NLL surrogate.
    """
    for module_iter in quantizer_modules:
        module_iter.enable_reconstruction_capture(token_mask=token_mask)

    try:
        autocast_enabled = input_ids.is_cuda
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            if _pipeline_enabled():
                _pipeline_local_forward(
                    model_obj,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    enable_backward=False,
                )
            else:
                _model_for_parts, backbone, _lm_head = _get_causal_lm_parts(model_obj)
                _backbone_forward(
                    backbone,
                    input_ids=input_ids,
                    attention_mask=_normalize_attention_mask_for_fast_path(attention_mask),
                    position_ids=position_ids,
                    use_cache=False,
                )

        sse_terms = []
        element_terms = []
        missing_modules = 0
        for module_iter in quantizer_modules:
            sse, elements = module_iter.consume_reconstruction_stats()
            if sse is None or elements is None:
                missing_modules += 1
                continue
            sse_terms.append(sse)
            element_terms.append(elements)
        if missing_modules or not sse_terms:
            raise RuntimeError(
                "Reconstruction control did not observe every local K/V quantizer: "
                f"missing={missing_modules}, observed={len(sse_terms)}, "
                f"expected={len(quantizer_modules)}"
            )
        reconstruction_sse = torch.stack(sse_terms).sum()
        local_elements = torch.stack(element_terms).sum().clamp_min(1.0)
        global_elements = local_elements.detach().clone()
        global_sse = reconstruction_sse.detach().clone()
        if _distributed and int(_data_parallel_world_size) > 1:
            _distributed_all_reduce(global_elements, op=torch.distributed.ReduceOp.SUM)
            _distributed_all_reduce(global_sse, op=torch.distributed.ReduceOp.SUM)
            # Optimizer gradients are averaged across DP ranks below. Scaling
            # the local SSE this way yields the exact global element-weighted
            # mean even when valid-token counts differ across ranks.
            backward_loss = (
                reconstruction_sse
                * float(_data_parallel_world_size)
                / global_elements.clamp_min(1.0)
            )
        else:
            backward_loss = reconstruction_sse / local_elements
        reconstruction_loss = global_sse / global_elements.clamp_min(1.0)
        (backward_loss / float(accum_steps_val)).backward()
        return reconstruction_loss.detach(), global_elements.detach()
    finally:
        for module_iter in quantizer_modules:
            module_iter.disable_reconstruction_capture()
            module_iter.clear_reconstruction_stats()


def train_micro_step(model, batch, accum_steps_val, stage_name=None):
    if not getattr(model, "training", True):
        model.train()
    global last_ce_weighted_loss_value, last_reconstruction_element_count_value
    global local_training_input_tokens_processed, local_training_supervised_tokens_processed
    debug_step = _env_flag("DISCOVER_DEBUG_TRAIN_STEP")

    try:
        if debug_step:
            rank_debug_print("DISCOVER_DEBUG_TRAIN_STEP", "train_micro_step: move batch start")
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels_host = batch["labels"]
        local_supervised_tokens = int((labels_host != -100).sum().item())
        labels = labels_host.to(device, non_blocking=True)
        local_training_input_tokens_processed += int(input_ids.numel())
        local_training_supervised_tokens_processed += local_supervised_tokens
        raw_token_mask = batch.get("attention_mask")
        if torch.is_tensor(raw_token_mask):
            reconstruction_token_mask = raw_token_mask.to(device, non_blocking=True)
        else:
            reconstruction_token_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        position_ids = (
            batch["position_ids"].to(device, non_blocking=True) if "position_ids" in batch else None
        )
        attention_mask_all_ones = batch.get("attention_mask_all_ones", False)
        if torch.is_tensor(attention_mask_all_ones):
            attention_mask_all_ones = bool(attention_mask_all_ones.item())
        else:
            attention_mask_all_ones = bool(attention_mask_all_ones)
        attention_mask = (
            None
            if attention_mask_all_ones and position_ids is None
            else (
                batch["attention_mask"].to(device, non_blocking=True)
                if "attention_mask" in batch
                else None
            )
        )
        if position_ids is not None:
            attention_mask = build_doc_attn_mask(position_ids, dtype=torch.bfloat16)
        if debug_step:
            rank_debug_print(
                "DISCOVER_DEBUG_TRAIN_STEP",
                "train_micro_step: move batch done",
                f"input_shape={tuple(input_ids.shape)}",
                f"labels_shape={tuple(labels.shape)}",
            )
    except Exception as e:
        print(f"Error: Could not move batch data to device {device}: {e}")
        if _distributed:
            raise
        return float("nan")

    try:
        # AMP: forward in bf16 for ~2x speedup; quantizer layers
        # internally cast to fp32 so numeric behaviour is preserved.
        if debug_step:
            rank_debug_print("DISCOVER_DEBUG_TRAIN_STEP", "train_micro_step: forward start")
        if experiment_control["training_objective"] == "kv_reconstruction_mse":
            reconstruction_loss, reconstruction_elements = _run_reconstruction_training_forward(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                token_mask=reconstruction_token_mask,
                accum_steps_val=accum_steps_val,
            )
            if not torch.isfinite(reconstruction_loss).all():
                raise RuntimeError("non-finite K/V reconstruction loss")
            last_reconstruction_element_count_value = reconstruction_elements.detach()
            return TrainStepResult(reconstruction_loss, ce_weighted_loss=None, valid=True)

        run_result = _run_loss_forward(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            use_cache=False,
            backward_scale=float(args.ce_loss_weight) / float(accum_steps_val),
            return_hidden_grad=True,
        )
        if len(run_result) == 4:
            loss, target_tokens, hidden_states, hidden_grad = run_result
            surrogate_loss = torch.sum(hidden_states * hidden_grad.detach())
        elif len(run_result) == 3:
            loss, target_tokens, surrogate_loss = run_result
            if surrogate_loss is None:
                if not _pipeline_enabled():
                    raise RuntimeError(
                        "Surrogate loss is missing from _run_loss_forward return path."
                    )
        else:
            raise RuntimeError(
                f"Unexpected _run_loss_forward return signature: {len(run_result)} values."
            )
        if debug_step:
            rank_debug_print("DISCOVER_DEBUG_TRAIN_STEP", "train_micro_step: forward done")

        if loss is None or target_tokens is None:
            msg = "invalid loss in training step"
            print(f"Warning: {msg}")
            if _distributed:
                raise RuntimeError(msg)
            return TrainStepResult(float("nan"), valid=False)

        if STRICT_TRAIN_LOSS_CHECKS:
            invalid_loss = torch.isnan(loss.detach()) | torch.isinf(loss.detach())
            if invalid_loss.item():
                msg = "invalid loss in training step"
                print(f"Warning: {msg}")
                if _distributed:
                    raise RuntimeError(msg)
                return TrainStepResult(float("nan"), valid=False)
            if target_tokens.detach().item() <= 0:
                msg = "no valid targets in micro-batch"
                print(f"Warning: {msg}")
                if _distributed:
                    raise RuntimeError(msg)
                return TrainStepResult(float("nan"), valid=False)

        if debug_step:
            rank_debug_print("DISCOVER_DEBUG_TRAIN_STEP", "train_micro_step: backward start")
        backward_loss = surrogate_loss
        if not _pipeline_enabled():
            backward_loss.backward()
        if debug_step:
            rank_debug_print("DISCOVER_DEBUG_TRAIN_STEP", "train_micro_step: backward done")

        loss_for_metrics = loss.detach()
        ce_weighted_loss = loss_for_metrics * float(args.ce_loss_weight)
        return TrainStepResult(loss_for_metrics, ce_weighted_loss=ce_weighted_loss, valid=True)

    except Exception as e:
        if _is_oom_exception(e):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError("Training step OOM. Lower --per_gpu_batch_size.") from e
        print(f"Error in training step: {str(e)}")
        if _distributed:
            raise
        raise RuntimeError(
            "Training step failed before the optimizer boundary; aborting so "
            "partial parameter gradients and threshold side sums cannot leak "
            "into the next accumulation window."
        ) from e


def apply_optimizer_step(
    stepper,
    optimizer_step_params_for_step,
    quantizer_modules_for_step,
    grad_scale=1.0,
):
    all_step_params = list(optimizer_step_params_for_step)
    local_pending_sides = has_pending_threshold_side_sums(quantizer_modules_for_step)
    direct_threshold_param_ids = set()
    for module_iter in quantizer_modules_for_step:
        direct_parameters = getattr(module_iter, "deferred_threshold_direct_parameters", None)
        if not callable(direct_parameters):
            continue
        direct_threshold_param_ids.update(id(param) for param in direct_parameters())

    # Ordinary gradients are reduced first.  Direct threshold parameters are
    # excluded because their complete gradient is materialized below; affine
    # endpoints stay here because codepoint gradients also reach them.
    ordinary_params = [
        param for param in all_step_params if id(param) not in direct_threshold_param_ids
    ]

    # The real threshold gradient does not exist until the raw side sums have
    # crossed the optimizer boundary.  Always clear those sums on every exit,
    # including no-grad and non-finite skip paths.
    try:
        global_pending_sides = local_pending_sides
        global_param_activity = [param.grad is not None for param in all_step_params]
        global_side_activity = None
        if _distributed:
            tensor_group = _tensor_parallel_reduce_group()
            side_activity_entries = threshold_side_activity_roster(
                quantizer_modules_for_step,
                include_all_enabled=True,
            )
            global_param_activity, global_side_activity = _reduce_optimizer_activity(
                all_step_params,
                side_activity_entries,
                tensor_group=tensor_group,
            )
            global_pending_sides = any(global_side_activity.values())
            global_has_grad = any(global_param_activity) or global_pending_sides
            if int(args.pipeline_model_parallel_size) <= 1 and (
                not all_step_params or (not ASSUME_QUANTIZER_GRADS and not global_has_grad)
            ):
                return False
        elif not all_step_params:
            return False

        if _distributed and ordinary_params:
            global_param_activity_by_id = {
                id(param): is_active
                for param, is_active in zip(all_step_params, global_param_activity)
            }
            _bucketed_reduce_quantizer_grads(
                ordinary_params,
                tensor_group=_tensor_parallel_reduce_group(),
                globally_active=[
                    global_param_activity_by_id[id(param)] for param in ordinary_params
                ],
            )

        if global_pending_sides:
            tensor_group = _tensor_parallel_reduce_group() if _distributed else None

            def reduce_tensor_sides(packed):
                if tensor_group is not None:
                    _all_reduce_group(
                        packed,
                        op=torch.distributed.ReduceOp.SUM,
                        group=tensor_group,
                    )

            def reduce_data_sides(packed):
                if _distributed:
                    _distributed_all_reduce(packed, op=torch.distributed.ReduceOp.AVG)

            finalize_deferred_threshold_grads(
                quantizer_modules_for_step,
                mode=args.threshold_grad_mode,
                tensor_reduce=reduce_tensor_sides if tensor_group is not None else None,
                data_reduce=reduce_data_sides if _distributed else None,
                global_activity=global_side_activity,
                include_all_enabled=_distributed,
            )

        params_with_grads = [param for param in all_step_params if param.grad is not None]
        pipeline_parallel = _distributed and int(args.pipeline_model_parallel_size) > 1
        if pipeline_parallel:
            if _pipeline_model_parallel_group is None:
                raise RuntimeError(
                    "Pipeline-parallel optimizer readiness requires a pipeline group."
                )
            # A stage without target quantizers or without a gradient in this
            # window is a neutral participant.  All stages update when any
            # stage has real work, and all stages skip only when none do.  This
            # keeps collective order and completed-step counts identical
            # without materializing zero gradients on the empty stage.
            pipeline_has_grad = torch.tensor(
                [1 if params_with_grads else 0],
                device=device,
                dtype=torch.int32,
            )
            _all_reduce_group(
                pipeline_has_grad,
                op=torch.distributed.ReduceOp.MAX,
                group=_pipeline_model_parallel_group,
            )
            if pipeline_has_grad.item() == 0:
                return False
        elif not params_with_grads:
            return False

        grads_for_step = [param.grad for param in params_with_grads]
        if CHECK_OPTIMIZER_GRADS:
            nonfinite_grad = torch.zeros(
                (),
                device=grads_for_step[0].device if grads_for_step else device,
                dtype=torch.float32,
            )
            used_foreach_check = False
            if grads_for_step and hasattr(torch, "_amp_foreach_non_finite_check_and_unscale_"):
                grad_multiplier = torch.full(
                    (),
                    float(grad_scale),
                    device=grads_for_step[0].device,
                    dtype=torch.float32,
                )
                try:
                    torch._amp_foreach_non_finite_check_and_unscale_(
                        grads_for_step,
                        nonfinite_grad,
                        grad_multiplier,
                    )
                    used_foreach_check = True
                except (RuntimeError, TypeError):
                    nonfinite_grad.zero_()
            if not used_foreach_check:
                if grad_scale != 1.0:
                    for grad in grads_for_step:
                        grad.mul_(grad_scale)
                for grad in grads_for_step:
                    nonfinite_grad.add_((~torch.isfinite(grad)).any().to(torch.float32))
            if _distributed:
                _distributed_all_reduce(nonfinite_grad, op=torch.distributed.ReduceOp.SUM)
                if pipeline_parallel:
                    if _pipeline_model_parallel_group is None:
                        raise RuntimeError(
                            "Pipeline-parallel non-finite sync requires a pipeline group."
                        )
                    _all_reduce_group(
                        nonfinite_grad,
                        op=torch.distributed.ReduceOp.SUM,
                        group=_pipeline_model_parallel_group,
                    )
            if nonfinite_grad.item() > 0:
                print("Warning: Detected NaN or Inf gradients, skipping optimizer step")
                return False
        elif grad_scale != 1.0:
            for grad in grads_for_step:
                grad.mul_(grad_scale)

        if params_with_grads:
            stepper.step()
            project_quantizer_modules(quantizer_modules_for_step)
        return True
    finally:
        clear_threshold_side_sums_(quantizer_modules_for_step)


# Runtime adapter: bind concrete training functions to the backend wrappers.
train_subsystem.configure(
    train_micro_step_fn=train_micro_step,
    apply_optimizer_step_fn=apply_optimizer_step,
)


def _train_step_loss_tensor(step_result):
    value = getattr(step_result, "loss", step_result)
    if isinstance(value, torch.Tensor):
        return value.detach()
    if value is None:
        return None
    try:
        return torch.tensor(float(value), device=device, dtype=torch.float32)
    except (TypeError, ValueError):
        return None


def _scalar_value(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return float("nan")
        return float(value.detach().float().item())
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _should_sync_train_loss(batch_idx, total_microbatches, micro_step):
    if TRAIN_LOSS_SYNC_INTERVAL == 1:
        return True
    if batch_idx == 0:
        return True
    if total_microbatches and batch_idx + 1 >= total_microbatches:
        return True
    if (
        TRAIN_LOSS_SYNC_INTERVAL > 1
        and micro_step > 0
        and micro_step % TRAIN_LOSS_SYNC_INTERVAL == 0
    ):
        return True
    return LOGGING_INTERVAL > 0 and micro_step > 0 and micro_step % LOGGING_INTERVAL == 0


def distributed_weighted_mean(total_value, total_count):
    if torch.is_tensor(total_value) or torch.is_tensor(total_count):
        total_tensor = (
            total_value.detach().to(device=device, dtype=torch.float64)
            if torch.is_tensor(total_value)
            else torch.tensor(float(total_value), device=device, dtype=torch.float64)
        )
        count_tensor = (
            total_count.detach().to(device=device, dtype=torch.float64)
            if torch.is_tensor(total_count)
            else torch.tensor(float(total_count), device=device, dtype=torch.float64)
        )
        stats = torch.stack([total_tensor.reshape(()), count_tensor.reshape(())])
    else:
        stats = torch.tensor(
            [float(total_value), float(total_count)],
            device=device,
            dtype=torch.float64,
        )
    if _distributed:
        _distributed_all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
    if stats[1].item() <= 0:
        return float("nan")
    return (stats[0] / stats[1]).item()


def stage_uses_gradient_checkpointing(seqlen):
    return (
        GRADIENT_CHECKPOINTING_FROM_SEQLEN > 0 and int(seqlen) >= GRADIENT_CHECKPOINTING_FROM_SEQLEN
    )


def run_periodic_eval(eval_step, label):
    rank0_print(f"\n{label} full held-out validation")
    try:
        return eval_subsystem.run_stages(
            model,
            eval_stages,
            "eval",
            "Val",
            writer_obj=writer,
            loss_tag="Loss/Val_step",
            ppl_tag="PPL/Val_step",
            step=eval_step,
        )
    finally:
        model.train()


# --- Training loop ---
best_val_ppl = None
best_val_loss = None
best_val_epoch = None
best_val_pass = None
best_quant_config_path = None
validation_history = []
early_stop_triggered = False
training_stop_reason = None
final_val_ppl = None
final_val_loss = None
final_eval_ppl = None
final_eval_loss = None
final_test_ppl = None
final_test_loss = None
final_train_loss = None
final_config_ppl = None
final_config_loss = None
epochs_completed = 0

# --- Token-budget curriculum training loop ---
num_curriculum_stages = len(training_stages)
total_training_units = total_epochs * num_curriculum_stages
unit_batch_counts = [
    len(training_stages[unit_idx % num_curriculum_stages]["train_dataloader"])
    for unit_idx in range(total_training_units)
]


def _stage_optimizer_step_count(stage):
    microbatch_count = len(stage["train_dataloader"])
    accum_steps = max(1, int(stage["accum_steps"]))
    return int(stage.get("stream_steps_per_pass") or _ceil_div(microbatch_count, accum_steps))


unit_optimizer_step_counts = [
    _stage_optimizer_step_count(training_stages[unit_idx % num_curriculum_stages])
    for unit_idx in range(total_training_units)
]
lr_schedule_total_steps = sum(unit_optimizer_step_counts)
curriculum_scheduler = CurriculumScheduler(total_epochs, num_curriculum_stages, unit_batch_counts)


def _optimizer_lr_for_completed_steps(completed_steps):
    base_lr = float(args.lr)
    if args.lr_schedule == "constant":
        return base_lr

    min_lr = base_lr * float(args.lr_min_ratio)
    denom = max(1, int(lr_schedule_total_steps) - 1)
    progress = max(0.0, min(1.0, float(completed_steps) / float(denom)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def _set_optimizer_lr_for_completed_steps(completed_steps):
    global last_optimizer_lr
    lr_value = _optimizer_lr_for_completed_steps(completed_steps)
    for group in getattr(optimizer, "param_groups", []):
        group["lr"] = lr_value
    last_optimizer_lr = float(lr_value)
    return lr_value


def _locate_resume_position(completed_optimizer_steps):
    remaining_steps = max(0, int(completed_optimizer_steps))
    completed_microbatches = 0
    for unit_idx, unit_steps in enumerate(unit_optimizer_step_counts):
        stage = training_stages[unit_idx % num_curriculum_stages]
        unit_microbatches = len(stage["train_dataloader"])
        accum_steps = max(1, int(stage["accum_steps"]))
        if remaining_steps >= unit_steps:
            completed_microbatches += unit_microbatches
            remaining_steps -= unit_steps
            continue
        skip_steps_in_unit = remaining_steps
        completed_microbatches += min(unit_microbatches, skip_steps_in_unit * accum_steps)
        return unit_idx, skip_steps_in_unit, completed_microbatches
    return total_training_units, 0, completed_microbatches


start_unit, resume_steps_in_start_unit, resume_micro_step_from_schedule = _locate_resume_position(
    global_step
)
if resume_enabled and global_step > 0 and resume_state_global_micro_step is None:
    global_micro_step = int(resume_micro_step_from_schedule)

rank0_print("Starting training")
val_worse_streak = 0
epochs_completed = 0
if isinstance(resume_training_state, dict):
    best_val_ppl = resume_training_state.get("best_val_ppl")
    best_val_loss = resume_training_state.get("best_val_loss")
    best_val_epoch = resume_training_state.get("best_val_epoch")
    best_val_pass = resume_training_state.get("best_val_pass")
    best_quant_config_path = resume_training_state.get("best_quant_config_path")
    final_val_ppl = resume_training_state.get("final_val_ppl")
    final_val_loss = resume_training_state.get("final_val_loss")
    loaded_history = resume_training_state.get("validation_history", [])
    if isinstance(loaded_history, list):
        validation_history = [dict(entry) for entry in loaded_history if isinstance(entry, dict)]
    early_stop_triggered = bool(resume_training_state.get("early_stop_triggered", False))
    loaded_stop_reason = resume_training_state.get("training_stop_reason")
    if isinstance(loaded_stop_reason, str) and loaded_stop_reason:
        training_stop_reason = loaded_stop_reason
    try:
        val_worse_streak = int(resume_training_state.get("val_worse_streak", 0) or 0)
    except (TypeError, ValueError):
        val_worse_streak = 0
    try:
        epochs_completed = int(resume_training_state.get("epochs_completed", 0) or 0)
    except (TypeError, ValueError):
        epochs_completed = 0
    rank0_print("Resume: restored validation tracking from training state.")
if resume_enabled and global_step > 0:
    epochs_completed = max(epochs_completed, min(start_unit, total_training_units))
next_periodic_eval_step = EVAL_EVERY_STEPS if EVAL_EVERY_STEPS > 0 else None
while next_periodic_eval_step is not None and next_periodic_eval_step <= global_step:
    next_periodic_eval_step += EVAL_EVERY_STEPS
optimizer_update_count = 0
if resume_enabled and global_step > 0:
    optimizer_update_count = global_step
    rank0_print(f"Resume: continuing optimizer-step counter from {global_step}.")
    planned_steps = sum(unit_optimizer_step_counts)
    if start_unit >= curriculum_scheduler.total_training_units:
        rank0_print(
            f"Resume: checkpoint already covers {global_step}/{planned_steps} planned optimizer steps; "
            "no additional training units are scheduled."
        )
    elif resume_steps_in_start_unit > 0:
        rank0_print(
            f"Resume: starting in unit {start_unit + 1}/{total_training_units} after "
            f"{resume_steps_in_start_unit} completed optimizer steps in that unit."
        )
    else:
        rank0_print(f"Resume: starting at unit {start_unit + 1}/{total_training_units}.")

current_optimizer_lr = _set_optimizer_lr_for_completed_steps(optimizer_update_count)
rank0_print(
    "LR schedule: "
    f"{args.lr_schedule}, base_lr={float(args.lr):g}, "
    f"min_lr={float(args.lr) * float(args.lr_min_ratio):g}, "
    f"total_optimizer_steps={int(lr_schedule_total_steps)}, "
    f"next_lr={current_optimizer_lr:g}"
)


def _finite_metric(value):
    return value is not None and math.isfinite(value)


def update_validation_tracking(
    val_metrics,
    config_path,
    epoch_marker,
    pass_marker,
    *,
    source,
):
    global best_val_ppl, best_val_loss, best_val_epoch, best_val_pass
    global best_quant_config_path, final_val_ppl, final_val_loss, val_worse_streak
    global early_stop_triggered, training_stop_reason

    final_val_loss = val_metrics["loss"]
    final_val_ppl = val_metrics["ppl"]
    best_before = best_val_ppl
    streak_before = val_worse_streak
    decision = update_validation_early_stopping(
        current_value=final_val_ppl,
        best_value=best_val_ppl,
        worse_streak=val_worse_streak,
        patience=args.early_stop_patience,
    )
    best_val_ppl = decision.best_value
    val_worse_streak = decision.worse_streak
    if decision.is_new_best:
        best_val_loss = final_val_loss
        best_val_epoch = epoch_marker
        best_val_pass = pass_marker
        best_quant_config_path = config_path

    stopping_criterion_met = bool(decision.should_stop)
    effective_should_stop = bool(
        stopping_criterion_met and int(global_step) < int(lr_schedule_total_steps or 0)
    )
    history_entry = {
        "source": str(source),
        "optimizer_step": int(global_step),
        "marker": int(epoch_marker),
        "pass_marker": int(pass_marker),
        "loss": (float(final_val_loss) if _finite_metric(final_val_loss) else None),
        "ppl": float(final_val_ppl) if _finite_metric(final_val_ppl) else None,
        "target_tokens": int(val_metrics.get("target_tokens", 0) or 0),
        "comparison": decision.comparison,
        "best_ppl_before": (float(best_before) if _finite_metric(best_before) else None),
        "best_ppl_after": (
            float(decision.best_value) if _finite_metric(decision.best_value) else None
        ),
        "worse_streak_before": int(streak_before),
        "worse_streak_after": int(decision.worse_streak),
        "config_path": config_path,
        "is_new_best": bool(decision.is_new_best),
        "stopping_criterion_met": stopping_criterion_met,
        "should_stop": effective_should_stop,
    }
    validation_history.append(history_entry)
    if effective_should_stop:
        early_stop_triggered = True
        training_stop_reason = "early_stopping"
    return effective_should_stop


def _advance_periodic_eval_schedule():
    global next_periodic_eval_step
    while next_periodic_eval_step is not None and global_step >= next_periodic_eval_step:
        next_periodic_eval_step += EVAL_EVERY_STEPS


def maybe_run_periodic_validation():
    if next_periodic_eval_step is None or global_step < next_periodic_eval_step:
        return False

    eval_step = int(next_periodic_eval_step)
    save_path = save_distributed_quant_config(f"step_{eval_step}")
    save_path = broadcast_main_path(save_path, f"step_{eval_step}_config_path")
    dist_barrier(f"step_{eval_step}_config_saved")
    val_metrics = run_periodic_eval(
        eval_step,
        f"Step {eval_step} (optimizer update {optimizer_update_count})",
    )
    should_stop = update_validation_tracking(
        val_metrics,
        save_path,
        eval_step,
        optimizer_update_count,
        source="periodic_steps",
    )
    state_path = save_training_state(eval_step, save_path)
    if state_path and is_main():
        rank0_print(f"Saved training state to: {state_path}")
    dist_barrier(f"step_{eval_step}_state_saved")
    _advance_periodic_eval_schedule()
    if should_stop:
        rank0_print(
            "Early stopping: periodic validation PPL exceeded best "
            f"for {val_worse_streak} consecutive eval intervals."
        )
    return should_stop


if _finite_metric(initial_periodic_val_metrics.get("ppl")):
    if isinstance(resume_training_state, dict):
        rank0_print("Resume: keeping validation tracking from the loaded training state.")
    else:
        update_validation_tracking(
            initial_periodic_val_metrics,
            resume_quant_config_path if resume_enabled else initial_quant_config_path,
            global_step if resume_enabled else 0,
            optimizer_update_count if resume_enabled else 0,
            source="initial",
        )
        rank0_print(
            "Full validation baseline set from Val_init: "
            f"loss={initial_periodic_val_metrics['loss']:.6f}, "
            f"ppl={initial_periodic_val_metrics['ppl']:.6f}, "
            f"target_tokens={initial_periodic_val_metrics['target_tokens']:,}"
        )

training_wall_start_time = time.perf_counter()

resume_early_stop_complete = bool(
    isinstance(resume_training_state, dict)
    and args.early_stop_patience > 0
    and val_worse_streak >= args.early_stop_patience
    and int(global_step) < int(lr_schedule_total_steps or 0)
)
if resume_early_stop_complete:
    early_stop_triggered = True
    training_stop_reason = "early_stopping"
    rank0_print(
        "Resume: checkpoint already satisfies the early-stopping criterion; "
        "skipping additional optimizer updates."
    )

training_unit_indices = (
    () if resume_early_stop_complete else range(start_unit, total_training_units)
)
for unit_idx in training_unit_indices:
    epoch = unit_idx // num_curriculum_stages
    stage = training_stages[unit_idx % num_curriculum_stages]
    stage_name = stage["name"]
    stage_seqlen = int(stage["seqlen"])
    stage_global_microbatch = int(stage["global_microbatch"])
    stage_local_microbatch = int(stage["local_microbatch"])
    stage_accum_steps = int(stage["accum_steps"])
    train_dataloader = data_subsystem.train_loader(stage)
    train_sampler = stage["train_sampler"]
    model.seqlen = stage_seqlen
    set_model_gradient_checkpointing(model, stage_uses_gradient_checkpointing(stage_seqlen))
    if train_sampler is not None:
        train_sampler.set_epoch(unit_idx)
    total_loss_epoch = torch.zeros((), device=device, dtype=torch.float64)
    num_valid_batches_epoch = torch.zeros((), device=device, dtype=torch.float64)
    train_total_microbatches = len(train_dataloader)
    train_total_steps = int(
        stage.get("stream_steps_per_pass") or _ceil_div(train_total_microbatches, stage_accum_steps)
    )
    resume_skip_optimizer_steps = (
        int(resume_steps_in_start_unit)
        if unit_idx == start_unit and resume_enabled and global_step > 0
        else 0
    )
    resume_skip_microbatches = min(
        train_total_microbatches,
        resume_skip_optimizer_steps * stage_accum_steps,
    )
    if resume_skip_optimizer_steps > 0:
        rank0_print(
            f"Resume: skipping {resume_skip_optimizer_steps} optimizer steps "
            f"({resume_skip_microbatches} microbatches) already covered in {stage_name}."
        )
    pbar = tqdm(
        total=train_total_steps,
        initial=min(resume_skip_optimizer_steps, train_total_steps),
        desc=(
            f"Pass {epoch + 1}/{total_epochs} "
            f"Stage {stage['stage_idx'] + 1}/{num_curriculum_stages} "
            f"{stage_name} sl{stage_seqlen} gmb{stage_global_microbatch} "
            f"lmb{stage_local_microbatch} acc{stage_accum_steps} steps"
        ),
        disable=not is_main(),
    )
    accum_count = 0
    _zero_stepper_grad(_backend_stepper, quantizer_modules)
    train_progress_interval = max(0, int(os.getenv("DISCOVER_TRAIN_PROGRESS_INTERVAL", "20")))
    train_batch_status_logging = _env_flag("DISCOVER_TRAIN_BATCH_STATUS")
    if train_batch_status_logging and _distributed:
        train_progress_interval = max(train_progress_interval, 1)

    for batch_idx, batch in enumerate(train_dataloader):
        if resume_skip_microbatches > 0 and batch_idx < resume_skip_microbatches:
            continue
        if train_batch_status_logging and (
            batch_idx == 0
            or (train_progress_interval > 0 and (batch_idx + 1) % train_progress_interval == 0)
            or batch_idx + 1 == train_total_microbatches
        ):
            rank0_print(
                f"Train unit {unit_idx + 1}/{total_training_units} stage {stage_name}: "
                f"microbatch {batch_idx + 1}/{train_total_microbatches} starting"
            )
        step_result = train_subsystem.train_step(
            model, batch, stage_accum_steps, stage_name=stage_name
        )
        if train_batch_status_logging and (
            batch_idx == 0
            or (train_progress_interval > 0 and (batch_idx + 1) % train_progress_interval == 0)
            or batch_idx + 1 == train_total_microbatches
        ):
            rank0_print(
                f"Train unit {unit_idx + 1}/{total_training_units} stage {stage_name}: "
                f"microbatch {batch_idx + 1}/{train_total_microbatches} finished"
            )
        loss_is_valid = bool(getattr(step_result, "valid", True))
        loss_tensor = _train_step_loss_tensor(step_result)

        if loss_is_valid:
            if loss_tensor is not None:
                total_loss_epoch.add_(loss_tensor.to(device=device, dtype=torch.float64))
            num_valid_batches_epoch.add_(1.0)
            accum_count += 1
        global_micro_step += 1

        loss_value = None
        ce_weighted_value = None
        should_sync_loss = loss_is_valid and _should_sync_train_loss(
            batch_idx,
            train_total_microbatches,
            global_micro_step,
        )
        if should_sync_loss:
            loss_value = _scalar_value(loss_tensor)
            ce_weighted_value = _scalar_value(getattr(step_result, "ce_weighted_loss", None))
            if ce_weighted_value is not None and math.isfinite(ce_weighted_value):
                last_ce_weighted_loss_value = ce_weighted_value
            if writer is not None and loss_value is not None and math.isfinite(loss_value):
                writer.add_scalar("Loss/Train_batch", loss_value, global_micro_step)
                if ce_weighted_value is not None and math.isfinite(ce_weighted_value):
                    writer.add_scalar(
                        "Loss/Train_CE_weighted_batch", ce_weighted_value, global_micro_step
                    )

            if is_main():
                local_count = _scalar_value(num_valid_batches_epoch)
                if local_count and local_count > 0:
                    avg_loss_value = _scalar_value(
                        total_loss_epoch / num_valid_batches_epoch.clamp_min(1.0)
                    )
                else:
                    avg_loss_value = float("nan")
                postfix = {
                    "Loss": f"{avg_loss_value:.4f}" if math.isfinite(avg_loss_value) else "nan"
                }
                if ce_weighted_value is not None and args.ce_loss_weight != 1.0:
                    postfix["CEw"] = (
                        f"{ce_weighted_value:.4f}" if math.isfinite(ce_weighted_value) else "nan"
                    )
                postfix["lr"] = f"{current_optimizer_lr:.2e}"
                pbar.set_postfix(postfix)

        if is_main() and global_micro_step % LOGGING_INTERVAL == 0 and loss_is_valid:
            if loss_value is None:
                loss_value = _scalar_value(loss_tensor)
            if ce_weighted_value is None:
                ce_weighted_value = _scalar_value(getattr(step_result, "ce_weighted_loss", None))
                if ce_weighted_value is not None and math.isfinite(ce_weighted_value):
                    last_ce_weighted_loss_value = ce_weighted_value
            if loss_value is None:
                loss_value = float("nan")
            if experiment_control["training_objective"] == "kv_reconstruction_mse":
                quant_backward_label = "local_reconstruction"
            elif (
                experiment_control["quantizer_parameterization"] == "full_precision_bucket_residual"
            ):
                quant_backward_label = "hard_bucket_residual"
            elif (
                experiment_control["quantizer_parameterization"]
                == "full_precision_chebyshev_residual"
            ):
                quant_backward_label = "continuous_adapter"
            else:
                quant_backward_label = "ste"
            log_parts = [
                f"\nUnit {unit_idx + 1}/{total_training_units}, Micro-step {global_micro_step}, "
                f"Optimizer step {global_step}, "
                f"Stage {stage_name}, global_microbatch={stage_global_microbatch}, "
                f"local_microbatch/GPU={stage_local_microbatch}, "
                f"accum_steps={stage_accum_steps}, Loss: {loss_value:.6f}, "
                f"lr={current_optimizer_lr:.6g}, "
                f"quant_backward={quant_backward_label}"
            ]
            if ce_weighted_value is not None and args.ce_loss_weight != 1.0:
                log_parts.append(f"ce_weighted={ce_weighted_value:.6f}")
            print(", ".join(log_parts))
            if writer is not None:
                writer.add_scalar("LR/Train", current_optimizer_lr, global_step)

        if accum_count == stage_accum_steps:
            current_optimizer_lr = _set_optimizer_lr_for_completed_steps(optimizer_update_count)
            stepped = train_subsystem.optimizer_step(
                grad_scale=1.0,
            )
            _zero_stepper_grad(_backend_stepper, quantizer_modules)
            accum_count = 0
            if stepped:
                if writer is not None:
                    writer.add_scalar(
                        "LR/Optimizer_step",
                        current_optimizer_lr,
                        optimizer_update_count + 1,
                    )
                optimizer_update_count += 1
                global_step = optimizer_update_count
                current_optimizer_lr = _set_optimizer_lr_for_completed_steps(optimizer_update_count)
                pbar.update(1)

                if maybe_run_periodic_validation():
                    break
            else:
                rank0_print(
                    "Warning: optimizer update was skipped; completed-step count, "
                    "LR schedule, validation cadence, and checkpoint step were not advanced."
                )

    if early_stop_triggered:
        pbar.close()
        break

    if accum_count > 0:
        # Compensate for partial accumulation so gradients reflect actual count.
        grad_scale = stage_accum_steps / accum_count
        current_optimizer_lr = _set_optimizer_lr_for_completed_steps(optimizer_update_count)
        stepped = train_subsystem.optimizer_step(
            grad_scale=grad_scale,
        )
        _zero_stepper_grad(_backend_stepper, quantizer_modules)
        accum_count = 0
        if stepped:
            if writer is not None:
                writer.add_scalar(
                    "LR/Optimizer_step",
                    current_optimizer_lr,
                    optimizer_update_count + 1,
                )
            optimizer_update_count += 1
            global_step = optimizer_update_count
            current_optimizer_lr = _set_optimizer_lr_for_completed_steps(optimizer_update_count)
            pbar.update(1)
            if maybe_run_periodic_validation():
                pbar.close()
                break
        else:
            rank0_print(
                "Warning: partial optimizer update was skipped; completed-step "
                "count and checkpoint step were not advanced."
            )
    pbar.close()

    avg_epoch_loss = distributed_weighted_mean(total_loss_epoch, num_valid_batches_epoch)
    if math.isfinite(avg_epoch_loss):
        rank0_print(
            f"\nUnit {unit_idx + 1}/{total_training_units} finished "
            f"({stage_name}, seqlen={stage_seqlen}), Average Loss: {avg_epoch_loss:.6f}"
        )
        if writer is not None:
            writer.add_scalar("Loss/Train_unit", avg_epoch_loss, unit_idx)
            writer.add_scalar(f"Loss/Train_stage/{stage_name}", avg_epoch_loss, unit_idx)
        final_train_loss = avg_epoch_loss
    else:
        rank0_print(
            f"\nUnit {unit_idx + 1} did not successfully process any valid training batches"
        )

    epochs_completed = unit_idx + 1

    is_pass_boundary = curriculum_scheduler.is_pass_boundary(unit_idx)
    should_run_eval = EVAL_EVERY_STAGE or (EVAL_EVERY_STEPS <= 0 and is_pass_boundary)
    if should_run_eval:
        save_path = save_distributed_quant_config(f"step_{global_step}")
        save_path = broadcast_main_path(save_path, f"step_{global_step}_config_path")
        dist_barrier(f"step_{global_step}_config_saved")
        pass_idx = curriculum_scheduler.pass_index(unit_idx)
        eval_step = unit_idx + 1 if EVAL_EVERY_STAGE else pass_idx
        boundary_label = f"Stage unit {unit_idx + 1}" if EVAL_EVERY_STAGE else f"Pass {pass_idx}"
        val_loss_tag = "Loss/Val_stage_unit" if EVAL_EVERY_STAGE else "Loss/Val_pass"
        val_ppl_tag = "PPL/Val_stage_unit" if EVAL_EVERY_STAGE else "PPL/Val_pass"
        eval_loss_tag = (
            "Loss/Val_holdout_stage_unit" if EVAL_EVERY_STAGE else "Loss/Val_holdout_pass"
        )
        eval_ppl_tag = "PPL/Val_holdout_stage_unit" if EVAL_EVERY_STAGE else "PPL/Val_holdout_pass"
        if run_train_validation:
            rank0_print(f"\n{boundary_label} curriculum validation")
            val_metrics = eval_subsystem.run_stages(
                model,
                training_stages,
                "val",
                "Val",
                writer_obj=writer,
                loss_tag=val_loss_tag,
                ppl_tag=val_ppl_tag,
                step=eval_step,
            )
            final_val_loss = val_metrics["loss"]
            final_val_ppl = val_metrics["ppl"]
        else:
            final_val_loss = None
            final_val_ppl = None

        rank0_print(f"\n{boundary_label} held-out validation")
        eval_metrics = eval_subsystem.run_stages(
            model,
            eval_stages,
            "eval",
            "Val",
            writer_obj=writer,
            loss_tag=eval_loss_tag,
            ppl_tag=eval_ppl_tag,
            step=eval_step,
        )
        final_eval_loss = eval_metrics["loss"]
        final_eval_ppl = eval_metrics["ppl"]
        final_val_loss = final_eval_loss
        final_val_ppl = final_eval_ppl

        should_stop = update_validation_tracking(
            eval_metrics,
            save_path,
            unit_idx + 1,
            pass_idx,
            source="stage" if EVAL_EVERY_STAGE else "pass",
        )
        state_path = save_training_state(global_step, save_path)
        if state_path and is_main():
            rank0_print(f"Saved training state to: {state_path}")
        dist_barrier(f"step_{global_step}_state_saved")
        if should_stop:
            patience_scope = "stage-level" if EVAL_EVERY_STAGE else "pass-level"
            rank0_print(
                f"Early stopping: {patience_scope} validation PPL exceeded best "
                f"for {val_worse_streak} consecutive eval intervals."
            )
            break
        if not _finite_metric(final_val_ppl):
            rank0_print(
                f"{boundary_label} validation produced invalid loss; skipping early stopping check."
            )

training_wall_end_time = time.perf_counter()
if training_stop_reason is None:
    training_stop_reason = (
        "max_steps"
        if int(global_step) >= int(lr_schedule_total_steps or 0)
        else "training_schedule_complete"
    )

# --- Training finished ---
final_config_path = None
if _distributed:
    best_path_payload = [best_quant_config_path if is_main() else None]
    torch.distributed.broadcast_object_list(best_path_payload, src=0)
    best_quant_config_path = best_path_payload[0]

if best_quant_config_path is not None:
    copy_failed = False
    if is_main():
        final_config_path = os.path.join(QUANT_CONFIG_DIR, "quant_config_final.pt")
        try:
            shutil.copyfile(best_quant_config_path, final_config_path)
        except Exception as e:
            print(f"Error saving best quantization config as final: {e}")
            final_config_path = None
            copy_failed = True
    if _distributed:
        copy_failed_payload = [copy_failed]
        torch.distributed.broadcast_object_list(copy_failed_payload, src=0)
        copy_failed = bool(copy_failed_payload[0])
    if copy_failed:
        raise RuntimeError(
            "Failed to copy the best-validation quantization checkpoint; "
            "refusing to substitute the stopping-step checkpoint."
        )
else:
    if args.early_stop_patience > 0 and bool(eval_stages):
        raise RuntimeError(
            "No finite validation checkpoint was selected while early stopping "
            "was enabled; refusing to save the last checkpoint as final."
        )
    final_config_path = save_distributed_quant_config("final")

if _distributed:
    path_payload = [final_config_path]
    torch.distributed.broadcast_object_list(path_payload, src=0)
    final_config_path = path_payload[0]
dist_barrier("final_config_ready")

final_training_state_path = save_training_state(global_step, final_config_path)
if final_training_state_path and is_main():
    rank0_print(f"Saved pre-final-eval training state to: {final_training_state_path}")
dist_barrier("pre_final_eval_training_state_ready")

final_eval_max_batches = max(0, int(args.final_eval_max_batches or 0))
if args.skip_final_eval:
    rank0_print("Skipping final held-out eval (--skip_final_eval).")
elif final_config_path:
    try:
        set_model_gradient_checkpointing(model, False)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        final_quant_config = load_quant_config(final_config_path)
        apply_quant_config(unwrap_model(model), final_quant_config, QuantizedLinearCls)
        if final_eval_max_batches > 0:
            rank0_print(f"Final eval is capped at {final_eval_max_batches} batch(es) per stage.")
        final_config_metrics = eval_subsystem.run_stages(
            model,
            test_stages,
            "eval",
            "Final eval",
            writer_obj=writer,
            loss_tag="Loss/Final_eval",
            ppl_tag="PPL/Final_eval",
            step=epochs_completed,
            max_batches_per_stage=final_eval_max_batches,
        )
        final_test_loss = final_config_metrics["loss"]
        final_test_ppl = final_config_metrics["ppl"]
        final_config_loss = final_test_loss
        final_config_ppl = final_test_ppl
    except Exception as e:
        print(f"Error evaluating final config eval-set perplexity: {e}", flush=True)
        final_test_loss = float("inf")
        final_test_ppl = float("inf")
        final_config_loss = float("inf")
        final_config_ppl = float("inf")

final_training_state_path = save_training_state(global_step, final_config_path)
if final_training_state_path and is_main():
    rank0_print(f"Saved final training state to: {final_training_state_path}")
dist_barrier("final_training_state_ready")

if writer is not None:
    writer.close()
metrics_parallel = "megatron" if _distributed else "single"
training_fingerprint_manifest = _fingerprint_manifest(training_stages)
validation_fingerprint_manifest = _fingerprint_manifest(eval_stages)
eval_fingerprint_manifest = _fingerprint_manifest(test_stages)
training_wall_seconds = _cumulative_training_wall_seconds()
current_run_wall_seconds = max(0.0, time.perf_counter() - _PROCESS_START_TIME)
training_gpu_hours = training_wall_seconds * int(_world_size) / 3600.0
current_run_gpu_hours = current_run_wall_seconds * int(_world_size) / 3600.0
global_training_input_tokens_processed = int(local_training_input_tokens_processed) * int(
    _data_parallel_world_size
)
global_training_supervised_tokens_processed = int(local_training_supervised_tokens_processed) * int(
    _data_parallel_world_size
)
dataset_revision_manifest = [
    {
        "path": source.get("path"),
        "name": source.get("name"),
        "split": source.get("split"),
        "revision": source.get("revision"),
        "revision_available": source.get("revision") is not None,
    }
    for source in DISCOVER_ACTIVE_SOURCES
]
_seen_control_param_ids = set()
control_trainable_parameter_count = 0
control_total_parameter_count = 0
for _control_module in quantizer_modules:
    for _control_param in _control_module.parameters(recurse=False):
        if id(_control_param) in _seen_control_param_ids:
            continue
        _seen_control_param_ids.add(id(_control_param))
        control_total_parameter_count += int(_control_param.numel())
        if _control_param.requires_grad:
            control_trainable_parameter_count += int(_control_param.numel())
physical_quant_table_count = sum(
    int(getattr(_control_module, "num_tables", 1)) for _control_module in quantizer_modules
)
learned_scalars_per_physical_table = (
    2
    if experiment_control["quantizer_parameterization"] == "uniform_affine_endpoints"
    else (
        ((2 * (1 << int(args.num_bits))) - 1)
        if experiment_control["quantizer_parameterization"]
        in {"full_precision_chebyshev_residual", "full_precision_bucket_residual"}
        else (
            ((1 << int(args.num_bits)) if args.train_quant_points else 0)
            + (((1 << int(args.num_bits)) - 1) if args.train_thresholds else 0)
        )
    )
)
control_manifest = {
    "schema_version": 1,
    "run_name": run_name,
    "logs_root": os.path.abspath(logs_root_dir),
    "config_dir_path": os.path.abspath(config_dir_path),
    "curriculum_cache_key": _curriculum_cache_key(),
    "curriculum_cache_path": os.path.abspath(_curriculum_cache_path()),
    "experiment_control": args.experiment_control,
    "training_objective": experiment_control["training_objective"],
    "loss_normalization": (
        "reconstructed_kv_scalar_count"
        if experiment_control["training_objective"] == "kv_reconstruction_mse"
        else "supervised_next_token_count"
    ),
    "quantizer_parameterization": experiment_control["quantizer_parameterization"],
    "threshold_grad_mode": args.threshold_grad_mode,
    "threshold_grad_aggregation": THRESHOLD_GRAD_AGGREGATION,
    "hard_forward_qdq": bool(experiment_control.get("hard_forward_qdq", True)),
    "hard_bucket_assignment": bool(experiment_control.get("hard_bucket_assignment", False)),
    "compressed_kv_cache": bool(experiment_control["compressed_kv_cache"]),
    "training_kv_cache_storage": "disabled_use_cache_false",
    "compression_semantics": (
        "hard_qdq_training_target_and_quantized_deployment_config"
        if experiment_control["compressed_kv_cache"]
        else (
            "hard_gated_identity_initialized_adapter_with_uncompressed_cache"
            if experiment_control["quantizer_parameterization"] == "full_precision_bucket_residual"
            else "continuous_identity_initialized_adapter_with_uncompressed_cache"
        )
    ),
    "cache_bits": (int(args.num_bits) if experiment_control["compressed_kv_cache"] else 16),
    "group_size": int(args.group_size),
    "table_axis": "physical_token_group",
    "k_activation_location": "post_rope",
    "v_activation_location": "post_v_projection",
    "reconstruction_target": (
        "pre_current_qdq_activation_in_same_hard_forward"
        if experiment_control["training_objective"] == "kv_reconstruction_mse"
        else None
    ),
    "reconstruction_gradient_scope": (
        "strictly_local_per_kv_quantizer"
        if experiment_control["training_objective"] == "kv_reconstruction_mse"
        else None
    ),
    "uniform_levels_constraint": (
        "equally_spaced_endpoints"
        if experiment_control["quantizer_parameterization"] == "uniform_affine_endpoints"
        else None
    ),
    "uniform_threshold_constraint": (
        "exact_adjacent_midpoints"
        if experiment_control["quantizer_parameterization"] == "uniform_affine_endpoints"
        else None
    ),
    "adapter_identity_initialization": (
        True
        if experiment_control["quantizer_parameterization"]
        in {"full_precision_chebyshev_residual", "full_precision_bucket_residual"}
        else None
    ),
    "adapter_parameter_budget_match": (
        "same_scalars_per_table_as_levels_plus_thresholds"
        if experiment_control["quantizer_parameterization"]
        in {"full_precision_chebyshev_residual", "full_precision_bucket_residual"}
        else None
    ),
    "adapter_residual_formula": (
        "x_plus_bucket_offset_selected_from_normalized_x"
        if experiment_control["quantizer_parameterization"] == "full_precision_bucket_residual"
        else None
    ),
    "adapter_cache_storage": (
        "bf16_or_fp16_model_activation_dtype"
        if not experiment_control["compressed_kv_cache"]
        else None
    ),
    "physical_quant_table_count": int(physical_quant_table_count),
    "learned_scalars_per_physical_table": int(learned_scalars_per_physical_table),
    "control_trainable_parameter_count": int(control_trainable_parameter_count),
    "control_total_parameter_count": int(control_total_parameter_count),
    "model_trainable_parameter_count": int(trainable_params),
    "attention_implementation_requested": args.attn_implementation,
    "attention_implementation_resolved": resolved_attn_implementation,
    "model_revision": model_revision,
    "model_revision_available": model_revision is not None,
    "tokenizer_revision": tokenizer_revision,
    "tokenizer_revision_available": tokenizer_revision is not None,
    "dataset_revisions": dataset_revision_manifest,
    "reference_tokens": int(args.reference_tokens),
    "optimizer_steps_planned": int(lr_schedule_total_steps or 0),
    "optimizer_steps_completed": int(global_step),
    "early_stop_patience": int(args.early_stop_patience),
    "early_stop_metric": "full_heldout_validation_ppl",
    "early_stop_comparator": "strictly_worse_than_global_best",
    "early_stop_equality_behavior": "reset_worse_streak",
    "early_stop_invalid_behavior": "reset_worse_streak_without_best_update",
    "early_stop_triggered": bool(early_stop_triggered),
    "training_stop_reason": training_stop_reason,
    "initial_quant_config_path": initial_quant_config_path,
    "final_quant_config_path": final_config_path,
    "training_input_tokens_processed": int(global_training_input_tokens_processed),
    "training_supervised_tokens_processed": int(global_training_supervised_tokens_processed),
    "training_wall_seconds": float(training_wall_seconds),
    "training_gpu_hours": float(training_gpu_hours),
    "current_run_wall_seconds": float(current_run_wall_seconds),
    "current_run_gpu_hours": float(current_run_gpu_hours),
    "gpu_count": int(_world_size),
    "optimizer": optimizer.__class__.__name__,
    "lr": float(args.lr),
    "optimizer_seed": int(args.seed),
    "data_seed": int(args.data_seed),
    "eval_seed": int(args.eval_seed),
    "resume_compatibility_key": {
        "experiment_control": args.experiment_control,
        "base_model": base_model,
        "num_bits": int(args.num_bits),
        "group_size": int(args.group_size),
        "dataset": args.dataset,
        "epochs": int(args.epochs),
        "train_steps_per_pass": int(args.train_steps_per_pass),
        "early_stop_patience": int(args.early_stop_patience),
        "eval_every_steps": int(EVAL_EVERY_STEPS),
        "eval_every_stage": bool(EVAL_EVERY_STAGE),
        "lr_schedule": args.lr_schedule,
        "lr_min_ratio": float(args.lr_min_ratio),
    },
}
metrics_payload = {
    "run_name": run_name,
    "config_dir_name": config_dir_name,
    "logs_root": os.path.abspath(logs_root_dir),
    "config_dir_path": os.path.abspath(config_dir_path),
    "curriculum_cache_key": _curriculum_cache_key(),
    "curriculum_cache_path": os.path.abspath(_curriculum_cache_path()),
    "auto_resume_enabled": not bool(args.no_auto_resume),
    "resume_enabled": bool(resume_enabled),
    "resume_quant_config_path": resume_quant_config_path,
    "resume_config_step": int(resume_config_step) if resume_config_step is not None else None,
    "resume_training_state_path": resume_training_state_path,
    "resume_training_state_loaded": isinstance(resume_training_state, dict),
    "resume_start_unit": int(start_unit),
    "resume_steps_in_start_unit": int(resume_steps_in_start_unit),
    "per_group_quant_tables": bool(experiment_control["compressed_kv_cache"]),
    "per_group_adaptation_tables": True,
    "final_training_state_path": final_training_state_path,
    "initial_quant_config_path": initial_quant_config_path,
    "final_quant_config_path": final_config_path,
    "base_model": base_model,
    "model_revision": model_revision,
    "model_revision_available": model_revision is not None,
    "tokenizer_revision": tokenizer_revision,
    "tokenizer_revision_available": tokenizer_revision is not None,
    "attention_implementation_requested": args.attn_implementation,
    "attention_implementation_resolved": resolved_attn_implementation,
    "dataset_revisions": dataset_revision_manifest,
    "model_type": model_type,
    "dataset": args.dataset,
    "experiment_control": args.experiment_control,
    "training_objective": experiment_control["training_objective"],
    "quantizer_parameterization": experiment_control["quantizer_parameterization"],
    "compressed_kv_cache": bool(experiment_control["compressed_kv_cache"]),
    "control_manifest": control_manifest,
    "control_manifest_path": os.path.join(config_dir_path, "experiment_manifest.json"),
    "num_bits": args.num_bits,
    "group_size": args.group_size,
    "boundary_window": float(args.boundary_window),
    "seed": int(args.seed),
    "optimizer_seed": int(args.seed),
    "data_seed": int(args.data_seed),
    "eval_seed": int(args.eval_seed),
    "training_fingerprint_manifest": training_fingerprint_manifest,
    "validation_fingerprint_manifest": validation_fingerprint_manifest,
    "eval_fingerprint_manifest": eval_fingerprint_manifest,
    "train_quant_points": bool(args.train_quant_points),
    "train_thresholds": bool(args.train_thresholds),
    "threshold_grad_mode": args.threshold_grad_mode,
    "threshold_grad_aggregation": THRESHOLD_GRAD_AGGREGATION,
    "kv_grouping_dim": "token",
    "reference_tokens": args.reference_tokens,
    "heldout_sizing": "fixed_chunks",
    "validation_chunks": int(DISCOVER_VALIDATION_CHUNKS),
    "eval_chunks": int(DISCOVER_EVAL_CHUNKS),
    "chat_wrap_pile": bool(args.chat_wrap_pile),
    "chat_wrap_pile_prompt": args.chat_wrap_pile_prompt,
    "train_val_ratio": train_val_ratio,
    "early_stopping_enabled": bool(args.early_stop_patience > 0 and bool(eval_stages)),
    "early_stop_patience": int(args.early_stop_patience),
    "early_stop_metric": "full_heldout_validation_ppl",
    "early_stop_comparator": "strictly_worse_than_global_best",
    "early_stop_equality_behavior": "reset_worse_streak",
    "early_stop_invalid_behavior": "reset_worse_streak_without_best_update",
    "early_stop_triggered": bool(early_stop_triggered),
    "training_stop_reason": training_stop_reason,
    "final_val_worse_streak": int(val_worse_streak),
    "validation_history": validation_history,
    "length_schedule": DISCOVER_LENGTH_SCHEDULE_NAME,
    "length_curriculum": False,
    "long_doc_truncation": bool(DISCOVER_LONG_DOC_TRUNCATION),
    "pack_train_to_seqlen": bool(PACK_TRAIN_TO_SEQLEN and not DISCOVER_LONG_DOC_TRUNCATION),
    "drop_train_pack_remainder": bool(
        DROP_TRAIN_PACK_REMAINDER and PACK_TRAIN_TO_SEQLEN and not DISCOVER_LONG_DOC_TRUNCATION
    ),
    "active_source_doc_modes": [
        str(source.get("doc_chunk_mode") or "") for source in DISCOVER_ACTIVE_SOURCES
    ],
    "active_source_min_doc_tokens": [
        source.get("min_doc_tokens") for source in DISCOVER_ACTIVE_SOURCES
    ],
    "active_source_require_doc_tokens_gt_seqlen": [
        bool(source.get("require_doc_tokens_gt_seqlen", False))
        for source in DISCOVER_ACTIVE_SOURCES
    ],
    "active_source_chat_wrap": [
        bool(source.get("chat_wrap", False)) for source in DISCOVER_ACTIVE_SOURCES
    ],
    "active_source_chat_user_prompt": [
        source.get("chat_user_prompt") for source in DISCOVER_ACTIVE_SOURCES
    ],
    "active_source_target_units": [
        source.get("target_unit", "tokens") for source in DISCOVER_ACTIVE_SOURCES
    ],
    "dataset_sources": [
        _normalize_source_for_signature(source) for source in DISCOVER_ACTIVE_SOURCES
    ],
    "eval_every_stage": bool(EVAL_EVERY_STAGE),
    "eval_every_steps": int(EVAL_EVERY_STEPS),
    "periodic_validation_full": True,
    "periodic_eval_target_batches": 0,
    "periodic_eval_target_batches_deprecated": int(PERIODIC_EVAL_TARGET_BATCHES),
    "skip_final_eval": bool(args.skip_final_eval),
    "final_eval_max_batches": int(final_eval_max_batches),
    "stream_train_steps_per_pass": int(args.train_steps_per_pass),
    "pile_train_mode": str(args.pile_train_mode),
    "materialized_train_chunks_per_stage": (
        int(args.train_steps_per_pass) * int(target_batch_size)
        if str(args.pile_train_mode) == "materialized"
        else 0
    ),
    "stream_train_step_unit": "optimizer_update",
    "stream_train_sequences_per_step": int(target_batch_size),
    "stream_train_num_workers": int(STREAM_TRAIN_NUM_WORKERS),
    "stream_train_prefetch_factor": int(
        STREAM_TRAIN_PREFETCH_FACTOR if STREAM_TRAIN_NUM_WORKERS > 0 else 0
    ),
    "tqdm_postfix_interval": int(TQDM_POSTFIX_INTERVAL),
    "train_loss_sync_interval": int(TRAIN_LOSS_SYNC_INTERVAL),
    "strict_train_loss_checks": bool(STRICT_TRAIN_LOSS_CHECKS),
    "optimizer_grad_check": bool(CHECK_OPTIMIZER_GRADS),
    "assume_quantizer_grads": bool(ASSUME_QUANTIZER_GRADS),
    "curriculum": [
        {
            "name": stage["name"],
            "seqlen": int(stage["seqlen"]),
            "token_budget": int(stage["token_budget"]),
            "effective_tokens": int(stage["effective_tokens"]),
            "streaming_train": bool(stage.get("streaming_train", False)),
            "train_effective_tokens": int(
                stage.get(
                    "train_effective_tokens",
                    sum(_sample_token_count(sample) for sample in stage["train_samples"]),
                )
            ),
            "val_effective_tokens": int(
                stage.get(
                    "val_effective_tokens",
                    sum(_sample_token_count(sample) for sample in stage["val_samples"]),
                )
            ),
            "collection_seed": int(stage.get("collection_seed", -1)),
            "val_split_seed": int(stage.get("val_split_seed") or -1),
            "train_samples": len(stage["train_samples"]),
            "val_samples": len(stage["val_samples"]),
            "global_sequence_budget": int(
                stage.get("global_sequence_budget", len(stage["train_samples"]))
            ),
            "local_sequence_budget": int(
                stage.get("local_sequence_budget", len(stage["train_samples"]))
            ),
            "reference_global_sequence_budget": int(
                stage.get("reference_global_sequence_budget", 0)
            ),
            "stream_steps_per_pass": int(stage.get("stream_steps_per_pass", 0)),
            "stream_microbatches_per_pass": int(stage.get("stream_microbatches_per_pass", 0)),
            "excluded_heldout_fingerprints": int(stage.get("excluded_heldout_fingerprints", 0)),
            "global_microbatch": int(stage["global_microbatch"]),
            "local_microbatch": int(stage["local_microbatch"]),
            "accum_steps": int(stage["accum_steps"]),
            "effective_batch_size": int(stage["effective_batch_size"]),
            "gradient_checkpointing": stage_uses_gradient_checkpointing(stage["seqlen"]),
        }
        for stage in training_stages
    ],
    "eval_curriculum": [
        {
            "name": stage["name"],
            "seqlen": int(stage["seqlen"]),
            "token_budget": int(stage["token_budget"]),
            "sample_budget": int(stage.get("sample_budget", len(stage["eval_samples"]))),
            "effective_tokens": int(stage["effective_tokens"]),
            "eval_samples": len(stage["eval_samples"]),
            "collection_seed": int(stage.get("collection_seed", -1)),
            "global_microbatch": int(stage["global_microbatch"]),
            "local_microbatch": int(stage["local_microbatch"]),
            "train_token_fraction": float(stage.get("train_token_fraction", 0.0)),
            "heldout_from_train": bool(stage.get("heldout_from_train", False)),
            "excluded_train_fingerprints": int(stage.get("excluded_train_fingerprints", 0)),
            "eval_fingerprints": int(stage.get("eval_fingerprints", 0)),
        }
        for stage in eval_stages
    ],
    "validation_curriculum": [
        {
            "name": stage["name"],
            "seqlen": int(stage["seqlen"]),
            "token_budget": int(stage["token_budget"]),
            "sample_budget": int(stage.get("sample_budget", len(stage["eval_samples"]))),
            "effective_tokens": int(stage["effective_tokens"]),
            "eval_samples": len(stage["eval_samples"]),
            "collection_seed": int(stage.get("collection_seed", -1)),
            "global_microbatch": int(stage["global_microbatch"]),
            "local_microbatch": int(stage["local_microbatch"]),
            "train_token_fraction": float(stage.get("train_token_fraction", 0.0)),
            "heldout_from_train": bool(stage.get("heldout_from_train", False)),
            "excluded_train_fingerprints": int(stage.get("excluded_train_fingerprints", 0)),
            "eval_fingerprints": int(stage.get("eval_fingerprints", 0)),
        }
        for stage in eval_stages
    ],
    "test_curriculum": [
        {
            "name": stage["name"],
            "seqlen": int(stage["seqlen"]),
            "token_budget": int(stage["token_budget"]),
            "sample_budget": int(stage.get("sample_budget", len(stage["eval_samples"]))),
            "effective_tokens": int(stage["effective_tokens"]),
            "test_samples": len(stage["eval_samples"]),
            "collection_seed": int(stage.get("collection_seed", -1)),
            "global_microbatch": int(stage["global_microbatch"]),
            "local_microbatch": int(stage["local_microbatch"]),
            "train_token_fraction": float(stage.get("train_token_fraction", 0.0)),
            "heldout_from_eval": bool(stage.get("heldout_from_eval", False)),
            "excluded_eval_fingerprints": int(stage.get("excluded_eval_fingerprints", 0)),
            "test_fingerprints": int(stage.get("test_fingerprints", 0)),
        }
        for stage in test_stages
    ],
    "eval_data_role": "validation",
    "eval_data_heldout_from_train": False,
    "eval_data_streamed_independently": True,
    "test_data_heldout_from_eval": True,
    "test_data_streamed_independently": True,
    "validation_data_role": "val",
    "eval_set_data_role": "eval",
    "final_config_eval_split": "eval",
    "final_config_storage_split": "test",
    "batch_size": target_batch_size,
    "target_effective_batch_size": target_batch_size,
    "per_gpu_batch_size": int(per_gpu_batch_size),
    "train_global_microbatch": int(train_global_microbatch),
    "train_accum_steps": int(train_accum_steps),
    "world_size": _world_size,
    "data_parallel_world_size": int(_data_parallel_world_size),
    "data_parallel_rank": int(_data_parallel_rank),
    "tensor_model_parallel_size": int(args.tensor_model_parallel_size),
    "pipeline_model_parallel_size": int(args.pipeline_model_parallel_size),
    "allow_linear_only_tensor_parallel": bool(args.allow_linear_only_tensor_parallel),
    "parallel": metrics_parallel,
    "runtime_backend": args.runtime_backend,
    "launcher": args.launcher,
    "effective_batch_size": target_batch_size,
    "stage_effective_batch_sizes": {
        stage["name"]: int(stage["effective_batch_size"]) for stage in training_stages
    },
    "stage_global_microbatches": {
        stage["name"]: int(stage["global_microbatch"]) for stage in training_stages
    },
    "stage_local_microbatches": {
        stage["name"]: int(stage["local_microbatch"]) for stage in training_stages
    },
    "stage_accum_steps": {stage["name"]: int(stage["accum_steps"]) for stage in training_stages},
    "gradient_checkpointing": GRADIENT_CHECKPOINTING_FROM_SEQLEN > 0,
    "gradient_checkpointing_from_seqlen": GRADIENT_CHECKPOINTING_FROM_SEQLEN,
    "stage_gradient_checkpointing": {
        stage["name"]: stage_uses_gradient_checkpointing(stage["seqlen"])
        for stage in training_stages
    },
    "max_accum_steps": max_accum_steps,
    "epochs_completed": epochs_completed,
    "curriculum_units_completed": epochs_completed,
    "curriculum_passes_completed": epochs_completed / max(1, num_curriculum_stages),
    "curriculum_units_total": total_training_units,
    "optimizer_steps_completed": int(global_step),
    "training_input_tokens_processed": int(global_training_input_tokens_processed),
    "training_supervised_tokens_processed": int(global_training_supervised_tokens_processed),
    "training_wall_seconds": float(training_wall_seconds),
    "training_gpu_hours": float(training_gpu_hours),
    "current_run_wall_seconds": float(current_run_wall_seconds),
    "current_run_gpu_hours": float(current_run_gpu_hours),
    "gpu_count": int(_world_size),
    "global_micro_step": int(global_micro_step),
    "backward_mode": (
        "local_hard_qdq_reconstruction"
        if experiment_control["training_objective"] == "kv_reconstruction_mse"
        else (
            "hard_bucket_residual_nll"
            if experiment_control["quantizer_parameterization"] == "full_precision_bucket_residual"
            else (
                "continuous_adapter_nll"
                if experiment_control["quantizer_parameterization"]
                == "full_precision_chebyshev_residual"
                else "ste_nll"
            )
        )
    ),
    "quantizer_input_grad": (
        "detached_for_local_reconstruction"
        if experiment_control["training_objective"] == "kv_reconstruction_mse"
        else (
            "identity_from_x_residual_path_routing_detached"
            if experiment_control["quantizer_parameterization"] == "full_precision_bucket_residual"
            else (
                "ordinary_autograd"
                if experiment_control["quantizer_parameterization"]
                == "full_precision_chebyshev_residual"
                else "ste"
            )
        )
    ),
    "ce_loss_weight": float(args.ce_loss_weight),
    "final_ce_weighted_loss": last_ce_weighted_loss_value,
    "last_reconstruction_element_count": _scalar_value(last_reconstruction_element_count_value),
    "lr": args.lr,
    "lr_schedule": args.lr_schedule,
    "lr_min_ratio": float(args.lr_min_ratio),
    "lr_min": float(args.lr) * float(args.lr_min_ratio),
    "lr_schedule_total_steps": int(lr_schedule_total_steps or 0),
    "final_optimizer_lr": float(last_optimizer_lr),
    "native_init_eval_ppl": native_init_eval_ppl,
    "native_init_eval_loss": native_init_eval_metrics["loss"],
    "native_init_eval_target_tokens": native_init_eval_metrics["target_tokens"],
    "native_init_eval_stages": native_init_eval_metrics["stages"],
    "native_init_eval_split": "eval",
    "native_initial_val_ppl": native_initial_val_ppl,
    "native_initial_val_loss": native_initial_periodic_val_metrics["loss"],
    "native_initial_val_target_tokens": native_initial_periodic_val_metrics["target_tokens"],
    "native_initial_val_stages": native_initial_periodic_val_metrics["stages"],
    "native_initial_val_split": "val",
    "init_eval_ppl": init_eval_ppl,
    "init_eval_loss": init_eval_metrics["loss"],
    "init_eval_target_tokens": init_eval_metrics["target_tokens"],
    "init_eval_stages": init_eval_metrics["stages"],
    "init_eval_split": "eval",
    "initial_val_loss": initial_periodic_val_metrics["loss"],
    "initial_val_ppl": initial_periodic_val_metrics["ppl"],
    "initial_val_target_tokens": initial_periodic_val_metrics["target_tokens"],
    "initial_val_stages": initial_periodic_val_metrics["stages"],
    "initial_val_split": "val",
    "initial_periodic_val_loss": initial_periodic_val_metrics["loss"],
    "initial_periodic_val_ppl": initial_periodic_val_metrics["ppl"],
    "initial_periodic_val_target_tokens": initial_periodic_val_metrics["target_tokens"],
    "initial_periodic_val_stages": initial_periodic_val_metrics["stages"],
    "final_train_loss": final_train_loss,
    "final_val_loss": final_val_loss,
    "final_val_ppl": final_val_ppl,
    "final_eval_loss": final_eval_loss,
    "final_eval_ppl": final_eval_ppl,
    "final_eval_set_loss": final_test_loss,
    "final_eval_set_ppl": final_test_ppl,
    "final_test_loss": final_test_loss,
    "final_test_ppl": final_test_ppl,
    "final_config_loss": final_config_loss,
    "final_config_ppl": final_config_ppl,
    "best_val_loss": best_val_loss,
    "best_val_ppl": best_val_ppl,
    "best_val_epoch": best_val_epoch,
    "best_val_unit": best_val_epoch,
    "best_val_pass": best_val_pass,
    "best_quant_config_path": best_quant_config_path,
    "final_config_selection": (
        "best_validation_checkpoint"
        if best_quant_config_path is not None
        else "last_checkpoint_no_finite_validation"
    ),
}
if is_main():
    save_metrics(metrics_path, metrics_payload)
    save_metrics(os.path.join(config_dir_path, "experiment_manifest.json"), control_manifest)
    print(f"Training finished! Final quantization config saved to: {final_config_path}")
dist_barrier("training_finished")
if torch.distributed.is_available() and torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()
