import inspect
import os
import sys
import threading

import torch
import torch.nn as nn

from beyond.quantization.table_precision import (
    TABLE_STORAGE_DTYPE_NAME,
    fp16_ste_roundtrip,
)

try:
    from beyond.quantization.triton_ste import (
        triton_group_quant_backward_ste as _triton_group_quant_backward_ste,
    )
    from beyond.quantization.triton_ste import (
        triton_group_quant_forward as _triton_group_quant_forward,
    )
    from beyond.quantization.triton_ste import (
        triton_group_quant_norm_backward_ste as _triton_group_quant_norm_backward_ste,
    )
    from beyond.quantization.triton_ste import (
        triton_group_quant_norm_forward as _triton_group_quant_norm_forward,
    )

    _TRITON_STE_QUANT_AVAILABLE = os.getenv("DISCOVER_TRITON_STE_QUANT", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
except Exception:
    _TRITON_STE_QUANT_AVAILABLE = False
    _triton_group_quant_forward = None
    _triton_group_quant_norm_forward = None
    _triton_group_quant_norm_backward_ste = None
    _triton_group_quant_backward_ste = None

THRESHOLD_GRAD_AGGREGATION = (
    "raw_side_sums_accumulate_then_global_activity_filter_then_tp_sum_dp_avg_then_combine_once_v3"
)

_THRESHOLD_GRAD_RAW_MODES = {
    "raw",
    "unrectified",
    "no_rect",
    "no-rect",
    "no_half_wave",
    "no-half-wave",
}


def combine_threshold_side_grads(g_left, g_right, mode=None):
    """Combine globally accumulated threshold-side gradients exactly once."""
    normalized_mode = (
        str(mode if mode is not None else os.getenv("DISCOVER_THRESHOLD_GRAD_MODE", "half_wave"))
        .strip()
        .lower()
    )
    if normalized_mode not in _THRESHOLD_GRAD_RAW_MODES:
        return torch.clamp(g_left, min=0.0) + torch.clamp(g_right, max=0.0)
    return g_left + g_right


def _packed_threshold_side_view(sum_left, sum_right):
    """Return a zero-copy ``[left, right]`` view when storage is adjacent."""
    if not (torch.is_tensor(sum_left) and torch.is_tensor(sum_right)):
        return None
    if (
        tuple(sum_left.shape) != tuple(sum_right.shape)
        or sum_left.dtype != sum_right.dtype
        or sum_left.device != sum_right.device
        or sum_left.layout != torch.strided
        or sum_right.layout != torch.strided
        or not sum_left.is_contiguous()
        or not sum_right.is_contiguous()
        or sum_left.numel() == 0
    ):
        return None
    try:
        same_storage = (
            sum_left.untyped_storage().data_ptr() == sum_right.untyped_storage().data_ptr()
        )
    except (AttributeError, RuntimeError):
        return None
    if not same_storage or sum_right.storage_offset() != (
        sum_left.storage_offset() + sum_left.numel()
    ):
        return None
    return sum_left.as_strided(
        (2,) + tuple(sum_left.shape),
        (sum_left.numel(),) + tuple(sum_left.stride()),
    )


def _has_threshold_side_sink(sink):
    return any(
        callable(getattr(sink, method_name, None))
        for method_name in (
            "accumulate_threshold_side_sums_packed_",
            "accumulate_threshold_side_sums_",
        )
    )


def _accumulate_threshold_side_sums(ctx, sum_left, sum_right):
    """Store raw side sums on the owning quantizer; never rectify per backward."""
    if not ctx.needs_input_grad[2]:
        return
    sink = getattr(ctx, "threshold_grad_sink", None)
    packed = _packed_threshold_side_view(sum_left, sum_right)
    accumulate_packed = getattr(sink, "accumulate_threshold_side_sums_packed_", None)
    if packed is not None and callable(accumulate_packed):
        accumulate_packed(packed)
        return
    accumulate = getattr(sink, "accumulate_threshold_side_sums_", None)
    if not callable(accumulate):
        raise RuntimeError(
            "Trainable thresholds require a deferred threshold-gradient sink. "
            "Use the owning quantizer when invoking the quantization Function."
        )
    accumulate(sum_left, sum_right)


_FAST_NORMALIZED_GROUP_QUANT_AVAILABLE = os.getenv(
    "DISCOVER_FAST_NORMALIZED_GROUP_QUANT",
    "1",
).lower() not in {
    "0",
    "false",
    "no",
    "off",
}

# Thread-local storage for deferred K quantization when using post-RoPE K QDQ.
_tls = threading.local()


def _env_enabled(name, default=True):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off", "disable", "disabled"}


def _quantizer_allows_qkv_inplace_merge(quantizer):
    unified_quant_layer = globals().get("UnifiedQuantLayer")
    return (
        unified_quant_layer is not None
        and isinstance(quantizer, unified_quant_layer)
        and _TRITON_STE_QUANT_AVAILABLE
        and _FAST_NORMALIZED_GROUP_QUANT_AVAILABLE
    )


def _can_rewrite_qkv_output_inplace(
    y,
    linear,
    k_quantizer,
    v_quantizer,
    k_changed,
    v_changed,
):
    if not _env_enabled("DISCOVER_QKV_INPLACE", True):
        return False
    # nn.Identity in tests and simple adapters can alias the caller's input.
    # The discover path wraps real projection modules, whose outputs are fresh.
    if isinstance(linear, nn.Identity):
        return False
    if y.requires_grad and y.is_leaf:
        return False
    if k_changed and not _quantizer_allows_qkv_inplace_merge(k_quantizer):
        return False
    if v_changed and not _quantizer_allows_qkv_inplace_merge(v_quantizer):
        return False
    return torch.is_tensor(y) and y.is_floating_point()


def _merge_quantized_qkv_output(
    y,
    linear,
    q_end,
    k_end,
    v_end,
    k_value,
    v_value,
    k_quantizer,
    v_quantizer,
    k_changed,
    v_changed,
):
    if not k_changed and not v_changed:
        return y
    if _can_rewrite_qkv_output_inplace(
        y,
        linear,
        k_quantizer,
        v_quantizer,
        k_changed,
        v_changed,
    ):
        if k_changed:
            y[..., q_end:k_end] = k_value
        if v_changed:
            y[..., k_end:v_end] = v_value
        return y

    if not k_changed:
        return torch.cat([y[..., :k_end], v_value], dim=-1)
    if not v_changed:
        return torch.cat([y[..., :q_end], k_value, y[..., k_end:v_end]], dim=-1)
    return torch.cat([y[..., :q_end], k_value, v_value], dim=-1)


def _prefer_triton_group_kernel(q_points):
    return q_points.dim() == 2 and 1 <= int(q_points.shape[1]) <= 64


def _can_use_fast_normalized_group_quant(x, q_points, thresholds, group_size):
    if not _FAST_NORMALIZED_GROUP_QUANT_AVAILABLE:
        return False
    if not (
        torch.is_tensor(x)
        and torch.is_tensor(q_points)
        and torch.is_tensor(thresholds)
        and x.is_cuda
        and q_points.is_cuda
        and thresholds.is_cuda
        and q_points.dtype == torch.float32
        and thresholds.dtype == torch.float32
        and x.device == q_points.device == thresholds.device
        and x.dim() == 3
        and q_points.dim() == 2
        and thresholds.dim() == 2
    ):
        return False
    group_size = int(group_size)
    if group_size <= 0 or int(x.shape[-1]) <= 0 or int(x.shape[-1]) % group_size != 0:
        return False
    num_tables = int(x.shape[-1]) // group_size
    if int(q_points.shape[0]) != num_tables:
        return False
    if int(thresholds.shape[0]) != num_tables:
        return False
    if int(q_points.shape[1]) != int(thresholds.shape[1]) + 1:
        return False
    if not (1 <= int(q_points.shape[0]) <= 512 and 1 <= int(q_points.shape[1]) <= 64):
        return False
    return (
        _TRITON_STE_QUANT_AVAILABLE
        and _triton_group_quant_norm_forward is not None
        and _triton_group_quant_norm_backward_ste is not None
        and _triton_group_quant_forward is not None
        and _triton_group_quant_backward_ste is not None
    )


def quantization_is_disabled() -> bool:
    return bool(getattr(_tls, "disable_quantization", False))


# --- Wrapper ---
class QuantizedLinear(nn.Module):
    """
    A wrapper for applying a quantizer module after a linear layer.
    When defer_quantize=True, stores the quantizer in TLS for later use
    (e.g., after RoPE application) and only performs the linear transform.
    """

    def __init__(self, linear, quantizer):
        super().__init__()
        self.linear = linear
        self.quantizer = quantizer
        self.defer_quantize = False

    def forward(self, x):
        out = self.linear(x)
        if quantization_is_disabled():
            _tls.active_k_quantizer = None
            return out
        if self.defer_quantize:
            _tls.active_k_quantizer = self.quantizer
            return out
        return self.quantizer(out)


class QuantizedQKVLinear(nn.Module):
    """
    Wrapper for fused QKV projection layers (e.g., `query_key_value`).
    Applies quantization to K and V slices only; Q slice stays in FP.
    Uses explicit Q/K/V split sizes when available; otherwise falls back to
    equal thirds for legacy fused projections.
    """

    def __init__(
        self, linear, k_quantizer: nn.Module = None, v_quantizer: nn.Module = None, qkv_splits=None
    ):
        super().__init__()
        self.linear = linear
        self.k_quantizer = k_quantizer
        self.v_quantizer = v_quantizer
        self.qkv_splits = tuple(qkv_splits) if qkv_splits is not None else None
        self.defer_k_quantize = False

    def forward(self, x):
        y = self.linear(x)
        # Work with 2D or 3D tensors; quantizers accept both as implemented
        if not torch.is_tensor(y):
            return y
        if quantization_is_disabled():
            _tls.active_k_quantizer = None
            return y
        last_dim = y.shape[-1]
        splits = _resolve_qkv_splits(last_dim, self.qkv_splits)
        if splits is None:
            # If shape is unexpected, skip quantization to avoid breaking forward
            return y

        q_dim, k_dim, v_dim = splits
        q_end = q_dim
        k_end = q_end + k_dim
        v_end = k_end + v_dim
        k_value = None
        v_value = None
        k_changed = False
        v_changed = False

        if self.k_quantizer is not None and self.defer_k_quantize:
            _tls.active_k_quantizer = self.k_quantizer
        elif self.k_quantizer is not None:
            k_value = self.k_quantizer(y[..., q_end:k_end])
            k_changed = True
        if self.v_quantizer is not None:
            v_value = self.v_quantizer(y[..., k_end:v_end])
            v_changed = True

        return _merge_quantized_qkv_output(
            y,
            self.linear,
            q_end,
            k_end,
            v_end,
            k_value,
            v_value,
            self.k_quantizer,
            self.v_quantizer,
            k_changed,
            v_changed,
        )


def _extract_linear_tensor_output(output):
    if not isinstance(output, (tuple, list)):
        return output

    if not output:
        raise ValueError("Quantized linear module received empty tuple/list output.")
    primary_output = output[0]
    if len(output) > 1 and torch.is_tensor(primary_output) and torch.is_tensor(output[1]):
        return primary_output + output[1]
    return primary_output


def _resolve_qkv_splits(last_dim, qkv_splits=None):
    if qkv_splits is not None:
        try:
            q_dim, k_dim, v_dim = [int(v) for v in qkv_splits]
        except Exception:
            q_dim = k_dim = v_dim = 0
        if q_dim > 0 and k_dim > 0 and v_dim > 0 and q_dim + k_dim + v_dim == int(last_dim):
            return q_dim, k_dim, v_dim

    if int(last_dim) % 3 != 0:
        return None
    h = int(last_dim) // 3
    return h, h, h


class MegatronQuantizedLinear(nn.Module):
    """
    Megatron-compatible quantization wrapper.
    Megatron module outputs may sometimes be a tuple; this adapter always returns
    the quantized tensor output expected by transformer attention blocks.
    """

    def __init__(self, linear, quantizer):
        super().__init__()
        self.linear = linear
        self.quantizer = quantizer
        self.defer_quantize = False

    def forward(self, *args, **kwargs):
        output = self.linear(*args, **kwargs)
        primary_output = _extract_linear_tensor_output(output)
        if quantization_is_disabled():
            _tls.active_k_quantizer = None
            return primary_output
        if self.defer_quantize:
            _tls.active_k_quantizer = self.quantizer
            return primary_output
        if not torch.is_tensor(primary_output):
            return primary_output
        quantized = self.quantizer(primary_output)
        return quantized


class MegatronQuantizedQKVLinear(nn.Module):
    """
    Megatron fused QKV wrapper that returns tensor output expected by transformer
    attention blocks.
    """

    def __init__(
        self, linear, k_quantizer: nn.Module = None, v_quantizer: nn.Module = None, qkv_splits=None
    ):
        super().__init__()
        self.linear = linear
        self.k_quantizer = k_quantizer
        self.v_quantizer = v_quantizer
        self.qkv_splits = tuple(qkv_splits) if qkv_splits is not None else None
        self.defer_k_quantize = False

    def forward(self, *args, **kwargs):
        output = self.linear(*args, **kwargs)
        primary_output = _extract_linear_tensor_output(output)
        if not torch.is_tensor(primary_output):
            return primary_output
        if quantization_is_disabled():
            _tls.active_k_quantizer = None
            return primary_output

        y = primary_output
        last_dim = y.shape[-1]
        splits = _resolve_qkv_splits(last_dim, self.qkv_splits)
        if splits is None:
            return y

        q_dim, k_dim, v_dim = splits
        q_end = q_dim
        k_end = q_end + k_dim
        v_end = k_end + v_dim
        k_value = None
        v_value = None
        k_changed = False
        v_changed = False

        if self.k_quantizer is not None and self.defer_k_quantize:
            _tls.active_k_quantizer = self.k_quantizer
        elif self.k_quantizer is not None:
            k_value = self.k_quantizer(y[..., q_end:k_end])
            k_changed = True
        if self.v_quantizer is not None:
            v_value = self.v_quantizer(y[..., k_end:v_end])
            v_changed = True

        return _merge_quantized_qkv_output(
            y,
            self.linear,
            q_end,
            k_end,
            v_end,
            k_value,
            v_value,
            self.k_quantizer,
            self.v_quantizer,
            k_changed,
            v_changed,
        )


# --- Quantization Function (hard forward, STE input backward) ---


class CustomQuantFunction(torch.autograd.Function):
    @staticmethod
    def _forward_group_tables(x_norm, q_points, thresholds):
        if q_points.dim() != 2 or thresholds.dim() != 2:
            raise ValueError(
                "Grouped quant tables require q_points and thresholds with "
                f"rank 2, got {tuple(q_points.shape)} and {tuple(thresholds.shape)}."
            )
        if x_norm.dim() < 2:
            raise ValueError("Grouped quant tables require x_norm group axis at dim -2.")
        num_tables, num_q_pts = q_points.shape
        if thresholds.shape != (num_tables, num_q_pts - 1):
            raise ValueError(
                f"Expected thresholds shape {(num_tables, num_q_pts - 1)}, "
                f"got {tuple(thresholds.shape)}."
            )
        if x_norm.shape[-2] != num_tables:
            raise ValueError(
                f"x_norm group axis has {x_norm.shape[-2]} tables, but q_points has {num_tables}."
            )

        outs = []
        bins = []
        for table_idx in range(num_tables):
            x_g = x_norm.select(-2, table_idx).contiguous()
            codes = torch.bucketize(x_g, thresholds[table_idx], right=True)
            codes = codes.clamp_(0, num_q_pts - 1)
            outs.append(q_points[table_idx][codes.to(torch.long)])
            bins.append(codes)
        return torch.stack(outs, dim=-2), torch.stack(bins, dim=-2)

    @staticmethod
    def _backward_group_tables(
        grad_output,
        x_norm,
        q_points,
        thresholds,
        bin_indices,
        eps,
    ):
        num_tables, num_q_pts = q_points.shape
        num_thresholds = thresholds.shape[-1]
        if thresholds.shape != (num_tables, num_q_pts - 1):
            raise ValueError(
                f"Expected thresholds shape {(num_tables, num_q_pts - 1)}, "
                f"got {tuple(thresholds.shape)}."
            )

        x_r = x_norm.reshape(-1, num_tables, x_norm.shape[-1])
        grad_r = grad_output.to(torch.float32).reshape_as(x_r)
        bin_r = bin_indices.to(torch.long).reshape_as(x_r)
        grad_q = torch.zeros_like(q_points)
        side_sums = torch.zeros(
            (2,) + tuple(thresholds.shape),
            device=thresholds.device,
            dtype=torch.float32,
        )
        side_sum_left, side_sum_right = side_sums.unbind(0)

        for table_idx in range(num_tables):
            x_f = x_r[:, table_idx, :].reshape(-1)
            grad_f = grad_r[:, table_idx, :].reshape(-1)
            bin_f = bin_r[:, table_idx, :].reshape(-1).clamp_(0, num_q_pts - 1)
            t_table = thresholds[table_idx]

            grad_q[table_idx] = torch.bincount(
                bin_f,
                weights=grad_f,
                minlength=num_q_pts,
            )[:num_q_pts].to(grad_q.dtype)

            if num_thresholds > 0:
                # Accumulate directly into the packed output planes.  They are
                # already zero-initialized, so per-table temporaries and copies
                # are unnecessary on the fallback path.
                sum_left = side_sum_left[table_idx]
                sum_right = side_sum_right[table_idx]

                right_bound_mask = bin_f < num_thresholds
                if right_bound_mask.any():
                    right_idx = bin_f[right_bound_mask]
                    x_right = x_f[right_bound_mask]
                    grad_right = grad_f[right_bound_mask]
                    t_right = t_table[right_idx]
                    diff_right = x_right - t_right
                    left_mask = (diff_right >= -eps) & (diff_right < 0)
                    if left_mask.any():
                        sum_left.index_add_(0, right_idx[left_mask], grad_right[left_mask])

                left_bound_mask = bin_f > 0
                if left_bound_mask.any():
                    left_idx = bin_f[left_bound_mask] - 1
                    x_left = x_f[left_bound_mask]
                    grad_left = grad_f[left_bound_mask]
                    t_left = t_table[left_idx]
                    diff_left = x_left - t_left
                    right_mask = (diff_left >= 0) & (diff_left <= eps)
                    if right_mask.any():
                        sum_right.index_add_(0, left_idx[right_mask], grad_left[right_mask])

        return grad_output.to(x_norm.dtype), grad_q, side_sum_left, side_sum_right

    @staticmethod
    def forward(ctx, x_norm, q_points, thresholds, eps, threshold_grad_sink):
        ctx.eps = eps
        ctx.threshold_grad_sink = threshold_grad_sink
        ctx.use_triton_ste_kernel = False
        if thresholds.requires_grad and not _has_threshold_side_sink(threshold_grad_sink):
            raise ValueError(
                "A deferred threshold-gradient sink is required when thresholds are trainable."
            )
        if (
            _prefer_triton_group_kernel(q_points)
            and _TRITON_STE_QUANT_AVAILABLE
            and _triton_group_quant_forward is not None
        ):
            triton_result = _triton_group_quant_forward(x_norm, q_points, thresholds)
            if triton_result is not None:
                xq_norm, bin_indices = triton_result
                ctx.use_triton_ste_kernel = True
                ctx.save_for_backward(x_norm, q_points, thresholds, bin_indices)
                return xq_norm

        thresholds_dev = thresholds.to(x_norm.device)
        q_points_dev = q_points.to(x_norm.device)
        if q_points_dev.dim() == 2:
            xq_norm, bin_indices = CustomQuantFunction._forward_group_tables(
                x_norm,
                q_points_dev,
                thresholds_dev,
            )
            ctx.save_for_backward(x_norm, q_points, thresholds, bin_indices)
            return xq_norm

        bin_indices = torch.bucketize(x_norm, thresholds_dev, right=True)
        ctx.save_for_backward(x_norm, q_points, thresholds, bin_indices)
        xq_norm = q_points_dev[bin_indices]
        return xq_norm

    @staticmethod
    def backward(ctx, grad_output):
        x_norm, q_points, thresholds, bin_indices = ctx.saved_tensors
        eps = ctx.eps

        if (
            getattr(ctx, "use_triton_ste_kernel", False)
            and _TRITON_STE_QUANT_AVAILABLE
            and _triton_group_quant_backward_ste is not None
        ):
            triton_result = _triton_group_quant_backward_ste(
                grad_output,
                x_norm,
                q_points,
                thresholds,
                bin_indices,
                eps,
                return_grad_x=bool(ctx.needs_input_grad[0]),
            )
            if triton_result is not None:
                grad_x, grad_q, sum_left, sum_right = triton_result
                _accumulate_threshold_side_sums(
                    ctx,
                    sum_left,
                    sum_right,
                )
                return grad_x, grad_q, None, None, None

        # Common flattened versions and device
        x_f = x_norm.reshape(-1)
        grad_f = grad_output.to(torch.float32).reshape(-1)
        bin_f = bin_indices.reshape(-1)
        current_device = x_norm.device

        # Ensure q_points and thresholds are on the correct device for subsequent operations
        q_points_dev = q_points.to(current_device)
        thresholds_dev = thresholds.to(current_device)

        if q_points_dev.dim() == 2:
            grad_x, grad_q, sum_left, sum_right = CustomQuantFunction._backward_group_tables(
                grad_output,
                x_norm,
                q_points_dev,
                thresholds_dev,
                bin_indices,
                eps,
            )
            if not ctx.needs_input_grad[0]:
                grad_x = None
            _accumulate_threshold_side_sums(
                ctx,
                sum_left,
                sum_right,
            )
            return grad_x, grad_q, None, None, None

        num_q_pts = q_points_dev.shape[0]
        num_thresholds = thresholds_dev.shape[0]
        if num_q_pts != num_thresholds + 1:
            raise ValueError(
                f"Expected thresholds to have length {num_q_pts - 1}, got {num_thresholds}."
            )

        # --- grad_q: per-bin sum update under hard assignment ---
        bin_f = bin_f.to(torch.long).clamp_(0, num_q_pts - 1)
        grad_q = torch.bincount(bin_f, weights=grad_f, minlength=num_q_pts)
        # Raw side statistics are accumulated across the complete optimizer
        # window.  Delta-q multiplication and half-wave/raw combination happen
        # only after DP reduction at the optimizer boundary.
        side_sums = torch.zeros(
            (2, num_thresholds),
            device=current_device,
            dtype=grad_f.dtype,
        )
        sum_left, sum_right = side_sums.unbind(0)
        if num_thresholds > 0:
            # Left side of threshold t_k: x in [t_k - eps, t_k), x is in bin k
            right_bound_mask = bin_f < num_thresholds
            if right_bound_mask.any():
                right_idx = bin_f[right_bound_mask]
                x_right = x_f[right_bound_mask]
                grad_right = grad_f[right_bound_mask]
                t_right = thresholds_dev[right_idx]
                diff_right = x_right - t_right
                left_mask = (diff_right >= -eps) & (diff_right < 0)
                if left_mask.any():
                    idx = right_idx[left_mask]
                    vals = grad_right[left_mask]
                    sum_left.index_add_(0, idx, vals)

            # Right side (including equality) of threshold t_k: x in [t_k, t_k + eps], x is in bin k+1
            left_bound_mask = bin_f > 0
            if left_bound_mask.any():
                left_idx = bin_f[left_bound_mask] - 1
                x_left = x_f[left_bound_mask]
                grad_left = grad_f[left_bound_mask]
                t_left = thresholds_dev[left_idx]
                diff_left = x_left - t_left
                right_mask = (diff_left >= 0) & (diff_left <= eps)
                if right_mask.any():
                    idx = left_idx[right_mask]
                    vals = grad_left[right_mask]
                    sum_right.index_add_(0, idx, vals)

        grad_x = grad_output.to(x_norm.dtype) if ctx.needs_input_grad[0] else None
        _accumulate_threshold_side_sums(
            ctx,
            sum_left,
            sum_right,
        )
        return grad_x, grad_q, None, None, None


class NormalizedGroupQuantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, q_points, thresholds, eps, group_size, threshold_grad_sink):
        group_size = int(group_size)
        ctx.threshold_grad_sink = threshold_grad_sink
        if thresholds.requires_grad and not _has_threshold_side_sink(threshold_grad_sink):
            raise ValueError(
                "A deferred threshold-gradient sink is required when thresholds are trainable."
            )
        orig_dtype = x.dtype
        B, S, H = x.shape
        num_tables = H // group_size
        x_g = x.contiguous().view(B, S, num_tables, group_size)

        use_triton = False
        if _TRITON_STE_QUANT_AVAILABLE and _triton_group_quant_norm_forward is not None:
            triton_norm_result = _triton_group_quant_norm_forward(
                x_g,
                q_points,
                thresholds,
            )
            if triton_norm_result is not None:
                out_g, bin_indices, mins, scale = triton_norm_result
                out = out_g.view(B, S, H).to(orig_dtype)
                use_triton = _triton_group_quant_norm_backward_ste is not None
                saved_input = x_g
                saved_aux = mins
                recompute_norm = True
            else:
                out = None
        else:
            out = None

        if out is None:
            saved_input = None
            saved_aux = None
            recompute_norm = False
            x_float = x.to(torch.float32)
            x_g = x_float.contiguous().view(B, S, num_tables, group_size)
            mn = x_g.amin(dim=-1, keepdim=True)
            mx = x_g.amax(dim=-1, keepdim=True)
            scale = (mx - mn).clamp_(min=1e-6)
            x_norm = ((x_g - mn) / scale).clamp_(0.0, 1.0)

            if (
                _prefer_triton_group_kernel(q_points)
                and _TRITON_STE_QUANT_AVAILABLE
                and _triton_group_quant_forward is not None
            ):
                triton_result = _triton_group_quant_forward(x_norm, q_points, thresholds)
                if triton_result is not None:
                    xq_norm, bin_indices = triton_result
                    use_triton = True
                else:
                    xq_norm = None
                    bin_indices = None
            else:
                xq_norm = None
                bin_indices = None

            if xq_norm is None:
                xq_norm, bin_indices = CustomQuantFunction._forward_group_tables(
                    x_norm,
                    q_points,
                    thresholds,
                )

            out = (xq_norm * scale + mn).view(B, S, H).to(orig_dtype)
            saved_input = x_norm
            saved_aux = torch.empty(0, device=x_norm.device, dtype=x_norm.dtype)
        ctx.eps = eps
        ctx.group_size = group_size
        ctx.input_dtype = orig_dtype
        ctx.use_triton_ste_kernel = use_triton
        ctx.recompute_norm_for_backward = recompute_norm
        ctx.save_for_backward(saved_input, q_points, thresholds, bin_indices, saved_aux, scale)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        saved_input, q_points, thresholds, bin_indices, saved_aux, scale = ctx.saved_tensors
        eps = ctx.eps

        if (
            getattr(ctx, "use_triton_ste_kernel", False)
            and getattr(ctx, "recompute_norm_for_backward", False)
            and _TRITON_STE_QUANT_AVAILABLE
            and _triton_group_quant_norm_backward_ste is not None
        ):
            grad_g = grad_output.contiguous().view_as(saved_input)
            triton_result = _triton_group_quant_norm_backward_ste(
                grad_g,
                saved_input,
                q_points,
                thresholds,
                bin_indices,
                saved_aux,
                scale,
                eps,
            )
            if triton_result is not None:
                grad_q, sum_left, sum_right = triton_result
                _accumulate_threshold_side_sums(
                    ctx,
                    sum_left,
                    sum_right,
                )
                grad_x = grad_output.to(ctx.input_dtype) if ctx.needs_input_grad[0] else None
                return grad_x, grad_q, None, None, None, None

        if getattr(ctx, "recompute_norm_for_backward", False):
            x_norm = ((saved_input.to(torch.float32) - saved_aux) / scale).clamp_(0.0, 1.0)
        else:
            x_norm = saved_input

        grad_g = grad_output.to(torch.float32).contiguous().view_as(x_norm)
        grad_norm = grad_g * scale
        if (
            getattr(ctx, "use_triton_ste_kernel", False)
            and _TRITON_STE_QUANT_AVAILABLE
            and _triton_group_quant_backward_ste is not None
        ):
            triton_result = _triton_group_quant_backward_ste(
                grad_norm,
                x_norm,
                q_points,
                thresholds,
                bin_indices,
                eps,
                return_grad_x=False,
            )
            if triton_result is not None:
                _grad_x_norm, grad_q, sum_left, sum_right = triton_result
                _accumulate_threshold_side_sums(
                    ctx,
                    sum_left,
                    sum_right,
                )
                grad_x = grad_output.to(ctx.input_dtype) if ctx.needs_input_grad[0] else None
                return grad_x, grad_q, None, None, None, None

        grad_x_norm, grad_q, sum_left, sum_right = CustomQuantFunction._backward_group_tables(
            grad_norm,
            x_norm,
            q_points,
            thresholds,
            bin_indices,
            eps,
        )
        del grad_x_norm
        _accumulate_threshold_side_sums(
            ctx,
            sum_left,
            sum_right,
        )
        grad_x = grad_output.to(ctx.input_dtype) if ctx.needs_input_grad[0] else None
        return grad_x, grad_q, None, None, None, None


class BaseQuantLayer(nn.Module):
    """
    Base class for quantization layers.
    """

    def __init__(
        self,
        num_bits=2,
        group_size=32,
        thresholds=None,
        table_axis="group",
        num_tables=1,
    ):
        super().__init__()
        table_axis = str(table_axis or "group").lower()
        if table_axis in {"per_group", "group"}:
            table_axis = "group"
        elif table_axis not in {"shared"}:
            raise ValueError(f"Unsupported quant table_axis: {table_axis}")
        num_tables = int(num_tables)
        if table_axis == "shared":
            num_tables = 1
        elif num_tables <= 0:
            raise ValueError("Grouped quant tables require num_tables > 0")

        self.num_bits = int(num_bits)

        if num_bits == 1:
            q_init = [0.0, 1.0]
        elif num_bits in (2, 3, 4, 5, 6):
            # Uniform init over [0, 1]: 2^bits levels (num_bits=6 -> 64 levels,
            # matching the CUDA kernel's kMaxQPoints=64 budget exactly).
            q_init = [i / ((1 << num_bits) - 1) for i in range(1 << num_bits)]
        else:
            raise ValueError(f"Unsupported num_bits: {num_bits}. Supported: 1, 2, 3, 4, 5, 6.")

        q_tensor = torch.tensor(sorted(q_init), dtype=torch.float32)
        if table_axis == "group":
            q_tensor = q_tensor.unsqueeze(0).repeat(num_tables, 1).contiguous()
        self.q_points = nn.Parameter(q_tensor)
        if thresholds is not None:
            thresholds_tensor = torch.tensor(thresholds, dtype=torch.float32)
            expected_dim = 2 if table_axis == "group" else 1
            if thresholds_tensor.dim() != expected_dim:
                raise ValueError(
                    f"Expected {expected_dim}-D thresholds for table_axis={table_axis}; "
                    f"got shape {tuple(thresholds_tensor.shape)}"
                )
            self.thresholds = nn.Parameter(thresholds_tensor)
        else:
            self.thresholds = nn.Parameter(
                (self.q_points.data[..., :-1] + self.q_points.data[..., 1:]) / 2.0
            )
        self.group_size = group_size
        self.table_axis = table_axis
        self.num_tables = num_tables
        self.table_storage_dtype = TABLE_STORAGE_DTYPE_NAME
        self.bandwidth = float(os.getenv("DISCOVER_BOUNDARY_WINDOW", "0.009"))
        self.register_buffer(
            "_threshold_side_sums",
            torch.zeros(
                (2,) + tuple(self.thresholds.shape),
                dtype=torch.float32,
                device=self.thresholds.device,
            ),
            persistent=False,
        )
        self._threshold_side_sum_updates = 0

    def _deferred_threshold_reference_parameter(self):
        return self._parameters.get("thresholds")

    def deferred_threshold_grad_enabled(self):
        parameter = self._deferred_threshold_reference_parameter()
        return parameter is not None and bool(parameter.requires_grad)

    def deferred_threshold_direct_parameters(self):
        """Parameters whose gradient comes only from deferred threshold stats."""
        parameter = self._parameters.get("thresholds")
        if parameter is None or not parameter.requires_grad:
            return ()
        return (parameter,)

    def deferred_threshold_update_parameters(self):
        """Parameters updated by the deferred threshold-gradient contribution."""
        parameter = self._parameters.get("thresholds")
        if parameter is None or not parameter.requires_grad:
            return ()
        return (parameter,)

    def _ensure_threshold_side_sums(self):
        reference = self._deferred_threshold_reference_parameter()
        buffer = getattr(self, "_threshold_side_sums", None)
        if reference is None or buffer is None:
            return None
        expected_shape = (2,) + tuple(self._deferred_threshold_shape())
        if (
            tuple(buffer.shape) != expected_shape
            or buffer.device != reference.device
            or buffer.dtype != torch.float32
        ):
            buffer = torch.zeros(
                expected_shape,
                device=reference.device,
                dtype=torch.float32,
            )
            self._buffers["_threshold_side_sums"] = buffer
        return buffer

    def _deferred_threshold_shape(self):
        parameter = self._parameters.get("thresholds")
        return tuple(parameter.shape) if parameter is not None else ()

    @torch.no_grad()
    def accumulate_threshold_side_sums_packed_(self, side_sums):
        if not self.deferred_threshold_grad_enabled():
            raise RuntimeError(
                "Received threshold-side statistics for a quantizer without "
                "trainable threshold degrees of freedom."
            )
        buffer = self._ensure_threshold_side_sums()
        if buffer is None:
            raise RuntimeError("Deferred threshold side-sum buffer is unavailable.")
        expected_shape = tuple(buffer.shape)
        if tuple(side_sums.shape) != expected_shape:
            raise ValueError(
                "Packed threshold side-sum shape mismatch: expected "
                f"{expected_shape}, got {tuple(side_sums.shape)}."
            )
        buffer.add_(side_sums.detach().to(device=buffer.device, dtype=buffer.dtype))
        self._threshold_side_sum_updates += 1

    @torch.no_grad()
    def accumulate_threshold_side_sums_(self, sum_left, sum_right):
        packed = _packed_threshold_side_view(sum_left, sum_right)
        if packed is not None:
            self.accumulate_threshold_side_sums_packed_(packed)
            return
        if not self.deferred_threshold_grad_enabled():
            raise RuntimeError(
                "Received threshold-side statistics for a quantizer without "
                "trainable threshold degrees of freedom."
            )
        buffer = self._ensure_threshold_side_sums()
        if buffer is None:
            raise RuntimeError("Deferred threshold side-sum buffer is unavailable.")
        expected_shape = tuple(buffer.shape[1:])
        if tuple(sum_left.shape) != expected_shape or tuple(sum_right.shape) != expected_shape:
            raise ValueError(
                "Threshold side-sum shape mismatch: expected "
                f"{expected_shape}, got {tuple(sum_left.shape)} and {tuple(sum_right.shape)}."
            )
        buffer[0].add_(sum_left.detach().to(device=buffer.device, dtype=buffer.dtype))
        buffer[1].add_(sum_right.detach().to(device=buffer.device, dtype=buffer.dtype))
        self._threshold_side_sum_updates += 1

    def has_pending_threshold_side_sums(self):
        return self._threshold_side_sum_updates > 0

    def threshold_side_sums(self, include_zeros=False):
        if not self.deferred_threshold_grad_enabled():
            return None
        if not include_zeros and not self.has_pending_threshold_side_sums():
            return None
        buffer = self._ensure_threshold_side_sums()
        if buffer is None:
            return None
        return buffer[0], buffer[1]

    @torch.no_grad()
    def clear_threshold_side_sums_(self):
        buffer = getattr(self, "_threshold_side_sums", None)
        if buffer is not None:
            buffer.zero_()
        self._threshold_side_sum_updates = 0

    @staticmethod
    @torch.no_grad()
    def _add_parameter_grad_(parameter, grad):
        if parameter is None or not parameter.requires_grad:
            return
        grad = grad.to(device=parameter.device, dtype=parameter.dtype)
        if parameter.grad is None:
            parameter.grad = grad.clone(memory_format=torch.preserve_format)
        else:
            parameter.grad.add_(grad)

    @torch.no_grad()
    def _route_deferred_threshold_grad_(self, threshold_grad):
        parameter = self._parameters.get("thresholds")
        if parameter is None or not parameter.requires_grad:
            return ()
        self._add_parameter_grad_(parameter, threshold_grad)
        return (parameter,)

    @torch.no_grad()
    def finalize_threshold_side_sums_(self, sum_left, sum_right, mode=None):
        if not self.deferred_threshold_grad_enabled():
            return ()
        q_points, thresholds = self.materialize_quant_tables()
        expected_shape = tuple(thresholds.shape)
        if tuple(sum_left.shape) != expected_shape or tuple(sum_right.shape) != expected_shape:
            raise ValueError(
                "Reduced threshold side-sum shape mismatch: expected "
                f"{expected_shape}, got {tuple(sum_left.shape)} and {tuple(sum_right.shape)}."
            )
        q_points = q_points.detach().to(device=sum_left.device, dtype=torch.float32)
        delta_q = q_points[..., 1:] - q_points[..., :-1]
        g_left = -delta_q * sum_left.to(torch.float32)
        g_right = -delta_q * sum_right.to(torch.float32)
        threshold_grad = combine_threshold_side_grads(g_left, g_right, mode=mode)
        return self._route_deferred_threshold_grad_(threshold_grad)

    def materialize_quant_tables(self):
        """Return deployment-representable tables backed by FP32 masters."""
        return fp16_ste_roundtrip(self.q_points), fp16_ste_roundtrip(self.thresholds)

    @torch.no_grad()
    def set_quant_tables(self, quant_points, thresholds=None):
        """Load an exported table into the native non-uniform parameters."""
        q_tensor = torch.as_tensor(
            quant_points,
            device=self.q_points.device,
            dtype=self.q_points.dtype,
        )
        if tuple(q_tensor.shape) != tuple(self.q_points.shape):
            raise ValueError(
                f"quant_points shape {tuple(q_tensor.shape)} does not match "
                f"{tuple(self.q_points.shape)}"
            )
        if thresholds is None:
            t_tensor = (q_tensor[..., :-1] + q_tensor[..., 1:]) * 0.5
        else:
            t_tensor = torch.as_tensor(
                thresholds,
                device=self.thresholds.device,
                dtype=self.thresholds.dtype,
            )
        if tuple(t_tensor.shape) != tuple(self.thresholds.shape):
            raise ValueError(
                f"thresholds shape {tuple(t_tensor.shape)} does not match "
                f"{tuple(self.thresholds.shape)}"
            )
        self.q_points.copy_(q_tensor)
        self.thresholds.copy_(t_tensor)

    def quantizer_parameterization(self):
        return "nonuniform_points_thresholds"


def has_pending_threshold_side_sums(modules):
    return any(
        callable(getattr(module, "has_pending_threshold_side_sums", None))
        and module.has_pending_threshold_side_sums()
        for module in modules
    )


@torch.no_grad()
def clear_threshold_side_sums_(modules):
    seen = set()
    for module in modules:
        if id(module) in seen:
            continue
        seen.add(id(module))
        clear = getattr(module, "clear_threshold_side_sums_", None)
        if callable(clear):
            clear()


@torch.no_grad()
def threshold_side_activity_roster(modules, *, include_all_enabled=False):
    """Return a deterministic, identity-deduplicated deferred-side roster.

    Entries are ``(module, left, right, pending)``.  With
    ``include_all_enabled=True``, enabled-but-idle modules are retained with
    zero-filled side views so every distributed rank can build the same fixed
    activity plane.
    """
    entries = []
    seen = set()
    for module in modules:
        module_id = id(module)
        if module_id in seen:
            continue
        seen.add(module_id)
        get_sides = getattr(module, "threshold_side_sums", None)
        if not callable(get_sides):
            continue
        sides = get_sides(include_zeros=include_all_enabled)
        if sides is None:
            continue
        is_pending = bool(
            callable(getattr(module, "has_pending_threshold_side_sums", None))
            and module.has_pending_threshold_side_sums()
        )
        entries.append((module, sides[0], sides[1], is_pending))
    return tuple(entries)


@torch.no_grad()
def finalize_deferred_threshold_grads(
    modules,
    *,
    mode=None,
    tensor_reduce=None,
    data_reduce=None,
    activity_reduce=None,
    global_activity=None,
    include_all_enabled=False,
):
    """Reduce raw side sums, then create parameter gradients once per update.

    Reducers must mutate their tensor in-place (or return a replacement tensor).
    ``tensor_reduce`` and ``data_reduce`` aggregate raw side sums;
    ``activity_reduce`` aggregates the compact per-module participation mask.
    Alternatively, ``global_activity`` can provide an already-reduced
    ``{id(module): bool}`` mapping and skip that collective.  The two activity
    inputs are mutually exclusive, and the mapping must cover the full roster.
    They are deliberately injectable so the training runtime can use its exact
    TP/DP process groups and CPU tests can verify the reduction order.  Each
    globally active module is rectified once; globally idle modules are only
    roster placeholders and retain ``grad=None``.
    """
    if activity_reduce is not None and global_activity is not None:
        raise ValueError("activity_reduce and global_activity are mutually exclusive")

    unique_modules = []
    seen = set()
    for module in modules:
        module_id = id(module)
        if module_id in seen:
            continue
        seen.add(module_id)
        unique_modules.append(module)

    # Validate ownership independently of this window's activity.  Otherwise a
    # shared parameter could pass silently whenever only one of its owners ran,
    # then be rectified twice in a later window where both ran.
    target_owners = {}
    for module in unique_modules:
        update_parameters = getattr(module, "deferred_threshold_update_parameters", None)
        if not callable(update_parameters):
            continue
        for parameter in update_parameters():
            owner = target_owners.setdefault(id(parameter), module)
            if owner is not module:
                raise RuntimeError(
                    "Deferred threshold update parameters must have one quantizer owner; "
                    "shared parameters must accumulate into one owner before rectification."
                )

    entries = threshold_side_activity_roster(
        unique_modules,
        include_all_enabled=include_all_enabled,
    )

    # Every distributed rank keeps the complete enabled-module roster, but only
    # modules that participated on at least one replica are rectified.  The
    # activity reduction is separate and compact so fixed side buckets do not
    # pay an extra full-size presence plane.
    if global_activity is None:
        globally_active = [entry[3] for entry in entries]
    else:
        missing_module_ids = [
            id(module)
            for module, _left, _right, _pending in entries
            if id(module) not in global_activity
        ]
        if missing_module_ids:
            raise ValueError(
                "global_activity is missing deferred-side module ids: "
                + ", ".join(str(module_id) for module_id in missing_module_ids)
            )
        globally_active = [
            bool(global_activity[id(module)]) for module, _left, _right, _pending in entries
        ]
    if activity_reduce is not None and entries:
        activity_buckets = {}
        for index, (_module, left, _right, _pending) in enumerate(entries):
            activity_buckets.setdefault(left.device, []).append(index)
        for device, indices in activity_buckets.items():
            activity = torch.tensor(
                [1 if globally_active[index] else 0 for index in indices],
                device=device,
                dtype=torch.int32,
            )
            replacement = activity_reduce(activity)
            if replacement is not None:
                activity = replacement
            reduced_flags = activity.gt(0).detach().cpu().tolist()
            for index, flag in zip(indices, reduced_flags):
                globally_active[index] = bool(flag)

    buckets = {}
    for index, entry in enumerate(entries):
        _module, left, _right, _pending = entry
        key = (left.device, left.dtype, tuple(left.shape))
        buckets.setdefault(key, []).append((index, entry))

    reduced_entries = []
    for bucket in buckets.values():
        packed = torch.stack(
            [side for _index, (_module, left, right, _pending) in bucket for side in (left, right)],
            dim=0,
        )
        if tensor_reduce is not None:
            replacement = tensor_reduce(packed)
            if replacement is not None:
                packed = replacement
        if data_reduce is not None:
            replacement = data_reduce(packed)
            if replacement is not None:
                packed = replacement
        for bucket_index, (entry_index, (module, _left, _right, _pending)) in enumerate(bucket):
            reduced_entries.append(
                (
                    module,
                    packed[2 * bucket_index],
                    packed[2 * bucket_index + 1],
                    globally_active[entry_index],
                )
            )

    # Consume the window before applying the nonlinear operation.  The packed
    # reduced tensors above own their storage, so this makes the helper
    # idempotent even if routing a later module's gradient raises an exception.
    for module, _left, _right, _is_active in reduced_entries:
        clear = getattr(module, "clear_threshold_side_sums_", None)
        if callable(clear):
            clear()

    finalized = 0
    for module, left, right, is_active in reduced_entries:
        if not is_active:
            continue
        finalize = getattr(module, "finalize_threshold_side_sums_", None)
        if callable(finalize):
            finalize(left, right, mode=mode)
            finalized += 1
    return finalized


class UnifiedQuantLayer(BaseQuantLayer):
    """
    Unified quantization layer with per-token head-dimension grouping.
    """

    _printed_one_group_sizes = False  # Initialize as a class attribute

    def __init__(
        self,
        num_bits=2,
        group_size=32,
        grouping_dim="token",
        thresholds=None,
        one_group=False,
        table_axis="group",
        num_tables=1,
        quant_width=None,
    ):
        table_axis = str(table_axis or "group").lower()
        if table_axis in {"per_group", "group"}:
            table_axis = "group"
        if one_group and table_axis == "group":
            raise ValueError("Grouped quant tables are not supported with one_group.")
        if table_axis == "group":
            if quant_width is not None:
                quant_width = int(quant_width)
                if group_size <= 0 or quant_width % int(group_size) != 0:
                    raise ValueError(
                        f"quant_width {quant_width} must be divisible by group_size {group_size}"
                    )
                inferred_tables = quant_width // int(group_size)
                if int(num_tables) not in (0, 1, inferred_tables):
                    raise ValueError(
                        f"num_tables={num_tables} does not match quant_width/group_size={inferred_tables}"
                    )
                num_tables = inferred_tables
            elif int(num_tables) <= 0:
                raise ValueError("Grouped quant tables require quant_width or num_tables.")
        super().__init__(
            num_bits=num_bits,
            group_size=group_size,
            thresholds=thresholds,
            table_axis=table_axis,
            num_tables=num_tables,
        )
        if grouping_dim != "token":
            raise ValueError("grouping_dim must be 'token'")
        self.grouping_dim = grouping_dim
        self.one_group = one_group
        self.quant_width = quant_width
        self._capture_reconstruction = False
        self._reconstruction_token_mask = None
        self._reconstruction_sse_terms = []
        self._reconstruction_element_terms = []
        # self._printed_one_group_sizes = False # Remove instance attribute

    def forward(self, x):
        if self._capture_reconstruction:
            # Reconstruction controls must remain valid when the enclosing
            # decoder layer is evaluated under gradient-checkpoint no-grad.
            # Detaching the activation and returning a detached hard-QDQ value
            # makes every recorded MSE strictly local to this physical table.
            with torch.enable_grad():
                out = self._forward_impl(x.detach())
                self._record_reconstruction(x.detach(), out)
            return out.detach()
        return self._forward_impl(x)

    def _forward_impl(self, x):
        orig_dtype, orig_shape = x.dtype, x.shape
        # Add batch dim if not present, standardizing to 3D tensor (B, S, H)
        if x.dim() == 2:
            x_3d = x.unsqueeze(0)
        elif x.dim() == 3:
            x_3d = x
        else:
            print(
                f"Warning: UnifiedQuantLayer received input with dim {x.dim()}, expected 2 or 3. Proceeding..."
            )
            x_3d = x

        B, S, H = x_3d.shape

        if self.one_group and not UnifiedQuantLayer._printed_one_group_sizes:
            print(f"Info: 'one_group' is active. Effective token group_size: {H}")
            UnifiedQuantLayer._printed_one_group_sizes = True

        x_quantized_S_H_dim = x_3d  # Default to original if quantization is skipped or fails
        # can_quantize = False # This variable is assigned but not used, can be removed if not needed later

        # Parameters are already on the right device; avoid redundant .to() calls.
        q_pts, thr = self.materialize_quant_tables()
        bw = self.bandwidth

        if self.one_group:
            # Quantize over the entire head dimension (H) as one group.
            x_float_3d = x_3d.float()
            mn = x_float_3d.amin(dim=-1, keepdim=True)
            mx = x_float_3d.amax(dim=-1, keepdim=True)
            scale = (mx - mn).clamp(min=1e-6)
            x_norm = ((x_float_3d - mn) / scale).clamp(0.0, 1.0)
            xq_norm = CustomQuantFunction.apply(x_norm, q_pts, thr, bw, self)
            x_quantized_S_H_dim = xq_norm * scale + mn
        elif self.group_size > 0 and H > 0 and H % self.group_size == 0:
            G = H // self.group_size
            if self.table_axis == "group" and q_pts.shape[0] != G:
                raise ValueError(
                    f"Grouped quant table count {q_pts.shape[0]} does not "
                    f"match input groups {G} for width {H} and group_size {self.group_size}."
                )
            if self.table_axis == "group" and _can_use_fast_normalized_group_quant(
                x_3d,
                q_pts,
                thr,
                self.group_size,
            ):
                x_quantized_S_H_dim = NormalizedGroupQuantFunction.apply(
                    x_3d,
                    q_pts,
                    thr,
                    bw,
                    int(self.group_size),
                    self,
                )
            else:
                x_float_3d = x_3d.float()
                x_g = x_float_3d.view(B, S, G, self.group_size)
                mn = x_g.amin(dim=-1, keepdim=True)
                mx = x_g.amax(dim=-1, keepdim=True)
                scale = (mx - mn).clamp(min=1e-6)
                x_norm = ((x_g - mn) / scale).clamp(0.0, 1.0)
                xq_norm = CustomQuantFunction.apply(x_norm, q_pts, thr, bw, self)
                xq = xq_norm * scale + mn
                x_quantized_S_H_dim = xq.view(B, S, H)

        out = x_quantized_S_H_dim.to(orig_dtype)
        # Remove batch dim if it was added by this layer
        if x.dim() == 2 and out.dim() == 3 and out.shape[0] == 1:
            out = out.squeeze(0)
        elif x.dim() > 2 and len(orig_shape) == x.dim() and out.shape[0] == B:
            pass  # Shape is already as expected or original was >2D
        # This condition seems to be a more specific case of the above, check if still needed or can be merged
        if len(orig_shape) == out.dim() - 1 and orig_shape == out.shape[1:]:
            if out.shape[0] == 1:
                out = out.squeeze(0)  # Remove the batch dimension added above.
        return out

    def enable_reconstruction_capture(self, token_mask=None):
        self._capture_reconstruction = True
        self._reconstruction_token_mask = token_mask
        self._reconstruction_sse_terms.clear()
        self._reconstruction_element_terms.clear()

    def disable_reconstruction_capture(self):
        self._capture_reconstruction = False
        self._reconstruction_token_mask = None

    def clear_reconstruction_stats(self):
        self._reconstruction_sse_terms.clear()
        self._reconstruction_element_terms.clear()

    def _record_reconstruction(self, target, quantized):
        difference = quantized.to(torch.float32) - target.to(torch.float32)
        mask = self._reconstruction_token_mask
        if mask is None:
            sse = difference.square().sum()
            elements = torch.tensor(
                float(difference.numel()),
                device=difference.device,
                dtype=torch.float32,
            )
        else:
            mask = mask.to(device=difference.device, dtype=torch.float32)
            if difference.dim() == 3 and mask.dim() == 2:
                if tuple(mask.shape) == tuple(difference.shape[:2]):
                    expanded_mask = mask.unsqueeze(-1)
                elif mask.numel() == int(difference.shape[0] * difference.shape[1]):
                    expanded_mask = mask.reshape(difference.shape[0], difference.shape[1], 1)
                else:
                    raise ValueError(
                        "Reconstruction token mask does not match quantizer input: "
                        f"mask={tuple(mask.shape)}, input={tuple(difference.shape)}"
                    )
            elif difference.dim() == 2 and mask.numel() == int(difference.shape[0]):
                expanded_mask = mask.reshape(difference.shape[0], 1)
            else:
                raise ValueError(
                    "Reconstruction token mask does not match quantizer input: "
                    f"mask={tuple(mask.shape)}, input={tuple(difference.shape)}"
                )
            sse = (difference.square() * expanded_mask).sum()
            elements = expanded_mask.sum() * int(difference.shape[-1])
        self._reconstruction_sse_terms.append(sse)
        self._reconstruction_element_terms.append(elements.detach())

    def consume_reconstruction_stats(self):
        if not self._reconstruction_sse_terms:
            return None, None
        sse = torch.stack(self._reconstruction_sse_terms).sum()
        elements = torch.stack(self._reconstruction_element_terms).sum()
        self.clear_reconstruction_stats()
        return sse, elements


class UniformAffineQuantLayer(UnifiedQuantLayer):
    """Hard uniform QDQ with exactly two learned endpoints per physical table.

    Input groups are normalized exactly as in :class:`UnifiedQuantLayer`.  The
    only learned values are ``affine_low`` and ``affine_high``.  Codepoints are
    always equally spaced between those endpoints and thresholds are always
    their exact midpoints, so this control cannot drift into Beyond's
    non-uniform parameterization.
    """

    endpoint_min_span = 1e-4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        q_shape = tuple(self.q_points.shape[:-1])
        device = self.q_points.device
        dtype = self.q_points.dtype
        del self.q_points
        del self.thresholds
        self.affine_low = nn.Parameter(torch.zeros(q_shape, device=device, dtype=dtype))
        self.affine_high = nn.Parameter(torch.ones(q_shape, device=device, dtype=dtype))
        self.register_buffer(
            "uniform_fractions",
            torch.linspace(0.0, 1.0, 1 << int(self.num_bits), device=device, dtype=dtype),
            persistent=False,
        )

    def quantizer_parameterization(self):
        return "uniform_affine_endpoints"

    def _deferred_threshold_reference_parameter(self):
        if self.affine_low.requires_grad:
            return self.affine_low
        if self.affine_high.requires_grad:
            return self.affine_high
        return None

    def deferred_threshold_grad_enabled(self):
        return bool(self.affine_low.requires_grad or self.affine_high.requires_grad)

    def deferred_threshold_direct_parameters(self):
        # Endpoints also receive the ordinary codepoint gradient, so they must
        # remain in the normal reducer before the deferred contribution is added.
        return ()

    def deferred_threshold_update_parameters(self):
        return tuple(
            parameter
            for parameter in (self.affine_low, self.affine_high)
            if parameter.requires_grad
        )

    def _deferred_threshold_shape(self):
        return tuple(self.affine_low.shape) + (self._num_levels - 1,)

    @torch.no_grad()
    def _route_deferred_threshold_grad_(self, threshold_grad):
        fractions = self.uniform_fractions.to(
            device=threshold_grad.device,
            dtype=torch.float32,
        )
        midpoint_fractions = (fractions[:-1] + fractions[1:]) * 0.5
        grad_low = (threshold_grad * (1.0 - midpoint_fractions)).sum(dim=-1)
        grad_high = (threshold_grad * midpoint_fractions).sum(dim=-1)
        routed = []
        if self.affine_low.requires_grad:
            self._add_parameter_grad_(self.affine_low, grad_low)
            routed.append(self.affine_low)
        if self.affine_high.requires_grad:
            self._add_parameter_grad_(self.affine_high, grad_high)
            routed.append(self.affine_high)
        return tuple(routed)

    def materialize_quant_tables(self):
        fractions = self.uniform_fractions
        q_points = (
            self.affine_low.unsqueeze(-1)
            + (self.affine_high - self.affine_low).unsqueeze(-1) * fractions
        )
        thresholds = (q_points[..., :-1] + q_points[..., 1:]) * 0.5
        return fp16_ste_roundtrip(q_points), fp16_ste_roundtrip(thresholds)

    @property
    def _num_levels(self):
        return int(getattr(self, "num_bits", 0) and (1 << int(self.num_bits)))

    @torch.no_grad()
    def project_parameters_(self):
        finite_low = torch.nan_to_num(self.affine_low, nan=0.0, posinf=1.0, neginf=0.0)
        finite_high = torch.nan_to_num(self.affine_high, nan=1.0, posinf=1.0, neginf=0.0)
        low = torch.minimum(finite_low, finite_high)
        high = torch.maximum(finite_low, finite_high)
        low = low.clamp(0.0, 1.0 - self.endpoint_min_span)
        high = high.clamp(self.endpoint_min_span, 1.0)
        high = torch.maximum(high, low + self.endpoint_min_span)
        low = torch.minimum(low, high - self.endpoint_min_span)
        self.affine_low.copy_(low)
        self.affine_high.copy_(high)

    @torch.no_grad()
    def set_quant_tables(self, quant_points, thresholds=None):
        q_tensor = torch.as_tensor(
            quant_points,
            device=self.affine_low.device,
            dtype=self.affine_low.dtype,
        )
        expected_shape = tuple(self.affine_low.shape) + (self._num_levels,)
        if tuple(q_tensor.shape) != expected_shape:
            raise ValueError(
                f"uniform quant_points shape {tuple(q_tensor.shape)} does not match {expected_shape}"
            )
        expected_q = torch.linspace(
            0.0,
            1.0,
            self._num_levels,
            device=q_tensor.device,
            dtype=q_tensor.dtype,
        )
        expected_q = q_tensor[..., :1] + (q_tensor[..., -1:] - q_tensor[..., :1]) * expected_q
        if not torch.allclose(q_tensor, expected_q, rtol=1e-5, atol=1e-6):
            raise ValueError("uniform-affine control refuses non-uniform quantization points")
        low = q_tensor[..., 0]
        high = q_tensor[..., -1]
        if (
            not torch.isfinite(q_tensor).all()
            or bool((low < 0.0).any())
            or bool((high > 1.0).any())
            or bool((high - low < self.endpoint_min_span).any())
        ):
            raise ValueError("uniform-affine control requires finite ordered endpoints in [0, 1]")
        midpoint_thresholds = (q_tensor[..., :-1] + q_tensor[..., 1:]) * 0.5
        if thresholds is not None:
            t_tensor = torch.as_tensor(
                thresholds,
                device=q_tensor.device,
                dtype=q_tensor.dtype,
            )
            if tuple(t_tensor.shape) != tuple(midpoint_thresholds.shape) or not torch.allclose(
                t_tensor,
                midpoint_thresholds,
                rtol=1e-5,
                atol=1e-6,
            ):
                raise ValueError("uniform-affine control requires midpoint thresholds")
        self.affine_low.copy_(low)
        self.affine_high.copy_(high)


class FullPrecisionChebyshevAdapter(UnifiedQuantLayer):
    """Parameter-matched, continuous full-precision KV adaptation control.

    Each physical token group receives ``2 * 2**num_bits - 1`` trainable
    Chebyshev coefficients: exactly the number of representation levels plus
    decision thresholds in the corresponding Beyond quantizer.  Coefficients
    are zero-initialized, so the adapter is an exact identity before training.
    The continuous output remains in the model activation dtype and therefore
    does *not* represent a compressed cache.

    For a min--max normalized scalar ``x`` and Chebyshev series ``s(x)``, the
    residual map is

        x' = x + x (1 - x) tanh(s(x)).

    This map is continuous, fixes both group endpoints, and stays in ``[0, 1]``
    for finite coefficients.  It provides a conservative way to separate the
    effect of an equally sized learned KV adaptation surface from low-bit hard
    discretization.
    """

    coefficient_clip = 8.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        table_shape = tuple(self.q_points.shape[:-1])
        device = self.q_points.device
        dtype = self.q_points.dtype
        del self.q_points
        del self.thresholds
        del self._threshold_side_sums
        self.num_coefficients = (2 * (1 << int(self.num_bits))) - 1
        self.coefficients = nn.Parameter(
            torch.zeros(
                table_shape + (self.num_coefficients,),
                device=device,
                dtype=dtype,
            )
        )

    def quantizer_parameterization(self):
        return "full_precision_chebyshev_residual"

    def _series(self, z):
        coefficients = self.coefficients.to(dtype=torch.float32)
        if self.table_axis == "group":
            if int(coefficients.shape[0]) != int(z.shape[-2]):
                raise ValueError(
                    f"Adapter table count {coefficients.shape[0]} does not match "
                    f"input groups {z.shape[-2]}."
                )

            def coefficient(degree):
                return coefficients[:, degree].view(1, 1, -1, 1)
        else:

            def coefficient(degree):
                return coefficients[degree]

        t_prev = torch.ones_like(z)
        result = coefficient(0) * t_prev
        if self.num_coefficients == 1:
            return result
        t_curr = z
        result = result + coefficient(1) * t_curr
        for degree in range(2, self.num_coefficients):
            t_next = (2.0 * z * t_curr) - t_prev
            result = result + coefficient(degree) * t_next
            t_prev, t_curr = t_curr, t_next
        return result

    def forward(self, x):
        if x.dim() == 2:
            x_3d = x.unsqueeze(0)
            added_batch = True
        elif x.dim() == 3:
            x_3d = x
            added_batch = False
        else:
            raise ValueError(
                "FullPrecisionChebyshevAdapter expects a 2-D or 3-D activation; "
                f"got shape {tuple(x.shape)}"
            )

        batch, sequence, width = x_3d.shape
        if self.one_group:
            groups = 1
            group_size = int(width)
        else:
            group_size = int(self.group_size)
            if group_size <= 0 or int(width) % group_size != 0:
                raise ValueError(
                    f"Activation width {width} must be divisible by group_size {group_size}."
                )
            groups = int(width) // group_size

        x_float = x_3d.to(torch.float32)
        x_grouped = x_float.reshape(batch, sequence, groups, group_size)
        group_min = x_grouped.amin(dim=-1, keepdim=True)
        group_span = (x_grouped.amax(dim=-1, keepdim=True) - group_min).clamp(min=1e-6)
        x_normalized = ((x_grouped - group_min) / group_span).clamp(0.0, 1.0)
        z = (2.0 * x_normalized) - 1.0
        series = self._series(z)
        normalized_residual = x_normalized * (1.0 - x_normalized) * torch.tanh(series)
        # Adding the residual to the original float-cast activation makes the
        # zero-coefficient initialization bit-exact after casting back, rather
        # than relying on a min--max normalize/de-normalize round trip.
        adapted = x_grouped + (group_span * normalized_residual)
        output = adapted.reshape(batch, sequence, width).to(dtype=x.dtype)
        if added_batch:
            output = output.squeeze(0)
        return output

    def export_adapter_state(self):
        return {
            "adapter_coefficients": self.coefficients.detach().cpu().clone().contiguous(),
            "num_coefficients": int(self.num_coefficients),
            "identity_initialized": True,
            "continuous_forward": True,
            "cache_storage": "model_activation_dtype",
        }

    @torch.no_grad()
    def set_adapter_state(self, coefficients):
        value = torch.as_tensor(
            coefficients,
            device=self.coefficients.device,
            dtype=self.coefficients.dtype,
        )
        if tuple(value.shape) != tuple(self.coefficients.shape):
            raise ValueError(
                f"adapter coefficients shape {tuple(value.shape)} does not match "
                f"{tuple(self.coefficients.shape)}"
            )
        if not torch.isfinite(value).all():
            raise ValueError("adapter coefficients must be finite")
        self.coefficients.copy_(value)

    @torch.no_grad()
    def project_parameters_(self):
        finite = torch.nan_to_num(
            self.coefficients,
            nan=0.0,
            posinf=self.coefficient_clip,
            neginf=-self.coefficient_clip,
        )
        self.coefficients.copy_(finite.clamp(-self.coefficient_clip, self.coefficient_clip))


class FullPrecisionBucketResidualAdapter(UnifiedQuantLayer):
    """Parameter-matched, uncompressed hard-gated KV residual control.

    For every physical token group, the activation is min--max normalized only
    to select one of ``2**num_bits`` buckets.  The selected trainable offset is
    added to the original activation without replacing it with a codepoint::

        u = clamp((x - min(x)) / span(x), 0, 1)
        j = bucketize(u, thresholds)
        x' = x + bucket_offsets[j]

    The offsets are zero-initialized, making the adapter an exact identity.
    Together with the decision thresholds, this gives the same number of
    trainable scalars per table as Beyond, while the adapted K/V values remain
    in the model activation dtype.  Bucket routing is hard, but the cache value
    itself is not discretized.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        offset_shape = tuple(self.q_points.shape)
        device = self.q_points.device
        dtype = self.q_points.dtype
        del self.q_points
        self.num_offsets = 1 << int(self.num_bits)
        self.bucket_offsets = nn.Parameter(torch.zeros(offset_shape, device=device, dtype=dtype))

    def quantizer_parameterization(self):
        return "full_precision_bucket_residual"

    def materialize_quant_tables(self):
        # The deferred threshold estimator only needs the output jump across
        # each boundary.  Here that jump is a[k + 1] - a[k].
        return self.bucket_offsets, self.thresholds

    def forward(self, x):
        if x.dim() == 2:
            x_3d = x.unsqueeze(0)
            added_batch = True
        elif x.dim() == 3:
            x_3d = x
            added_batch = False
        else:
            raise ValueError(
                "FullPrecisionBucketResidualAdapter expects a 2-D or 3-D "
                f"activation; got shape {tuple(x.shape)}"
            )

        batch, sequence, width = x_3d.shape
        if self.one_group:
            groups = 1
            group_size = int(width)
        else:
            group_size = int(self.group_size)
            if group_size <= 0 or int(width) % group_size != 0:
                raise ValueError(
                    f"Activation width {width} must be divisible by group_size {group_size}."
                )
            groups = int(width) // group_size

        if self.table_axis == "group" and int(self.bucket_offsets.shape[0]) != groups:
            raise ValueError(
                f"Adapter table count {self.bucket_offsets.shape[0]} does not "
                f"match input groups {groups}."
            )

        x_float = x_3d.to(torch.float32)
        x_grouped = x_float.reshape(batch, sequence, groups, group_size)
        group_min = x_grouped.amin(dim=-1, keepdim=True)
        group_span = (x_grouped.amax(dim=-1, keepdim=True) - group_min).clamp(min=1e-6)
        x_normalized = ((x_grouped - group_min) / group_span).clamp(0.0, 1.0)

        # ``x`` already supplies the exact identity input gradient.  Detaching
        # the routing input prevents CustomQuantFunction's identity STE from
        # adding a second input-gradient path, while retaining offset gradients
        # and the same deferred threshold-boundary statistics as Beyond.
        selected_offsets = CustomQuantFunction.apply(
            x_normalized.detach(),
            self.bucket_offsets,
            self.thresholds,
            self.bandwidth,
            self,
        )
        adapted = x_grouped + selected_offsets
        output = adapted.reshape(batch, sequence, width).to(dtype=x.dtype)
        if added_batch:
            output = output.squeeze(0)
        return output

    def export_adapter_state(self):
        return {
            "bucket_offsets": self.bucket_offsets.detach().cpu().clone().contiguous(),
            "bucket_thresholds": self.thresholds.detach().cpu().clone().contiguous(),
            "num_offsets": int(self.num_offsets),
            "identity_initialized": True,
            "hard_bucket_assignment": True,
            "continuous_forward": False,
            "cache_storage": "model_activation_dtype",
        }

    @torch.no_grad()
    def set_bucket_residual_state(self, offsets, thresholds):
        offset_value = torch.as_tensor(
            offsets,
            device=self.bucket_offsets.device,
            dtype=self.bucket_offsets.dtype,
        )
        threshold_value = torch.as_tensor(
            thresholds,
            device=self.thresholds.device,
            dtype=self.thresholds.dtype,
        )
        if tuple(offset_value.shape) != tuple(self.bucket_offsets.shape):
            raise ValueError(
                f"bucket offset shape {tuple(offset_value.shape)} does not match "
                f"{tuple(self.bucket_offsets.shape)}"
            )
        if tuple(threshold_value.shape) != tuple(self.thresholds.shape):
            raise ValueError(
                f"bucket threshold shape {tuple(threshold_value.shape)} does not "
                f"match {tuple(self.thresholds.shape)}"
            )
        if not torch.isfinite(offset_value).all() or not torch.isfinite(threshold_value).all():
            raise ValueError("bucket residual state must be finite")
        if bool((threshold_value < 0.0).any()) or bool((threshold_value > 1.0).any()):
            raise ValueError("bucket thresholds must lie in [0, 1]")
        if threshold_value.shape[-1] > 1 and bool(
            (threshold_value[..., 1:] <= threshold_value[..., :-1]).any()
        ):
            raise ValueError("bucket thresholds must be strictly increasing")
        self.bucket_offsets.copy_(offset_value)
        self.thresholds.copy_(threshold_value)


def quantize_post_rope_k_tensor(k_rot, quantizer, unsqueeze_dim=None):
    """Apply a flattened per-token K quantizer after RoPE to a 4-D K tensor."""
    if quantizer is None or not torch.is_tensor(k_rot) or k_rot.dim() != 4:
        return k_rot
    try:
        sequence_dim = int(unsqueeze_dim)
    except (TypeError, ValueError):
        sequence_dim = 2 if int(k_rot.shape[1]) >= int(k_rot.shape[2]) else 1
    if sequence_dim == 2:
        B, S, H, D = k_rot.shape
        k_flat = k_rot.reshape(B, S, H * D)
        return quantizer(k_flat).reshape(B, S, H, D)
    if sequence_dim != 1:
        raise ValueError(f"Unsupported RoPE sequence dimension: {sequence_dim}")
    B, H, S, D = k_rot.shape
    k_flat = k_rot.transpose(1, 2).reshape(B, S, H * D)
    return quantizer(k_flat).reshape(B, S, H, D).transpose(1, 2)


def _pop_active_k_quantizer():
    quantizer = getattr(_tls, "active_k_quantizer", None)
    if quantizer is not None:
        _tls.active_k_quantizer = None
    return quantizer


def install_post_rope_k_quantization(
    model,
    *,
    quantized_linear_cls=QuantizedLinear,
    quantized_qkv_linear_cls=QuantizedQKVLinear,
    log_fn=print,
):
    """Defer K projection QDQ and install the shared post-RoPE application hook."""
    k_count = 0
    for name, module in model.named_modules():
        if isinstance(module, quantized_linear_cls) and "k_proj" in name:
            module.defer_quantize = True
            k_count += 1
        elif (
            isinstance(module, quantized_qkv_linear_cls)
            and getattr(module, "k_quantizer", None) is not None
        ):
            module.defer_k_quantize = True
            k_count += 1

    attention_layers = [
        module for module in model.modules() if type(module).__name__.endswith("Attention")
    ]
    if not attention_layers:
        raise RuntimeError("No Attention module found; cannot guarantee post-RoPE K quantization")

    architecture_module = sys.modules[type(attention_layers[0]).__module__]
    original_apply_rope = getattr(architecture_module, "apply_rotary_pos_emb", None)
    if original_apply_rope is None:
        raise RuntimeError(
            f"{architecture_module.__name__} has no apply_rotary_pos_emb; "
            "cannot guarantee post-RoPE K quantization"
        )
    if getattr(original_apply_rope, "_beyond_post_rope_quant_patch", False):
        if log_fn:
            log_fn(f"quant_after_rope: reused {architecture_module.__name__}.apply_rotary_pos_emb")
        return k_count

    default_unsqueeze_dim = None
    try:
        parameter = inspect.signature(original_apply_rope).parameters.get("unsqueeze_dim")
        if parameter is not None and parameter.default is not inspect.Parameter.empty:
            default_unsqueeze_dim = parameter.default
    except (TypeError, ValueError):
        pass

    def _rope_then_quant(*rope_args, **kwargs):
        result = original_apply_rope(*rope_args, **kwargs)
        quantizer = _pop_active_k_quantizer()
        if quantizer is None:
            return result
        unsqueeze_dim = kwargs.get("unsqueeze_dim", default_unsqueeze_dim)
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            q_rot, k_rot = result[0], result[1]
            k_rot = quantize_post_rope_k_tensor(
                k_rot,
                quantizer,
                unsqueeze_dim=unsqueeze_dim,
            )
            if isinstance(result, tuple):
                return (q_rot, k_rot, *result[2:])
            return [q_rot, k_rot, *result[2:]]
        if torch.is_tensor(result):
            return quantize_post_rope_k_tensor(
                result,
                quantizer,
                unsqueeze_dim=unsqueeze_dim,
            )
        raise RuntimeError("Unsupported apply_rotary_pos_emb signature for post-RoPE K QDQ")

    _rope_then_quant._beyond_post_rope_quant_patch = True
    _rope_then_quant._beyond_original_apply_rope = original_apply_rope
    architecture_module.apply_rotary_pos_emb = _rope_then_quant
    if log_fn:
        log_fn(f"quant_after_rope: deferred {k_count} K quantizers")
        log_fn(f"quant_after_rope: patched {architecture_module.__name__}.apply_rotary_pos_emb")
    return k_count
