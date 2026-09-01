# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: E741, F403

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""SM103 mixed-input decode kernel for independent non-uniform K/V LUTs.

The production vLLM launcher consumes block-64 or block-128 HND packed-cache
pages directly and dispatches this path on SM103. It consumes 4-bit/G32 K and V
with independent 16-entry FP16/BF16 codebooks and supports Hq32,
Hkv4-or-8, D128 decode shapes. Block 128 is preferred; block 64 is an explicit
prefix-cache specialization.
"""

import math
from functools import partial
from typing import Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op

try:
    from cutlass.cute.nvgpu import OperandMajorMode
except ImportError:  # CUTLASS DSL 4.4.x, as bundled by vLLM 0.21.
    from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import *

from quant.fa4_cute.beyond_decode_policy import (
    conversion_stage_cap,
    modeled_splits,
    packed_kv_stage_cap,
    scheduled_pages_per_split,
    select_cluster_splits,
    select_reduction_kind,
)

# Kernel invariants
mma_modes = (0, 1, 2)
mma_dice = (None, None, None)  # (MMA, #MMA_M, #MMA_K)
cpy_dice = (None,) + mma_dice  # (CPY, #CPY_MMA, #CPY_M, #CPY_K)
warp_threads = 32
warpgroup_warps = 4
warpgroup_threads = 128

# Math helpers
log2_e = math.log2(math.e)  # change exponential base
use_tensor_ssa_math = False  # experimental
fadd2 = cute.arch.add_packed_f32x2
fmul2 = cute.arch.mul_packed_f32x2
ffma2 = cute.arch.fma_packed_f32x2
exp2 = partial(cute.math.exp2, fastmath=True)
warp_fmax = partial(cute.arch.warp_redux_sync, kind="fmax", nan=True)
if hasattr(cute.arch, "atomic_fmax"):
    atomic_fmax = cute.arch.atomic_fmax
else:
    import cutlass.cutlass_dsl as cutlass_dsl

    @dsl_user_op
    def atomic_fmax(ptr, val, *, sem=None, scope=None, loc=None, ip=None):
        """CUTLASS DSL 4.4 compatibility for signed FP32 atomic max."""
        intval = llvm.bitcast(
            T.i32(), val.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
        )
        def then_body():
            return cute.arch.atomic_min(
                ptr,
                cutlass.Uint32(intval),
                sem=sem,
                scope=scope,
                loc=loc,
                ip=ip,
            )

        def else_body():
            return cute.arch.atomic_max(
                ptr,
                cutlass.Int32(intval),
                sem=sem,
                scope=scope,
                loc=loc,
                ip=ip,
            )
        old_intval = cutlass_dsl.if_generate(
            cutlass.Int32(intval) < 0,
            then_body,
            else_body,
            [],
            [cutlass.Int32],
            loc=loc,
            ip=ip,
        )
        return cutlass.Float32(
            llvm.bitcast(
                T.f32(), old_intval.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
            )
        )


@dsl_user_op
def map_cluster_smem_ptr(
    smem_ptr: cute.Pointer,
    peer_cta_rank: cutlass.Int32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Int32:
    """Map a local shared-memory address into a peer CTA in the cluster."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                smem_ptr.toint(loc=loc, ip=ip).ir_value(),
                peer_cta_rank.ir_value(),
            ],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def store_cluster_f32(
    value: cutlass.Float32,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank: cutlass.Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store FP32 into peer DSM and account for it on the peer mbarrier."""
    remote_smem = map_cluster_smem_ptr(
        smem_ptr, peer_cta_rank, loc=loc, ip=ip
    ).ir_value()
    remote_mbar = map_cluster_smem_ptr(
        mbar_ptr, peer_cta_rank, loc=loc, ip=ip
    ).ir_value()
    llvm.inline_asm(
        None,
        [remote_smem, value.ir_value(loc=loc, ip=ip), remote_mbar],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 "
        "[$0], $1, [$2];",
        "r,f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def store_cluster_i64(
    value: cutlass.Int64,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank: cutlass.Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store four packed FP16/BF16 values into peer DSM."""
    remote_smem = map_cluster_smem_ptr(
        smem_ptr, peer_cta_rank, loc=loc, ip=ip
    ).ir_value()
    remote_mbar = map_cluster_smem_ptr(
        mbar_ptr, peer_cta_rank, loc=loc, ip=ip
    ).ir_value()
    llvm.inline_asm(
        None,
        [
            remote_smem,
            remote_mbar,
            value.ir_value(loc=loc, ip=ip),
        ],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.s64 "
        "[$0], $2, [$1];",
        "r,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def store_cluster_i64x2(
    value_0: cutlass.Int64,
    value_1: cutlass.Int64,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank: cutlass.Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store eight packed FP16/BF16 values into peer DSM."""
    remote_smem = map_cluster_smem_ptr(
        smem_ptr, peer_cta_rank, loc=loc, ip=ip
    ).ir_value()
    remote_mbar = map_cluster_smem_ptr(
        mbar_ptr, peer_cta_rank, loc=loc, ip=ip
    ).ir_value()
    llvm.inline_asm(
        None,
        [
            remote_smem,
            remote_mbar,
            value_0.ir_value(loc=loc, ip=ip),
            value_1.ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        ".reg .v2 .b64 values;\n\t"
        "mov.b64 values.x, $2;\n\t"
        "mov.b64 values.y, $3;\n\t"
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b64 "
        "[$0], values, [$1];\n\t"
        "}\n",
        "r,r,l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def tensor_element_ptr(
    tensor: cute.Tensor,
    coord,
    *,
    loc=None,
    ip=None,
) -> cute.Pointer:
    """Return the shared-memory pointer for one logical tensor element."""
    return tensor.iterator + cute.crd2idx(coord, tensor.layout, loc=loc, ip=ip)


smem_fmax = partial(atomic_fmax, sem="relaxed", scope="cta")
gmem_fmax = partial(atomic_fmax, sem="relaxed", scope="gpu")


class MixedInputFusedMultiHeadAttentionDecode:
    def __init__(
        self,
        headdim,
        block_scaledim,  # headdim per scale factor; scale factor shape is (batches, heads_k, seqlen, headdim / block_scaledim)
        heads_per_kv,
        grouped_head_tile,  # GQA packing tile size, can be less than group size
        page_size=128,
        convert_warpgroups=1,  # Multiple warpgroups striding on convert stages
        reduction_kind="parallel",
        lut_value_dtype=cutlass.Float16,
        kv_stage_cap=8,
        cvt_stage_cap=8,
        single_split_direct=False,
        reduction_splits=1,
        adaptive_split_base=0,
        adaptive_extra_tasks=0,
        packed_page_stride_i32=0,
        physical_page_capacity=0,
        page_table_stride=0,
        q_batch_stride=0,
        task_major_grid=False,
        cluster_reduction=False,
        direct_smem_store=False,
        wait_for_pdl_writer=False,
    ):
        self.headdim = headdim
        self.page_size = page_size
        self.heads_per_kv = heads_per_kv
        self.grouped_head_tile = grouped_head_tile
        self.block_scaledim = block_scaledim
        self.scaledim = headdim // block_scaledim
        self.statdim = self.scaledim * 2  # interleaved (scale, minimum)
        self.convert_warpgroups = convert_warpgroups
        self.single_split_direct = single_split_direct
        self.reduction_splits = reduction_splits
        self.adaptive_split_base = adaptive_split_base
        self.adaptive_extra_tasks = adaptive_extra_tasks
        self.packed_page_stride_i32 = packed_page_stride_i32
        self.physical_page_capacity = physical_page_capacity
        self.page_table_stride = page_table_stride
        self.q_batch_stride = q_batch_stride
        self.task_major_grid = task_major_grid
        self.cluster_reduction = cluster_reduction
        self.cluster_size = reduction_splits if cluster_reduction else 1
        self.direct_smem_store = direct_smem_store
        self.wait_for_pdl_writer = wait_for_pdl_writer
        assert convert_warpgroups == 2
        assert page_size in (64, 128)
        assert 1 <= reduction_splits <= warp_threads
        assert adaptive_split_base >= 0 and adaptive_extra_tasks >= 0
        assert packed_page_stride_i32 >= 0
        assert physical_page_capacity >= 0
        assert page_table_stride >= 0
        assert q_batch_stride >= 0
        if adaptive_extra_tasks:
            assert adaptive_split_base >= 1
            assert reduction_splits == adaptive_split_base + 1
            assert not single_split_direct
        if cluster_reduction:
            assert 2 <= self.cluster_size <= 16
            assert adaptive_extra_tasks == 0
            assert not single_split_direct
            assert not task_major_grid
        if direct_smem_store:
            assert single_split_direct
        self.reduction_kind = reduction_kind
        assert reduction_kind in (
            "parallel",
            "cta4",
            "warp_parallel",
            "warp_parallel2",
        )
        self.reducer_warps_per_cta = (
            4 if reduction_kind in ("cta4", "warp_parallel2") else 1
        )
        self.lut_value_dtype = lut_value_dtype
        self.kv_stage_cap = kv_stage_cap
        self.cvt_stage_cap = cvt_stage_cap
        assert lut_value_dtype in (cutlass.Float16, cutlass.BFloat16)

        assert headdim % block_scaledim == 0
        assert grouped_head_tile % 8 == 0 and 0 < grouped_head_tile <= 32

        warpgroup_id = 0

        self.softmax_warpgroup_id = warpgroup_id
        warpgroup_id += 1

        self.cvt_warpgroup_ids = tuple(
            range(warpgroup_id, warpgroup_id + convert_warpgroups)
        )
        warpgroup_id += convert_warpgroups

        # Why 2 MMA+TMA warps when not MMA bound?
        # Less register pressure per warp promotes concise SASS
        # hides MMA 'switching' latency that gets exposed with less concise SASS
        # We would have 2 leftover warps if we do warpgroup reg realloc
        # and less register pressure gives more realloc flexibility
        self.mma_kq_warp_id = warpgroup_id * warpgroup_warps + 0
        self.mma_vp_warp_id = warpgroup_id * warpgroup_warps + 1
        self.tma_kv_warp_id = warpgroup_id * warpgroup_warps + 2
        self.tma_qo_warp_id = warpgroup_id * warpgroup_warps + 3
        self.mma_tma_warpgroup_id = warpgroup_id
        warpgroup_id += 1

        self.threads_per_cta = warpgroup_id * warpgroup_threads

        self.use_reg_reconfig = grouped_head_tile > 16
        max_regs_per_wg_thread = 64 * 1024 // warpgroup_threads  # 64K regs per SM
        self.mma_tma_regs = 72
        self.cvt_regs = 112
        self.softmax_regs = (
            max_regs_per_wg_thread
            - self.mma_tma_regs
            - self.cvt_regs * convert_warpgroups
        )
        self.softmax_regs = max(128, min(256, self.softmax_regs))
        assert (
            self.mma_tma_regs
            + self.softmax_regs
            + self.cvt_regs * convert_warpgroups
        ) <= max_regs_per_wg_thread or not self.use_reg_reconfig

        self.bs_stages = 2
        self.sp_stages = 1
        self.o_stages = 1

    def can_implement(
        self,
        problem_shape,
        kv_splits,
        q_dtype,
        kv_dtype,
        o_dtype,
        acc_dtype,
    ):
        b, h_q, h_k, s_k, d = problem_shape

        if d != 128:
            raise testing.CantImplementError(
                f"fused reduction requires headdim=128, got {d}"
            )
        if kv_splits > warp_threads:
            raise testing.CantImplementError(
                "independent reduction supports at most one split per warp lane; "
                f"got kv_splits={kv_splits}"
            )

        if kv_dtype is cutlass.Float8E4M3:
            raise ValueError("use Float8E4M3FN instead of Float8E4M3")

        if d % 64 != 0:
            raise testing.CantImplementError(f"headdim({d}) must be multiple of 64")

        if h_q % h_k != 0:
            raise testing.CantImplementError(
                f"heads_q({h_q}) must be a multiple of heads_k({h_k})"
            )
        if h_q % self.reducer_warps_per_cta != 0:
            raise testing.CantImplementError(
                "query heads must be divisible by reducer warps per CTA; "
                f"got heads_q={h_q}, warps={self.reducer_warps_per_cta}"
            )

        align_scale_bits = 128  # TMA requirement
        if self.statdim * q_dtype.width < align_scale_bits:
            align_seq = align_scale_bits // (self.statdim * q_dtype.width)
            if s_k % align_seq != 0:
                raise testing.CantImplementError(
                    f"seqlen({s_k}) must be a multiple of {align_seq}"
                )

        if kv_dtype.width < 8 and d % 128 != 0:  # TMA requirement
            raise testing.CantImplementError(
                f"headdim({d}) must be multiple of 128 for {kv_dtype} KV"
            )

        if s_k % self.page_size != 0:
            raise testing.CantImplementError(
                "paged prototype requires seqlen"
                f"({s_k}) to be a multiple of page_size={self.page_size}"
            )

    @cute.jit
    def __call__(
        self,
        problem_shape: Tuple[
            cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32
        ],  # b, h_q, h_k, s_k, d
        kv_splits: cutlass.Int32,  # threadblocks per sequence
        q_iter: cute.Pointer,
        k_iter: cute.Pointer,
        v_iter: cute.Pointer,
        k_scale_iter: cute.Pointer,
        v_scale_iter: cute.Pointer,
        k_qpoint_iter: cute.Pointer,
        v_qpoint_iter: cute.Pointer,
        page_table_iter: cute.Pointer,
        seq_lens_iter: cute.Pointer,
        o_iter: cute.Pointer,
        o_partial_iter: cute.Pointer,  # partial O per kv split
        m_partial_iter: cute.Pointer,  # partial colmax_s per kv split
        l_partial_iter: cute.Pointer,  # partial colsum_p per kv split
        scale_qs: cutlass.Float32,
        scale_o: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        ##############################
        # TiledMma creation
        ##############################
        mma_dtype = q_iter.dtype
        acc_dtype = cutlass.Float32
        assert o_partial_iter.dtype in (cutlass.Float16, cutlass.BFloat16)
        assert m_partial_iter.dtype is acc_dtype
        assert l_partial_iter.dtype is acc_dtype

        # Block tile sets the granularity at which threadblocks consume work
        blk_tile_s = self.page_size
        blk_tile_h = self.grouped_head_tile
        blk_tile_d = self.headdim
        blk_tile_shd = (blk_tile_s, blk_tile_h, blk_tile_d)

        # MMA tile sets the granularity at which TMAs + MMAs are issued
        mma_tile_n = self.grouped_head_tile
        mma_tile_mnk_kq = (blk_tile_s, mma_tile_n, self.headdim)
        mma_tile_mnk_vp = (self.headdim, mma_tile_n, blk_tile_s)

        # GEMM1: (S_K, H_R, D, (H_K, B))
        tiled_mma_kq = sm100_utils.make_trivial_tiled_mma(
            mma_dtype,
            OperandMajorMode.K,  # K
            OperandMajorMode.K,  # Q
            acc_dtype,
            tcgen05.CtaGroup.ONE,
            mma_tile_mnk_kq[:2],
            tcgen05.OperandSource.TMEM,  # converted K in tmem
        )

        # GEMM2: (D, H_R, S_K, (H_K, B))
        tiled_mma_vp = sm100_utils.make_trivial_tiled_mma(  #
            mma_dtype,
            OperandMajorMode.K,  # V
            OperandMajorMode.MN,  # P
            acc_dtype,
            tcgen05.CtaGroup.ONE,
            mma_tile_mnk_vp[:2],
            tcgen05.OperandSource.TMEM,  # converted V in tmem
        )

        # Calculate Q stages
        self.q_stages = blk_tile_d // mma_tile_mnk_kq[2]

        # Perf heuristics
        max_cvt_stages = min(
            self.cvt_stage_cap,
            4
            if self.grouped_head_tile == 32
            and mma_tile_mnk_kq[2] == 128
            else 8,
        )
        # A D128 split CTA consumes only a handful of pages. Deep Int4 staging
        # can reserve almost the entire SM without useful lookahead.
        max_kv_stages = self.kv_stage_cap

        # Calculate KV tmem stages
        tmem_alloc_cols = mma_tile_n * self.sp_stages
        tmem_alloc_cols += (
            mma_tile_n
            * self.o_stages
            * (blk_tile_d // mma_tile_mnk_vp[0])
        )
        tmem_capacity = 512
        cvt_stage_cols_k = mma_tile_mnk_kq[2] * mma_dtype.width // 32
        cvt_stage_cols_v = mma_tile_mnk_vp[2] * mma_dtype.width // 32
        self.cvt_stages = (tmem_capacity - tmem_alloc_cols) // (
            cvt_stage_cols_k + cvt_stage_cols_v
        )
        self.cvt_stages = min(self.cvt_stages, max_cvt_stages)

        tmem_alloc_cols += self.cvt_stages * (
            cvt_stage_cols_k + cvt_stage_cols_v
        )
        self.tmem_alloc_cols = 2 ** math.ceil(
            math.log2(tmem_alloc_cols)
        )  # Tmem alloc must be PO2

        print(f"\tcvt stages: {self.cvt_stages}")

        # Calculate KV smem stages
        self.mbarrier_reserved_bytes = 768
        smem_alloc_bits = self.mbarrier_reserved_bytes * 8
        smem_alloc_bits += mma_tile_n * acc_dtype.width  # colmax
        smem_alloc_bits += (
            self.statdim
            * blk_tile_s
            * self.bs_stages
            * mma_dtype.width
            * 2
        )  # independent K/V block statistics when converters overlap
        smem_alloc_bits += mma_tile_n * warpgroup_warps * acc_dtype.width  # colsum
        smem_alloc_bits += (
            mma_tile_n
            * mma_tile_mnk_kq[2]
            * self.q_stages
            * mma_dtype.width
        )  # Q
        smem_alloc_bits += (
            mma_tile_mnk_kq[0]
            * mma_tile_n
            * self.sp_stages
            * mma_dtype.width
        )  # P

        smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        kv_smem_dtype = cutlass.Int8 if k_iter.dtype.width < 8 else k_iter.dtype
        self.kv_stages = (smem_capacity * 8 - smem_alloc_bits) // (
            mma_tile_mnk_kq[0]
            * mma_tile_mnk_kq[2]
            * kv_smem_dtype.width
            * 2
        )
        self.kv_stages = min(self.kv_stages, max_kv_stages)

        print(f"\tkv stages: {self.kv_stages}")

        ##############################
        # TMA creation
        ##############################
        b, h_q, h_k, s_k, d = problem_shape
        h_r = h_q // h_k
        page_size = self.page_size
        logical_pages = s_k // page_size
        physical_pages = b * logical_pages
        if cutlass.const_expr(self.physical_page_capacity > 0):
            physical_pages = self.physical_page_capacity

        if cutlass.const_expr(self.q_batch_stride > 0):
            q_layout = cute.make_layout(
                shape=(h_r, d, (h_k, b)),
                stride=(d, 1, (h_r * d, self.q_batch_stride)),
            )
        else:
            q_layout = cute.make_ordered_layout(
                shape=(h_r, d, (h_k, b)), order=(1, 0, (2, 3))
            )
        q = cute.make_tensor(q_iter, q_layout)

        if cutlass.const_expr(self.packed_page_stride_i32 > 0):
            packed_code_page_stride = self.packed_page_stride_i32 * 8
            k_layout = cute.make_layout(
                shape=(page_size, d, h_k, physical_pages),
                stride=(d, 1, page_size * d, packed_code_page_stride),
            )
            v_layout = cute.make_layout(
                shape=(d, page_size, h_k, physical_pages),
                stride=(1, d, page_size * d, packed_code_page_stride),
            )
        else:
            k_layout = cute.make_ordered_layout(
                shape=(page_size, d, h_k, physical_pages), order=(1, 0, 2, 3)
            )
            v_layout = cute.make_ordered_layout(
                shape=(d, page_size, h_k, physical_pages), order=(0, 1, 2, 3)
            )

        k = cute.make_tensor(k_iter, k_layout)
        assert k_iter.dtype is not q_iter.dtype

        v = cute.make_tensor(v_iter, v_layout)
        assert v_iter.dtype is k_iter.dtype

        o_partial = cute.make_tensor(
            o_partial_iter,
            cute.make_ordered_layout(
                shape=(d, h_r, (h_k, b), kv_splits), order=(0, 1, (2, 3), 4)
            ),
        )
        m_partial = cute.make_tensor(
            m_partial_iter,
            cute.make_ordered_layout(
                shape=(h_r, (h_k, b), kv_splits),
                order=(0, (1, 2), 3),
            ),
        )
        assert m_partial_iter.dtype is acc_dtype

        l_partial = cute.make_tensor(
            l_partial_iter,
            cute.make_ordered_layout(
                shape=(h_r, (h_k, b), kv_splits),
                order=(0, (1, 2), 3),
            ),
        )
        assert l_partial_iter.dtype is acc_dtype
        align_scale_bits = 128  # TMA requirement
        if cutlass.const_expr(self.packed_page_stride_i32 > 0):
            scale_layout = cute.make_layout(
                shape=(self.statdim, page_size, h_k, physical_pages),
                stride=(
                    1,
                    self.statdim,
                    self.statdim * page_size,
                    self.packed_page_stride_i32 * 2,
                ),
            )
        elif cutlass.const_expr(self.statdim * mma_dtype.width >= align_scale_bits):
            scale_layout = cute.make_ordered_layout(
                shape=(self.statdim, page_size, h_k, physical_pages),
                order=(0, 1, 2, 3),
            )
        else:
            align_seq = align_scale_bits // (self.statdim * mma_dtype.width)
            page_s = (align_seq, page_size // align_seq)
            scale_layout = cute.make_ordered_layout(
                shape=(self.statdim, page_s, h_k, physical_pages),
                order=(0, (1, 2), 3, 4),
            )

        k_scale = cute.make_tensor(k_scale_iter, scale_layout)
        assert k_scale_iter.dtype is mma_dtype

        v_scale = cute.make_tensor(v_scale_iter, scale_layout)
        assert v_scale_iter.dtype is mma_dtype

        # K conversion needs all four G32 entries owned by one LUT lane, so
        # keep group contiguous and fetch them as one 64-bit vector.  V assigns
        # one group to each warp and retains code-contiguous storage.
        k_qpoint_layout = cute.make_ordered_layout(
            shape=(self.scaledim, 16, h_k), order=(0, 1, 2)
        )
        v_qpoint_layout = cute.make_ordered_layout(
            shape=(16, self.scaledim, h_k), order=(0, 1, 2)
        )
        k_qpoint = cute.make_tensor(k_qpoint_iter, k_qpoint_layout)
        v_qpoint = cute.make_tensor(v_qpoint_iter, v_qpoint_layout)
        assert k_qpoint_iter.dtype is self.lut_value_dtype
        assert v_qpoint_iter.dtype is k_qpoint_iter.dtype

        if cutlass.const_expr(self.page_table_stride > 0):
            page_table_layout = cute.make_layout(
                shape=(logical_pages, b),
                stride=(1, self.page_table_stride),
            )
        else:
            page_table_layout = cute.make_ordered_layout(
                shape=(logical_pages, b), order=(0, 1)
            )
        page_table = cute.make_tensor(page_table_iter, page_table_layout)
        assert page_table_iter.dtype is cutlass.Int32
        seq_lens = cute.make_tensor(seq_lens_iter, cute.make_layout((b,)))
        assert seq_lens_iter.dtype is cutlass.Int32

        # (MMA, MMA_M/N, MMA_K, Stages)
        smem_layout_q = sm100_utils.make_smem_layout_b(
            tiled_mma_kq, mma_tile_mnk_kq, q_iter.dtype, self.q_stages
        )
        smem_layout_k = sm100_utils.make_smem_layout_a(
            tiled_mma_kq, mma_tile_mnk_kq, kv_smem_dtype, self.kv_stages
        )
        smem_layout_v = sm100_utils.make_smem_layout_a(
            tiled_mma_vp,
            mma_tile_mnk_vp,
            kv_smem_dtype,
            self.kv_stages,
            is_k_major=False,
        )  # V is always headdim-major (GEMM2 M-major) in gmem+smem

        smem_layout_bs = cute.make_layout(
            (self.statdim, blk_tile_s, self.bs_stages)
        )

        o_store_dtype = o_iter.dtype
        assert o_partial_iter.dtype is o_store_dtype

        smem_layout_atom_o = tcgen05.make_smem_layout_atom(
            tcgen05.mma.SmemLayoutAtomKind.MN_SW128, o_store_dtype
        )
        smem_layout_o = cute.flat_divide(
            cute.tile_to_shape(
                smem_layout_atom_o,
                (blk_tile_d, blk_tile_h),
                order=(1, 0),
            ),
            (mma_tile_mnk_vp[0], mma_tile_n),
        )

        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()

        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            q,
            cute.select(smem_layout_q, mma_modes),
            mma_tile_mnk_kq,
            tiled_mma_kq,
        )
        tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            k,
            cute.select(smem_layout_k, mma_modes),
            mma_tile_mnk_kq,
            tiled_mma_kq,
            internal_type=kv_smem_dtype,
        )
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            v,
            cute.select(smem_layout_v, mma_modes),
            mma_tile_mnk_vp,
            tiled_mma_vp,
            internal_type=kv_smem_dtype,
        )
        tma_atom_ks, tma_tensor_ks = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_load_op,
            k_scale,
            cute.select(smem_layout_bs, mode=[0, 1]),
            smem_layout_bs.shape[:2],
        )
        tma_atom_vs, tma_tensor_vs = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_load_op,
            v_scale,
            cute.select(smem_layout_bs, mode=[0, 1]),
            smem_layout_bs.shape[:2],
        )
        # K scale and V scale will have the same TMA tensor (coord tensor)
        # only difference is base ptr which is stored in copy atom
        tma_tensor_bs = tma_tensor_ks
        o_final = cute.make_tensor(
            o_iter,
            cute.make_ordered_layout(
                shape=(d, h_r, (h_k, b)), order=(0, 1, (2, 3))
            ),
        )

        ##############################
        # Decode Kernel launch
        ##############################
        scale_qs_log2_e = scale_qs * log2_e

        n_tiles = cute.ceil_div(h_r, blk_tile_h)
        l_tiles = b * h_k
        # D128/GQA4-or-8 fits one grouped-query tile per KV head. Keep head and
        # batch in distinct grid axes to avoid flattened-task divide/modulo.
        grid = (kv_splits, h_k, b)
        if cutlass.const_expr(self.task_major_grid):
            assert self.adaptive_extra_tasks == 0
            # Traverse flattened (batch, KV-head) tasks before split index.
            # Packed pages place all heads of one physical page together, so
            # this order probes whether adjacent task CTAs improve TMA/L2
            # locality without changing work partitioning or arithmetic.
            grid = (l_tiles, kv_splits, n_tiles)
        if cutlass.const_expr(self.adaptive_extra_tasks > 0):
            # A linear launch lets adjacent split counts share one exact SM
            # wave without launching max_splits * tasks CTAs and spilling a
            # handful of no-op blocks into a second wave. Supported shapes fit
            # one grouped-query tile per KV head.
            adaptive_ctas = (
                self.adaptive_split_base * l_tiles
                + self.adaptive_extra_tasks
            )
            grid = (adaptive_ctas, 1, 1)
        decode_cluster = (
            [self.cluster_size, 1, 1]
            if self.cluster_reduction
            else [1, 1, 1]
        )
        self.decode(
            blk_tile_shd,
            mma_tile_mnk_kq,
            mma_tile_mnk_vp,
            tiled_mma_kq,
            tiled_mma_vp,
            q_iter.dtype,
            smem_layout_q,
            tma_atom_q,
            tma_tensor_q,
            k_iter.dtype,
            smem_layout_k,
            tma_atom_k,
            tma_tensor_k,
            v_iter.dtype,
            smem_layout_v,
            tma_atom_v,
            tma_tensor_v,
            smem_layout_bs,
            tma_atom_ks,
            tma_atom_vs,
            tma_tensor_bs,
            k_qpoint,
            v_qpoint,
            page_table,
            seq_lens,
            o_store_dtype,
            smem_layout_o,
            o_partial,
            m_partial,
            l_partial,
            o_final,
            scale_qs,
            scale_qs_log2_e,
            scale_o,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=decode_cluster,
            stream=stream,
            min_blocks_per_mp=1,
            # Production decode may be launched as a PDL consumer of the
            # immediately preceding packed-cache writer. Standalone decode
            # keeps ordinary stream ordering and pays no dependency wait.
            use_pdl=self.wait_for_pdl_writer,
        )

        if cutlass.const_expr(
            not self.single_split_direct and not self.cluster_reduction
        ):
            o_reduce = cute.make_tensor(o_iter, cute.make_layout((d, h_q, b)))
            o_partial_reduce = cute.make_tensor(
                o_partial_iter, cute.make_layout((d, h_q, b, kv_splits))
            )
            m_partial_reduce = cute.make_tensor(
                m_partial_iter, cute.make_layout((h_q, b, kv_splits))
            )
            l_partial_reduce = cute.make_tensor(
                l_partial_iter, cute.make_layout((h_q, b, kv_splits))
            )
            if cutlass.const_expr(self.reduction_kind == "parallel"):
                reduction_op = self.reduction_parallel_weights
                reduction_threads = 160
            elif cutlass.const_expr(self.reduction_kind == "cta4"):
                reduction_op = self.reduction_cta4_weights
                reduction_threads = 512
            else:
                reduction_op = self.reduction_warp_parallel_weights
                reduction_threads = warp_threads * self.reducer_warps_per_cta
            reduction_grid = [
                1,
                cute.ceil_div(h_q, self.reducer_warps_per_cta),
                b,
            ]
            reduction_op(
                o_reduce,
                o_partial_reduce,
                m_partial_reduce,
                l_partial_reduce,
                seq_lens,
                scale_o,
            ).launch(
                grid=reduction_grid,
                block=[reduction_threads, 1, 1],
                cluster=[1, 1, 1],
                stream=stream,
                use_pdl=True,
            )

    @cute.kernel
    def decode(
        self,
        # MMA
        blk_tile_shd: cute.Tile,
        mma_tile_mnk_kq: cute.Tile,
        mma_tile_mnk_vp: cute.Tile,
        tiled_mma_kq: cute.TiledMma,
        tiled_mma_vp: cute.TiledMma,
        # Q
        q_dtype: Type[cutlass.Numeric],
        smem_layout_q: cute.ComposedLayout,
        tma_atom_q: cute.CopyAtom,
        mQ: cute.Tensor,
        # K
        k_dtype: Type[cutlass.Numeric],
        smem_layout_k: cute.ComposedLayout,
        tma_atom_k: cute.CopyAtom,
        mK: cute.Tensor,
        # V
        v_dtype: Type[cutlass.Numeric],
        smem_layout_v: cute.ComposedLayout,
        tma_atom_v: cute.CopyAtom,
        mV: cute.Tensor,
        # K/V block scale
        smem_layout_bs: cute.Layout,
        tma_atom_ks: cute.CopyAtom,
        tma_atom_vs: cute.CopyAtom,
        mBS: cute.Tensor,
        mKQP: cute.Tensor,
        mVQP: cute.Tensor,
        mPT: cute.Tensor,
        mSeqLens: cute.Tensor,
        # O
        o_dtype: Type[cutlass.Numeric],
        smem_layout_o: cute.ComposedLayout,
        mO_partial: cute.Tensor,
        # Rest
        mM_partial: cute.Tensor,
        mL_partial: cute.Tensor,
        mO_final: cute.Tensor,
        scale_qs: cutlass.Float32,
        scale_qs_log2_e: cutlass.Float32,
        scale_o: cutlass.Float32,
    ):
        # Read special registers
        block_x, block_y, block_z = cute.arch.block_idx()
        kv_splits = self.reduction_splits
        kv_split_idx = block_x
        head_idx = block_y
        batch_idx = block_z
        coord_hr = 0
        coord_hb = head_idx + batch_idx * mK.shape[2]
        if cutlass.const_expr(self.task_major_grid):
            coord_hb = block_x
            kv_split_idx = block_y
            coord_hr = block_z
            head_idx = coord_hb % mK.shape[2]
            batch_idx = coord_hb // mK.shape[2]
        if cutlass.const_expr(self.adaptive_extra_tasks > 0):
            larger_splits = self.adaptive_split_base + 1
            larger_ctas = self.adaptive_extra_tasks * larger_splits
            if block_x < larger_ctas:
                coord_hb = block_x // larger_splits
                kv_split_idx = block_x % larger_splits
                kv_splits = larger_splits
            else:
                smaller_cta = block_x - larger_ctas
                coord_hb = (
                    self.adaptive_extra_tasks
                    + smaller_cta // self.adaptive_split_base
                )
                kv_split_idx = smaller_cta % self.adaptive_split_base
                kv_splits = self.adaptive_split_base
            coord_hr = 0
            head_idx = coord_hb % mK.shape[2]
            batch_idx = coord_hb // mK.shape[2]
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(tidx // warp_threads)
        warpgroup_idx = cute.arch.make_warp_uniform(tidx // warpgroup_threads)
        warpgroup_tidx = tidx % warpgroup_threads
        warpgroup_widx = warp_idx % warpgroup_warps
        init_warp = 0
        # No multicast
        mcast_coord = 0
        mcast_layout = cute.make_layout((1, 1, 1, 1))  # vmnk

        # Alias types
        mma_dtype = q_dtype
        acc_dtype = cutlass.Float32
        kv_smem_dtype = cutlass.Int8 if k_dtype.width < 8 else k_dtype

        # Shapes for MMA tile indexing (Read TMA partition for example)
        blk_tile_s, blk_tile_h, blk_tile_d = blk_tile_shd
        mma_tile_m_kq, mma_tile_n, mma_tile_k_kq = mma_tile_mnk_kq
        mma_tile_m_vp, mma_tile_n_vp, mma_tile_k_vp = mma_tile_mnk_vp
        assert mma_tile_n == mma_tile_n_vp
        tiles_dm, tiles_sk = cute.ceil_div(
            (blk_tile_d, blk_tile_s), (mma_tile_m_vp, mma_tile_k_vp)
        )
        tiles_dk, tiles_sm = cute.ceil_div(
            (blk_tile_d, blk_tile_s), (mma_tile_k_kq, mma_tile_m_kq)
        )
        seqlen_k = mSeqLens[batch_idx]
        tiles_s = cute.ceil_div(seqlen_k, blk_tile_s)
        iters_s = cute.ceil_div(tiles_s - kv_split_idx, kv_splits)
        prefetch_iters = self.sp_stages - 1
        if iters_s < prefetch_iters:
            prefetch_iters = iters_s
        assert tiles_sm == 1
        assert tiles_dm * tiles_sk == 1

        # Runtime checks
        exit_early = kv_split_idx >= tiles_s
        if cutlass.const_expr(self.wait_for_pdl_writer):
            # Splits visit pages round-robin. Only the split that owns the
            # final logical page can observe the current-token store; all
            # other CTAs may overlap historical-cache work with the writer.
            writer_split_idx = (tiles_s - 1) % kv_splits
            if not exit_early and kv_split_idx == writer_split_idx:
                cute.arch.griddepcontrol_wait()
        lane_value_idx = lane_idx
        head_lane_idx = lane_idx
        lane_store_max = mma_tile_n == warp_threads or lane_idx < mma_tile_n
        if cutlass.const_expr(self.page_size == 64):
            # Non-WS M64 maps two independent four-head score fragments onto
            # the two 16-lane halves of each warp.
            lane_value_idx = lane_idx % 16
            head_lane_idx = (
                (lane_idx // 16) * (mma_tile_n // 2) + lane_value_idx
            )
            lane_store_max = lane_value_idx < mma_tile_n // 2

        # Smem alloc helper
        svector_align = 16
        stensor_align = 128
        smem = utils.SmemAllocator()

        ##############################
        # Prefetch TMA descriptor
        ##############################
        if warp_idx == init_warp and not exit_early:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_ks)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_vs)
        init_warp += 1

        ##############################
        # Tmem Allocation
        ##############################
        tmem_ptr_smem_ptr = smem.allocate_array(cutlass.Int32)
        if warp_idx == init_warp and not exit_early:
            cute.arch.alloc_tmem(self.tmem_alloc_cols, tmem_ptr_smem_ptr)
        init_warp += 1

        ##############################
        # Pipeline Allocation + Init
        ##############################
        # Allocate Mbarriers
        q_pipeline_ptr = smem.allocate_array(cutlass.Int64, self.q_stages * 2)
        kv_pipeline_ptr = smem.allocate_array(cutlass.Int64, self.kv_stages * 2)
        v_kv_pipeline_ptr = smem.allocate_array(
            cutlass.Int64, self.kv_stages * 2
        )
        bs_pipeline_ptr = smem.allocate_array(cutlass.Int64, self.bs_stages * 2)
        v_bs_pipeline_ptr = smem.allocate_array(
            cutlass.Int64, self.bs_stages * 2
        )
        cvt_pipeline_ptr = smem.allocate_array(cutlass.Int64, self.cvt_stages * 2)
        v_cvt_pipeline_ptr = smem.allocate_array(
            cutlass.Int64, self.cvt_stages * 2
        )
        s_pipeline_ptr = smem.allocate_array(cutlass.Int64, self.sp_stages * 2)
        p_pipeline_ptr = smem.allocate_array(cutlass.Int64, self.sp_stages * 2)
        o_pipeline_ptr = smem.allocate_array(cutlass.Int64, self.o_stages * 2)
        # Declare named barriers
        softmax_nbar = pipeline.NamedBarrier(
            barrier_id=1, num_threads=warpgroup_threads
        )
        mma_kq_nbar = pipeline.NamedBarrier(barrier_id=2, num_threads=64)
        mma_vp_nbar = pipeline.NamedBarrier(barrier_id=3, num_threads=64)
        direct_epilogue_nbar = pipeline.NamedBarrier(
            barrier_id=4, num_threads=warpgroup_threads + warp_threads
        )

        # Alias thread cooperatives
        elect_one_cooperative = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        warpgroup_cooperative = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, warpgroup_threads
        )
        mma_group = elect_one_cooperative
        tma_group = elect_one_cooperative
        cvt_group = warpgroup_cooperative
        softmax_group = warpgroup_cooperative

        # Initialize pipelines
        q_producer, q_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.q_stages,
            producer_group=tma_group,
            consumer_group=softmax_group,  # Reuse Q consumer mbarriers to sync O store
            tx_count=cute.size_in_bytes(q_dtype, cute.select(smem_layout_q, mma_modes)),
            barrier_storage=q_pipeline_ptr,
            tidx=mcast_coord,
            cta_layout_vmnk=mcast_layout,
            defer_sync=True,
        ).make_participants()
        kv_producer, kv_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.kv_stages,
            producer_group=tma_group,
            consumer_group=cvt_group,
            tx_count=cute.size_in_bytes(k_dtype, cute.select(smem_layout_k, mma_modes)),
            barrier_storage=kv_pipeline_ptr,
            tidx=mcast_coord,
            cta_layout_vmnk=mcast_layout,
            defer_sync=True,
        ).make_participants()
        v_kv_producer, v_kv_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.kv_stages,
            producer_group=tma_group,
            consumer_group=cvt_group,
            tx_count=cute.size_in_bytes(
                v_dtype, cute.select(smem_layout_v, mma_modes)
            ),
            barrier_storage=v_kv_pipeline_ptr,
            tidx=mcast_coord,
            cta_layout_vmnk=mcast_layout,
            defer_sync=True,
        ).make_participants()
        bs_producer, bs_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.bs_stages,
            producer_group=tma_group,
            consumer_group=cvt_group,
            tx_count=cute.size_in_bytes(
                mma_dtype, cute.select(smem_layout_bs, mode=[0, 1])
            ),
            barrier_storage=bs_pipeline_ptr,
            tidx=mcast_coord,
            cta_layout_vmnk=mcast_layout,
            defer_sync=True,
        ).make_participants()
        v_bs_producer, v_bs_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.bs_stages,
            producer_group=tma_group,
            consumer_group=cvt_group,
            tx_count=cute.size_in_bytes(
                mma_dtype, cute.select(smem_layout_bs, mode=[0, 1])
            ),
            barrier_storage=v_bs_pipeline_ptr,
            tidx=mcast_coord,
            cta_layout_vmnk=mcast_layout,
            defer_sync=True,
        ).make_participants()
        cvt_producer, cvt_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.cvt_stages,
            producer_group=cvt_group,
            consumer_group=mma_group,
            barrier_storage=cvt_pipeline_ptr,
            defer_sync=True,
        ).make_participants()
        v_cvt_producer, v_cvt_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.cvt_stages,
            producer_group=cvt_group,
            consumer_group=mma_group,
            barrier_storage=v_cvt_pipeline_ptr,
            defer_sync=True,
        ).make_participants()
        s_producer, s_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.sp_stages,
            producer_group=mma_group,
            consumer_group=softmax_group,
            barrier_storage=s_pipeline_ptr,
            defer_sync=True,
        ).make_participants()
        p_producer, p_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.sp_stages,
            producer_group=softmax_group,
            consumer_group=mma_group,
            barrier_storage=p_pipeline_ptr,
            defer_sync=True,
        ).make_participants()
        o_producer, o_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.o_stages,
            producer_group=mma_group,
            consumer_group=softmax_group,
            barrier_storage=o_pipeline_ptr,
            defer_sync=True,
        ).make_participants()

        # Ensure visibility of local mbarrier inits and tmem alloc
        cute.arch.sync_threads()

        ##############################
        # MMA Partition + Allocate
        ##############################
        # Threadblock slice
        thrblk_mma_kq = tiled_mma_kq.get_slice(0)
        thrblk_mma_vp = tiled_mma_vp.get_slice(0)

        # M - colmax
        sM_layout = cute.make_layout(
            shape=(mma_tile_m_kq, mma_tile_n), stride=(0, 1)
        )
        sM = smem.allocate_tensor(acc_dtype, sM_layout, svector_align)
        tCsM = thrblk_mma_kq.partition_C(sM)

        # L - colsum
        sL_layout = cute.make_layout(
            shape=(mma_tile_m_kq, mma_tile_n, warpgroup_warps),
            stride=(0, 1, mma_tile_n),
        )
        sL = smem.allocate_tensor(acc_dtype, sL_layout, svector_align)
        tCsL = thrblk_mma_kq.partition_C(sL)

        if cutlass.const_expr(self.cluster_reduction):
            # Publish split outputs at the runtime dtype. This is the same
            # rounding boundary as the independent split workspace, while a
            # packed 64-bit cluster store carries four adjacent heads.
            cluster_o_dtype = o_dtype
            s_cluster_o = smem.allocate_tensor(
                cluster_o_dtype,
                cute.make_layout(
                    shape=(blk_tile_d, mma_tile_n, self.cluster_size),
                    stride=(
                        mma_tile_n,
                        1,
                        blk_tile_d * mma_tile_n,
                    ),
                ),
                stensor_align,
            )
            s_cluster_m = smem.allocate_tensor(
                acc_dtype,
                cute.make_layout((mma_tile_n, self.cluster_size)),
                svector_align,
            )
            s_cluster_l = smem.allocate_tensor(
                acc_dtype,
                cute.make_layout((mma_tile_n, self.cluster_size)),
                svector_align,
            )
            s_cluster_weights = smem.allocate_tensor(
                acc_dtype,
                cute.make_layout((mma_tile_n, self.cluster_size + 1)),
                svector_align,
            )
            cluster_mbar_ptr = smem.allocate_array(
                cutlass.Int64, num_elems=1
            )

        # BS - block scale
        sBS = smem.allocate_tensor(
            mma_dtype, smem_layout_bs, stensor_align
        )  # (SCALE, TILE_S, bs_stages)
        sVBS = smem.allocate_tensor(
            mma_dtype, smem_layout_bs, stensor_align
        )

        # Q
        tBsQ = smem.allocate_tensor(
            q_dtype, smem_layout_q.outer, stensor_align, smem_layout_q.inner
        )  # (MMA, #MMA_N, #MMA_K, q_stages)

        # K
        tAsK = smem.allocate_tensor(
            kv_smem_dtype, smem_layout_k.outer, stensor_align, smem_layout_k.inner
        )  # (MMA, #MMA_M, #MMA_K, kv_stages)
        tAtK_cvt_shape = tiled_mma_kq.partition_shape_A(
            (mma_tile_m_kq, mma_tile_k_kq, self.cvt_stages)
        )  # (MMA, #MMA_M, #MMA_K, cvt_stages)
        tAtK_cvt = thrblk_mma_kq.make_fragment_A(tAtK_cvt_shape)

        # V
        tAsV = smem.allocate_tensor(
            kv_smem_dtype,
            smem_layout_v.outer,
            stensor_align,
            smem_layout_v.inner,
        )
        tAtV_cvt_shape = tiled_mma_vp.partition_shape_A(
            (mma_tile_m_vp, mma_tile_k_vp, self.cvt_stages)
        )  # (MMA, #MMA_M, #MMA_K, cvt_stages)
        tAtV_cvt = thrblk_mma_vp.make_fragment_A(tAtV_cvt_shape)

        # S
        tCtS_shape = tiled_mma_kq.partition_shape_C(
            (mma_tile_m_kq, mma_tile_n, self.sp_stages)
        )
        tCtS = thrblk_mma_kq.make_fragment_C(
            tCtS_shape
        )  # (MMA_MN, #MMA_M=1, #MMA_N=1, sp_stages)

        # P - Treat MN C tile of BMM0 as NM B tile of BMM1
        # (MMA_NK, #MMA_N, #MMA_K=MMA_TILE_M/MMA_K, sp_stages)
        mma_tile_nm = (None, mma_tile_n, mma_tile_m_kq)
        tBsP_nm_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_vp, mma_tile_nm, mma_dtype, self.sp_stages
        )
        tBsP_nm = smem.allocate_tensor(
            mma_dtype, tBsP_nm_layout.outer, stensor_align, tBsP_nm_layout.inner
        )

        # Tile for NK B tile iteration
        # (MMA_NK, #MMA_N, #MMA_K=MMA_TILE_K/MMA_K, #TILES_SK=MMA_TILE_M/MMA_TILE_K, sp_stages)
        tBsP_nk_tile = thrblk_mma_vp.partition_shape_B(
            (mma_tile_n, mma_tile_k_vp)
        )
        tBsP_nk = cute.local_tile(tBsP_nm, tBsP_nk_tile, (0, 0, None, None))

        # Reshape NM B tile of BMM1 to become MN C tile of BMM0
        # (MMA_NK, #MMA_N, #MMA_K=MMA_TILE_M/MMA_K, sp_stages) ->
        # (MMA_MN, #MMA_M, #MMA_N, sp_stages)
        tCsP_tile = cute.make_ordered_layout(tCtS_shape, order=((2, 0), 3, 1, 4))
        tCsP = cute.composition(tBsP_nm, tCsP_tile)

        # O
        sO_iterator = cute.recast_ptr(
            tBsQ.iterator, smem_layout_o.inner, dtype=o_dtype
        )  # Reuse QKV smem for O TMA store
        sO_mma = cute.make_tensor(
            sO_iterator, smem_layout_o.outer
        )  # (MMA_TILE_M, MMA_TILE_N, #TILE_DM, #TILE_HN)
        tCsO = thrblk_mma_vp.partition_C(
            sO_mma
        )  # (MMA, #MMA_M, #MMA_N, #TILE_DM, #TILE_HN)
        tCtO = thrblk_mma_vp.make_fragment_C(tCsO.shape)

        # Tmem tensor allocation
        tmem_ptr = cute.arch.retrieve_tmem_ptr(cutlass.Int32, 16, tmem_ptr_smem_ptr)
        tmem_offset = 0

        tAtK_cvt = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=mma_dtype), tAtK_cvt.layout
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tAtK_cvt)
        tAtV_cvt = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=mma_dtype),
            tAtV_cvt.layout,
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tAtV_cvt)

        tCtS = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype), tCtS.layout
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtS)

        tCtO = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype), tCtO.layout
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtO)

        print(
            f"\t{tmem_offset} tmem cols used, {self.tmem_alloc_cols} tmem cols allocated"
        )
        assert tmem_offset <= self.tmem_alloc_cols

        if cutlass.const_expr(self.cluster_reduction):
            cluster_rank = cute.arch.block_idx_in_cluster()
            cluster_slot = cluster_rank
            if cluster_rank == 0 and tidx == 0:
                cute.arch.mbarrier_init(cluster_mbar_ptr, 1)
            cute.arch.mbarrier_init_fence()

            # Only rank 0 owns the receive barrier. Arm its exact transaction
            # count before arriving at the cluster barrier; the barrier then
            # publishes both initialization and expectation to every peer.
            if cluster_rank == 0 and tidx == 0:
                cluster_o_bytes = (
                    blk_tile_d * self.heads_per_kv * o_dtype.width // 8
                )
                expected_bytes = self.cluster_size * (
                    cluster_o_bytes + 2 * self.heads_per_kv * 4
                )
                cute.arch.mbarrier_arrive_and_expect_tx(
                    cluster_mbar_ptr, expected_bytes
                )
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        ##############################
        # Exit early
        ##############################
        if exit_early:
            if cutlass.const_expr(self.cluster_reduction):
                if tidx < blk_tile_d:
                    for head_group in cutlass.range_constexpr(
                        self.heads_per_kv // 4
                    ):
                        if cutlass.const_expr(self.heads_per_kv == 4):
                            store_cluster_i64(
                                cutlass.Int64(0),
                                tensor_element_ptr(
                                    s_cluster_o,
                                    (tidx, head_group * 4, cluster_slot),
                                ),
                                cluster_mbar_ptr,
                                cutlass.Int32(0),
                            )
                        else:
                            store_cluster_i64x2(
                                cutlass.Int64(0),
                                cutlass.Int64(0),
                                tensor_element_ptr(
                                    s_cluster_o,
                                    (tidx, head_group * 4, cluster_slot),
                                ),
                                cluster_mbar_ptr,
                                cutlass.Int32(0),
                            )
                if tidx < self.heads_per_kv:
                    store_cluster_f32(
                        cutlass.Float32(-math.inf),
                        tensor_element_ptr(
                            s_cluster_m, (tidx, cluster_slot)
                        ),
                        cluster_mbar_ptr,
                        cutlass.Int32(0),
                    )
                    store_cluster_f32(
                        cutlass.Float32(0.0),
                        tensor_element_ptr(
                            s_cluster_l, (tidx, cluster_slot)
                        ),
                        cluster_mbar_ptr,
                        cutlass.Int32(0),
                    )
            noop = None  # early return not supported # noqa: F841

        ##############################
        # TMA KV Dispatch
        ##############################
        elif warp_idx == self.tma_kv_warp_id:
            # Free registers
            if cutlass.const_expr(self.use_reg_reconfig):
                cute.arch.setmaxregister_decrease(self.mma_tma_regs)

            # Apply block tiler and slice
            gK = cute.local_tile(
                mK, tiler=(blk_tile_s, blk_tile_d), coord=(0, 0, head_idx, None)
            )  # (TILE_S, TILE_D, #PHYSICAL_PAGE)
            gV = cute.local_tile(
                mV, tiler=(blk_tile_d, blk_tile_s), coord=(0, 0, head_idx, None)
            )  # (TILE_D, TILE_S, #PHYSICAL_PAGE)

            # Apply MMA tiler and MMA partition
            gK_mma = cute.flat_divide(
                gK, (mma_tile_m_kq, mma_tile_k_kq)
            )  # (MMA_TILE_M, MMA_TILE_K, #TILE_SM, #TILE_DK, #TILE_S)
            gV_mma = cute.flat_divide(
                gV, (mma_tile_m_vp, mma_tile_k_vp)
            )  # (MMA_TILE_M, MMA_TILE_K, #TILE_DM, #TILE_SK, #TILE_S)
            tAgK = thrblk_mma_kq.partition_A(
                gK_mma
            )  # (MMA, #MMA_M, #MMA_K, #TILE_SM, #TILE_DK, #TILE_S)
            tAgV = thrblk_mma_vp.partition_A(
                gV_mma
            )  # (MMA, #MMA_M, #MMA_K, #TILE_DM, #TILE_SK, #TILE_S)

            # #TILE_SM=TILE_S/MMA_TILE_M, #TILE_HN=TILE_H/MMA_TILE_N, #TILE_DK=TILE_D/MMA_TILE_K
            # #TILE_DM=TILE_D/MMA_TILE_M, #TILE_HN=TILE_H/MMA_TILE_N, #TILE_SK=TILE_S/MMA_TILE_K
            #
            # Example with TILE_S=MMA_TILE_M=128, TILE_H=MMA_TILE_N=8, MMA_TILE_K=64, TILE_D=512
            # BMM1: MMA=128x8x16, #MMA_M=1, #MMA_N=1, #MMA_K=4, #TILE_SM=1, #TILE_HN=1, #TILE_DK=8, #TILE_S=S/128
            # BMM2: MMA=128x8x16, #MMA_M=1, #MMA_N=1, #MMA_K=4, #TILE_DM=4, #TILE_HN=1, #TILE_SK=2, #TILE_S=S/128

            # TMA partition
            # (MMA, #MMA_M, #MMA_K, Rest...) -> (TMA, Rest...)
            tGSsK, tGSgK = cute.nvgpu.cpasync.tma_partition(
                tma_atom_k,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(tAsK, 0, 3),
                gmem_tensor=cute.group_modes(tAgK, 0, 3),
            )

            tGSsV, tGSgV = cute.nvgpu.cpasync.tma_partition(
                tma_atom_v,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(tAsV, 0, 3),
                gmem_tensor=cute.group_modes(tAgV, 0, 3),
            )

            #
            # Sequence loop
            #
            for s in cutlass.range(kv_split_idx, tiles_s, kv_splits):
                physical_page = mPT[s, batch_idx]
                tGSgK_s = tGSgK[None, None, None, physical_page]
                for dk in cutlass.range_constexpr(tiles_dk):
                    k_handle = kv_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_k,
                        tGSgK_s[None, 0, dk],
                        tGSsK[None, k_handle.index],
                        tma_bar_ptr=k_handle.barrier,
                    )
                tGSgV_s = tGSgV[None, None, None, physical_page]
                for sk in cutlass.range_constexpr(tiles_sk):
                    for dm in cutlass.range_constexpr(tiles_dm):
                        v_handle = v_kv_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_v,
                            tGSgV_s[None, dm, sk],
                            tGSsV[None, v_handle.index],
                            tma_bar_ptr=v_handle.barrier,
                        )

        ##############################
        # TMA QO Dispatch
        ##############################
        elif warp_idx == self.tma_qo_warp_id:
            # Free registers
            if cutlass.const_expr(self.use_reg_reconfig):
                cute.arch.setmaxregister_decrease(self.mma_tma_regs)

            # Apply block tiler and slice
            gQ = cute.local_tile(
                mQ, tiler=(blk_tile_h, blk_tile_d), coord=(coord_hr, 0, coord_hb)
            )  # (TILE_H, TILE_D)
            gBS = cute.local_tile(
                mBS,
                tiler=(self.statdim, blk_tile_s),
                coord=(0, 0, head_idx, None),
            )  # (SCALE, TILE_S, #PHYSICAL_PAGE)

            # Apply MMA tiler and MMA partition
            gQ_mma = cute.flat_divide(
                gQ, (mma_tile_n, mma_tile_k_kq)
            )  # (MMA_TILE_N, MMA_TILE_K, #TILE_HN, #TILE_DK)
            tBgQ = thrblk_mma_kq.partition_B(
                gQ_mma
            )  # (MMA, #MMA_N, #MMA_K, #TILE_HN, #TILE_DK)

            # TMA partition
            tGSsQ, tGSgQ = cute.nvgpu.cpasync.tma_partition(
                tma_atom_q,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(tBsQ, 0, 3),
                gmem_tensor=cute.group_modes(tBgQ, 0, 3),
            )

            tGSsBS, tGSgBS = cute.nvgpu.cpasync.tma_partition(
                tma_atom_ks,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(sBS, 0, 2),
                gmem_tensor=cute.group_modes(gBS, 0, 2),
            )
            tGSsVBS, _ = cute.nvgpu.cpasync.tma_partition(
                tma_atom_vs,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(sVBS, 0, 2),
                gmem_tensor=cute.group_modes(gBS, 0, 2),
            )

            # K statistics gate the first dequantized K tile. Issue that small
            # transfer before Q so its latency overlaps the Q TMA instead of
            # extending the decode startup critical path.
            first_physical_page = mPT[kv_split_idx, batch_idx]
            first_bs_handle = bs_producer.acquire_and_advance()
            cute.copy(
                tma_atom_ks,
                tGSgBS[None, first_physical_page],
                tGSsBS[None, first_bs_handle.index],
                tma_bar_ptr=first_bs_handle.barrier,
            )

            # Load Q
            for dk in cutlass.range_constexpr(tiles_dk):
                q_handle = q_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_q,
                    tGSgQ[None, 0, dk],  # stages_q == tiles_dk by construction
                    tGSsQ[None, dk],
                    tma_bar_ptr=q_handle.barrier,
                )

            # Sequence Loop
            for s in cutlass.range(kv_split_idx, tiles_s, kv_splits):
                physical_page = mPT[s, batch_idx]
                if s != kv_split_idx:
                    bs_handle = bs_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_ks,
                        tGSgBS[None, physical_page],
                        tGSsBS[None, bs_handle.index],
                        tma_bar_ptr=bs_handle.barrier,
                    )
                v_bs_handle = v_bs_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_vs,
                    tGSgBS[None, physical_page],
                    tGSsVBS[None, v_bs_handle.index],
                    tma_bar_ptr=v_bs_handle.barrier,
                )

            # Pace the external PDL reducer with the final output store.
            if cutlass.const_expr(not self.single_split_direct):
                for dm in cutlass.range_constexpr(tiles_dm):
                    q_producer.acquire_and_advance()
            elif cutlass.const_expr(self.direct_smem_store):
                direct_epilogue_nbar.arrive_and_wait()
                for direct_head in cutlass.range_constexpr(self.heads_per_kv):
                    sO_head = sO_mma[None, direct_head, 0, 0]
                    sO_vec = cute.local_tile(
                        sO_head, (4,), (lane_idx,)
                    )
                    direct_o_vec = cute.make_rmem_tensor((4,), o_dtype)
                    cute.autovec_copy(sO_vec, direct_o_vec)
                    gO_head = mO_final[None, direct_head, coord_hb]
                    gO_vec = cute.local_tile(
                        gO_head, (4,), (lane_idx,)
                    )
                    cute.autovec_copy(direct_o_vec, gO_vec)

        ##############################
        # Convert Dispatch
        ##############################
        elif warpgroup_idx in self.cvt_warpgroup_ids:
            convert_phase = (
                warpgroup_idx - self.cvt_warpgroup_ids[0]
            ) % self.convert_warpgroups
            assert tiles_dk == 1
            assert tiles_dm == 1

            # Free registers
            if cutlass.const_expr(self.use_reg_reconfig):
                cute.arch.setmaxregister_decrease(self.cvt_regs)

            # Intermediate convert type
            cvt_type = cutlass.Float32
            if cutlass.const_expr(
                mma_dtype in (cutlass.Float16, cutlass.BFloat16)
                and k_dtype in (cutlass.Int4, cutlass.Int8)
            ):
                cvt_type = mma_dtype

            v_output_producer = v_cvt_producer
            v_input_consumer = v_kv_consumer
            v_stats_consumer = v_bs_consumer

            # Construct tiled copy and partition K
            mma_k_bits = mma_tile_k_kq * mma_dtype.width
            tmem_store_atom_k = cute.make_copy_atom(
                tcgen05.St16x256bOp(tcgen05.Repetition(mma_k_bits // 256)),
                mma_dtype,
            )
            smem_load_atom_k = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x16x8bOp(
                    num_matrices=mma_tile_m_kq // 32,
                    unpack_bits=(k_dtype.width if k_dtype.width < 8 else None),
                ),
                kv_smem_dtype,
            )
            tmem_store_k = tcgen05.make_tmem_copy(
                tmem_store_atom_k, tAtK_cvt[mma_dice + (0,)]
            )
            thr_store_k = tmem_store_k.get_slice(warpgroup_tidx)
            tKrK_cvt_shape = thr_store_k.partition_S(tAtK_cvt).shape[:-1]
            tKtK_cvt = thr_store_k.partition_D(tAtK_cvt)

            smem_load_k = cute.make_tiled_copy_S(smem_load_atom_k, tmem_store_k)
            thr_load_k = smem_load_k.get_slice(warpgroup_tidx)
            tKsK = thr_load_k.partition_S(tAsK)
            tKrK_shape = thr_load_k.partition_D(tAsK).shape[:-1]
            assert cute.size(tKrK_shape) % 4 == 0

            # Construct tiled copy and partition V
            mma_v_bits = mma_tile_k_vp * mma_dtype.width
            tmem_store_atom_v = cute.make_copy_atom(
                tcgen05.St16x256bOp(tcgen05.Repetition(mma_v_bits // 256)),
                mma_dtype,
            )
            smem_load_op_v = cute.nvgpu.warp.LdMatrix16x16x8bOp(
                transpose=True,
                num_matrices=mma_tile_k_vp // 64,
                unpack_bits=(v_dtype.width if v_dtype.width < 8 else None),
            )
            smem_load_atom_v = cute.make_copy_atom(smem_load_op_v, kv_smem_dtype)

            tmem_store_v = tcgen05.make_tmem_copy(
                tmem_store_atom_v, tAtV_cvt[mma_dice + (0,)]
            )
            thr_store_v = tmem_store_v.get_slice(warpgroup_tidx)
            tVrV_cvt_shape = thr_store_v.partition_S(tAtV_cvt).shape[:-1]
            tVtV_cvt = thr_store_v.partition_D(tAtV_cvt)

            smem_load_v = cute.make_tiled_copy_S(smem_load_atom_v, tmem_store_v)
            thr_load_v = smem_load_v.get_slice(warpgroup_tidx)
            tVsV = thr_load_v.partition_S(tAsV)
            tVrV_shape = thr_load_v.partition_D(tAsV).shape[:-1]

            # Partition interleaved K (scale, minimum) statistics.
            sKS_layout = cute.make_layout(
                shape=(
                    blk_tile_s,
                    (self.block_scaledim, self.scaledim),
                    self.bs_stages,
                ),
                stride=(self.statdim, (0, 2), blk_tile_s * self.statdim),
            )
            sKS = cute.make_tensor(sBS.iterator, sKS_layout)
            sKM = cute.make_tensor(sBS.iterator + 1, sKS_layout)
            sKS_mma = cute.group_modes(
                cute.flat_divide(sKS, (mma_tile_m_kq, mma_tile_k_kq)),
                2,
                4,
            )  # (MMA_TILE_M, MMA_TILE_K, (#TILE_SM, #TILE_DK), bs_stages)
            sKM_mma = cute.group_modes(
                cute.flat_divide(sKM, (mma_tile_m_kq, mma_tile_k_kq)),
                2,
                4,
            )
            tAsKS = thrblk_mma_kq.partition_A(
                sKS_mma
            )  # (MMA, #MMA_M, #MMA_K, (#TILE_SM, #TILE_DK), bs_stages)
            tAsKM = thrblk_mma_kq.partition_A(sKM_mma)
            tKsKS = thr_load_k.partition_D(
                tAsKS
            )  # (CPY, CPY_MMA, CPY_M, CPY_K, #TILE, bs_stages)
            tKsKM = thr_load_k.partition_D(tAsKM)

            # Partition interleaved V (scale, minimum) statistics.
            sVS_layout = cute.make_layout(
                shape=(
                    (self.block_scaledim, self.scaledim),
                    blk_tile_s,
                    self.bs_stages,
                ),
                # V statistics are group-major: for one G32 table, all token
                # (scale, minimum) pairs are contiguous. K stays token-major.
                stride=((0, 2 * blk_tile_s), 2, blk_tile_s * self.statdim),
            )
            sVS = cute.make_tensor(sVBS.iterator, sVS_layout)
            sVM = cute.make_tensor(sVBS.iterator + 1, sVS_layout)
            sVS_mma = cute.group_modes(
                cute.flat_divide(sVS, (mma_tile_m_vp, mma_tile_k_vp)),
                2,
                4,
            )  # (MMA_TILE_M, MMA_TILE_K, (#TILE_DM, #TILE_SK), bs_stages)
            sVM_mma = cute.group_modes(
                cute.flat_divide(sVM, (mma_tile_m_vp, mma_tile_k_vp)),
                2,
                4,
            )
            tAsVS = thrblk_mma_vp.partition_A(
                sVS_mma
            )  # (MMA, #MMA_M, #MMA_K, (#TILE_DM, #TILE_SK), bs_stages)
            tAsVM = thrblk_mma_vp.partition_A(sVM_mma)
            tVsVS = thr_load_v.partition_D(
                tAsVS
            )  # (CPY, CPY_MMA, CPY_M, CPY_K, #TILE, bs_stages)
            tVsVM = thr_load_v.partition_D(tAsVM)
            # Hoist each lane-owned LUT entry as a duplicated runtime-dtype
            # pair. Two independent shuffles can then be joined directly
            # into one FP16x2/BF16x2 register with a constant PRMT.
            k_lane_qpoints = cute.make_rmem_tensor(
                (self.scaledim,), cutlass.Int32
            )
            k_lane_qpoints.fill(cutlass.Int32(0))
            v_lane_qpoint = cutlass.Int32(0)
            # Codes are unsigned nibbles, so warp shuffle sources are
            # always lanes 0..15. Lanes 16..31 never serve as a source.
            if lane_idx < 16:
                if convert_phase == 0:
                    for group_idx in cutlass.range_constexpr(
                        self.scaledim
                    ):
                        qpoint_pair = cute.make_rmem_tensor((2,), mma_dtype)
                        qpoint = mma_dtype(
                            mKQP[group_idx, lane_idx, head_idx]
                        )
                        qpoint_pair[0] = qpoint
                        qpoint_pair[1] = qpoint
                        k_lane_qpoints[group_idx] = cute.recast_tensor(
                            qpoint_pair, cutlass.Int32
                        ).load()[0]
                if convert_phase == 1:
                    qpoint_pair = cute.make_rmem_tensor((2,), mma_dtype)
                    qpoint = mma_dtype(
                        mVQP[lane_idx, warpgroup_widx, head_idx]
                    )
                    qpoint_pair[0] = qpoint
                    qpoint_pair[1] = qpoint
                    v_lane_qpoint = cute.recast_tensor(
                        qpoint_pair, cutlass.Int32
                    ).load()[0]

            #
            # Sequence loop
            #
            for s in cutlass.range(prefetch_iters + iters_s):
                if s < iters_s:
                    do_k_convert = convert_phase == 0
                    if do_k_convert:
                        # Load K scale
                        bs_handle = bs_consumer.wait_and_advance()
                        tKrKS = cute.make_rmem_tensor_like(
                            tKsKS[cpy_dice + (None, 0)]
                        )  # 'like' preserves 0 strides
                        tKrKM = cute.make_rmem_tensor_like(
                            tKsKM[cpy_dice + (None, 0)]
                        )
                        cute.autovec_copy(
                            tKsKS[cpy_dice + (None, bs_handle.index)], tKrKS
                        )
                        cute.autovec_copy(
                            tKsKM[cpy_dice + (None, bs_handle.index)], tKrKM
                        )
                        bs_handle.release()

                        # Convert and scale K.
                        for dk in cutlass.range(tiles_dk, unroll=2):
                            tKrK = cute.make_rmem_tensor(tKrK_shape, kv_smem_dtype)
                            tKrK_cvt = cute.make_rmem_tensor(
                                tKrK_cvt_shape, mma_dtype
                            )

                            kv_handle = kv_consumer.wait_and_advance()
                            cute.copy(
                                thr_load_k,
                                tKsK[cpy_dice + (kv_handle.index,)],
                                tKrK,
                            )
                            kv_handle.release()

                            # The LUT consumes the raw unsigned nibble.  The copy
                            # fragment has already unpacked each Int4 element into
                            # the low nibble of an Int8 lane, so sign extension is
                            # unnecessary (and would be masked off immediately).

                            scale_k_tensor = tKrKS[cpy_dice + (dk,)]
                            minimum_k_tensor = tKrKM[cpy_dice + (dk,)]
                            scale_k = scale_k_tensor.load()
                            minimum_k = minimum_k_tensor.load()
                            packed_codes = cute.recast_tensor(
                                tKrK, cutlass.Int32
                            ).load()
                            qvals = cute.make_rmem_tensor(
                                (cute.size(tKrK) // 2,), cutlass.Int32
                            )
                            elems_per_group = cute.size(tKrK) // (
                                self.scaledim * 2
                            )
                            for pair_idx in cutlass.range_constexpr(
                                cute.size(tKrK) // 2
                            ):
                                i = pair_idx * 2
                                packed_code = packed_codes[pair_idx // 2]
                                code_0 = packed_code >> ((pair_idx % 2) * 16)
                                code_1 = packed_code >> (
                                    (pair_idx % 2) * 16 + 8
                                )
                                if cutlass.const_expr(self.page_size == 64):
                                    # M64 LdMatrix local K coordinates are
                                    # ((a4,b2),c8): consecutive c pairs map
                                    # to G32 tables 0,0,1,1,2,2,3,3.
                                    elems_per_group_m64 = (
                                        cute.size(tKrK) // self.scaledim
                                    )
                                    group_0 = i // elems_per_group_m64
                                    group_1 = (
                                        i + 1
                                    ) // elems_per_group_m64
                                else:
                                    group_0 = (
                                        i // elems_per_group
                                    ) % self.scaledim
                                    group_1 = (
                                        (i + 1) // elems_per_group
                                    ) % self.scaledim
                                qpoint_0 = cute.arch.shuffle_sync(
                                    k_lane_qpoints[group_0], code_0
                                )
                                qpoint_1 = cute.arch.shuffle_sync(
                                    k_lane_qpoints[group_1],
                                    code_1,
                                )
                                qvals[pair_idx] = cute.arch.prmt(
                                    qpoint_0, qpoint_1, 0x5410
                                )
                            qvals_ssa = cute.recast_tensor(
                                qvals, mma_dtype
                            ).load().reshape(tKrK_shape)
                            if cutlass.const_expr(self.page_size == 64):
                                # Keep the grouped zero-stride statistics'
                                # tuple profile; flattening is numerically
                                # wrong for the M64 MMA fragment.
                                qvals_ssa = qvals_ssa.reshape(
                                    scale_k_tensor.shape
                                )
                            tKrK_ssa = (
                                qvals_ssa.to(cvt_type).to(mma_dtype) * scale_k
                                + minimum_k
                            )
                            tKrK_cvt.store(
                                tKrK_ssa.reshape(tKrK_cvt_shape)
                            )
                            cvt_handle = cvt_producer.acquire_and_advance()
                            cute.copy(
                                thr_store_k,
                                tKrK_cvt,
                                tKtK_cvt[cpy_dice + (cvt_handle.index,)],
                            )
                            cute.arch.fence_view_async_tmem_store()
                            cvt_handle.commit()

                if s >= prefetch_iters:
                    do_v_convert = convert_phase == 1
                    if do_v_convert:
                        # Load V scale
                        bs_handle = v_stats_consumer.wait_and_advance()
                        tVrVS = cute.make_rmem_tensor_like(
                            tVsVS[cpy_dice + (None, 0)]
                        )  # 'like' preserves 0 strides
                        tVrVM = cute.make_rmem_tensor_like(
                            tVsVM[cpy_dice + (None, 0)]
                        )
                        cute.autovec_copy(
                            tVsVS[cpy_dice + (None, bs_handle.index)], tVrVS
                        )
                        cute.autovec_copy(
                            tVsVM[cpy_dice + (None, bs_handle.index)], tVrVM
                        )
                        bs_handle.release()

                        # Convert and scale V
                        for dmsk in cutlass.range(
                            tiles_dm * tiles_sk, unroll=2
                        ):
                            tVrV = cute.make_rmem_tensor(tVrV_shape, kv_smem_dtype)
                            tVrV_cvt = cute.make_rmem_tensor(
                                tVrV_cvt_shape, mma_dtype
                            )

                            kv_handle = v_input_consumer.wait_and_advance()
                            cute.copy(
                                thr_load_v,
                                tVsV[cpy_dice + (kv_handle.index,)],
                                tVrV,
                            )
                            kv_handle.release()

                            # As for K, decode the copied Int4 lane as an
                            # unsigned low nibble; no signed conversion is needed.

                            scale_v_tensor = tVrVS[cpy_dice + (dmsk,)]
                            minimum_v_tensor = tVrVM[cpy_dice + (dmsk,)]
                            scale_v = scale_v_tensor.load()
                            minimum_v = minimum_v_tensor.load()
                            packed_codes = cute.recast_tensor(
                                tVrV, cutlass.Int32
                            ).load()
                            qvals = cute.make_rmem_tensor(
                                (cute.size(tVrV) // 2,), cutlass.Int32
                            )
                            for pair_idx in cutlass.range_constexpr(
                                cute.size(tVrV) // 2
                            ):
                                packed_code = packed_codes[pair_idx // 2]
                                code_0 = packed_code >> ((pair_idx % 2) * 16)
                                code_1 = packed_code >> (
                                    (pair_idx % 2) * 16 + 8
                                )
                                qpoint_0 = cute.arch.shuffle_sync(
                                    v_lane_qpoint, code_0
                                )
                                qpoint_1 = cute.arch.shuffle_sync(
                                    v_lane_qpoint,
                                    code_1,
                                )
                                qvals[pair_idx] = cute.arch.prmt(
                                    qpoint_0, qpoint_1, 0x5410
                                )
                            qvals_ssa = cute.recast_tensor(
                                qvals, mma_dtype
                            ).load().reshape(tVrV_shape)
                            tVrV_ssa = (
                                qvals_ssa.to(cvt_type).to(mma_dtype) * scale_v
                                + minimum_v
                            )
                            tVrV_cvt.store(
                                tVrV_ssa.reshape(tVrV_cvt_shape)
                            )
                            cvt_handle = v_output_producer.acquire_and_advance()
                            cute.copy(
                                thr_store_v,
                                tVrV_cvt,
                                tVtV_cvt[cpy_dice + (cvt_handle.index,)],
                            )
                            cute.arch.fence_view_async_tmem_store()
                            cvt_handle.commit()

        ##############################
        # MMA KQ Dispatch
        ##############################
        elif warp_idx == self.mma_kq_warp_id:
            # Free registers
            if cutlass.const_expr(self.use_reg_reconfig):
                cute.arch.setmaxregister_decrease(self.mma_tma_regs)

            # Setup mma descriptors
            tBsQ_desc = thrblk_mma_kq.make_fragment_B(tBsQ)
            # Wait for Q
            for dk in cutlass.range_constexpr(tiles_dk):
                q_consumer.wait_and_advance()

            # Sequence loop
            s_token = True  # Producer always acquires first
            for s in cutlass.range(iters_s):
                # BMM1
                k_token = cvt_consumer.try_wait()
                s_handle = s_producer.acquire_and_advance(s_token)
                tiled_mma_kq.set(tcgen05.Field.ACCUMULATE, False)
                for dk in cutlass.range_constexpr(tiles_dk):
                    is_last_iter = dk == tiles_dk - 1
                    k_handle = cvt_consumer.wait_and_advance(k_token)
                    if is_last_iter:
                        mma_kq_nbar.arrive()
                    for mma_k in cutlass.range_constexpr(tAtK_cvt.shape[2]):
                        cute.gemm(
                            tiled_mma_kq,
                            tCtS[mma_dice + (s_handle.index,)],
                            tAtK_cvt[
                                None,
                                None,
                                mma_k,
                                k_handle.index,
                            ],
                            tBsQ_desc[None, None, mma_k, dk],
                            tCtS[mma_dice + (s_handle.index,)],
                        )
                        if dk == 0 and mma_k == 0:
                            tiled_mma_kq.set(tcgen05.Field.ACCUMULATE, True)
                    k_handle.release()
                    if not is_last_iter:
                        k_token = cvt_consumer.try_wait()
                s_handle.commit()

                # Advance and wait for BMM 2
                if s > 0:
                    mma_vp_nbar.arrive_and_wait()
                    s_token = s_producer.try_acquire()

        ##############################
        # MMA VP Dispatch
        ##############################
        elif warp_idx == self.mma_vp_warp_id:
            # Free registers
            if cutlass.const_expr(self.use_reg_reconfig):
                cute.arch.setmaxregister_decrease(self.mma_tma_regs)

            # Setup mma descriptors
            tiled_mma_vp.set(tcgen05.Field.ACCUMULATE, True)
            tBsP_desc = thrblk_mma_vp.make_fragment_B(tBsP_nk)
            vp_cvt_consumer = v_cvt_consumer

            # Advance and wait for BMM1
            mma_kq_nbar.arrive_and_wait()

            # Sequence loop
            p_token = False
            o_token = True  # Producer always acquires first
            for s in cutlass.range(iters_s):
                # Advance and wait for BMM1
                if s < iters_s - 1:
                    mma_kq_nbar.arrive_and_wait()
                    p_token = p_consumer.try_wait()

                # BMM2
                v_token = vp_cvt_consumer.try_wait()
                p_handle = p_consumer.wait_and_advance(p_token)
                o_handle = o_producer.acquire_and_advance(o_token)
                for sk in cutlass.range_constexpr(tiles_sk):
                    for dm in cutlass.range_constexpr(tiles_dm):
                        is_last_iter = sk == tiles_sk - 1 and dm == tiles_dm - 1
                        v_handle = vp_cvt_consumer.wait_and_advance(v_token)
                        # Signal BMM1 to start
                        if is_last_iter:
                            mma_vp_nbar.arrive()
                        for mma_k in cutlass.range_constexpr(tAtV_cvt.shape[2]):
                            cute.gemm(
                                tiled_mma_vp,
                                tCtO[mma_dice + (dm, 0)],
                                tAtV_cvt[None, None, mma_k, v_handle.index],
                                tBsP_desc[None, None, mma_k, sk, p_handle.index],
                                tCtO[mma_dice + (dm, 0)],
                            )
                        v_handle.release()
                        if not is_last_iter:
                            v_token = vp_cvt_consumer.try_wait()
                p_handle.release()
                o_handle.commit()
                o_token = o_producer.try_acquire()

            # Wait for signal to dealloc tmem, then dealloc
            o_producer.tail()
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(tmem_ptr, self.tmem_alloc_cols)

        ##############################
        # Softmax + Correction Dispatch
        ##############################
        elif warpgroup_idx == self.softmax_warpgroup_id:
            # Alloc registers
            if cutlass.const_expr(self.use_reg_reconfig):
                cute.arch.setmaxregister_increase(self.softmax_regs)

            # Construct tiled copies
            tmem_op_width = 32
            tmem_op_repeat_o = tcgen05.Repetition(
                mma_tile_n * acc_dtype.width // tmem_op_width
            )
            tmem_op_repeat_s = tcgen05.Repetition(
                mma_tile_n
                * acc_dtype.width
                // tmem_op_width
                // (2 if self.page_size == 64 else 1)
            )
            # A 64-row QK accumulator is exposed as two 16-row TMEM
            # sub-fragments; M128 uses the ordinary 32-row load atom.
            tmem_load_op_s = tcgen05.Ld32x32bOp(tmem_op_repeat_s)
            if cutlass.const_expr(self.page_size == 64):
                tmem_load_op_s = tcgen05.Ld16x32bx2Op(tmem_op_repeat_s)
            tmem_load_atom_s = cute.make_copy_atom(
                tmem_load_op_s, acc_dtype
            )
            tmem_load_s = tcgen05.make_tmem_copy(
                tmem_load_atom_s, tCtS[mma_dice + (0,)]
            )
            thr_load_s = tmem_load_s.get_slice(warpgroup_tidx)
            cS = cute.make_identity_tensor((mma_tile_m_kq, mma_tile_n))
            tCsCoord = thrblk_mma_kq.partition_C(cS)
            tScS = thr_load_s.partition_D(tCsCoord)

            tmem_store_atom_o = cute.make_copy_atom(
                tcgen05.St32x32bOp(tmem_op_repeat_o), acc_dtype
            )
            tmem_store_o = tcgen05.make_tmem_copy(
                tmem_store_atom_o, tCtO[mma_dice + (0, 0)]
            )
            thr_store_o = tmem_store_o.get_slice(warpgroup_tidx)

            tmem_load_atom_o = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tmem_op_repeat_o), acc_dtype
            )
            tmem_load_o = tcgen05.make_tmem_copy(
                tmem_load_atom_o, tCtO[mma_dice + (0, 0)]
            )
            thr_load_o = tmem_load_o.get_slice(warpgroup_tidx)

            # Partition S and P
            tStS = thr_load_s.partition_S(
                tCtS
            )  # (CPY, #CPY_MMA, #CPY_M, #CPY_N, stages_sp)
            tSsP = thr_load_s.partition_D(
                tCsP
            )  # (CPY, #CPY_MMA, #CPY_M, #CPY_N, stages_sp)

            # Partition O
            tStO = thr_load_o.partition_S(
                tCtO
            )  # (CPY, #CPY_MMA, #CPY_M, #CPY_N, #TILE_DM, #TILE_HN)
            tSsO = thr_load_o.partition_D(
                tCsO
            )  # (CPY, #CPY_MMA, #CPY_M, #CPY_N, #TILE_DM, #TILE_HN)
            tSrO = cute.make_rmem_tensor(tSsO.shape, acc_dtype)
            cO = cute.make_identity_tensor((mma_tile_m_vp, mma_tile_n))
            tCcO = thrblk_mma_vp.partition_C(cO)
            tOcO = thr_load_o.partition_D(tCcO)
            if cutlass.const_expr(self.cluster_reduction):
                tCsClusterO = thrblk_mma_vp.partition_C(
                    s_cluster_o[None, None, cluster_slot]
                )
                tSsClusterO = thr_load_o.partition_D(tCsClusterO)

            # Partition colmax and initialize in RF
            tSsM = thr_load_s.partition_D(tCsM)  # (CPY, #CPY_MMA, #CPY_M, #CPY_N)
            tSrM_prev = cute.make_rmem_tensor_like(tSsM)
            tSrM_prev.fill(-cutlass.Float32.inf)

            # Partition colsum and initialize in RF
            # Each thread maintains a local colsum in RF, smem reduction happens after loop
            tSsL = thr_load_s.partition_D(
                tCsL
            )  # (CPY, #CPY_MMA, #CPY_M, #CPY_N, WARPS)
            tSrL = cute.make_rmem_tensor_like(tSsL[cpy_dice + (0,)])
            tSrL.fill(cutlass.Float32(0))

            assert warp_threads >= cute.size(tSsM)
            # get gmem colmax + colsum to store to
            inbound_hr = (
                coord_hr * blk_tile_h + head_lane_idx < mM_partial.shape[0]
            )
            gM_partial = cute.local_tile(
                mM_partial,
                tiler=(mma_tile_n, 1),
                coord=(coord_hr, coord_hb, kv_split_idx),
            )
            gL_partial = cute.local_tile(
                mL_partial,
                tiler=(mma_tile_n, 1),
                coord=(coord_hr, coord_hb, kv_split_idx),
            )

            # Initialize O
            tSrO.fill(cutlass.Float32(0))
            cute.copy(thr_store_o, tSrO, tStO)

            #
            # Sequence loop
            #
            for s in cutlass.range(iters_s):
                # Load S from tmem
                s_handle = s_consumer.wait_and_advance()
                tSrS = cute.make_rmem_tensor(tSsP.shape[:-1], acc_dtype)
                cute.copy(
                    tmem_load_s, tStS[cpy_dice + (s_handle.index,)], tSrS
                )
                cute.arch.fence_view_async_tmem_load()
                s_handle.release()

                # Only the final logical page can contain padding.  Page-table
                # splits interleave logical pages, so recover the global page
                # index before masking residual tokens.
                logical_tile = kv_split_idx + s * kv_splits
                if logical_tile == tiles_s - 1 and seqlen_k % blk_tile_s != 0:
                    for i in cutlass.range_constexpr(cute.size(tSrS)):
                        coord_s, _ = tScS[i]
                        if coord_s + logical_tile * blk_tile_s >= seqlen_k:
                            tSrS[i] = -cutlass.Float32.inf

                # Reduce colmax in warp RF
                tSrM = cute.make_rmem_tensor_like(tSsM)
                tSrM_lane = cutlass.Float32(0)  # Avoid dynamic register indexing
                for i in cutlass.range_constexpr(cute.size(tSrS)):
                    if cutlass.const_expr(self.page_size == 64):
                        tSrM[i] = cute.arch.warp_reduction_max(
                            tSrS[i], threads_in_group=16
                        )
                    else:
                        tSrM[i] = warp_fmax(tSrS[i])
                    if i == lane_value_idx:
                        # Online softmax must retain the running maximum.  A
                        # page-local maximum can decrease on a later page;
                        # using it directly makes exp(m_prev - m_page) exceed
                        # one and repeatedly amplifies the accumulated
                        # denominator.  Long contexts can then overflow even
                        # though every score and the dense reference remain
                        # finite.  Every softmax warp carries the same
                        # previously published head maximum, so fold it into
                        # the value reduced across the warpgroup here.
                        tSrM_lane = cute.arch.fmax(tSrM[i], tSrM_prev[i])

                # Publish one local maximum per softmax warp. Reuse the colsum
                # scratch, then let warp 0 perform a deterministic four-way
                # reduction. Two phases are required because the second makes
                # warp 0's exact result visible to all warps.
                if lane_store_max:
                    sL[0, head_lane_idx, warpgroup_widx] = tSrM_lane

                softmax_nbar.arrive_and_wait()
                if warpgroup_widx == 0 and lane_store_max:
                    warp_maxima = sL[0, head_lane_idx, None]
                    warpgroup_max = warp_maxima[0]
                    for max_warp in cutlass.range_constexpr(
                        1, warpgroup_warps
                    ):
                        warpgroup_max = cute.arch.fmax(
                            warpgroup_max,
                            warp_maxima[max_warp],
                        )
                    sM[0, head_lane_idx] = warpgroup_max

                softmax_nbar.arrive_and_wait()
                cute.autovec_copy(tSsM, tSrM)

                # Compute online softmax
                tSrP = cute.make_rmem_tensor(tSsP.shape[:-1], mma_dtype)
                if cutlass.const_expr(use_tensor_ssa_math):
                    tSrP_f32 = exp2(scale_qs_log2_e * (tSrS.load() - tSrM.load()))
                    tSrP.store(tSrP_f32.to(mma_dtype))  # convert
                else:
                    tSrP_f32 = cute.make_rmem_tensor(tSrS.shape, acc_dtype)
                    for i in cutlass.range_constexpr(0, cute.size(tSrS), 2):
                        p_f32x2 = fadd2(
                            (tSrS[i], tSrS[i + 1]), (-tSrM[i], -tSrM[i + 1])
                        )
                        p_f32x2 = fmul2(p_f32x2, (scale_qs_log2_e, scale_qs_log2_e))
                        tSrP_f32[i] = exp2(p_f32x2[0])
                        tSrP_f32[i + 1] = exp2(p_f32x2[1])
                    tSrP.store(tSrP_f32.load().to(mma_dtype))

                # Store P to smem
                p_handle = p_producer.acquire_and_advance()
                cute.autovec_copy(tSrP, tSsP[cpy_dice + (p_handle.index,)])
                if cutlass.const_expr(not self.single_split_direct):
                    cute.arch.fence_view_async_shared()
                p_handle.commit()

                # Compute correction and correct colsum
                if cutlass.const_expr(use_tensor_ssa_math):
                    correction = exp2(
                        scale_qs_log2_e * (tSrM_prev.load() - tSrM.load())
                    )
                    tSrL.store(tSrL.load() * correction + tSrP_f32)
                else:
                    correction = cute.make_rmem_tensor_like(tSrM)
                    for i in cutlass.range_constexpr(0, cute.size(tSrM), 2):
                        c_f32x2 = fadd2(
                            (tSrM_prev[i], tSrM_prev[i + 1]), (-tSrM[i], -tSrM[i + 1])
                        )
                        c_f32x2 = fmul2(c_f32x2, (scale_qs_log2_e, scale_qs_log2_e))
                        c_f32x2 = (exp2(c_f32x2[0]), exp2(c_f32x2[1]))
                        correction[i] = c_f32x2[0]
                        correction[i + 1] = c_f32x2[1]
                        l_f32x2 = ffma2(
                            c_f32x2,
                            (tSrL[i], tSrL[i + 1]),
                            (tSrP_f32[i], tSrP_f32[i + 1]),
                        )
                        tSrL[i] = l_f32x2[0]
                        tSrL[i + 1] = l_f32x2[1]

                correction_o = correction
                if cutlass.const_expr(self.page_size == 64):
                    # PV produces all eight heads per D lane. Gather the other
                    # half-warp's four corrections to match that fragment.
                    correction_o = cute.make_rmem_tensor(
                        (mma_tile_n,), acc_dtype
                    )
                    for i in cutlass.range_constexpr(mma_tile_n // 2):
                        correction_peer = cute.arch.shuffle_sync_bfly(
                            correction[i], offset=16
                        )
                        if lane_idx < 16:
                            correction_o[i] = correction[i]
                            correction_o[i + mma_tile_n // 2] = (
                                correction_peer
                            )
                        else:
                            correction_o[i] = correction_peer
                            correction_o[i + mma_tile_n // 2] = correction[i]

                # Correct O
                if s > 0:
                    # Wait for O
                    o_handle = o_consumer.wait_and_advance()

                    # Apply correction
                    for dm in cutlass.range_constexpr(tiles_dm):
                        tSrO_dm = cute.make_rmem_tensor(
                            tSsO[cpy_dice + (0, 0)].shape, acc_dtype
                        )
                        cute.copy(
                            thr_load_o,
                            tStO[cpy_dice + (dm, 0)],
                            tSrO_dm,
                        )

                        for i in cutlass.range_constexpr(0, cute.size(tSrO_dm), 2):
                            o_f32x2 = fmul2(
                                (tSrO_dm[i], tSrO_dm[i + 1]),
                                (correction_o[i], correction_o[i + 1]),
                            )
                            tSrO_dm[i] = o_f32x2[0]
                            tSrO_dm[i + 1] = o_f32x2[1]

                        cute.copy(
                            thr_store_o,
                            tSrO_dm,
                            tStO[cpy_dice + (dm, 0)],
                        )

                    # Notify MMA
                    cute.arch.fence_view_async_tmem_store()
                    o_handle.release()

                # Update colmax
                tSrM_prev.store(tSrM.load())

            #
            # Softmax Epilogue
            #

            # Reduce colsum in warp RF
            tSrL_lane = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(cute.size(tSrL)):
                tSrL[i] = cute.arch.warp_reduction_sum(
                    tSrL[i],
                    threads_in_group=(
                        16 if self.page_size == 64 else warp_threads
                    ),
                )
                if i == lane_value_idx:
                    tSrL_lane = tSrL[i]

            # Store partial colsum in smem
            if lane_store_max:
                tSsL[cpy_dice + (warpgroup_widx,)][lane_value_idx] = tSrL_lane

            # Wait for colsum
            softmax_nbar.arrive_and_wait()

            tSrFinalScale = cute.make_rmem_tensor_like(tSsM)
            tSrFinalScale.fill(cutlass.Float32(1.0))
            if cutlass.const_expr(self.single_split_direct):
                inv_l_lane = cutlass.Float32(0.0)
                if lane_store_max and inbound_hr:
                    sL_lane_wg = sL[0, head_lane_idx, None]
                    sL_lane = (
                        sL_lane_wg[0]
                        + sL_lane_wg[1]
                        + sL_lane_wg[2]
                        + sL_lane_wg[3]
                    )
                    inv_l_lane = cute.arch.rcp_approx(sL_lane)
                for i in cutlass.range_constexpr(cute.size(tSrFinalScale)):
                    scale_source_lane = i
                    if cutlass.const_expr(self.page_size == 64):
                        scale_source_lane = i + (lane_idx // 16) * 16
                    tSrFinalScale[i] = cute.arch.shuffle_sync(
                        inv_l_lane, scale_source_lane
                    )
            elif cutlass.const_expr(self.cluster_reduction):
                if (
                    warpgroup_widx == 0
                    and head_lane_idx < self.heads_per_kv
                ):
                    sL_lane_wg = sL[0, head_lane_idx, None]
                    sL_lane = (
                        sL_lane_wg[0]
                        + sL_lane_wg[1]
                        + sL_lane_wg[2]
                        + sL_lane_wg[3]
                    )
                    sM_lane = sM[0, head_lane_idx]
                    store_cluster_f32(
                        sM_lane,
                        tensor_element_ptr(
                            s_cluster_m,
                            (head_lane_idx, cluster_slot),
                        ),
                        cluster_mbar_ptr,
                        cutlass.Int32(0),
                    )
                    store_cluster_f32(
                        sL_lane,
                        tensor_element_ptr(
                            s_cluster_l,
                            (head_lane_idx, cluster_slot),
                        ),
                        cluster_mbar_ptr,
                        cutlass.Int32(0),
                    )
            else:
                if warpgroup_widx == 0 and lane_store_max and inbound_hr:
                    sL_lane_wg = sL[0, head_lane_idx, None]
                    sL_lane = (
                        sL_lane_wg[0]
                        + sL_lane_wg[1]
                        + sL_lane_wg[2]
                        + sL_lane_wg[3]
                    )
                    sM_lane = sM[0, head_lane_idx] * scale_qs
                    gL_partial[head_lane_idx] = sL_lane
                    gM_partial[head_lane_idx] = sM_lane

            tSrFinalScaleO = tSrFinalScale
            if cutlass.const_expr(
                self.page_size == 64 and self.single_split_direct
            ):
                tSrFinalScaleO = cute.make_rmem_tensor(
                    (mma_tile_n,), acc_dtype
                )
                for i in cutlass.range_constexpr(mma_tile_n // 2):
                    final_scale_peer = cute.arch.shuffle_sync_bfly(
                        tSrFinalScale[i], offset=16
                    )
                    if lane_idx < 16:
                        tSrFinalScaleO[i] = tSrFinalScale[i]
                        tSrFinalScaleO[i + mma_tile_n // 2] = (
                            final_scale_peer
                        )
                    else:
                        tSrFinalScaleO[i] = final_scale_peer
                        tSrFinalScaleO[i + mma_tile_n // 2] = (
                            tSrFinalScale[i]
                        )

            o_handle = o_consumer.wait_and_advance()
            cute.copy(thr_load_o, tStO, tSrO)
            cute.arch.fence_view_async_tmem_load()
            o_handle.release()  # Final release signals tmem dealloc

            # Store O to smem
            for dm in cutlass.range_constexpr(tiles_dm):
                tOrO_dm = tSrO[cpy_dice + (dm, 0)]

                if cutlass.const_expr(self.single_split_direct):
                    for i in cutlass.range_constexpr(cute.size(tOrO_dm)):
                        tOrO_dm[i] *= tSrFinalScaleO[i] * scale_o
                    tOrO_store = cute.make_rmem_tensor(tOrO_dm.shape, o_dtype)
                    tOrO_store.store(tOrO_dm.load().to(o_dtype))
                    if cutlass.const_expr(self.direct_smem_store):
                        cute.autovec_copy(
                            tOrO_store, tSsO[cpy_dice + (dm, 0)]
                        )
                    else:
                        # TCGEN gives each softmax thread all eight head values
                        # at one D coordinate. Transpose that 8x32 warp fragment
                        # in registers so every lane can issue one aligned
                        # D-vector store instead of scalar/head-strided stores.
                        packed_heads = cute.recast_tensor(
                            tOrO_store, cutlass.Int32
                        )
                        direct_values_per_store = (
                            4 if self.heads_per_kv == 4 else 8
                        )
                        direct_d_groups = (
                            warp_threads // direct_values_per_store
                        )
                        direct_head = lane_idx // direct_d_groups
                        direct_d_group = lane_idx % direct_d_groups
                        direct_head_pair = direct_head // 2
                        direct_o_packed = cute.make_rmem_tensor(
                            (direct_values_per_store // 2,), cutlass.Int32
                        )
                        direct_prmt = 0x5410
                        if direct_head % 2 == 1:
                            direct_prmt = 0x7632
                        for direct_pair_head in cutlass.range_constexpr(
                            self.heads_per_kv // 2
                        ):
                            for direct_pair in cutlass.range_constexpr(
                                direct_values_per_store // 2
                            ):
                                direct_lane_0 = (
                                    direct_d_group * direct_values_per_store
                                    + direct_pair * 2
                                )
                                direct_lane_1 = direct_lane_0 + 1
                                direct_value_0 = cute.arch.shuffle_sync(
                                    packed_heads[direct_pair_head],
                                    direct_lane_0,
                                )
                                direct_value_1 = cute.arch.shuffle_sync(
                                    packed_heads[direct_pair_head],
                                    direct_lane_1,
                                )
                                direct_packed_value = cute.arch.prmt(
                                    direct_value_0,
                                    direct_value_1,
                                    direct_prmt,
                                )
                                if direct_head_pair == direct_pair_head:
                                    direct_o_packed[direct_pair] = (
                                        direct_packed_value
                                    )
                        warp_d_base = cute.arch.shuffle_sync(tOcO[0][0], 0)
                        direct_coord_d = (
                            warp_d_base
                            + direct_d_group * direct_values_per_store
                        )
                        if direct_head < self.heads_per_kv:
                            direct_o_ptr = cute.make_ptr(
                                o_dtype,
                                (
                                    mO_final.iterator
                                    + cute.crd2idx(
                                        (
                                            direct_coord_d
                                            + dm * mma_tile_m_vp,
                                            direct_head,
                                            coord_hb,
                                        ),
                                        mO_final.layout,
                                    )
                                ).toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            )
                            direct_o_src = cute.recast_tensor(
                                direct_o_packed, o_dtype
                            )
                            direct_o_dst = cute.make_tensor(
                                direct_o_ptr,
                                cute.make_layout(
                                    (direct_values_per_store,)
                                ),
                            )
                            direct_o_atom = cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                o_dtype,
                                num_bits_per_copy=(
                                    direct_values_per_store * o_dtype.width
                                ),
                            )
                            cute.copy(
                                direct_o_atom, direct_o_src, direct_o_dst
                            )
                elif cutlass.const_expr(self.cluster_reduction):
                    tOsClusterO_dm = tSsClusterO
                    if cutlass.const_expr(self.heads_per_kv == 4):
                        tOrO_store = cute.make_rmem_tensor(
                            tOrO_dm.shape, o_dtype
                        )
                        tOrO_store.store(tOrO_dm.load().to(o_dtype))
                        packed_cluster_o = cute.recast_tensor(
                            tOrO_store, cutlass.Int64
                        ).load()
                        store_cluster_i64(
                            packed_cluster_o[0],
                            tensor_element_ptr(
                                tOsClusterO_dm,
                                cute.idx2crd(0, tOsClusterO_dm.shape),
                            ),
                            cluster_mbar_ptr,
                            cutlass.Int32(0),
                        )
                    else:
                        tOrO_store = cute.make_rmem_tensor(
                            tOrO_dm.shape, o_dtype
                        )
                        tOrO_store.store(tOrO_dm.load().to(o_dtype))
                        packed_cluster_o = cute.recast_tensor(
                            tOrO_store, cutlass.Int64
                        ).load()
                        store_cluster_i64x2(
                            packed_cluster_o[0],
                            packed_cluster_o[1],
                            tensor_element_ptr(
                                tOsClusterO_dm,
                                cute.idx2crd(0, tOsClusterO_dm.shape),
                            ),
                            cluster_mbar_ptr,
                            cutlass.Int32(0),
                        )
                else:
                    tOrO_store = cute.make_rmem_tensor(tOrO_dm.shape, o_dtype)
                    tOrO_store.store(tOrO_dm.load().to(o_dtype))
                    if cutlass.const_expr(not self.single_split_direct):
                        # The split workspace is D-contiguous just like the
                        # final output.  Reuse the warp-register transpose from
                        # the single-split epilogue so each lane emits one
                        # aligned D vector instead of head-strided scalars.
                        packed_heads = cute.recast_tensor(
                            tOrO_store, cutlass.Int32
                        )
                        direct_values_per_store = (
                            4 if self.heads_per_kv == 4 else 8
                        )
                        direct_d_groups = (
                            warp_threads // direct_values_per_store
                        )
                        direct_head = lane_idx // direct_d_groups
                        direct_d_group = lane_idx % direct_d_groups
                        direct_head_pair = direct_head // 2
                        direct_o_packed = cute.make_rmem_tensor(
                            (direct_values_per_store // 2,), cutlass.Int32
                        )
                        direct_prmt = 0x5410
                        if direct_head % 2 == 1:
                            direct_prmt = 0x7632
                        for direct_pair_head in cutlass.range_constexpr(
                            self.heads_per_kv // 2
                        ):
                            for direct_pair in cutlass.range_constexpr(
                                direct_values_per_store // 2
                            ):
                                direct_lane_0 = (
                                    direct_d_group * direct_values_per_store
                                    + direct_pair * 2
                                )
                                direct_lane_1 = direct_lane_0 + 1
                                direct_value_0 = cute.arch.shuffle_sync(
                                    packed_heads[direct_pair_head],
                                    direct_lane_0,
                                )
                                direct_value_1 = cute.arch.shuffle_sync(
                                    packed_heads[direct_pair_head],
                                    direct_lane_1,
                                )
                                direct_packed_value = cute.arch.prmt(
                                    direct_value_0,
                                    direct_value_1,
                                    direct_prmt,
                                )
                                if direct_head_pair == direct_pair_head:
                                    direct_o_packed[direct_pair] = (
                                        direct_packed_value
                                    )
                        warp_d_base = cute.arch.shuffle_sync(tOcO[0][0], 0)
                        direct_coord_d = (
                            warp_d_base
                            + direct_d_group * direct_values_per_store
                        )
                        if direct_head < self.heads_per_kv:
                            direct_o_ptr = cute.make_ptr(
                                o_dtype,
                                (
                                    mO_partial.iterator
                                    + cute.crd2idx(
                                        (
                                            direct_coord_d
                                            + dm * mma_tile_m_vp,
                                            direct_head,
                                            coord_hb,
                                            kv_split_idx,
                                        ),
                                        mO_partial.layout,
                                    )
                                ).toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            )
                            direct_o_src = cute.recast_tensor(
                                direct_o_packed, o_dtype
                            )
                            direct_o_dst = cute.make_tensor(
                                direct_o_ptr,
                                cute.make_layout(
                                    (direct_values_per_store,)
                                ),
                            )
                            direct_o_atom = cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                o_dtype,
                                num_bits_per_copy=(
                                    direct_values_per_store * o_dtype.width
                                ),
                            )
                            cute.copy(
                                direct_o_atom,
                                direct_o_src,
                                direct_o_dst,
                            )
                if cutlass.const_expr(not self.single_split_direct):
                    cute.arch.fence_view_async_shared()

                    # Pace the split reducer's PDL.
                    q_consumer.release()
                    q_consumer.advance()

            if cutlass.const_expr(
                self.single_split_direct and self.direct_smem_store
            ):
                direct_epilogue_nbar.arrive_and_wait()

        if cutlass.const_expr(self.single_split_direct):
            # Padded FULL CUDA-graph rows have no K/V tiles.  After every
            # role-specialized pipeline has drained, deterministically
            # initialize the caller-owned output instead of depending on the
            # empty softmax epilogue's intermediate state.
            if seqlen_k <= 0:
                cute.arch.sync_threads()
                for linear_idx in cutlass.range(
                    tidx,
                    blk_tile_d * self.heads_per_kv,
                    self.threads_per_cta,
                ):
                    empty_d = linear_idx % blk_tile_d
                    empty_head = linear_idx // blk_tile_d
                    mO_final[empty_d, empty_head, coord_hb] = (
                        mO_final.element_type(0)
                    )

        if cutlass.const_expr(self.cluster_reduction):
            # All role-specialized warps have now published their split
            # partials. Rank 0 waits on the exact DSM transaction count,
            # computes one weight vector per real GQA head, and writes final O.
            cute.arch.cluster_arrive_relaxed()
            if cluster_rank == 0:
                if warp_idx == 0:
                    cute.arch.mbarrier_wait(cluster_mbar_ptr, phase=0)
                if tidx < self.heads_per_kv:
                    cluster_head = tidx
                    head_max = -cutlass.Float32.inf
                    for split_idx in cutlass.range_constexpr(
                        self.cluster_size
                    ):
                        head_max = cute.arch.fmax(
                            head_max,
                            s_cluster_m[cluster_head, split_idx],
                        )
                    head_denom = cutlass.Float32(0.0)
                    for split_idx in cutlass.range_constexpr(
                        self.cluster_size
                    ):
                        correction = exp2(
                            scale_qs_log2_e
                            * (
                                s_cluster_m[cluster_head, split_idx]
                                - head_max
                            )
                        )
                        s_cluster_weights[cluster_head, split_idx] = correction
                        head_denom += (
                            correction
                            * s_cluster_l[cluster_head, split_idx]
                        )
                    s_cluster_weights[
                        cluster_head, self.cluster_size
                    ] = cute.arch.rcp_approx(head_denom)

                cute.arch.sync_threads()
                if cutlass.const_expr(self.heads_per_kv == 8):
                    cluster_head = (tidx % 4) * 2
                    cluster_d = tidx // 4
                    cluster_acc_0 = cutlass.Float32(0.0)
                    cluster_acc_1 = cutlass.Float32(0.0)
                    for split_idx in cutlass.range_constexpr(
                        self.cluster_size
                    ):
                        cluster_weight_0 = s_cluster_weights[
                            cluster_head, split_idx
                        ]
                        cluster_weight_1 = s_cluster_weights[
                            cluster_head + 1, split_idx
                        ]
                        cluster_acc_0, cluster_acc_1 = ffma2(
                            (cluster_weight_0, cluster_weight_1),
                            (
                                cutlass.Float32(
                                    s_cluster_o[
                                        cluster_d, cluster_head, split_idx
                                    ]
                                ),
                                cutlass.Float32(
                                    s_cluster_o[
                                        cluster_d,
                                        cluster_head + 1,
                                        split_idx,
                                    ]
                                ),
                            ),
                            (cluster_acc_0, cluster_acc_1),
                        )
                    cluster_scale_0 = (
                        scale_o
                        * s_cluster_weights[
                            cluster_head, self.cluster_size
                        ]
                    )
                    cluster_scale_1 = (
                        scale_o
                        * s_cluster_weights[
                            cluster_head + 1, self.cluster_size
                        ]
                    )
                    cluster_value_0, cluster_value_1 = fmul2(
                        (cluster_acc_0, cluster_acc_1),
                        (cluster_scale_0, cluster_scale_1),
                    )
                    if seqlen_k <= 0:
                        cluster_value_0 = cutlass.Float32(0.0)
                        cluster_value_1 = cutlass.Float32(0.0)
                    mO_final[cluster_d, cluster_head, coord_hb] = (
                        mO_final.element_type(cluster_value_0)
                    )
                    mO_final[cluster_d, cluster_head + 1, coord_hb] = (
                        mO_final.element_type(cluster_value_1)
                    )
                else:
                    scalar_cluster_head = tidx % self.heads_per_kv
                    scalar_cluster_d = tidx // self.heads_per_kv
                    scalar_cluster_acc = cutlass.Float32(0.0)
                    for split_idx in cutlass.range_constexpr(
                        self.cluster_size
                    ):
                        scalar_cluster_acc += (
                            s_cluster_weights[
                                scalar_cluster_head, split_idx
                            ]
                            * cutlass.Float32(
                                s_cluster_o[
                                    scalar_cluster_d,
                                    scalar_cluster_head,
                                    split_idx,
                                ]
                            )
                        )
                    scalar_cluster_acc *= (
                        scale_o
                        * s_cluster_weights[
                            scalar_cluster_head, self.cluster_size
                        ]
                    )
                    if seqlen_k <= 0:
                        scalar_cluster_acc = cutlass.Float32(0.0)
                    mO_final[
                        scalar_cluster_d,
                        scalar_cluster_head,
                        coord_hb,
                    ] = mO_final.element_type(scalar_cluster_acc)
            cute.arch.cluster_wait()
        elif cutlass.const_expr(not self.single_split_direct):
            cute.arch.griddepcontrol_launch_dependents()
        return

    @cute.kernel
    def reduction_parallel_weights(
        self,
        o: cute.Tensor,
        o_partial: cute.Tensor,
        m_partial: cute.Tensor,
        l_partial: cute.Tensor,
        seq_lens: cute.Tensor,
        scale_o: cutlass.Float32,
    ):
        """Combine at most 32 splits for one D128 query head.

        A dedicated warp maps one lane to each compile-time split, publishes
        raw correction weights plus one inverse denominator, and resets the
        persistent max workspace. The remaining four warps prefetch one D128
        output vector while that work runs, then perform register-only FMAs.
        Final LSE is intentionally omitted from this inference-only path.
        """
        _, coord_h, coord_b = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        dim = tidx - warp_threads
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        num_splits = self.reduction_splits
        if cutlass.const_expr(self.adaptive_extra_tasks > 0):
            heads_k = o.shape[1] // self.heads_per_kv
            kv_task = (
                coord_b * heads_k + coord_h // self.heads_per_kv
            )
            num_splits = self.adaptive_split_base
            if kv_task < self.adaptive_extra_tasks:
                num_splits += 1
        runtime_tiles = cute.ceil_div(seq_lens[coord_b], self.page_size)
        if runtime_tiles < num_splits:
            num_splits = runtime_tiles

        cute.arch.fence_acq_rel_cta()
        cute.arch.griddepcontrol_wait()

        smem = utils.SmemAllocator()
        weights = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((64,)),
            byte_alignment=128,
        )
        if warp_idx == 0:
            if lane_idx < self.reduction_splits:
                weights[lane_idx] = cutlass.Float32(0.0)
            local_max = -cutlass.Float32.inf
            if lane_idx < num_splits:
                local_max = m_partial[coord_h, coord_b, lane_idx]
            # One lane already owns every split, so reducing the partial
            # maxima here is cheaper than contended global atomics in every
            # producer CTA and removes persistent max-workspace state.
            m_head = local_max
            reduction_steps = math.ceil(math.log2(self.reduction_splits))
            for red_iter in cutlass.range_constexpr(reduction_steps):
                peer_lane = lane_idx ^ (1 << red_iter)
                peer_max = cute.arch.shuffle_sync(m_head, peer_lane)
                m_head = cute.arch.fmax(m_head, peer_max)
            denom_lane = cutlass.Float32(0.0)
            if lane_idx < num_splits:
                split_idx = lane_idx
                correction = exp2(
                    log2_e * (local_max - m_head)
                )
                weights[split_idx] = correction
                denom_lane += correction * l_partial[coord_h, coord_b, split_idx]

            denom = denom_lane
            for red_iter in cutlass.range_constexpr(reduction_steps):
                peer_lane = lane_idx ^ (1 << red_iter)
                denom += cute.arch.shuffle_sync(denom, peer_lane)
            if lane_idx == 0:
                inv_denom = cutlass.Float32(0.0)
                if num_splits > 0:
                    inv_denom = cute.arch.rcp_approx(denom)
                weights[32] = inv_denom

        partial_values = cute.make_rmem_tensor(
            (self.reduction_splits,), cutlass.Float32
        )
        if tidx >= warp_threads:
            for split_idx in cutlass.range_constexpr(self.reduction_splits):
                partial_values[split_idx] = cutlass.Float32(0.0)
                if split_idx < num_splits:
                    partial_values[split_idx] = cutlass.Float32(
                        o_partial[dim, coord_h, coord_b, split_idx]
                    )

        cute.arch.sync_threads()

        if tidx >= warp_threads:
            acc = cutlass.Float32(0.0)
            for split_idx in cutlass.range_constexpr(self.reduction_splits):
                acc += weights[split_idx] * partial_values[split_idx]
            output_value = scale_o * weights[32] * acc
            if runtime_tiles <= 0:
                output_value = cutlass.Float32(0.0)
            o[dim, coord_h, coord_b] = o.element_type(output_value)

        cute.arch.griddepcontrol_launch_dependents()
        return

    @cute.kernel
    def reduction_cta4_weights(
        self,
        o: cute.Tensor,
        o_partial: cute.Tensor,
        m_partial: cute.Tensor,
        l_partial: cute.Tensor,
        seq_lens: cute.Tensor,
        scale_o: cutlass.Float32,
    ):
        """Reduce four D128 query heads with one 512-thread CTA.

        Four weight warps first compute one split-softmax vector per head. The
        same CTA then maps 128 threads to every head, retaining the output
        parallelism of the five-warp-per-head reducer without its extra warp or
        32-CTA launch topology.
        """
        _, coord_h_tile, coord_b = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(tidx // warp_threads)
        heads_per_cta = 4
        num_splits = self.reduction_splits
        runtime_tiles = cute.ceil_div(seq_lens[coord_b], self.page_size)
        if runtime_tiles < num_splits:
            num_splits = runtime_tiles
        reduction_steps = math.ceil(math.log2(self.reduction_splits))

        cute.arch.fence_acq_rel_cta()
        cute.arch.griddepcontrol_wait()

        smem = utils.SmemAllocator()
        weights = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((heads_per_cta, 64)),
            byte_alignment=128,
        )
        if warp_idx < heads_per_cta:
            coord_h_weights = coord_h_tile * heads_per_cta + warp_idx
            if lane_idx < self.reduction_splits:
                weights[warp_idx, lane_idx] = cutlass.Float32(0.0)
            local_max = -cutlass.Float32.inf
            if lane_idx < num_splits:
                local_max = m_partial[
                    coord_h_weights, coord_b, lane_idx
                ]
            m_head = local_max
            for red_iter in cutlass.range_constexpr(reduction_steps):
                peer_lane = lane_idx ^ (1 << red_iter)
                m_head = cute.arch.fmax(
                    m_head, cute.arch.shuffle_sync(m_head, peer_lane)
                )
            correction = cutlass.Float32(0.0)
            denom = cutlass.Float32(0.0)
            if lane_idx < num_splits:
                correction = exp2(log2_e * (local_max - m_head))
                weights[warp_idx, lane_idx] = correction
                denom = correction * l_partial[
                    coord_h_weights, coord_b, lane_idx
                ]
            for red_iter in cutlass.range_constexpr(reduction_steps):
                peer_lane = lane_idx ^ (1 << red_iter)
                denom += cute.arch.shuffle_sync(denom, peer_lane)
            if lane_idx == 0:
                weights[warp_idx, 32] = cute.arch.rcp_approx(denom)

        cute.arch.sync_threads()

        local_head = tidx // 128
        coord_d = tidx % 128
        coord_h = coord_h_tile * heads_per_cta + local_head
        acc = cutlass.Float32(0.0)
        for split_idx in cutlass.range_constexpr(self.reduction_splits):
            acc += weights[local_head, split_idx] * cutlass.Float32(
                o_partial[coord_d, coord_h, coord_b, split_idx]
            )
        acc *= scale_o * weights[local_head, 32]
        if runtime_tiles <= 0:
            acc = cutlass.Float32(0.0)
        o[coord_d, coord_h, coord_b] = o.element_type(acc)
        cute.arch.griddepcontrol_launch_dependents()
        return

    @cute.kernel
    def reduction_warp_parallel_weights(
        self,
        o: cute.Tensor,
        o_partial: cute.Tensor,
        m_partial: cute.Tensor,
        l_partial: cute.Tensor,
        seq_lens: cute.Tensor,
        scale_o: cutlass.Float32,
    ):
        """Combine one D128 query head with one warp and no shared memory.

        Split lanes evaluate one correction each and reduce the max and
        denominator through the minimum power-of-two butterfly. Every output
        lane then reuses those register weights for four dimensions. This is
        the same split-softmax reduction as ``reduction_parallel_weights``
        without a five-warp CTA, shared staging, or a CTA barrier.
        """
        _, coord_h_tile, coord_b = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(tidx // warp_threads)
        coord_h = coord_h_tile * self.reducer_warps_per_cta + warp_idx
        lane_idx = cute.arch.lane_idx()
        num_splits = self.reduction_splits
        if cutlass.const_expr(self.adaptive_extra_tasks > 0):
            heads_k = o.shape[1] // self.heads_per_kv
            kv_task = (
                coord_b * heads_k + coord_h // self.heads_per_kv
            )
            num_splits = self.adaptive_split_base
            if kv_task < self.adaptive_extra_tasks:
                num_splits += 1
        runtime_tiles = cute.ceil_div(seq_lens[coord_b], self.page_size)
        if runtime_tiles < num_splits:
            num_splits = runtime_tiles
        reduction_steps = math.ceil(math.log2(self.reduction_splits))

        cute.arch.fence_acq_rel_cta()
        cute.arch.griddepcontrol_wait()

        local_max = -cutlass.Float32.inf
        if lane_idx < num_splits:
            local_max = m_partial[coord_h, coord_b, lane_idx]
        m_head = local_max
        for red_iter in cutlass.range_constexpr(reduction_steps):
            peer_lane = lane_idx ^ (1 << red_iter)
            peer_max = cute.arch.shuffle_sync(m_head, peer_lane)
            m_head = cute.arch.fmax(m_head, peer_max)

        correction = cutlass.Float32(0.0)
        denom = cutlass.Float32(0.0)
        if lane_idx < num_splits:
            correction = exp2(log2_e * (local_max - m_head))
            denom = correction * l_partial[coord_h, coord_b, lane_idx]
        for red_iter in cutlass.range_constexpr(reduction_steps):
            peer_lane = lane_idx ^ (1 << red_iter)
            denom += cute.arch.shuffle_sync(denom, peer_lane)
        inv_denom = cutlass.Float32(0.0)
        if num_splits > 0:
            inv_denom = cute.arch.rcp_approx(cute.arch.shuffle_sync(denom, 0))

        acc_0 = cutlass.Float32(0.0)
        acc_1 = cutlass.Float32(0.0)
        acc_2 = cutlass.Float32(0.0)
        acc_3 = cutlass.Float32(0.0)
        for split_idx in cutlass.range_constexpr(self.reduction_splits):
            if split_idx < num_splits:
                weight = (
                    cute.arch.shuffle_sync(correction, split_idx) * inv_denom
                )
                acc_0 += weight * cutlass.Float32(
                    o_partial[lane_idx, coord_h, coord_b, split_idx]
                )
                acc_1 += weight * cutlass.Float32(
                    o_partial[lane_idx + 32, coord_h, coord_b, split_idx]
                )
                acc_2 += weight * cutlass.Float32(
                    o_partial[lane_idx + 64, coord_h, coord_b, split_idx]
                )
                acc_3 += weight * cutlass.Float32(
                    o_partial[lane_idx + 96, coord_h, coord_b, split_idx]
                )
        if runtime_tiles <= 0:
            acc_0 = cutlass.Float32(0.0)
            acc_1 = cutlass.Float32(0.0)
            acc_2 = cutlass.Float32(0.0)
            acc_3 = cutlass.Float32(0.0)
        o[lane_idx, coord_h, coord_b] = o.element_type(scale_o * acc_0)
        o[lane_idx + 32, coord_h, coord_b] = o.element_type(scale_o * acc_1)
        o[lane_idx + 64, coord_h, coord_b] = o.element_type(scale_o * acc_2)
        o[lane_idx + 96, coord_h, coord_b] = o.element_type(scale_o * acc_3)
        cute.arch.griddepcontrol_launch_dependents()
        return
