# SPDX-License-Identifier: Apache-2.0
"""Eager original-position prefill over the runner's physical KV read stream."""

from collections.abc import Callable
from itertools import accumulate

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.typing import Float32, Int32
from flashinfer import BatchPrefillWithPagedKVCacheWrapper
from flashinfer.cute_dsl.attention import AttentionVariant, BatchPrefillCuteDSLWrapper

from sglang.srt.mem_cache.layout.paged_view import packed_page_indices, paged_kv_view
from sglang.srt.model_executor.sparse_query import SparseQueryMetadata


class OriginalQueryPositions(AttentionVariant):
    """Mask keys after each query's original position, including repeated queries."""

    has_score_mod = True

    def __init__(self, positions: torch.Tensor) -> None:
        self._positions = positions

    @property
    def extra_params(self) -> torch.Tensor:
        return self._positions

    @cute.jit
    def score_mod(self, score, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx):
        offset = Int32(self.params[batch_idx])
        return cutlass.select_(
            kv_idx > self.params[offset + qo_idx], -Float32.inf, score
        )


class CuTeSparsePrefill:
    """A per-prefill CuTe plan with FA2 fallback for unsupported layer options.

    Token addresses must already be physical, page-aligned within each request,
    and internally contiguous within each page. Referenced KV pages must contain
    finite values, including unused final-page rows, as required by CuTe TMA.
    The caller retains this object through the prefill; all tensors are created
    and consumed on the caller's current CUDA stream. Position storage is linear
    in the total query count plus batch size, with a 256-entry tail for masked
    query lanes. Its capacity is bucketed to bound compilation variants.
    An optional fallback planner runs once, on the current stream, before the
    first unsupported layer; its captured tensors are retained until then.
    """

    def __init__(
        self,
        fallback: BatchPrefillWithPagedKVCacheWrapper,
        query: SparseQueryMetadata,
        token_indices: torch.Tensor,
        kv_lens: list[int],
        page_size: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        plan_fallback: Callable[[], None] | None = None,
    ) -> None:
        query.validate(kv_lens)
        self.plan_fallback = plan_fallback
        self.fallback = fallback
        self.page_size = page_size
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        device = query.positions.device
        query_offsets = list(accumulate(query.query_lens, initial=0))
        batch_size = len(query.query_lens)
        num_queries = query_offsets[-1]
        # Invalid query lanes may read beyond their request's positions; only
        # the final request needs padding to keep those loads in bounds.
        capacity = 1 << (batch_size + num_queries + 256 - 1).bit_length()
        positions = torch.full((capacity,), -1.0, device=device, dtype=torch.float32)
        positions[:batch_size] = torch.tensor(
            [batch_size + offset for offset in query_offsets[:-1]],
            dtype=torch.float32,
            device=device,
        )
        positions[batch_size : batch_size + num_queries] = query.positions
        self.variant = OriginalQueryPositions(positions)
        self.qo_indptr = torch.tensor(
            query_offsets,
            dtype=torch.int32,
            device=device,
        )
        page_counts = [(length + page_size - 1) // page_size for length in kv_lens]
        self.kv_indptr = torch.tensor(
            list(accumulate(page_counts, initial=0)), dtype=torch.int32, device=device
        )
        self.page_indices = packed_page_indices(token_indices, kv_lens, page_size)
        self.last_page_len = torch.tensor(
            [(length - 1) % page_size + 1 for length in kv_lens],
            dtype=torch.int32,
            device=device,
        )
        self.plans: dict[float, BatchPrefillCuteDSLWrapper] = {}

    def forward(
        self,
        q: torch.Tensor,
        kv: tuple[torch.Tensor, torch.Tensor],
        *,
        causal: bool,
        sm_scale: float,
        window_left: int,
        logits_soft_cap: float,
        k_scale: float | None,
        v_scale: float | None,
    ) -> torch.Tensor:
        """Attend selected queries, retaining FA2 for windows/softcaps/KV scales."""
        if (
            not causal
            or window_left != -1
            or logits_soft_cap != 0.0
            or k_scale not in (None, 1.0)
            or v_scale not in (None, 1.0)
        ):
            if self.plan_fallback is not None:
                self.plan_fallback()
                self.plan_fallback = None
            return self.fallback.forward(
                q,
                kv,
                causal=causal,
                sm_scale=sm_scale,
                window_left=window_left,
                logits_soft_cap=logits_soft_cap,
                k_scale=k_scale,
                v_scale=v_scale,
            )
        if sm_scale not in self.plans:
            wrapper = BatchPrefillCuteDSLWrapper(
                torch.empty(1024, dtype=torch.uint8, device=q.device)
            )
            wrapper.plan(
                self.qo_indptr,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_dim,
                # Sorted positions may repeat. Suffix-causal scheduling would
                # hide valid keys for repeated queries near the end of KV.
                causal=False,
                sm_scale=sm_scale,
                q_data_type=torch.bfloat16,
                kv_data_type=torch.bfloat16,
                variant=self.variant,
                page_size=self.page_size,
                paged_kv_indptr=self.kv_indptr,
                paged_kv_indices=self.page_indices,
                paged_kv_last_page_len=self.last_page_len,
            )
            self.plans[sm_scale] = wrapper
        k, v = kv
        return self.plans[sm_scale].run_paged(
            q,
            (
                paged_kv_view(k, self.page_size, self.num_kv_heads, self.head_dim),
                paged_kv_view(v, self.page_size, self.num_kv_heads, self.head_dim),
            ),
        )
