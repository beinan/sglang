"""Experimental full-MHA Qwen3 prefill using request-owned MP CacheBlend spans."""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

import msgspec
import torch

if TYPE_CHECKING:
    from lmcache.integration.sglang.lmcache_mp_metadata import (
        LMCacheBlendLookupOperation,
    )
    from lmcache.integration.sglang.unified_lmcache_mp_connector import (
        UnifiedLMCacheMPConnector,
    )
    from lmcache.v1.multiprocess.custom_types import CBMatchResult

    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)
CALIBRATION_LAYERS = 4


def select_blend_matches(
    lookup: LMCacheBlendLookupOperation | None,
    prefix_len: int,
    allocated_end: int | None = None,
) -> list[CBMatchResult]:
    """Select complete matches outside the shared, immutable cached prefix.

    Only matches contained in [prefix_len, allocated_end) may be transferred.
    The optional end bounds the currently allocated complete pages; omitting it
    checks whether any later chunk can blend. Matches crossing either boundary
    are recomputed rather than clipped because the connector accepts only
    complete matches from the lookup result. Short requests have no lookup.
    """
    if lookup is None:
        return []
    result = lookup.blend_result
    return [
        match
        for match in result.non_prefix_segments + result.segmented_prefix_segments
        if match.cur_st >= prefix_len
        and (allocated_end is None or match.cur_ed <= allocated_end)
    ]


class BlendRotaryRegistration:
    """Share registration of an immutable model-owned rotary table across prefills."""

    def __init__(self, connector: UnifiedLMCacheMPConnector) -> None:
        self._connector = connector
        self._table: torch.Tensor | None = None
        self._is_neox_style: bool | None = None

    def ensure_registered(
        self,
        table: torch.Tensor,
        *,
        is_neox_style: bool,
    ) -> None:
        """Register a new table or layout, retaining the table for its IPC lifetime.

        The model must replace the tensor when changing its storage or geometry.
        A failed registration remains retryable on the next call.
        """
        if self._table is table and self._is_neox_style == is_neox_style:
            return
        # Retain the old IPC allocation until replacement is acknowledged.
        self._is_neox_style = None
        self._connector.register_blend_rope(table, is_neox_style=is_neox_style)
        self._table = table
        self._is_neox_style = is_neox_style


class BlendPrefillPlan(msgspec.Struct, frozen=True):
    """Retrieve into scheduler-owned slots, then recompute selected query rows.

    The cache owner retains each lookup until checkpoint or cancellation. The
    blocking device completion below deliberately keeps destination ownership
    simple; asynchronous overlap requires a separate scheduler contract.
    """

    connector: UnifiedLMCacheMPConnector
    rotary_registration: BlendRotaryRegistration
    lookups: tuple[LMCacheBlendLookupOperation | None, ...]
    recompute_ratio: float
    page_size: int
    blended_requests: set[str]

    def execute(
        self, model: torch.nn.Module, forward_batch: ForwardBatch, **kwargs
    ) -> LogitsProcessorOutput:
        """Use sparse prefill for allocated plain-text Qwen3 prompt chunks.

        Unsupported request modes use the ordinary dense model forward without
        loading sparse KV. Approximate results are excluded from cache writeback.
        Active LoRA batches retain the dense rows used to prepare adapter routing.
        """
        if any(uid is not None for uid in forward_batch.lora_ids):
            return model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )

        from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
        from sglang.srt.layers.logits_processor import LogitsMetadata
        from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
        from sglang.srt.model_executor.forward_context import (
            get_attn_backend,
            get_req_to_token_pool,
            get_token_to_kv_pool,
        )
        from sglang.srt.model_executor.forward_mode import (
            CaptureHiddenMode,
            ForwardMode,
        )
        from sglang.srt.model_executor.sparse_query import SparseQueryMetadata
        from sglang.srt.models.qwen3 import Qwen3ForCausalLM

        fb = forward_batch
        backend = get_attn_backend()
        pool = get_token_to_kv_pool()
        if (
            type(model) is not Qwen3ForCausalLM
            or type(backend) is not FlashInferAttnBackend
            or fb.forward_mode != ForwardMode.EXTEND
            or fb.return_logprob
            or fb.capture_hidden_mode != CaptureHiddenMode.NULL
            or fb.return_hidden_states_before_norm
            or fb.input_embeds is not None
            or any(fb.mm_inputs or [])
            or fb.spec_info is not None
            or fb.attn_cp_metadata is not None
            or any(
                value is not None and value is not False for value in kwargs.values()
            )
            or model.model.layers_to_capture
            or model.capture_aux_hidden_states
            or model.model.start_layer != 0
            or model.model.end_layer <= CALIBRATION_LAYERS
            or not model.pp_group.is_first_rank
            or not model.pp_group.is_last_rank
            or pool.layer_router.read(0).key().dtype != torch.bfloat16
        ):
            return model.forward(fb.input_ids, fb.positions, fb, **kwargs)

        body = model.model
        rope = body.layers[0].self_attn.rotary_emb
        self.rotary_registration.ensure_registered(
            rope.cos_sin_cache, is_neox_style=rope.is_neox_style
        )
        slots = fb.out_cache_loc
        covered = torch.zeros(slots.numel(), device=slots.device, dtype=torch.bool)
        retrieved = False
        offset = 0
        for row, lookup, length, prefix_len in zip(
            fb.req_pool_indices_cpu,
            self.lookups,
            fb.extend_seq_lens_cpu,
            fb.extend_prefix_lens_cpu,
            strict=True,
        ):
            aligned_length = (prefix_len + length) // self.page_size * self.page_size
            matches = select_blend_matches(lookup, prefix_len, aligned_length)
            if matches:
                # The request table contains both borrowed prefix and newly
                # allocated suffix slots. Transfers must not write shared KV.
                request_slots = get_req_to_token_pool().req_to_token[
                    row, :aligned_length
                ]
                indices = self.connector.device_indices_by_group(request_slots)
                future = self.connector.submit_blend_retrieve(
                    lookup,
                    indices,
                    matches=matches,
                    producer_stream=torch.cuda.current_stream(),
                )
                if future.result():
                    retrieved = True
                    for match in matches:
                        start = offset + match.cur_st - prefix_len
                        end = offset + match.cur_ed - prefix_len
                        covered[start:end] = True
            offset += length
        if not retrieved:
            return model.forward(fb.input_ids, fb.positions, fb, **kwargs)

        cached_keys = pool.layer_router.read(CALIBRATION_LAYERS - 1).key()[slots]
        hidden = body.embed_tokens(fb.input_ids)
        residual_batch.start(fb)
        compact = None
        try:
            for i in range(CALIBRATION_LAYERS):
                hidden = body.layers[i](fb.positions, hidden, fb)
            current_keys = pool.layer_router.read(CALIBRATION_LAYERS - 1).key()[slots]
            delta = (
                (current_keys.float() - cached_keys.float()).flatten(1).square().mean(1)
            )
            selected_mask = ~covered
            query_lens = []
            offset = 0
            for length in fb.extend_seq_lens_cpu:
                matched = covered[offset : offset + length].nonzero().flatten() + offset
                selected_mask[offset + length - 1] = True
                count = int(len(matched) * self.recompute_ratio)
                if count:
                    selected_mask[matched[delta[matched].topk(count).indices]] = True
                query_lens.append(int(selected_mask[offset : offset + length].sum()))
                offset += length
            selected = selected_mask.nonzero().flatten()
            self.blended_requests.update(
                op.request_id
                for op, query_len, length in zip(
                    self.lookups, query_lens, fb.extend_seq_lens_cpu, strict=True
                )
                if op is not None and query_len != length
            )
            handoff = residual_batch.to_pp(hidden, fb)
            compact = copy.copy(fb)
            compact.prefill_plan = None
            compact.positions = fb.positions[selected]
            compact.input_ids = fb.input_ids[selected]
            compact.iteration_ids.bind(compact, backend.runner_ids, cols=selected)
            compact.extend_num_tokens = len(selected)
            compact.extend_seq_lens_cpu = query_lens
            compact.extend_seq_lens = torch.tensor(
                query_lens, device=slots.device, dtype=torch.int32
            )
            compact.extend_start_loc = (
                compact.extend_seq_lens.cumsum(0).to(torch.int32)
                - compact.extend_seq_lens
            )
            compact.sparse_query = (
                SparseQueryMetadata(tuple(query_lens), compact.positions)
                if len(selected) != len(slots)
                else None
            )
            backend.init_forward_metadata(compact)
            hidden = body.layers[CALIBRATION_LAYERS].attn_boundary.from_pp(
                PPProxyTensors(
                    {
                        "hidden_states": handoff["hidden_states"][selected],
                        "residual": handoff["residual"][selected],
                    }
                ),
                compact,
            )
            for i in range(CALIBRATION_LAYERS, body.end_layer):
                hidden = body.layers[i](compact.positions, hidden, compact)
            normalized = residual_batch.final_norm(hidden, compact, body.norm)
            logger.info(
                "CacheBlend prefill requests=%s tokens=%d matched=%d queries=%s",
                fb.rids,
                len(slots),
                int(covered.sum()),
                query_lens,
            )
            return model.logits_processor(
                compact.input_ids,
                normalized,
                model.lm_head,
                LogitsMetadata.from_forward_batch(compact),
            )
        finally:
            residual_batch.clear(fb)
            if compact is not None:
                residual_batch.clear(compact)
            backend.init_forward_metadata(fb)
