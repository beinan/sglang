import importlib.util
import sys
from concurrent.futures import Future
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    IncLockRefResult,
    InitLoadBackParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage.lmcache.blend_prefill import (
    BlendPrefillPlan,
    BlendRotaryRegistration,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


@pytest.fixture
def cache_class():
    # Exercise the real cache module without installing the optional MP service.
    metadata = ModuleType("lmcache.integration.sglang.lmcache_mp_metadata")
    metadata.LMCacheExternalFlow = mock.Mock()
    metadata.LMCachePendingStore = mock.Mock()
    connector = ModuleType("lmcache.integration.sglang.unified_lmcache_mp_connector")
    connector.UnifiedLMCacheMPConnector = mock.Mock()
    with mock.patch.dict(
        sys.modules, {metadata.__name__: metadata, connector.__name__: connector}
    ):
        spec = importlib.util.find_spec(
            "sglang.srt.mem_cache.storage.lmcache.lmcache_unified_radix_cache"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.LMCacheUnifiedRadixCache


def _cache_and_request(cache_class, *, swa_enabled, swa_cursor=0):
    cache = cache_class.__new__(cache_class)
    cache.is_swa_enabled = swa_enabled
    cache._declined_loads = set()
    cache._blend_ratio = None
    cache._blended_requests = set()
    cache.lmcache_connector = SimpleNamespace(aligned_swa_window_size=lambda: 4)
    cache._blend_rotary_registration = BlendRotaryRegistration(cache.lmcache_connector)
    flow = SimpleNamespace(
        load=SimpleNamespace(local_hit_tokens=2, device_indices=torch.arange(2, 8)),
        loaded_skip_tokens=0,
        total_hit=8,
        key=RadixKey(list(range(8))),
        prefix_published=False,
        mamba_value=None,
        free_mamba_after_load=False,
        request_mamba_value=None,
        allocated_request_mamba_for_load=False,
        load_req=None,
    )
    cache._external_flows = {"request": flow}
    req = SimpleNamespace(
        rid="request",
        kv=ReqKvInfo(
            req_pool_idx=0,
            kv_allocated_len=12,
            cache_protected_len=8,
            component_evicted_seqlens={ComponentType.AUXILIARY_SWA: 6},
        ),
        priority=0,
        last_node=None,
        lock=None,
    )
    if swa_enabled:
        req.kv.set_evicted_seqlen(ComponentType.SWA, swa_cursor)
    return cache, req, flow


@pytest.mark.parametrize("swa_cursor", [0, 6])
def test_external_swa_load_preserves_independent_cursors(cache_class, swa_cursor):
    cache, req, flow = _cache_and_request(
        cache_class, swa_enabled=True, swa_cursor=swa_cursor
    )

    cache._publish_external_loaded_prefix(req, token_ids_len=12)

    assert flow.prefix_published
    assert req.kv.cache_protected_len == 2
    assert req.kv.component_evicted_seqlens == {
        ComponentType.SWA: max(swa_cursor, 4),
        ComponentType.AUXILIARY_SWA: 6,
    }


@pytest.mark.parametrize("swa_enabled", [False, True])
def test_mamba_publication_snapshots_component_cursors(cache_class, swa_enabled):
    cache, req, flow = _cache_and_request(cache_class, swa_enabled=swa_enabled)
    flow.mamba_value = torch.tensor([1], dtype=torch.int64)
    cache.req_to_token_pool = mock.Mock()
    cache.req_to_token_pool.req_to_token = torch.arange(12).reshape(1, 12)
    cache.insert = mock.Mock(return_value=InsertResult(prefix_len=8))
    cache.inc_lock_ref = mock.Mock(return_value=IncLockRefResult(node_id=42))
    matched = MatchResult(
        device_prefix_len=8,
        last_device_node=42,
        last_host_node=None,
        best_match_node=None,
    )

    with (
        mock.patch.object(UnifiedRadixCache, "match_prefix", return_value=matched),
        mock.patch.object(cache, "path_device_indices", return_value=torch.arange(8)),
    ):
        cache._publish_external_loaded_prefix(req, token_ids_len=12)

    insert_params = cache.insert.call_args.args[0]
    expected = {ComponentType.AUXILIARY_SWA: 6}
    if swa_enabled:
        expected[ComponentType.SWA] = 4
    assert insert_params.component_evicted_seqlens == expected
    assert (
        insert_params.component_evicted_seqlens is not req.kv.component_evicted_seqlens
    )
    req.kv.set_evicted_seqlen(ComponentType.AUXILIARY_SWA, 99)
    assert insert_params.get_evicted_seqlen(ComponentType.AUXILIARY_SWA) == 6
    assert req.kv.cache_protected_len == 8
    assert req.lock.receipt.node_id == 42
    assert flow.prefix_published
    assert flow.mamba_value is None


@pytest.mark.parametrize("first_completed", ["load", "store", "no_store"])
@pytest.mark.parametrize("load_success", [True, False])
def test_aborted_request_waits_for_load_and_store(
    cache_class, first_completed, load_success
):
    cache, req, flow = _cache_and_request(cache_class, swa_enabled=False)
    handle = CacheRequestHandle(rid=req.rid, attempt_id=0)
    req.cache_request_handle = handle
    flow.lookup = SimpleNamespace(request_id=req.rid)
    flow.load_req = req
    flow.load_completed = False
    flow.released_skip_tokens = 0
    flow.anchor_node = flow.anchor_lock = None
    flow.load.ready = False
    flow.load.result = None
    cache.prefetch_loaded_tokens_by_reqid = {}
    cache.prefetch_loaded_storage_start_by_reqid = {}
    cache._session_finish_requested = set()
    store = SimpleNamespace(request_id=req.rid, ready=False)
    has_store = first_completed != "no_store"
    cache._pending_stores = (
        [SimpleNamespace(operation=store, node_id=42, lock_params=object())]
        if has_store
        else []
    )
    cache._pending_store_counts = {req.rid: 1} if has_store else {}
    cache.token_to_kv_pool_allocator = mock.Mock()
    cache.dec_lock_ref = mock.Mock()
    connector = cache.lmcache_connector = mock.Mock()
    connector.ready_prefix_count.side_effect = lambda operations: sum(
        op.ready for op in operations
    )

    def complete_load(operation):
        operation.result = load_success
        return load_success

    connector.complete_load.side_effect = complete_load
    with mock.patch.object(UnifiedRadixCache, "on_release"):
        cache.on_release(req, checkpointed=False)
    cache.check_hicache_events()
    connector.finish_request.assert_not_called()
    cache.token_to_kv_pool_allocator.free.assert_not_called()

    if has_store:
        first = flow.load if first_completed == "load" else store
        first.ready = True
        cache.check_hicache_events()
        connector.finish_request.assert_not_called()
    flow.load.ready = store.ready = True
    cache.check_hicache_events()
    cache.check_hicache_events()
    connector.finish_request.assert_called_once_with(req.rid)
    connector.complete_load.assert_called_once_with(flow.load)
    cache.token_to_kv_pool_allocator.free.assert_called_once()
    torch.testing.assert_close(
        cache.token_to_kv_pool_allocator.free.call_args.args[0],
        flow.load.device_indices,
    )
    assert not cache.has_pending_cache_operations()


def _deferred_load_cache(cache_class, *, mamba_enabled=False):
    cache, req, flow = _cache_and_request(cache_class, swa_enabled=False)
    req.cache_request_handle = CacheRequestHandle(rid=req.rid, attempt_id=0)
    req.full_untruncated_fill_ids = list(range(8))
    req.extra_key, req.cache_salt = None, ""
    req.finished = lambda: True
    flow.lookup = SimpleNamespace(request_id=req.rid)
    flow.load = None
    flow.local_hit_tokens = 2
    flow.load_completed = flow.cancelled = flow.retire_requested = False
    flow.released_skip_tokens = 0
    cache._mamba_component = object() if mamba_enabled else None
    cache._forward_stream = "forward"
    cache._pending_stores, cache._pending_store_counts = [], {}
    cache._session_finish_requested = set()
    cache.prefetch_loaded_tokens_by_reqid = {}
    cache.prefetch_loaded_storage_start_by_reqid = {}
    cache.tree_core = SimpleNamespace(is_eagle=False)
    cache.page_size = 2
    cache.req_to_token_pool = mock.Mock()
    allocator = cache.token_to_kv_pool_allocator = mock.Mock()
    allocator.available_size.return_value = 32
    allocator.alloc.return_value = torch.arange(2, 8)
    allocator.state_slots.alloc.side_effect = [torch.tensor([11]), torch.tensor([12])]
    cache.inc_lock_ref = mock.Mock(return_value=IncLockRefResult(node_id=42))
    cache.dec_lock_ref = mock.Mock()
    future = Future()
    operation = SimpleNamespace(
        request_id=req.rid,
        future=future,
        result=None,
        device_indices=allocator.alloc.return_value,
        local_hit_tokens=2,
    )
    operation.query = lambda: operation.result is not None or future.done()
    connector = cache.lmcache_connector = mock.Mock()
    connector.chunk_size = 2
    connector.operation_timeout = 0
    connector.submit_load.return_value = operation
    connector.get_store_start.return_value = 8
    connector.ready_prefix_count.side_effect = lambda operations: sum(
        op.query() for op in operations
    )

    def decline_load(load, stream):
        # A failed admission/drain can cache False without finishing device work.
        load.result = False
        return False

    def complete_load(load, **kwargs):
        load.result = load.future.result(timeout=0)
        return load.result

    connector.prepare_load_on_stream.side_effect = decline_load
    connector.complete_load.side_effect = complete_load
    return cache, req, flow, operation, future


@pytest.mark.parametrize("mamba_enabled", [False, True])
@pytest.mark.parametrize("load_success", [False, True])
def test_declined_load_retains_slots_until_transfer_completes(
    cache_class, mamba_enabled, load_success
):
    cache, req, flow, operation, future = _deferred_load_cache(
        cache_class, mamba_enabled=mamba_enabled
    )
    local = MatchResult(
        device_prefix_len=2,
        last_device_node=42,
        last_host_node=None,
        best_match_node=None,
    )
    with mock.patch.object(UnifiedRadixCache, "match_prefix", return_value=local):
        assert cache.init_load_back(InitLoadBackParams(42, 6, req=req)) == (0, 42)
        assert cache.check_prefetch_progress(req.cache_request_handle)
        assert (
            cache.match_prefix(MatchPrefixParams(flow.key, req=req, cow_mamba=True))
            == local
        )
        assert req.kv.mamba_cow_src_index is None
        if mamba_enabled:
            # Dense fallback may allocate new mutable state while H2D is pending.
            req.kv.mamba_pool_idx = torch.tensor(99)
        with mock.patch.object(UnifiedRadixCache, "checkpoint"):
            cache.checkpoint(req, up_to=8)
    assert req.kv.cache_protected_len == 8
    cache.check_hicache_events()
    cache.lmcache_connector.complete_load.assert_not_called()
    cache.lmcache_connector.finish_request.assert_not_called()
    cache.token_to_kv_pool_allocator.free.assert_not_called()
    cache.dec_lock_ref.assert_not_called()
    state_free = cache.token_to_kv_pool_allocator.state_slots.free
    if mamba_enabled:
        state_free.assert_called_once()
        torch.testing.assert_close(state_free.call_args.args[0], torch.tensor([12]))
    future.set_result(load_success)
    cache.check_hicache_events()
    cache.check_hicache_events()
    cache.lmcache_connector.finish_request.assert_called_once_with(req.rid)
    cache.token_to_kv_pool_allocator.free.assert_called_once()
    torch.testing.assert_close(
        cache.token_to_kv_pool_allocator.free.call_args.args[0],
        operation.device_indices,
    )
    cache.dec_lock_ref.assert_called_once()
    if mamba_enabled:
        assert state_free.call_count == 2
        torch.testing.assert_close(state_free.call_args.args[0], torch.tensor([11]))
        assert req.kv.mamba_pool_idx.item() == 99
    assert not cache.has_pending_cache_operations()


def test_reset_waits_for_all_transfers_before_releasing_any_slots(cache_class):
    cache, req, flow, operation, future = _deferred_load_cache(cache_class)
    local = MatchResult(
        device_prefix_len=2,
        last_device_node=42,
        last_host_node=None,
        best_match_node=None,
    )
    with mock.patch.object(UnifiedRadixCache, "match_prefix", return_value=local):
        cache.init_load_back(InitLoadBackParams(42, 6, req=req))
    stored = Future()
    cache._pending_stores = [
        SimpleNamespace(
            operation=SimpleNamespace(future=stored), node_id=43, lock_params=object()
        )
    ]
    with mock.patch.object(UnifiedRadixCache, "reset") as reset_tree:
        with pytest.raises(TimeoutError):
            cache.reset()
        future.set_result(True)
        with pytest.raises(TimeoutError):
            cache.reset()
        cache.token_to_kv_pool_allocator.free.assert_not_called()
        cache.dec_lock_ref.assert_not_called()
        cache.lmcache_connector.end_all_sessions.assert_not_called()
        reset_tree.assert_not_called()
        stored.set_result(True)
        cache.reset()
        cache.reset()
    cache.token_to_kv_pool_allocator.free.assert_called_once()
    assert cache.dec_lock_ref.call_count == 2
    assert not cache.has_pending_cache_operations()


@pytest.mark.parametrize("local_tokens", [0, 2])
def test_sparse_lookup_does_not_publish_a_contiguous_prefix(cache_class, local_tokens):
    """Disjoint external hits survive admission without inflating prefix coverage."""
    cache, req, flow, _, _ = _deferred_load_cache(cache_class)
    cache._blend_ratio = 0.25
    flow.total_hit = None
    flow.anchor_node = flow.anchor_lock = None
    flow.lookup.lock_start = 0
    flow.lookup.blend_result = SimpleNamespace(
        non_prefix_segments=[SimpleNamespace(cur_st=4, cur_ed=8)],
        segmented_prefix_segments=[],
    )
    cache.lmcache_connector.poll_lookup.return_value = 0
    local = MatchResult(
        device_prefix_len=local_tokens,
        last_device_node=42,
        last_host_node=None,
        best_match_node=None,
    )
    with mock.patch.object(UnifiedRadixCache, "match_prefix", return_value=local):
        assert cache.check_prefetch_progress(req.cache_request_handle)
        assert cache.prefetch_loaded_tokens_by_reqid[req.cache_request_handle] == 0
        assert cache.match_prefix(MatchPrefixParams(flow.key, req=req)) == local
    plan = cache.prefill_execution_plan([req.rid], [local_tokens])
    assert plan.lookups == (flow.lookup,)
    mixed = cache.prefill_execution_plan([req.rid, "short"], [local_tokens, 0])
    assert mixed.lookups == (flow.lookup, None)
    with mock.patch.object(UnifiedRadixCache, "on_release"):
        cache.on_release(req, checkpointed=False)
    assert not cache.has_pending_cache_operations()
    cache.lmcache_connector.finish_request.assert_called_once_with(req.rid)


def test_approximate_prefill_is_not_published_or_stored(cache_class):
    """A repeated prompt must not acquire an approximate result as an exact hit."""
    cache, req, flow, _, _ = _deferred_load_cache(cache_class)
    cache._blended_requests.add(req.rid)
    flow.total_hit = 0
    flow.lookup.lock_start = 0
    with mock.patch.object(UnifiedRadixCache, "checkpoint") as publish:
        cache.checkpoint(req, up_to=8)
    publish.assert_not_called()
    cache.lmcache_connector.submit_store.assert_not_called()
    cache.lmcache_connector.finish_request.assert_called_once_with(req.rid)
    assert not cache.has_pending_cache_operations()
    with mock.patch.object(UnifiedRadixCache, "on_release"):
        cache.on_release(req, checkpointed=True)
    assert req.rid not in cache._blended_requests


def test_rotary_registration_reuses_only_the_same_table_and_layout():
    connector = mock.Mock()
    registration = BlendRotaryRegistration(connector)
    table = torch.zeros(8, 4)
    registration.ensure_registered(table, is_neox_style=True)
    registration.ensure_registered(table, is_neox_style=True)
    connector.register_blend_rope.assert_called_once_with(table, is_neox_style=True)

    replacement = table.clone()
    registration.ensure_registered(replacement, is_neox_style=True)
    registration.ensure_registered(replacement, is_neox_style=False)
    assert connector.register_blend_rope.call_count == 3
    connector.register_blend_rope.assert_called_with(replacement, is_neox_style=False)


def test_rotary_registration_retries_after_failure():
    connector = mock.Mock()
    registration = BlendRotaryRegistration(connector)
    table = torch.zeros(8, 4)
    registration.ensure_registered(table, is_neox_style=True)
    connector.register_blend_rope.side_effect = [
        RuntimeError("registration failed"),
        None,
    ]
    with pytest.raises(RuntimeError, match="registration failed"):
        registration.ensure_registered(table.clone(), is_neox_style=True)
    registration.ensure_registered(table, is_neox_style=True)
    assert connector.register_blend_rope.call_count == 3


def test_blend_matches_never_overwrite_shared_prefix():
    """A span crossing the shared prefix must be recomputed, not transferred."""
    from sglang.srt.mem_cache.storage.lmcache.blend_prefill import select_blend_matches

    prefix = SimpleNamespace(cur_st=0, cur_ed=4)
    crossing = SimpleNamespace(cur_st=2, cur_ed=6)
    suffix = SimpleNamespace(cur_st=4, cur_ed=8)
    tail = SimpleNamespace(cur_st=8, cur_ed=12)
    lookup = SimpleNamespace(
        blend_result=SimpleNamespace(
            non_prefix_segments=[prefix, crossing, suffix],
            segmented_prefix_segments=[tail],
        )
    )
    assert select_blend_matches(lookup, 4) == [suffix, tail]
    assert select_blend_matches(None, 0) == []


def test_partial_recompute_publishes_only_exact_loaded_prefix(cache_class):
    """Remote prefix slots are adopted before the approximate suffix is released."""
    cache, req, flow, operation, _ = _deferred_load_cache(cache_class)
    cache._blended_requests.add(req.rid)
    flow.load = operation
    flow.load_completed = True
    flow.anchor_node = flow.anchor_lock = None
    operation.result = True
    published = []

    def adopt_prefix(request, *, up_to):
        assert request.kv.cache_protected_len == 2
        published.append(up_to)
        request.kv.protect_prefix(up_to)

    with mock.patch.object(UnifiedRadixCache, "checkpoint", side_effect=adopt_prefix):
        cache.checkpoint(req, up_to=12)
    assert published == [8]
    assert req.kv.cache_protected_len == 8
    cache.lmcache_connector.submit_store.assert_not_called()
    cache.token_to_kv_pool_allocator.free.assert_not_called()
    assert not cache.has_pending_cache_operations()


@pytest.mark.parametrize("lora_ids", [["adapter", None], [None, "adapter"]])
def test_blend_lora_batch_uses_dense_forward(lora_ids):
    """Mixed adapter/base requests keep their original rows and forward options."""
    connector = mock.Mock()
    plan = BlendPrefillPlan(
        connector, BlendRotaryRegistration(connector), (None, None), 0.25, 64, set()
    )
    batch = SimpleNamespace(
        lora_ids=lora_ids, input_ids=torch.arange(8), positions=torch.arange(8)
    )
    model = mock.Mock()

    result = plan.execute(model, batch, pp_proxy_tensors=None)

    assert result is model.forward.return_value
    model.forward.assert_called_once_with(
        batch.input_ids, batch.positions, batch, pp_proxy_tensors=None
    )
    assert connector.mock_calls == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


@pytest.mark.parametrize("blend_enabled", [False, True])
@pytest.mark.parametrize("loaded_prefix", [False, True])
@pytest.mark.parametrize("phase", ["partial", "final", "finished"])
def test_checkpoint_preserves_blend_lookup_until_final_prefill(
    cache_class, blend_enabled, loaded_prefix, phase
):
    """Dense intermediate chunks must not discard later sparse matches."""
    cache, req, flow, operation, _ = _deferred_load_cache(cache_class)
    cache._blend_ratio = 0.5 if blend_enabled else None
    req.full_untruncated_fill_ids = list(range(40))
    req.finished = lambda: phase == "finished"
    flow.anchor_node = flow.anchor_lock = None
    flow.lookup.lock_start = 0
    flow.lookup.blend_result = SimpleNamespace(
        non_prefix_segments=[SimpleNamespace(cur_st=24, cur_ed=32)],
        segmented_prefix_segments=[],
    )
    if loaded_prefix:
        flow.load = operation
        flow.load_req = req
        flow.load_completed = True
        operation.result = True
    up_to = 40 if phase == "final" else 16
    with (
        mock.patch.object(UnifiedRadixCache, "checkpoint"),
        mock.patch.object(cache_class, "_submit_store"),
    ):
        cache.checkpoint(req, up_to=up_to)
    plan = cache.prefill_execution_plan([req.rid], [16])
    retained = blend_enabled and phase == "partial"
    assert (plan is not None) == retained
    if retained:
        assert plan.lookups == (flow.lookup,)
        cache.lmcache_connector.free_lookup_locks.assert_not_called()
        cache.lmcache_connector.finish_request.assert_not_called()
        with mock.patch.object(UnifiedRadixCache, "on_release"):
            cache.on_release(req, checkpointed=False)
        assert not cache.has_pending_cache_operations()
        cache.lmcache_connector.finish_request.assert_called_once_with(req.rid)
        cache.token_to_kv_pool_allocator.free.assert_not_called()
    if phase == "finished":
        cache.lmcache_connector.finish_request.assert_called_once_with(req.rid)


def test_final_blend_checkpoint_preserves_published_dense_chunks(cache_class):
    """The final approximate suffix must not shorten an earlier exact checkpoint."""
    cache, req, flow, operation, _ = _deferred_load_cache(cache_class)
    cache._blend_ratio = 0.5
    cache._blended_requests.add(req.rid)
    req.full_untruncated_fill_ids = list(range(40))
    req.kv.protect_prefix(16)
    flow.load = operation
    flow.load_completed = flow.prefix_published = True
    flow.anchor_node = flow.anchor_lock = None
    operation.result = True
    with mock.patch.object(UnifiedRadixCache, "checkpoint") as publish:
        cache.checkpoint(req, up_to=40)
    publish.assert_not_called()
    assert req.kv.cache_protected_len == 16
    cache.lmcache_connector.submit_store.assert_not_called()
    cache.lmcache_connector.finish_request.assert_called_once_with(req.rid)
