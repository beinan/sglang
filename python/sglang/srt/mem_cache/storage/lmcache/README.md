# LMCache Connector for SGLang

This document describes how to use LMCache as KV Cache Management Backend for SGLang engine.
For more details about LMCache, please refer to: https://lmcache.ai

## Install LMCache

### Method 1: with pip

```bash
pip install lmcache
```

### Method 2: from source

Clone LMCache project:

```bash
git clone https://github.com/LMCache/LMCache
```

Install:

```bash
cd LMCache
pip install -e . --no-build-isolation
```


## Use LMCache

SGLang uses `LMCacheUnifiedRadixCache` with LMCache's multiprocess connector.
The standalone LMCache daemon owns the external cache and can survive SGLang
process restarts. Daemon host and port come from the LMCache YAML config.

Terminal 1 — start the LMCache daemon:

```bash
lmcache server \
  --host 127.0.0.1 --port 5556 \
  --l1-size-gb 4 \
  --eviction-policy LRU
```

Use the bundled `example_config_mp.yaml` (or any YAML setting `mp_host` / `mp_port`):

Terminal 2 — start SGLang:

```bash
python -m sglang.launch_server \
  --model-path MODEL \
  --enable-lmcache \
  --lmcache-config-file example_config_mp.yaml
```

For full LMCache config options see https://docs.lmcache.ai/api_reference/configurations.html.

If a cache retrieve cannot be admitted, SGLang falls back to local prefill without
publishing the retrieved prefix. Its private KV slots and Mamba checkpoint stay
allocated until the transfer completes, even if admission timed out. This keeps
late remote writes away from slots reused by another request.

Cache reset and shutdown drain outstanding loads and stores before releasing
buffers. If draining raises an error, reset propagates it and retains ownership
so cleanup can be retried after the transfer completes.

## Experimental MP CacheBlend

This section applies to the experimental integration branch, not a released
SGLang/LMCache pair. It requires the SGLang sparse-query and scheduler changes
and an LMCache server/client with protocol-v2 rotary component support.

Start the blend server with a chunk size matching SGLang's page size:

```bash
lmcache server --host 127.0.0.1 --port 5556 \
  --engine-type blend --chunk-size 64 --l1-size-gb 4 --eviction-policy LRU
```

Save the following as `blend.yaml`:

```yaml
mp_host: 127.0.0.1
mp_port: 5556
extra_config:
  lmcache.mp.mp_transfer_mode: lmcache_driven
  lmcache.sglang.blend_recompute_ratio: 0.5
```

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B \
  --enable-lmcache --lmcache-config-file blend.yaml \
  --page-size 64 --dtype bfloat16 --attention-backend flashinfer \
  --tp-size 1 --disable-cuda-graph --disable-overlap-schedule \
  --chunked-prefill-size 8192
```

The tested path uses plain-text Qwen3 full attention, BF16, FlashInfer, one GPU,
and prompts that fit in one prefill chunk. Active LoRA, input log probabilities,
hidden-state capture, and unsupported model/request modes use dense prefill.

The first four layers recompute all suffix rows. Later layers recompute unmatched
rows, each request's final row, and the configured fraction of matched rows with
the largest key changes. The ratio is a fraction of matched rows, not total
model work. Approximate results are excluded from radix and remote cache
writeback. The input token order is unchanged.

Omit `lmcache.sglang.blend_recompute_ratio` to use ordinary prefix caching.
`1.0` recomputes all rows and is useful as a correctness control. The `0.5`
example is an experiment setting, not a recommended quality/performance default:
measure answer quality and TTFT against both ordinary prefix caching and a
no-cache reference on the same questions. A non-prefix lookup hit alone does
not establish selective execution; check scatter metrics and the
`CacheBlend prefill` log's query counts.
