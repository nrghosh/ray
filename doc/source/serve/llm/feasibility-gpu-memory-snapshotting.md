# GPU Memory Snapshotting for Ray Serve LLM: Feasibility Analysis

## Executive Summary

GPU memory snapshotting is a technique that can reduce inference server startup times by 10x by serializing the complete program state (model weights, compiled kernels, KV cache structures, CUDA graphs) to storage and restoring it on new replicas instead of re-initializing from scratch.

**Feasibility Assessment: Partially Feasible with Significant Engineering Effort**

- **Short-term (achievable now)**: 2-4x speedup via torch.compile cache persistence and model weight pre-staging
- **Medium-term (requires vLLM coordination)**: 5-7x speedup via CUDA graph serialization and memory allocation restoration
- **Long-term (research required)**: 10x speedup via full GPU state snapshotting (requires CUDA/driver-level support)

---

## 1. Current Ray Serve LLM Startup Architecture

### 1.1 Startup Phase Breakdown

Based on the current implementation in `python/ray/llm/_internal/serve/engines/vllm/vllm_engine.py`, the startup sequence consists of:

| Phase | Description | Typical Duration | % of Total |
|-------|-------------|------------------|------------|
| **Node Provisioning** | GPU instance spin-up (if needed) | 30-120s | Variable |
| **Image Download** | Container/runtime image pull | 10-60s | Variable |
| **Ray Initialization** | Process spawning, library imports | 10-20s | 5-10% |
| **Model Download** | Weights from HuggingFace/S3/GCS | 30-300s | 15-40% |
| **Weight Loading** | CPU → GPU memory transfer | 10-60s | 5-15% |
| **Torch Compile** | JIT compilation of model kernels | 60-300s | 20-40% |
| **Memory Profiling** | vLLM KV cache sizing inference | 5-15s | 2-5% |
| **CUDA Graph Capture** | Pre-capture graphs for batch sizes | 30-120s | 10-20% |
| **Warmup** | Initial inference runs | 5-10s | 2-5% |

**Total typical startup: 3-10 minutes** for production models (8B-70B parameters)

### 1.2 Current Architecture Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Ray Serve Application                        │
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │ LLMServer   │  │ LLMServer   │  │ LLMServer   │  (Replicas)     │
│  │  Replica 1  │  │  Replica 2  │  │  Replica N  │                 │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                 │
│         │                │                │                         │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐                 │
│  │ VLLMEngine  │  │ VLLMEngine  │  │ VLLMEngine  │                 │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                 │
│         │                │                │                         │
│  ┌──────▼──────────────────────────────────────────────────┐       │
│  │              vLLM AsyncLLM + Ray Executor               │       │
│  │  ┌─────────────────────────────────────────────────┐    │       │
│  │  │  GPU Memory Layout:                             │    │       │
│  │  │  ┌──────────┐ ┌──────────┐ ┌────────────────┐  │    │       │
│  │  │  │  Model   │ │  CUDA    │ │   KV Cache     │  │    │       │
│  │  │  │ Weights  │ │  Graphs  │ │   Blocks       │  │    │       │
│  │  │  │  (FP16)  │ │          │ │   (PagedAttn)  │  │    │       │
│  │  │  └──────────┘ └──────────┘ └────────────────┘  │    │       │
│  │  └─────────────────────────────────────────────────┘    │       │
│  └─────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 Existing State Management Mechanisms

Ray Serve LLM already has several mechanisms that partially address this problem:

#### Sleep/Wakeup (in `vllm_engine.py:626-656`)
```python
class VLLMSleepConfig:
    level: int = 1  # Level 1: Offload weights to CPU RAM
                    # Level 2: Discard both weights and KV cache
```
- **Level 1**: Model weights → CPU RAM, KV cache discarded
- **Level 2**: Everything discarded (deeper sleep)
- **Wakeup**: Selective restoration via tags ("weights", "kv_cache")

#### Pause/Resume (in `vllm_engine.py:657-693`)
```python
class VLLMPauseConfig:
    wait_for_inflight_requests: bool = False
    clear_cache: bool = True  # Preserve cache for faster resume
```
- Halts generation while keeping weights in GPU memory
- Faster than sleep for temporary pauses

#### KV Transfer Infrastructure (`python/ray/llm/_internal/serve/engines/vllm/kv_transfer/`)
- LMCache connector for distributed KV cache sharing
- NIXL connector for NVIDIA inference transfer library
- Enables cross-replica KV cache transfer

---

## 2. What GPU Memory Snapshotting Would Require

### 2.1 State Components to Snapshot

| Component | Size (8B Model) | Size (70B Model) | Serializable? |
|-----------|-----------------|------------------|---------------|
| Model Weights (FP16) | ~16 GB | ~140 GB | Yes |
| Model Weights (INT4) | ~4 GB | ~35 GB | Yes |
| CUDA Graphs | ~1-5 GB | ~5-20 GB | **Partially** |
| Compiled Kernels | ~100 MB | ~500 MB | Yes (cache dir) |
| KV Cache Blocks | ~4-16 GB | ~20-80 GB | **Yes** |
| Memory Allocator State | ~1 MB | ~10 MB | **Difficult** |
| Scheduler State | ~10 KB | ~100 KB | Yes |

### 2.2 Technical Challenges by Component

#### Model Weights (Feasible)
```python
# Current approach in sleep/wakeup
await engine_client.sleep(level=1)  # Weights → CPU
await engine_client.wake_up(tags=["weights"])  # CPU → GPU
```
- Already supported via vLLM sleep/wakeup
- Could extend to serialize to disk instead of CPU RAM
- **Speedup potential**: Skip model download, direct GPU load from local NVMe

#### Torch Compile Cache (Feasible - Already Supported)
```python
# From deployment-initialization.md
engine_kwargs={
    "compilation_config": {
        "cache_dir": "/home/ray/.cache/vllm/torch_compile_cache/model-cache",
    }
}
```
- Directory-based caching already exists
- Can pre-download from S3 via CloudDownloader callback
- **Speedup potential**: 60-300s → 5-10s

#### CUDA Graphs (Challenging)
```
Current state:
- CUDA graphs are captured during startup for various batch sizes
- Capture includes kernel launch parameters, memory addresses
- Memory addresses are NOT portable across processes

Snapshotting challenge:
- CUDA graphs contain device pointers that are process-specific
- Need to either:
  1. Re-capture graphs on restore (loses benefit)
  2. Use CUDA driver APIs to serialize/restore (not publicly available)
  3. Re-map memory addresses during restore (complex)
```
**Speedup potential if solved**: 30-120s → 1-5s (significant)

#### KV Cache Block Allocator (Challenging)
```
PagedAttention allocates fixed-size blocks for KV cache.
Snapshotting requires:
1. Saving block allocation table
2. Saving allocated block contents (if warm restore desired)
3. Restoring allocator with same memory layout

Challenge: Memory fragmentation and address consistency
```

#### vLLM Scheduler State (Feasible)
```python
# Scheduler state is relatively simple:
- pending_requests: List[Request]
- running_requests: List[Request]
- block_manager state
```
- Could serialize with cloudpickle
- Useful for warm restores with preserved request state

---

## 3. Implementation Approaches

### 3.1 Approach A: Enhanced Sleep/Wakeup (Lowest Risk)

**Concept**: Extend existing sleep mechanism to serialize to storage instead of CPU RAM.

```python
# Proposed extension
class VLLMSnapshotConfig:
    snapshot_path: str  # Local NVMe or network storage
    include_weights: bool = True
    include_kv_cache: bool = False  # Optional warm cache preservation
    compression: str = "lz4"  # Fast compression

async def snapshot(self, config: VLLMSnapshotConfig) -> str:
    """Serialize engine state to storage."""
    # 1. Pause generation
    await self.pause(wait_for_inflight_requests=True)

    # 2. Serialize weights to storage (instead of CPU)
    await self._serialize_weights(config.snapshot_path)

    # 3. Optionally serialize KV cache
    if config.include_kv_cache:
        await self._serialize_kv_cache(config.snapshot_path)

    # 4. Return snapshot metadata
    return snapshot_id

async def restore(self, snapshot_id: str) -> None:
    """Restore engine state from snapshot."""
    # Skip normal initialization, load directly
    await self._deserialize_weights(snapshot_id)
    await self.resume()
```

**Pros**:
- Builds on existing vLLM primitives
- Minimal risk to stability
- Can be implemented incrementally

**Cons**:
- Doesn't address CUDA graph capture time
- Doesn't address torch compile time (though cache already exists)
- Still requires memory profiling

**Expected Speedup**: 2-3x

### 3.2 Approach B: Full State Serialization (Medium Risk)

**Concept**: Serialize complete engine state including CUDA graphs using experimental APIs.

```python
# Requires coordination with vLLM upstream

class FullSnapshotConfig:
    path: str
    include_cuda_graphs: bool = True
    include_memory_layout: bool = True

async def create_full_snapshot(engine, config: FullSnapshotConfig):
    """
    1. Quiesce engine (no inflight requests)
    2. Export model state_dict
    3. Export CUDA graph topology (not addresses)
    4. Export memory allocation map
    5. Export scheduler state
    6. Package into single snapshot file
    """

async def restore_from_full_snapshot(path: str) -> VLLMEngine:
    """
    1. Allocate GPU memory with same layout
    2. Load weights directly to GPU
    3. Re-capture CUDA graphs (fast, deterministic)
    4. Restore scheduler state
    """
```

**Key insight**: Even if we can't serialize CUDA graphs directly, we can:
1. Record the graph capture parameters (batch sizes, seq lengths)
2. Pre-allocate memory with known layout
3. Re-capture graphs deterministically (faster than profiling + capture)

**Expected Speedup**: 4-6x

### 3.3 Approach C: CRIU/GPU Checkpoint (Highest Risk, Highest Reward)

**Concept**: Use CUDA-aware checkpoint/restore at the process level.

```
External dependencies:
- NVIDIA CUDA Checkpoint (experimental, not public)
- CRIU with GPU support (experimental)
- Container-level checkpointing (Kubernetes + GPU)
```

**Current State**:
- NVIDIA has internal CUDA checkpoint/restore capabilities
- Not yet available as public API
- Some support in recent CUDA drivers (12.x+) for MIG/vGPU

**Ray Integration Path**:
```python
# Future possibility with CUDA checkpoint API
async def cuda_checkpoint(replica_actor) -> str:
    """
    Uses NVIDIA's CUDA checkpoint API to:
    1. Freeze GPU execution
    2. Serialize all GPU state (memory, graphs, streams)
    3. Write to checkpoint file
    """

async def cuda_restore(checkpoint_path: str) -> Actor:
    """
    1. Create new process
    2. Restore GPU state from checkpoint
    3. Resume execution
    """
```

**Expected Speedup**: 10x+ (full initialization skip)

**Risks**:
- Depends on NVIDIA releasing public APIs
- May have GPU architecture requirements
- Complex interaction with Ray's actor model

---

## 4. Impact Assessment

### 4.1 Performance Impact

| Approach | Startup Time Reduction | Model Size Scaling |
|----------|------------------------|-------------------|
| Current (no snapshot) | Baseline | O(n) with parameters |
| Approach A (Enhanced Sleep) | 2-3x | O(n) with parameters |
| Approach B (Full State) | 4-6x | O(n) with parameters |
| Approach C (CUDA Checkpoint) | 10x | O(1) constant |

### 4.2 Use Case Impact

| Use Case | Current Pain | With Snapshotting |
|----------|--------------|-------------------|
| **Autoscaling** | 3-10 min lag on scale-up | 30s-2min lag |
| **Spot Instance Recovery** | Full restart, high cost | Fast recovery |
| **Blue-Green Deployment** | Long rollout windows | Rapid switchover |
| **Multi-tenant Scaling** | Cold start per tenant | Warm cache sharing |
| **Development/Testing** | Slow iteration cycles | Fast model reload |

### 4.3 Resource Impact

| Resource | Impact |
|----------|--------|
| **Storage** | +16-150 GB per model per snapshot |
| **Network** | Initial snapshot distribution overhead |
| **Memory** | Potential CPU RAM reduction (no offload buffer) |
| **GPU** | No change in steady-state utilization |

---

## 5. Implementation Scope

### 5.1 Phase 1: Torch Compile Cache Optimization (2-4 weeks)

**Already partially implemented. Enhancements:**

1. **Automatic cache generation workflow**
   ```python
   # New CLI tool
   ray llm generate-compile-cache \
       --model meta-llama/Llama-3-70B \
       --output s3://bucket/compile-cache/
   ```

2. **Built-in cache distribution**
   - Integrate with CloudDownloader
   - Automatic cache versioning by vLLM version

3. **Cache validation**
   - Verify cache compatibility before use
   - Fall back to compilation on mismatch

**Files to modify:**
- `python/ray/llm/_internal/common/callbacks/cloud_downloader.py`
- `python/ray/llm/_internal/serve/core/configs/llm_config.py`
- New: `python/ray/llm/_internal/serve/utils/compile_cache_utils.py`

### 5.2 Phase 2: Weight Snapshot System (4-8 weeks)

**New snapshot/restore API:**

1. **Snapshot creation**
   ```python
   from ray.serve.llm import LLMConfig, create_snapshot

   config = LLMConfig(model_loading_config={"model_id": "llama-70b", ...})
   snapshot_uri = await create_snapshot(
       config,
       output_path="s3://bucket/snapshots/",
       include_weights=True
   )
   ```

2. **Snapshot-based deployment**
   ```python
   config = LLMConfig(
       model_loading_config={
           "model_id": "llama-70b",
           "snapshot_uri": "s3://bucket/snapshots/llama-70b-v1",
       }
   )
   ```

3. **Direct GPU loading**
   - Use RunAI streamer with sharded weights
   - Memory-mapped loading from NVMe

**Files to create/modify:**
- New: `python/ray/llm/_internal/serve/snapshot/`
  - `snapshot_manager.py`
  - `weight_serializer.py`
  - `snapshot_config.py`
- `python/ray/llm/_internal/serve/engines/vllm/vllm_engine.py`
- `python/ray/llm/_internal/serve/core/server/llm_server.py`

### 5.3 Phase 3: CUDA Graph Optimization (8-12 weeks)

**Requires vLLM upstream coordination:**

1. **Graph capture parameter recording**
   - Record batch sizes, sequence lengths used for capture
   - Store as metadata with snapshot

2. **Deterministic graph re-capture**
   - Use recorded parameters for fast re-capture
   - Skip memory profiling with known layout

3. **Memory layout restoration**
   - Allocate KV cache blocks with deterministic addresses
   - Pre-compute block allocation map

**Integration points:**
- vLLM `CUDAGraphRunner` class
- vLLM `BlockAllocator` interface
- Ray `python/ray/llm/_internal/serve/engines/vllm/`

### 5.4 Phase 4: Full GPU State (Research, 12+ weeks)

**Requires external dependencies:**

1. **NVIDIA CUDA Checkpoint API**
   - Monitor NVIDIA roadmap for public API
   - Prototype with experimental features

2. **Container-level checkpointing**
   - Investigate Kubernetes GPU checkpoint support
   - CRIU with CUDA extensions

3. **Ray Actor checkpointing**
   - Integration with Ray's actor fault tolerance
   - Automatic snapshot on scale-down

---

## 6. Risks and Mitigations

### 6.1 Technical Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| CUDA graph non-portability | High | Focus on fast re-capture, not serialization |
| vLLM version incompatibility | Medium | Version-tagged snapshots, validation |
| Storage bandwidth bottleneck | Medium | Compression, parallel loading, NVMe |
| Memory layout changes | High | Validation checksums, fallback to full init |
| GPU architecture dependencies | Medium | Architecture-specific snapshots |

### 6.2 Operational Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Snapshot storage costs | Low | Lifecycle policies, deduplication |
| Snapshot staleness | Medium | Automatic refresh on model update |
| Security (model weights exposure) | Medium | Encryption at rest, access controls |
| Debugging complexity | Medium | Detailed logging, validation mode |

---

## 7. Recommendations

### 7.1 Short-term (Next Quarter)

1. **Productionize torch.compile cache workflow**
   - Automatic cache generation in CI/CD
   - Documentation and best practices
   - Integration tests

2. **Improve weight loading**
   - Default to RunAI streamer for S3/GCS models
   - Pre-sharded weight support
   - Parallel multi-GPU loading

### 7.2 Medium-term (Next 6 Months)

3. **Implement weight snapshot system**
   - Snapshot/restore API
   - Integration with autoscaler
   - Spot instance recovery integration

4. **Optimize CUDA graph capture**
   - Work with vLLM team on deterministic capture
   - Graph parameter recording and replay

### 7.3 Long-term (Next Year)

5. **Investigate CUDA checkpoint**
   - Track NVIDIA roadmap
   - Prototype with experimental APIs
   - Container-level checkpointing research

6. **Full integration**
   - Automatic snapshot management
   - Cross-replica state sharing
   - Predictive pre-warming

---

## 8. Conclusion

GPU memory snapshotting is a promising technique for reducing Ray Serve LLM startup times, with the potential for 10x improvements. However, the full vision requires CUDA-level support that is not yet publicly available.

**Recommended path forward:**
1. Maximize existing mechanisms (torch.compile cache, RunAI streamer) for 2-3x improvement
2. Build weight snapshot infrastructure for 3-5x improvement
3. Partner with vLLM on CUDA graph optimization for 5-7x improvement
4. Monitor NVIDIA checkpoint APIs for eventual 10x improvement

The existing sleep/wakeup and KV transfer infrastructure in Ray Serve LLM provides a strong foundation for incremental improvements while the ecosystem matures toward full GPU state checkpointing.

---

## 9. Ray Data LLM Batch Workloads: Extended Analysis

### 9.1 Applicability to Batch Inference

The GPU memory snapshotting techniques analyzed above apply equally to Ray Data LLM batch workloads, but with different cost-benefit tradeoffs.

**Ray Data LLM Architecture** (`python/ray/llm/_internal/batch/`):
- Uses `ActorPoolStrategy` with vLLM/SGLang engines per actor
- Each actor initializes a full engine (same startup cost as Serve replicas)
- Default: `max_tasks_in_flight_per_actor=4`, `max_concurrent_batches=8`

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Ray Data Pipeline                            │
│                                                                     │
│  ds.map_batches(VLLMEngineStage, compute=ActorPoolStrategy(...))   │
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │ GPU Actor 1 │  │ GPU Actor 2 │  │ GPU Actor N │  (Actor Pool)   │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │                 │
│  │ │  vLLM   │ │  │ │  vLLM   │ │  │ │  vLLM   │ │                 │
│  │ │ Engine  │ │  │ │ Engine  │ │  │ │ Engine  │ │                 │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.2 Batch vs. Serve: When Does Startup Matter?

| Factor | Ray Serve | Ray Data Batch |
|--------|-----------|----------------|
| **Job Duration** | Indefinite (always-on) | Minutes to hours |
| **Startup Amortization** | Amortized over many requests | Amortized over dataset |
| **Failure Frequency** | Rare (health checks) | Common (spot preemption) |
| **Recovery Urgency** | High (SLA, latency) | Medium (throughput) |
| **Cost Model** | Per-request latency | Per-token throughput |

**Key Insight**: For batch jobs, initialization cost matters primarily when:
1. **Job duration < 10x startup time**: Short jobs (< 30 min) feel the pain
2. **High failure rate**: Spot instances with 10-20% preemption rate
3. **Large actor pools**: N actors × startup time = significant cluster cost

### 9.3 Current Ray Data Fault Tolerance

From `python/ray/llm/_internal/batch/stages/vllm_engine_stage.py:701-722`:

```python
except _VLLM_FATAL_ERRORS as e:  # EngineDeadError
    # Fatal engine errors indicate the vLLM engine subprocess is dead
    # but the Ray actor is still alive.
    #
    # Fix: exit the actor so Ray can restart it with a fresh engine.
    # Ray Data's max_restarts=-1 (default) will create a replacement
    # actor, and task retries will go to healthy actors.
    logger.error(f"[vLLM] Fatal engine error, exiting actor: {e}")
    os._exit(1)
```

**Current behavior on failure:**
1. Actor dies → Ray detects failure
2. Ray creates replacement actor
3. **Full engine re-initialization** (3-10 min)
4. Failed tasks retried on healthy actors

**Gap**: No mechanism to speed up step 3.

### 9.4 Proposed Approaches for Batch Resilience

#### Approach 1: Hot Standby Pool ("Stay Ready")

**Concept**: Maintain extra initialized GPU actors that can immediately take over on failure.

```python
# Proposed configuration
class WarmPoolConfig:
    min_warm_replicas: int = 1  # Always keep N actors ready
    preemption_buffer: float = 0.1  # 10% extra for spot recovery
    warmup_on_idle: bool = True  # Use idle time for pre-warming

# Usage in Ray Data
ds.map_batches(
    VLLMEngineStage,
    compute=ActorPoolStrategy(
        min_size=8,
        max_size=16,
        warm_pool=WarmPoolConfig(min_warm_replicas=2),
    ),
)
```

**Architecture**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                        Actor Pool with Warm Standby                 │
│                                                                     │
│  Active Pool (processing)          Warm Pool (idle, ready)         │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐   ┌─────┐ ┌─────┐                │
│  │ A1  │ │ A2  │ │ A3  │ │ A4  │   │ W1  │ │ W2  │                │
│  │ 🔥  │ │ 🔥  │ │ 🔥  │ │ 🔥  │   │ 💤  │ │ 💤  │                │
│  └─────┘ └─────┘ └─────┘ └─────┘   └─────┘ └─────┘                │
│                                          │                          │
│  On A2 failure:                          │                          │
│  ┌─────┐         ┌─────┐ ┌─────┐   ┌─────┤ ┌─────┐                │
│  │ A1  │   ❌    │ A3  │ │ A4  │   │ W1 ─┼─│ W2  │                │
│  │ 🔥  │         │ 🔥  │ │ 🔥  │   │ 🔥  │ │ 💤  │                │
│  └─────┘         └─────┘ └─────┘   └─────┘ └─────┘                │
│                                     ↑ Instant promotion             │
│                                     + Background: new W3 warming    │
└─────────────────────────────────────────────────────────────────────┘
```

**Implementation scope:**

| Component | Effort | Location |
|-----------|--------|----------|
| Warm pool state tracking | Medium | `actor_pool_map_operator.py` |
| Actor promotion logic | Medium | `autoscaling_actor_pool.py` |
| Background warming | Low | New: `warm_pool_manager.py` |
| Sleep/wakeup integration | Low | Already exists in vLLM |
| Configuration API | Low | `ActorPoolStrategy` extension |

**Cost-Benefit Analysis:**

| Factor | Impact |
|--------|--------|
| **GPU cost** | +10-25% (warm standby overhead) |
| **Recovery time** | ~0s (instant promotion) vs 3-10 min |
| **Memory overhead** | Warm actors hold weights in GPU |
| **Complexity** | Medium (pool state management) |

**When to use:**
- High-value batch jobs where downtime is expensive
- Spot instances with high preemption rates
- Jobs with strict completion deadlines

#### Approach 2: Fast-Replace Mechanism ("Accelerated Autoscaling")

**Concept**: Speed up actor replacement without holding extra GPUs idle.

```python
# Proposed configuration
class FastReplaceConfig:
    snapshot_uri: str  # Pre-built snapshot location
    use_local_nvme: bool = True  # Cache snapshot locally
    parallel_init: bool = True  # Initialize while draining failed actor

# Integrates with existing ActorPoolStrategy
ds.map_batches(
    VLLMEngineStage,
    compute=ActorPoolStrategy(
        min_size=8,
        max_size=16,
    ),
    fast_replace=FastReplaceConfig(
        snapshot_uri="s3://bucket/snapshots/llama-70b",
    ),
)
```

**Architecture**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                     Fast Replace on Failure                         │
│                                                                     │
│  Timeline (current):                                                │
│  ───────────────────────────────────────────────────────────────►   │
│  │ Failure │ Provision │ Download │ Load │ Compile │ Ready │       │
│  │         │   Node    │  Model   │ GPU  │ Graphs  │       │       │
│  │    0s   │   30s     │  120s    │ 60s  │  180s   │ ~6min │       │
│                                                                     │
│  Timeline (with fast-replace):                                      │
│  ───────────────────────────────────────────────────────────────►   │
│  │ Failure │ Provision │ Load Snapshot │ Ready │                   │
│  │         │   Node    │  from NVMe    │       │                   │
│  │    0s   │   30s     │     30s       │ ~1min │                   │
│                                                                     │
│  Speedup: ~6x faster recovery                                       │
└─────────────────────────────────────────────────────────────────────┘
```

**Implementation scope:**

| Component | Effort | Location |
|-----------|--------|----------|
| Snapshot creation CLI | Medium | New: `ray llm snapshot` |
| Snapshot-aware engine init | Medium | `vllm_engine_stage.py` |
| Local NVMe caching | Low | Node init callback |
| Parallel initialization | Medium | `actor_pool_map_operator.py` |
| S3/GCS snapshot storage | Low | Existing infra |

**Cost-Benefit Analysis:**

| Factor | Impact |
|--------|--------|
| **GPU cost** | +0% (no idle capacity) |
| **Recovery time** | ~1 min vs 3-10 min (still some delay) |
| **Storage cost** | +$5-50/month per model snapshot |
| **Complexity** | Medium (snapshot management) |

**When to use:**
- Cost-sensitive workloads
- Moderate preemption rates
- Jobs where 1-min recovery is acceptable

### 9.5 Comparison: Hot Standby vs. Fast-Replace

| Dimension | Hot Standby | Fast-Replace |
|-----------|-------------|--------------|
| **Recovery Time** | ~0s (instant) | ~60s |
| **GPU Cost Overhead** | +10-25% | +0% |
| **Storage Cost** | +0% | +$5-50/mo |
| **Implementation Effort** | Medium | Medium |
| **Best For** | High-value, time-critical | Cost-sensitive, tolerant |

### 9.6 Recommendation: Focus Priority

**Should we focus on Serve or Batch?**

| Factor | Serve Priority | Batch Priority |
|--------|----------------|----------------|
| User pain frequency | High (every scale event) | Medium (only on failures) |
| Latency sensitivity | Critical (SLAs) | Lower (throughput-focused) |
| Existing workarounds | Limited | Over-provision, retry |
| Implementation reuse | Foundation for batch | Builds on serve work |

**Recommendation**: **Start with Serve, extend to Batch**

1. The core snapshotting mechanisms (torch compile cache, weight snapshots) benefit both
2. Serve has higher urgency due to autoscaling requirements
3. Batch can leverage the same infrastructure with pool-specific extensions

**Phased approach:**

| Phase | Serve Benefit | Batch Benefit |
|-------|--------------|---------------|
| Phase 1: Compile cache | 2-3x faster startup | 2-3x faster actor init |
| Phase 2: Weight snapshots | 3-5x faster startup | 3-5x faster recovery |
| Phase 3: Hot standby | N/A (already has replicas) | Instant failover |
| Phase 4: Fast-replace | Faster autoscaling | Faster spot recovery |

### 9.7 Batch-Specific Implementation Details

**Changes to Ray Data actor pool** (`python/ray/data/_internal/`):

```python
# actor_pool_map_operator.py - proposed additions

class ActorPoolMapOperator:
    def __init__(
        self,
        ...,
        warm_pool_config: Optional[WarmPoolConfig] = None,
        fast_replace_config: Optional[FastReplaceConfig] = None,
    ):
        self._warm_pool = WarmPoolManager(warm_pool_config)
        self._fast_replace = FastReplaceManager(fast_replace_config)

    def _handle_actor_failure(self, actor_id: str):
        if self._warm_pool.has_warm_actor():
            # Instant promotion
            replacement = self._warm_pool.promote_warm_actor()
            self._reassign_tasks(actor_id, replacement)
            self._warm_pool.start_background_warming()
        elif self._fast_replace.enabled():
            # Fast snapshot-based init
            self._fast_replace.start_replacement(actor_id)
        else:
            # Current behavior: full re-init
            self._start_replacement_actor(actor_id)
```

**Changes to vLLM batch stage** (`python/ray/llm/_internal/batch/stages/`):

```python
# vllm_engine_stage.py - proposed additions

class VLLMEngineStage:
    def __init__(
        self,
        ...,
        snapshot_uri: Optional[str] = None,
    ):
        self._snapshot_uri = snapshot_uri

    def _init_engine(self):
        if self._snapshot_uri:
            # Fast path: load from snapshot
            return self._load_from_snapshot(self._snapshot_uri)
        else:
            # Current path: full initialization
            return self._init_engine_from_scratch()
```

---

## 10. Conclusion (Updated)

GPU memory snapshotting benefits both Ray Serve and Ray Data LLM workloads:

**For Ray Serve**: Primary focus on reducing autoscaling latency and improving responsiveness to demand changes. Implementation provides foundation for batch workloads.

**For Ray Data Batch**: Two complementary approaches address failure recovery:
1. **Hot Standby**: Instant failover at cost of idle GPU capacity
2. **Fast-Replace**: 6x faster recovery with minimal cost overhead

**Recommended priority:**
1. Implement core snapshotting for Serve (benefits both)
2. Add Fast-Replace for batch (low cost, high impact)
3. Add Hot Standby as optional premium feature for time-critical batch jobs

The incremental nature of these improvements allows Ray to deliver value at each phase while building toward the full vision of near-instant GPU workload recovery.

---

## References

- [Ray Serve LLM Deployment Initialization Guide](./user-guides/deployment-initialization.md)
- [vLLM CUDA Graphs Design](https://docs.vllm.ai/en/latest/design/cuda_graphs.html)
- [Ray Serve LLM Benchmarks](https://github.com/anyscale/ray-serve-llm-perf-examples)
- [RunAI Model Streamer](https://docs.vllm.ai/en/stable/models/extensions/runai_model_streamer.html)
- [LMCache for KV Transfer](https://github.com/LMCache/LMCache)
- [Ray Data Actor Pool Fault Tolerance](https://docs.ray.io/en/latest/data/api/doc/ray.data.ActorPoolStrategy.html)
