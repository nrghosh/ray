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

## References

- [Ray Serve LLM Deployment Initialization Guide](./user-guides/deployment-initialization.md)
- [vLLM CUDA Graphs Design](https://docs.vllm.ai/en/latest/design/cuda_graphs.html)
- [Ray Serve LLM Benchmarks](https://github.com/anyscale/ray-serve-llm-perf-examples)
- [RunAI Model Streamer](https://docs.vllm.ai/en/stable/models/extensions/runai_model_streamer.html)
- [LMCache for KV Transfer](https://github.com/LMCache/LMCache)
