# A12-A15: Phase 1 Speedup (Colibri Techniques Integration)

**Timeline:** Oct 20 — Nov 15, 2026  
**Target:** >0.50 tok/s (Phase 1B) → >0.85 tok/s (Phase 1C)  
**Hardware:** Windows machine with Intel Arc GPU + NPU, 5 Ollama models cached  
**Baseline:** Colibri's 0.05-0.1 tok/s on CPU

---

## A12: Unified Inference Path Design

**Goal:** Create single code path for all 9 models (OLMoE, GLM, Inkling, K3, DeepSeek, Qwen, Mixtral)

### A12.1: Refactor execution_service.py
```python
# New unified inference path
class UnifiedInferenceExecution:
    def forward(self, input_tokens, batch_size):
        """
        Per-token inference flow:
        1. Route experts (which experts fire for this batch)
        2. Batch-Union (deduplicate expert loads)
        3. Place (decide where each expert lives: VRAM/RAM/disk)
        4. Overlap (async I/O while computing)
        5. Learn (track routing history)
        """
        
        # Step 1: Route
        routing = self.router(input_tokens)  # shape: (batch, num_layers, num_experts)
        
        # Step 2: Batch-Union (expert deduplication)
        unique_experts_per_layer = self._deduplicate_routing(routing)
        
        # Step 3: Place (learned hot-store)
        placements = self.expert_store.get_placement(unique_experts_per_layer)
        
        # Step 4: Overlap (prefetch + async)
        expert_data = self._async_load_experts(placements)
        
        # Step 5: Learn
        self.learning_cache.record_routing(routing)
        
        # Compute
        return self._compute_layer(expert_data, routing)
```

### A12.2: Create TieredExpertStore
```python
class TieredExpertStore:
    """
    3-tier expert placement:
    - VRAM: GPU device memory (if Arc GPU available)
    - RAM: Learned hot-store (pinned based on routing history)
    - Disk: NVMe primary storage
    """
    
    def __init__(self, model_path, learning_cache_path='.openagent_usage'):
        self.disk_path = model_path
        self.ram_hot_store = {}  # Hottest K experts
        self.learning_cache = self._load_learning_cache(learning_cache_path)
        self.lru = LRUCache(max_size=16)  # 16 hottest experts in RAM
    
    def get_placement(self, expert_ids):
        """Return {expert_id: placement} where placement ∈ {VRAM, RAM, DISK}"""
        placements = {}
        for expert_id in expert_ids:
            if expert_id in self.ram_hot_store:
                placements[expert_id] = 'RAM'
            elif self._should_load_to_ram(expert_id):
                placements[expert_id] = 'RAM'
            else:
                placements[expert_id] = 'DISK'
        return placements
    
    def _load_learning_cache(self, path):
        """Load .openagent_usage file (routing history)"""
        # Format: {layer: {expert: count}}
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {}
    
    def _should_load_to_ram(self, expert_id):
        """Decide if expert should be in RAM based on frequency"""
        count = self.learning_cache.get(expert_id, 0)
        return count > HEAT_THRESHOLD
    
    def save_learning_cache(self):
        """Persist routing history for next session"""
        with open('.openagent_usage', 'w') as f:
            json.dump(self.learning_cache, f)
```

### A12.3: Remove model-specific code
- Find all `if model_type == 'qwen'` or `native_qwen35` imports in hot paths
- Replace with calls to UnifiedInferenceExecution
- Mark removed files for deprecation (can delete in Phase 2)

**Tests:**
- All 9 models route through unified path
- No model-specific logic in forward()
- Same outputs as before (numerically)

---

## A13: Batch-Union + Async I/O

**Goal:** Reduce expert loads from N×M (batch×unique_experts) to M (unique_experts only)

### A13.1: Implement Batch-Union
```python
class BatchUnionOptimizer:
    def deduplicate_routing(self, routing_decisions):
        """
        Input: routing[batch_size, num_layers, num_experts] (one-hot per layer)
        Output: unique_experts_per_layer[num_layers] + position_indices
        
        Example:
        - Batch of 4, layer 0: routing = [[1,0,0,0], [0,1,0,0], [1,0,0,0], [0,1,0,0]]
        - Unique experts: {0, 1}
        - Load experts 0 and 1 ONCE (not 4 times)
        - Position indices tell us which positions use which expert
        """
        unique_per_layer = {}
        position_indices = {}
        
        for layer_id, routing_layer in enumerate(routing_decisions):
            # Find unique experts fired in this layer
            unique_experts = set()
            for pos, expert_flags in enumerate(routing_layer):
                fired = [i for i, flag in enumerate(expert_flags) if flag]
                unique_experts.update(fired)
            
            unique_per_layer[layer_id] = unique_experts
            position_indices[layer_id] = self._map_positions(routing_layer, unique_experts)
        
        return unique_per_layer, position_indices
    
    def load_batch_union(self, unique_experts_per_layer):
        """Load each unique expert ONCE, store in shared buffer"""
        results = {}
        for layer_id, expert_ids in unique_experts_per_layer.items():
            # One pread per expert (not per position)
            expert_data = {}
            for expert_id in expert_ids:
                expert_data[expert_id] = self._read_expert(expert_id)
            results[layer_id] = expert_data
        return results
```

### A13.2: Async I/O Pool
```python
class AsyncIOPool:
    def __init__(self, max_concurrent=8):
        self.max_concurrent = max_concurrent
        self.queue = asyncio.Queue(maxsize=max_concurrent)
        self.pending = {}
    
    async def async_load_expert(self, expert_id, placement):
        """Load expert from disk/RAM asynchronously"""
        if placement == 'RAM':
            return self.ram_cache[expert_id]  # Instant
        
        # Disk read in background
        if expert_id in self.pending:
            return await self.pending[expert_id]  # Already loading
        
        future = asyncio.create_task(self._disk_read(expert_id))
        self.pending[expert_id] = future
        return await future
    
    async def _disk_read(self, expert_id):
        """Actual disk I/O"""
        while len(self.pending) >= self.max_concurrent:
            await asyncio.sleep(0.001)  # Throttle
        
        # Non-blocking read
        data = await self._pread_async(expert_id)
        del self.pending[expert_id]
        return data
```

**Tests:**
- Batch-union reduces reads (measure: 1 expert per unique vs N per batch)
- Async I/O doesn't block forward() (measure: compute/I/O overlap)
- Multi-layer prefetch hides I/O (measure: 71%+ hit rate)

---

## A14: Lookahead Prefetch + Learning Cache

**Goal:** Router runs one layer ahead; predict next layer's experts before needed

### A14.1: Lookahead Thread
```python
class LookaheadPrefetcher:
    def __init__(self, router, expert_store, prefetch_layers=2):
        self.router = router
        self.expert_store = expert_store
        self.prefetch_layers = prefetch_layers
        self.prefetch_thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self.current_batch = None
        self.next_batch_queue = asyncio.Queue()
    
    def _prefetch_loop(self):
        """Run in background: predict next layer's experts and load them"""
        while True:
            batch = self.next_batch_queue.get()
            
            # Predict next N layers
            for layer_offset in range(1, self.prefetch_layers + 1):
                predicted_routing = self.router.predict_ahead(batch, offset=layer_offset)
                unique_experts = self._deduplicate(predicted_routing)
                
                # Load in background
                for expert_id in unique_experts:
                    asyncio.run(self.expert_store.async_load(expert_id))
    
    def forward(self, input_tokens):
        """
        Main forward: compute layer N while layer N+1 is prefetching
        """
        # Queue next batch for prefetch
        self.next_batch_queue.put(input_tokens)
        
        # Current layer computes (prefetch from previous turn happens in parallel)
        return self._compute_current_layer(input_tokens)
```

### A14.2: Learning Cache Persistence
```python
class LearningCache:
    def __init__(self, cache_file='.openagent_usage'):
        self.cache_file = cache_file
        self.routing_history = {}  # {layer: {expert: count}}
    
    def record_routing(self, routing_decisions):
        """Track which experts fired"""
        for layer_id, routing in enumerate(routing_decisions):
            if layer_id not in self.routing_history:
                self.routing_history[layer_id] = {}
            
            for expert_id in routing.fired_experts:
                self.routing_history[layer_id][expert_id] = \
                    self.routing_history[layer_id].get(expert_id, 0) + 1
    
    def get_hot_experts(self, layer_id, top_k=16):
        """Return hottest K experts in this layer (learned from history)"""
        if layer_id not in self.routing_history:
            return []
        
        experts = self.routing_history[layer_id]
        return sorted(experts.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    def save(self):
        """Persist to disk after each turn"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.routing_history, f)
    
    def load(self):
        """Load on startup (placement improves over time)"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                self.routing_history = json.load(f)
```

**Tests:**
- Prefetch runs one layer ahead (no blocking)
- 71%+ hit rate (predicted experts already loaded)
- Learning cache improves placement (measure: fewer disk reads over N batches)

---

## A15: Model Integration + API Exposure

**Goal:** Wire all 9 models through unified path; expose metrics via API

### A15.1: Update Model Adapters
```python
# Before (model-specific):
# in native_qwen35.py:
def forward(tokens):
    experts = load_qwen35_experts(tokens)  # Qwen-specific
    return qwen35_compute(tokens, experts)

# After (unified):
# in native_qwen35.py:
def forward(tokens):
    return unified_inference.forward(tokens, model_config=self.config)

# Same for all 9 families:
# - native_glm5.py
# - native_glm_dsa.py
# - native_deepseek.py
# - native_qwen4.py
# - native_mixtral.py
# - etc.
```

### A15.2: API Endpoints
```python
# FastAPI routes in adaptive_engine_routes.py

@router.get("/api/adaptive-engine/placement")
async def get_placement():
    """Current expert placement (VRAM/RAM/disk)"""
    return {
        "layer_0": {"expert_0": "RAM", "expert_1": "DISK", ...},
        ...
    }

@router.get("/api/adaptive-engine/routing-heat")
async def get_routing_heat():
    """Which experts are firing most"""
    return learning_cache.get_hot_experts(top_k=20)

@router.get("/api/adaptive-engine/learn")
async def get_learning_stats():
    """Learning cache statistics"""
    return {
        "batches_processed": learning_cache.total_batches,
        "hot_experts_pinned": len(learning_cache.get_hot_experts()),
        "disk_reads_saved": learning_cache.batch_union_savings,
        "prefetch_hit_rate": learning_cache.prefetch_hit_rate,
    }

@router.get("/api/adaptive-engine/plan")
async def get_hardware_plan():
    """Hardware detection and optimization plan"""
    return {
        "cpu_cores": os.cpu_count(),
        "ram_available_gb": psutil.virtual_memory().available / 1024**3,
        "disk_paths": ["/primary/path", "/secondary/path"],
        "disk_bandwidth_mbps": [500, 300],
        "arc_gpu_detected": torch.xpu.is_available(),
        "recommendation": "2-layer prefetch, 16-expert hot-store, 8-concurrent-reads",
    }
```

### A15.3: Performance Metrics
```python
class PerformanceMetrics:
    def __init__(self):
        self.tokens_per_second = 0
        self.ttft_ms = 0
        self.bytes_read = 0
        self.cache_hit_rate = 0
        self.compute_time_ms = 0
        self.io_time_ms = 0
    
    def measure(self):
        """Record one forward pass"""
        start = time.perf_counter()
        
        # Forward pass
        output = unified_inference.forward(...)
        
        end = time.perf_counter()
        
        self.compute_time_ms = (end - start) * 1000
        self.tokens_per_second = len(output) / (end - start)
        self.cache_hit_rate = expert_store.get_hit_rate()
        self.bytes_read = expert_store.get_bytes_read()
        
        return output
    
    def report(self):
        return {
            "tok/s": self.tokens_per_second,
            "ttft_ms": self.ttft_ms,
            "compute_ms": self.compute_time_ms,
            "io_ms": self.io_time_ms,
            "cache_hit_rate": self.cache_hit_rate,
            "bytes_read_mb": self.bytes_read / 1024 / 1024,
        }
```

**Tests:**
- All 9 models forward through unified path
- API endpoints return correct data
- Metrics improve with prefetch/learning cache

---

## Success Checklist

### A12 (Unified Path)
- [ ] UnifiedInferenceExecution implemented
- [ ] TieredExpertStore created
- [ ] All model-specific hot-path code removed
- [ ] All 9 models still work (same outputs)
- [ ] Tests: 12/12 PASS (A02 + A10 tests, unchanged)

### A13 (Batch-Union + Async)
- [ ] BatchUnionOptimizer reduces reads
- [ ] AsyncIOPool doesn't block forward()
- [ ] Measure: 1 expert load per unique (vs N per batch)
- [ ] Measure: 20%+ I/O reduction

### A14 (Prefetch + Learning)
- [ ] LookaheadPrefetcher running
- [ ] 71%+ hit rate measured
- [ ] LearningCache persists .openagent_usage
- [ ] Placement improves over N batches

### A15 (Integration + API)
- [ ] All 9 models use unified path
- [ ] `/api/adaptive-engine/*` endpoints working
- [ ] PerformanceMetrics recording
- [ ] Speed: >0.50 tok/s (Phase 1B target)

---

## Performance Timeline

| Phase | Target | Start | End |
|-------|--------|-------|-----|
| **A12** | Source unified | Oct 20 | Oct 23 |
| **A13** | Batch-union working | Oct 23 | Oct 26 |
| **A14** | Prefetch + learning | Oct 26 | Oct 29 |
| **A15** | All 9 models + API | Oct 29 | Nov 1 |
| **Tuning** | >0.50 tok/s | Nov 1 | Nov 15 |

---

## Work Location

✅ **LOCAL WINDOWS MACHINE ONLY** (no cloud container)

Clone from GitHub, work locally, commit locally, push when ready.

```bash
cd D:\openagent
git pull origin main  # Get Phase 1 A01-A11 code
# A12-A15 work starts here
```
