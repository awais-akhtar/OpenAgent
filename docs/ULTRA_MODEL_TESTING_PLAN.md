# G22: Ultra Model Testing Plan

## Current State

**Status**: Routes wired; runtime and Demo isolation unverified

## Test Coverage Needed

### 1. Model Session Lifecycle
```python
def test_ultra_model_manifest_registration():
    """POST /api/ultra-model/manifests"""

def test_ultra_model_plan_generation():
    """POST /api/ultra-model/plan - generate launch plan"""

def test_ultra_model_launch():
    """POST /api/ultra-model/launch - start model session"""

def test_ultra_model_session_status():
    """GET /api/ultra-model/sessions/{id} - check readiness"""

def test_ultra_model_session_stop():
    """POST /api/ultra-model/sessions/{id}/stop - graceful shutdown"""

def test_ultra_model_calibrate():
    """POST /api/ultra-model/sessions/{id}/calibrate - runtime config"""
```

### 2. Safety Tests
```python
def test_ultra_model_requires_explicit_launch_approval():
    """Launch operations require approval"""

def test_ultra_model_respects_resource_limits():
    """Memory and CPU limits enforced"""

def test_ultra_model_stops_cleanly():
    """Process terminates properly on stop"""

def test_ultra_model_demo_isolation():
    """Demo mode doesn't call real backend"""
```

### 3. Failure Recovery
```python
def test_ultra_model_timeout_handling():
    """Launch timeout returns meaningful error"""

def test_ultra_model_resource_exhaustion():
    """OOM or CPU throttle handled gracefully"""

def test_ultra_model_restart_capability():
    """Can restart after failure"""
```

## Quick Checklist

- [ ] Model manifest registry working
- [ ] Plan generation produces valid config
- [ ] Launch succeeds with compatible model
- [ ] Session stop kills process properly
- [ ] Calibration applies settings
- [ ] Demo mode isolated from real execution
- [ ] Error messages helpful
- [ ] Timeouts enforced
- [ ] Resource limits checked
- [ ] No zombie processes

## Next Steps

1. Add test_g22_ultra_model_lifecycle.py
2. Mock runtime for testing
3. Verify Demo mode isolation
4. Measure actual startup time
5. Test with real model (if available)
