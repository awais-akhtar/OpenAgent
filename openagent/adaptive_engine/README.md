# Adaptive engine source map

The adaptive engine separates shared lifecycle policy from family-specific
model arithmetic.

- `runtime.py` and `execution_service.py` own application/runtime integration.
- `universal_engine.py` is the shared exact-adapter session boundary.
- `source_scope.py` records fail-closed Phase 1 route and evidence decisions.
- `execution_factories.py` owns explicit construction paths.
- `wholemodel_validation/` owns evidence manifests and comparisons.
- `k3_execution/` and `v41_execution/` contain family execution subsystems.
- `platform_certification/` owns platform-specific receipt tooling.

New family support is incomplete until configuration, inventory, loader,
execution consumer, cache/TP ownership, ABI/factory/CLI/gateway wiring,
recovery, and evidence tooling form one admitted route. A registry entry alone
does not establish support.
