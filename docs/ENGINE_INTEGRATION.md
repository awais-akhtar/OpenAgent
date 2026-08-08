# Adaptive Engine Integration

OpenAgent keeps the adaptive compiler and local execution engine under `openagent/adaptive_engine`. The standalone engine package can evolve independently, while the application continues to use the same gateway and storage interfaces.

The application runtime now includes the engine's configuration-aware resident sessions. A model is reused only when its path, device, device properties, and scheduler options have the same fingerprint. Changing a CPU tuning option or batching setting therefore triggers a correct reload instead of silently reusing an incompatible session.

Optional CPU properties are filtered against the installed OpenVINO device before compilation. This keeps tuning hints useful on newer runtimes without making older drivers fail model loading. `auto` direct execution targets prefer a verified OpenVINO GPU, then NPU, then CPU, with a clear CPU fallback when no accelerator is exposed.

The engine continues to run inside the existing local boundary. Hardware discovery, compilation caches, model weights, and benchmark records remain local. No telemetry is sent by the adaptive runtime.

The extracted package supports its own standalone storage and environment names. The OpenAgent application intentionally retains `OpenAgentStore`, existing route contracts, and current imports so this integration does not split the application state model.
