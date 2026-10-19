"""Explicit Phase 1 source owners; never runtime or numerical certificates.

References are data. Reading this catalog does not import a decoder, inspect a
device, open a checkpoint, execute a comparison, or change admission policy.
Consumers must still admit the complete selected model and its runtime policy.
"""
from copy import deepcopy
from functools import lru_cache
from dataclasses import asdict
import ast
from pathlib import Path
from .publisher_family_contracts import PUBLISHER_VARIANTS
from .mixtral_inventory import MIXTRAL_PUBLISHER_VARIANTS
from .publisher_source_pins import K3_REVISION, V41_REVISION


_COMMON = {
    'serving': ('execution_factories', 'execution_service.NativeExecutionService',
                'publisher_provider.NativePublisherProvider'),
    'generation_recovery': ('service_generation_checkpoint', 'generation_checkpoint_files.AtomicGenerationCheckpoint'),
    'scheduling': ('admission_scheduler.AdmissionScheduler', 'executor_budget.BudgetedExecutorFactory',
                   'execution_queue.NativeExecutionQueue'),
    'durability': ('continuation_store.DurableContinuationStore',
                   'indexed_continuation_store.IndexedContinuationStore',
                   'cache_file_recovery'),
    'distributed_recovery': ('distributed_execution.DistributedExecutionService',
        'distributed_recovery.launch_recovered_distributed_execution'),
    'correctness': ('wholemodel_validation.runner', 'platform_certification.decoder_matrix'),
    # Colibri-specific protocol surfaces are intentionally explicit.  A
    # family/backend row must not imply these capabilities merely because it
    # exposes ordinary token generation.
    'colibri_surfaces': (
        'colibri_capabilities.require_colibri_capability:single-admission-boundary',
        'brio.http-options-questions-schema:configured-native-partial',
        'brio.jev-systemone-contract:json-text-and-alias-input-served-model-output-partial',
        'brio.cache-slot-cross-request-snapshots:missing',
        'brio.cli-terminal-and-browser-clients:missing',
        'brio.direct-k3-v41-scoring:rejected-pending-lifecycle-composition',
        'api.openai-chat-completions-and-legacy-completions:text-sse-local-subset',
        'api.anthropic-messages-and-event-protocol:text-sse-route-scoped-x-api-key-partial',
        'grammar.gbnf-constrained-generation:restricted-not-admitted',
        'grammar.per-family-tool-call-protocol:restricted-not-admitted',
        'protocol.colibri-ready-mux-caps-and-legacy-stdin:missing',
        'routing.telemetry-coli-usage-and-expert-atlas:source-observation-only',
        'storage.multi-disk-mirror-and-partial-planner:restricted-not-admitted',
        'cache.cache-route-topk-substitution:source-observation-only',
        'operations.health-and-prometheus:bounded-fifo-native-http-queue-with-streaming-without-priority',
        'family-api.explicit-openagent-entrypoints-not-colibri-wire-parity',
    ),
}

# The dependency order is part of the P02 contract.  Keep it in one place so
# generators, validators and tooling cannot silently disagree about what a
# complete family/backend/format row means.
_P02_CHAIN_DIMENSIONS = (
    'configuration', 'inventory', 'loading', 'execution', 'cache_tp',
    'native_abi', 'factory', 'cli', 'gateway', 'recovery', 'tooling',
)


_FAMILIES = {
    'mixtral': {
        'identities': ('mixtral',),
        'publisher_contract': 'mixtral_inventory.MIXTRAL_PUBLISHER_VARIANTS',
        'configuration': ('native_mixtral_decoder.configuration', 'mixtral_inventory.publisher_contract'),
        'inventory': ('mixtral_inventory.admit_mixtral_inventory', 'resident_family_inventory.admit_resident_inventory'),
        'loading': ('native_checkpoint.read_native_text_checkpoint', 'torch_mixtral_decoder.TorchMixtralDecoder.from_directory'),
        'native': ('native_mixtral_decoder.NativeMixtralDecoder',),
        'device': ('torch_mixtral_decoder.TorchMixtralDecoder',),
        'streaming': ('mixtral_streamed_experts', 'torch_mixtral_experts', 'expert_packing'),
        'position': 'GQA; publisher default RoPE; separate admitted linear RoPE and sliding-window contracts',
        'media': (),
        'native_abi': ('mixtral_public_adapters.NativeMixtralPublicAdapters', 'native_mixtral_ranges', 'native/segment_edge_compat.c'),
        'cache': ('mixtral_cache_recovery', 'mixtral_range_cache', 'torch_mixtral_cache'),
        'cli': ('native_family_cli.PublisherSession',),
        'comparison': ('wholemodel_validation.adapters', 'wholemodel_validation.cache_readers'),
    },
    'olmoe': {
        'identities': ('olmoe',),
        'publisher_contract': 'olmoe.OlmoeConfig and SparseExpertManifest',
        'configuration': ('olmoe.OlmoeConfig', 'olmoe.validate_olmoe_tensors'),
        'inventory': ('olmoe.OlmoeModel._tensor', 'native_checkpoint'),
        'loading': ('native_leased_olmoe_segments.NativeLeasedOlmoeSegmentEngine',),
        'native': ('native_olmoe_segments.NativeOlmoeSegmentEngine',),
        'device': (),
        'streaming': ('native_leased_olmoe_segments', 'expert_runtime.SparseExpertManifest'),
        'position': 'Contiguous token positions with explicit bounded segment ranges',
        'media': (),
        'native_abi': ('segment_edge_adapters.NativeOlmoePublicAdapters', 'segment_edge_runtime'),
        'cache': ('native_olmoe_segments.NativeOlmoeSegmentSession',),
        'cli': ('native_olmoe_cli',),
        'comparison': ('wholemodel_validation.adapters', 'wholemodel_validation.cache_readers'),
    },
    'qwen35': {
        'identities': ('qwen3_5_moe', 'qwen3_5_moe_text'),
        'publisher_contract': 'publisher_family_contracts.PUBLISHER_VARIANTS[qwen3_5_moe]',
        'configuration': ('publisher_family_contracts.component_contract', 'independent_qwen35'),
        'inventory': ('resident_family_inventory.admit_resident_inventory', 'publisher_conditional_loader.load_publisher_conditional_decoder'),
        'loading': ('native_checkpoint.read_native_text_checkpoint', 'publisher_conditional_loader.load_publisher_conditional_decoder'),
        'native': ('native_qwen35.NativeQwen35TextDecoder',),
        'device': ('torch_qwen35_decoder.TorchQwen35Decoder',),
        'streaming': ('qwen35_publisher_experts', 'qwen35_streamed_experts', 'device_publisher_projections'),
        'position': 'Gated delta/full attention; interleaved three-axis MRoPE; physical capacity separate from semantic position bound',
        'media': ('qwen35_multimodal.Qwen35MultimodalDecoder', 'native_qwen35_vision', 'publisher_media_processor.PublisherMediaProcessor'),
        'native_abi': ('qwen35_public_adapters', 'native_qwen35_ranges', 'native/segment_edge_compat.c'),
        'cache': ('qwen35_cache_recovery', 'torch_qwen35_cache', 'compound_cache'),
        'cli': ('native_family_cli.PublisherSession', 'native_conditional_cli.PublisherSession'),
        'comparison': ('wholemodel_validation.qwen35_publisher_caches', 'wholemodel_validation.qwen35_media'),
    },
    'qwen4': {
        'identities': ('qwen4_exp', 'qwen4_exp_text'),
        'publisher_contract': 'publisher_family_contracts.PUBLISHER_VARIANTS[qwen4_exp]',
        'configuration': ('publisher_family_contracts.component_contract', 'independent_qwen4'),
        'inventory': ('native_qwen4_ranges.qwen4_tensor_inventory', 'publisher_conditional_loader.load_publisher_conditional_decoder'),
        'loading': ('native_checkpoint.read_native_text_checkpoint', 'publisher_conditional_loader.load_publisher_conditional_decoder'),
        'native': ('native_qwen4.NativeQwen4TextDecoder',),
        'device': ('torch_qwen4_decoder.TorchQwen4Decoder',),
        'streaming': ('qwen4_publisher_experts', 'qwen4_streamed_experts', 'qwen4_ple_store', 'device_publisher_projections'),
        'position': 'QSA/gated delta; lexical token and convolution histories; default/linear and admitted three-axis rotary positions',
        'media': ('qwen4_multimodal.Qwen4MultimodalDecoder', 'qwen4_vision', 'publisher_media_processor.PublisherMediaProcessor'),
        'native_abi': ('native_qwen4_public_adapters.NativeQwen4PublicAdapters', 'native_qwen4_public_text', 'native/qwen4_public_adapters.c'),
        'cache': ('qwen4_cache_recovery', 'torch_qwen4_cache', 'compound_cache'),
        'cli': ('native_family_cli.PublisherSession', 'native_conditional_cli.PublisherSession'),
        'comparison': ('wholemodel_validation.qwen4_publisher_caches', 'wholemodel_validation.native_qwen4_caches', 'wholemodel_validation.qwen4_media'),
    },
    'glm5-next': {
        'identities': ('glm5_next', 'glm5_next_text'),
        'publisher_contract': 'publisher_family_contracts.PUBLISHER_VARIANTS[glm5_next]',
        'configuration': ('publisher_glm5_config', 'independent_glm5_next.Glm5NextTextContract'),
        'inventory': ('publisher_glm5_loader.admit_publisher_glm5_inventory', 'resident_family_inventory.admit_resident_inventory'),
        'loading': ('publisher_glm5_loader', 'native_checkpoint.read_native_text_checkpoint',
                    'family_source_binding.bind_decoder_to_snapshot'),
        'native': ('native_glm5_next.NativeGlm5NextTextDecoder',),
        'device': ('torch_glm5_decoder.TorchGlm5NextDecoder',),
        'streaming': ('glm5_publisher_experts', 'device_glm5_publisher_experts', 'device_publisher_projections'),
        'position': 'KDA/MLA, shared DSA index masks, KPool, MHC and NoPE contiguous positions',
        'media': ('glm5_multimodal.Glm5MultimodalDecoder', 'glm5_vision', 'publisher_media_processor.PublisherMediaProcessor'),
        'native_abi': ('native_family_public_adapters.NativeFamilyPublicAdapters',
                       'family_source_binding.require_decoder_snapshot',
                       'native_family_public_text', 'native/segment_family_models.c'),
        'cache': ('flat_native_session', 'compound_cache'),
        'cli': ('native_family_cli.PublisherSession', 'native_conditional_cli.PublisherSession'),
        'comparison': ('wholemodel_validation.publisher_family_candidate', 'wholemodel_validation.publisher_family_caches'),
    },
    'inkling': {
        'identities': ('inkling_mm_model', 'inkling_text'),
        'publisher_contract': 'publisher_family_contracts.PUBLISHER_VARIANTS[inkling_mm_model]',
        'configuration': ('publisher_family_contracts.component_contract', 'independent_inkling'),
        'inventory': ('resident_family_inventory.admit_resident_inventory', 'inkling_mtp.admit_mtp_inventory', 'publisher_conditional_loader.load_publisher_conditional_decoder'),
        'loading': ('native_checkpoint.read_native_text_checkpoint', 'publisher_conditional_loader.load_publisher_conditional_decoder',
                    'family_source_binding.bind_decoder_to_snapshot'),
        'native': ('native_inkling.NativeInklingDecoder',),
        'device': ('torch_inkling_decoder.TorchInklingDecoder',),
        'streaming': ('inkling_publisher_experts', 'device_inkling_publisher_experts', 'device_publisher_projections'),
        'position': 'Hybrid/sliding attention, relative position profiles, complete FIR history and contiguous occupied positions',
        'media': ('inkling_multimodal.InklingMultimodalDecoder', 'inkling_media', 'inkling_mtp.InklingMTP', 'publisher_media_processor.PublisherMediaProcessor'),
        'native_abi': ('native_family_public_adapters.NativeFamilyPublicAdapters',
                       'family_source_binding.require_decoder_snapshot',
                       'native_family_public_text', 'native/segment_family_models.c'),
        'cache': ('flat_native_session', 'torch_inkling_cache', 'compound_cache', 'grouped_continuation'),
        'cli': ('native_family_cli.PublisherSession', 'native_conditional_cli.PublisherSession'),
        'comparison': ('wholemodel_validation.publisher_family_candidate', 'wholemodel_validation.publisher_family_caches'),
    },
    'glm-dsa': {
        'identities': ('glm_moe_dsa',),
        'publisher_contract': 'Exact glm_moe_dsa configuration and caller-pinned complete local artifacts',
        'configuration': ('independent_glm_dsa',),
        'inventory': ('resident_family_inventory.admit_resident_inventory', 'native_glm_dsa_ranges.admit_inventory'),
        'loading': ('native_checkpoint.read_native_text_checkpoint', 'torch_glm_dsa_decoder.TorchGlmDsaDecoder.from_directory'),
        'native': ('native_glm_dsa.NativeGlmDsaDecoder',),
        'device': ('torch_glm_dsa_decoder.TorchGlmDsaDecoder',),
        'streaming': ('glm_dsa_streamed_experts', 'device_glm_dsa_experts', 'torch_glm_dsa_experts'),
        'position': 'Compressed latent/rotary/indexer state; default or static YaRN rotary policy',
        'media': (),
        'native_abi': ('glm_dsa_public_adapters', 'native_glm_dsa_ranges', 'native/segment_edge_compat.c'),
        'cache': ('native_glm_dsa_cache', 'torch_inkling_cache'),
        'cli': ('native_family_cli.PublisherSession',),
        'comparison': ('wholemodel_validation.glm_dsa_caches',),
    },
    'kimi-k3': {
        'identities': ('kimi_k3', 'kimi_linear'),
        'publisher_contract': 'publisher_checkpoint.contracts.K3_REVISION',
        'configuration': ('publisher_checkpoint.contracts.build_contracts', 'k3_execution.executor.K3Config'),
        'inventory': ('publisher_checkpoint.service.PublisherCheckpoint', 'publisher_checkpoint.k3_adapter'),
        'loading': ('publisher_checkpoint.k3_adapter',),
        'native': ('k3_execution.executor.K3Execution',),
        'device': ('k3_execution.bf16_arithmetic.K3BF16Arithmetic', 'k3_execution.fla_mixer', 'k3_execution.metal_mixer'),
        'streaming': ('k3_execution.packed', 'k3_execution.publisher_projections', 'k3_execution.device_projections'),
        'position': 'Occupied token positions, compressed MLA, complete KDA/convolution state and independent batch slots',
        'media': ('k3_execution.preprocessing', 'k3_execution.video_input', 'k3_execution.vision', 'k3_execution.exact_admission.K3ExactEvidence'),
        'native_abi': ('k3_execution.executor.K3Execution._bind_cpu',
                       'publisher_public_adapters.NativePublisherPublicAdapters',
                       'native/segment_edge_compat.c', 'k3_execution/attention.c',
                       'native_kimi.c'),
        'cache': ('k3_execution.checkpoints', 'scalable_execution.k3_compressed'),
        'cli': ('publisher_k3_cli.PublisherSession', 'publisher_public_cli.PublisherSession'),
        'comparison': ('wholemodel_validation.k3', 'wholemodel_validation.k3_media'),
    },
    'deepseek-v41': {
        'identities': ('deepseek_v41', 'deepseek_v41_text'),
        'publisher_contract': 'publisher_checkpoint.contracts.V41_REVISION',
        'configuration': ('publisher_checkpoint.v41_profile', 'publisher_checkpoint.contracts.build_contracts'),
        'inventory': ('publisher_checkpoint.service.PublisherCheckpoint', 'publisher_checkpoint.v41_adapter'),
        'loading': ('publisher_checkpoint.v41_adapter',),
        'native': ('v41_execution.publisher.PublisherV41Execution',),
        'device': ('v41_execution.device_execution.DevicePublisherV41Execution', 'v41_execution.device_arithmetic.DevicePublisherV41Arithmetic'),
        'streaming': ('v41_execution.packed', 'v41_execution.fp8_projection', 'v41_execution.bf16_projections', 'tensor_parallel'),
        'position': 'Compressed attention ownership, Engram history, shared index masks and complete target/draft positions',
        'media': ('v41_execution.publisher_vision', 'v41_execution.device_vision', 'v41_execution.publisher_mtp', 'v41_execution.dspark', 'v41_execution.speculative'),
        'native_abi': ('v41_execution.publisher_arithmetic.NativePublisherV41Arithmetic',
                       'publisher_public_adapters.NativePublisherPublicAdapters',
                       'native/segment_edge_compat.c', 'v41_execution/kernels.c',
                       'v41_execution/fp8_projection.c'),
        'cache': ('v41_execution.publisher_checkpoints', 'v41_execution.dspark_cache', 'v41_execution.speculative_cache', 'v41_generation_recovery', 'distributed_generation_checkpoint'),
        'cli': ('publisher_v41_cli.PublisherSession', 'publisher_public_cli.PublisherSession'),
        'comparison': ('wholemodel_validation.v41_publisher', 'wholemodel_validation.v41'),
    },
}


def source_family_for_identity(model_type):
    """Resolve one exact checkpoint identity through the canonical source catalog.

    Loader and CLI admission code used to keep independent identity-to-family
    dictionaries.  That allowed a new identity to enter one entrypoint without
    entering its P02 inventory gate.  Ambiguous or undeclared identities now
    fail closed at the shared lookup.
    """
    if type(model_type) is not str or not model_type:
        raise ValueError('exact checkpoint model identity required for source-family admission')
    matches = tuple(family for family, contract in _FAMILIES.items()
                    if model_type in contract.get('identities', ()))
    if len(matches) != 1:
        raise ValueError('checkpoint identity is absent from or ambiguous in the P02 source catalog: '
                         + model_type)
    return matches[0]


# P02 closure dimensions that cannot be inferred safely from the common
# service catalog.  Every family row must name its concrete cache/TP owner,
# gateway route, recovery readers/writers and source-only tooling adapters.
# Keeping these explicit prevents a family from appearing complete merely
# because it has a decoder and a generic factory.
_FAMILY_CLOSURE = {
    'mixtral': {
        'cache_tp': ('mixtral_cache_recovery', 'mixtral_range_cache', 'tensor_parallel'),
        'factory': ('execution_factories', 'native_family_cli.PublisherSession'),
        'gateway': ('publisher_provider.NativePublisherProvider', 'execution_service.NativeExecutionService'),
        'recovery': ('mixtral_cache_recovery', 'service_generation_checkpoint'),
        'tooling': ('wholemodel_validation.runner', 'platform_certification.decoder_matrix', 'native_family_cli'),
    },
    'olmoe': {
        'cache_tp': ('native_olmoe_segments.NativeOlmoeSegmentSession', 'tensor_parallel'),
        'factory': ('execution_factories', 'native_olmoe_cli'),
        'gateway': ('execution_service.NativeExecutionService', 'segment_edge_runtime'),
        'recovery': ('native_olmoe_segments', 'service_generation_checkpoint'),
        'tooling': ('wholemodel_validation.runner', 'wholemodel_validation.olmoe_native',
                    'native_olmoe_cli', 'platform_certification.decoder_matrix'),
    },
    'qwen35': {
        'cache_tp': ('qwen35_cache_recovery', 'torch_qwen35_cache', 'tensor_parallel', 'device_publisher_projections'),
        'factory': ('execution_factories', 'publisher_conditional_loader.load_publisher_conditional_decoder'),
        'gateway': ('publisher_provider.NativePublisherProvider', 'native_conditional_cli.PublisherSession'),
        'recovery': ('qwen35_cache_recovery', 'compound_cache', 'service_generation_checkpoint'),
        'tooling': ('wholemodel_validation.qwen35_media', 'wholemodel_validation.qwen35_publisher_caches', 'wholemodel_validation.qwen35_media_cli'),
    },
    'qwen4': {
        'cache_tp': ('qwen4_cache_recovery', 'wholemodel_validation.native_qwen4_caches', 'tensor_parallel', 'qwen4_ple_store'),
        'factory': ('execution_factories', 'publisher_conditional_loader.load_publisher_conditional_decoder'),
        'gateway': ('publisher_provider.NativePublisherProvider', 'native_conditional_cli.PublisherSession'),
        'recovery': ('qwen4_cache_recovery', 'compound_cache', 'service_generation_checkpoint'),
        'tooling': ('wholemodel_validation.qwen4_media', 'wholemodel_validation.qwen4_publisher_caches', 'wholemodel_validation.qwen4_media_cli'),
    },
    'glm5-next': {
        'cache_tp': ('flat_native_session', 'compound_cache', 'tensor_parallel', 'device_publisher_projections'),
        'factory': ('execution_factories', 'publisher_conditional_loader.load_publisher_conditional_decoder'),
        'gateway': ('publisher_provider.NativePublisherProvider', 'native_conditional_cli.PublisherSession'),
        'recovery': ('flat_native_session', 'compound_cache', 'service_generation_checkpoint'),
        'tooling': ('wholemodel_validation.publisher_family_candidate', 'wholemodel_validation.publisher_family_caches', 'native_family_cli'),
    },
    'inkling': {
        'cache_tp': ('flat_native_session', 'torch_inkling_cache', 'tensor_parallel', 'grouped_continuation'),
        'factory': ('execution_factories', 'publisher_conditional_loader.load_publisher_conditional_decoder'),
        'gateway': ('publisher_provider.NativePublisherProvider', 'native_conditional_cli.PublisherSession'),
        'recovery': ('flat_native_session', 'grouped_continuation', 'service_generation_checkpoint'),
        'tooling': ('wholemodel_validation.publisher_family_candidate', 'wholemodel_validation.publisher_family_caches', 'native_family_cli'),
    },
    'glm-dsa': {
        'cache_tp': ('native_glm_dsa_cache', 'tensor_parallel', 'device_glm_dsa_experts'),
        'factory': ('execution_factories', 'native_family_cli.PublisherSession'),
        'gateway': ('publisher_provider.NativePublisherProvider', 'execution_factories'),
        'recovery': ('native_glm_dsa_cache', 'service_generation_checkpoint'),
        'tooling': ('wholemodel_validation.glm_dsa_caches', 'platform_certification.decoder_matrix', 'native_family_cli'),
    },
    'kimi-k3': {
        'cache_tp': ('k3_execution.checkpoints', 'scalable_execution.k3_compressed', 'k3_execution.publisher_projections'),
        'factory': ('execution_factories', 'execution_factories.create_public_k3_execution_service',
                    'publisher_k3_cli.PublisherSession', 'publisher_public_cli.PublisherSession'),
        'gateway': ('publisher_k3_cli.PublisherSession', 'publisher_public_cli.PublisherSession',
                    'execution_factories'),
        'recovery': ('k3_execution.checkpoints', 'service_generation_checkpoint', 'distributed_recovery'),
        'tooling': ('wholemodel_validation.k3', 'wholemodel_validation.k3_media',
                    'k3_execution.exact_evidence_files', 'publisher_public_adapters'),
    },
    'deepseek-v41': {
        'cache_tp': ('v41_execution.publisher_checkpoints', 'distributed_generation_checkpoint', 'tensor_parallel'),
        'factory': ('execution_factories', 'execution_factories.create_public_publisher_v41_execution_service',
                    'publisher_v41_cli.PublisherSession', 'publisher_public_cli.PublisherSession'),
        'gateway': ('publisher_v41_cli.PublisherSession', 'publisher_public_cli.PublisherSession',
                    'publisher_provider.NativePublisherProvider'),
        'recovery': ('v41_generation_recovery', 'distributed_generation_checkpoint', 'service_generation_checkpoint'),
        'tooling': ('wholemodel_validation.v41_publisher', 'wholemodel_validation.v41',
                    'platform_certification.decoder_matrix', 'publisher_public_adapters'),
    },
}

# P13 source evidence obligations are attached to each generated execution row,
# not inferred from the presence of a family-level adapter.  These entries
# describe the source contract that a later evidence run must satisfy; they
# deliberately do not assert that an observation, artifact pin, placement, or
# comparison has already been produced.  Exact per-backend/per-format schemas
# are not interchangeable across rows, so the row remains unverified until a
# runner binds a concrete receipt to this obligation.
_EVIDENCE_OBLIGATIONS = {
    # Function-level schema owners are intentionally used rather than a
    # family-level "cache exists" label. Their schemas depend on admitted
    # configuration and are not instantiated by this source catalog.
    'mixtral': dict(observation_schema='wholemodel_validation.adapters', cache_schema='wholemodel_validation.cache_readers.export_cache',
                    report_schema='openagent.wholemodel-validation.v1', media_adapter=None, media_report_schema=None,
                    pin='wholemodel_validation.runner.pinned_artifacts', placement='platform_certification.decoder_matrix'),
    'olmoe': dict(observation_schema='wholemodel_validation.olmoe_native.read_complete_cache',
                  cache_schema='wholemodel_validation.olmoe_native.cache_schema',
                  report_schema='openagent.wholemodel-validation.v1', media_adapter=None, media_report_schema=None,
                  pin='wholemodel_validation.runner.pinned_artifacts', placement='platform_certification.decoder_matrix'),
    'qwen35': dict(observation_schema='wholemodel_validation.qwen35_publisher_caches', cache_schema='wholemodel_validation.qwen35_publisher_caches.schema_for',
                   report_schema='openagent.wholemodel-validation.v1', media_adapter='wholemodel_validation.qwen35_media',
                   media_report_schema='openagent.qwen35-media-comparison.v1', pin='wholemodel_validation.runner.pinned_artifacts', placement='platform_certification.decoder_matrix'),
    'qwen4': dict(observation_schema='wholemodel_validation.qwen4_publisher_caches', cache_schema='wholemodel_validation.qwen4_publisher_caches.schema_for',
                  report_schema='openagent.wholemodel-validation.v1', media_adapter='wholemodel_validation.qwen4_media',
                  media_report_schema='openagent.qwen4-media-comparison.v1', pin='wholemodel_validation.runner.pinned_artifacts', placement='platform_certification.decoder_matrix'),
    'glm5-next': dict(observation_schema='wholemodel_validation.publisher_family_candidate', cache_schema='wholemodel_validation.publisher_family_caches.schema',
                      report_schema='openagent.wholemodel-validation.v1', media_adapter=None, media_report_schema=None,
                      pin='wholemodel_validation.runner.pinned_artifacts', placement='platform_certification.decoder_matrix'),
    'inkling': dict(observation_schema='wholemodel_validation.publisher_family_candidate', cache_schema='wholemodel_validation.publisher_family_caches.schema',
                    report_schema='openagent.wholemodel-validation.v1', media_adapter=None, media_report_schema=None,
                    pin='wholemodel_validation.runner.pinned_artifacts', placement='platform_certification.decoder_matrix'),
    'glm-dsa': dict(observation_schema='wholemodel_validation.glm_dsa_caches', cache_schema='wholemodel_validation.glm_dsa_caches.schema',
                    report_schema='openagent.wholemodel-validation.v1', media_adapter=None, media_report_schema=None,
                    pin='wholemodel_validation.runner.pinned_artifacts', placement='platform_certification.decoder_matrix'),
    'kimi-k3': dict(observation_schema='k3_execution.observations.observation_schema', cache_schema='wholemodel_validation.k3.k3_cache_schema',
                    report_schema='openagent.wholemodel-validation.v1', media_adapter='wholemodel_validation.k3_media',
                    media_report_schema='openagent.k3-media-wholemodel.v2', pin='k3_execution.exact_admission.K3ExactEvidence',
                    placement='wholemodel_validation.k3'),
    'deepseek-v41': dict(observation_schema='wholemodel_validation.v41_publisher', cache_schema='wholemodel_validation.v41_publisher.publisher_v41_cache_schema',
                         report_schema='openagent.wholemodel-validation.v1', media_adapter=None, media_report_schema=None,
                         pin='wholemodel_validation.runner.pinned_artifacts', placement='platform_certification.decoder_matrix'),
}

_MEDIA_EVIDENCE_RESTRICTION = (
    'No row-specific full-media execution adapter and comparison report is '
    'implemented; wholemodel_validation.runner.validate_execution_binding '
    'rejects this media tuple before importing candidate adapters.'
)


def _evidence_obligation(family, model_type, backend, arithmetic,
                         source_format, persisted_format, media):
    """Return explicit P13 requirements without manufacturing a receipt."""
    source = _EVIDENCE_OBLIGATIONS[family]
    # Schema owners in this catalogue are family-level source references, not
    # instantiated adapters for every generated execution tuple. Preserve the
    # exact tuple beside each reference so downstream tooling cannot silently
    # widen a family schema to a different backend/format row. This is a
    # source-time binding only; it deliberately grants no evidence credit.
    row_binding = dict(family=family, model_type=model_type, backend=backend,
                       arithmetic=arithmetic, source_format=source_format,
                       persisted_format=persisted_format, media=bool(media))
    return dict(
        classification='source-obligation-unverified',
        evidence_state='not-captured',
        completion_claim=False,
        row_binding=row_binding,
        observation_schema=source['observation_schema'],
        physical_cache_schema=(source['cache_schema'], 'complete-unprojected-physical-cache-required'),
        artifact_pins=(source['pin'], 'config', 'weights-or-shards', 'tokenizer', 'runtime-dependencies'),
        placement_evidence=(source['placement'], 'per-operation-placement-required'),
        media_adapter=source['media_adapter'],
        media_adapter_missing=bool(media and source['media_adapter'] is None),
        media_restriction=(
            _MEDIA_EVIDENCE_RESTRICTION
            if media and source['media_adapter'] is None else None),
        comparison_adapter=source['observation_schema'],
        comparison_report_schema=(source['media_report_schema'] if media else source['report_schema']),
        comparison_report_owner='wholemodel_validation.runner.report_receipt_binding',
        report_receipt_schema='openagent.wholemodel-report-binding.v1',
        adapter_binding=dict(scope='exact-execution-row', row=row_binding,
                             owner_scope='family-schema-reference-only',
                             receipt_required=True),
        receipt_present=False,
        executed=False,
        numerical_parity_verified=False,
        platform_certified=False)


# Persisted expert formats are a separate axis from source tensor dtype.  The
# matrix must carry this axis explicitly instead of treating ``packed-expert``
# as proof that every packer is consumable by every family.
_PERSISTED_FORMAT_POLICIES = {
    'mixtral': ('float32', 'float16', 'int4-row', 'q4_0', 'q5_0', 'q8_0',
                'e8-lattice-caller-v1'),
    # OLMoE's sparse runtime resolves these two quantized expert formats
    # and its tested extended repacks through the native expert kernel
    # registry before opening expert storage.
    'olmoe': ('float32', 'float16', 'int8-row', 'int4-row',
              'int2-row', 'int3-grouped-g64', 'int4-grouped-g64',
              'q4_0', 'q4_1', 'q5_0', 'q5_1', 'q8_0', 'q8_1',
              'mxfp4-e2m1-g32-ue8m0', 'fp8-e4m3-b128-f32'),
    'qwen35': ('float32', 'float16', 'int4-row', 'int4-grouped-g64', 'fp8-e4m3-b128-f32',
               'e8-lattice-caller-v1'),
    'qwen4': ('float32', 'float16', 'int4-row', 'int4-grouped-g64', 'fp8-e4m3-b128-f32',
              'e8-lattice-caller-v1'),
    'glm5-next': ('float32', 'float16', 'int4-row', 'int4-grouped-g64', 'fp8-e4m3-b128-f32'),
    'inkling': ('float32', 'float16', 'int4-row', 'int4-grouped-g64', 'fp8-e4m3-b128-f32'),
    'glm-dsa': ('float32', 'float16', 'int4-row', 'int4-grouped-g64'),
    'kimi-k3': ('float32', 'mxfp4-e2m1-g32-ue8m0'),
    'deepseek-v41': ('bf16', 'fp8', 'fp4', 'float32',
                     'fp8-e4m3-b128-f32', 'e8-lattice-caller-v1', 'iq3-format6-98-v1'),
}

# These formats are part of the Colibri comparison surface even where no
# family admits them.  Keep explicit rejected rows in every family policy so
# the source matrix records the decision at the family/backend/format level;
# a global route entry alone would make omission indistinguishable from an
# unreviewed capability.
for _family in tuple(_PERSISTED_FORMAT_POLICIES):
    _PERSISTED_FORMAT_POLICIES[_family] = tuple(_PERSISTED_FORMAT_POLICIES[_family]) + (
        'int4-rans256-g0', 'fp8-e4m3-b128-ue8m0')


# Source-level classifications used by the Phase 1 closure review.  These
# values describe the admitted source contract only; they do not certify
# numerical parity, hardware placement, or an executed backend.
_CONTRACT_CASES = {
    'mixtral': dict(backends=('native', 'device', 'public-range'), formats=('fp32', 'q4_0', 'q5_0', 'q8_0'),
                    positions=('default-rope', 'linear-rope', 'sliding-window'),
                    source_dtypes=('F32','F16','BF16'), optional_heads=('tied-embedding',),
                    arithmetic=('fp32',), media='none', evidence='source-paths-only'),
    'olmoe': dict(backends=('native', 'public-range'),
                  formats=('int8-row','q4_0','q4_1','q5_0','q5_1','q8_0','q8_1'),
                  positions=('contiguous',), source_dtypes=('F32','F16','BF16'), optional_heads=(),
                  arithmetic=('fp32',), media='none', evidence='source-paths-only'),
    'qwen35': dict(backends=('native', 'device', 'public-range'), formats=('fp32', 'packed-expert'),
                   positions=('interleaved-mrope', 'physical-capacity-bound'),
                   source_dtypes=('F32','F16','BF16','F8_E4M3'), optional_heads=('vision','tied-embedding'),
                   arithmetic=('fp32',), media='vision-text', evidence='source-paths-only'),
    'qwen4': dict(backends=('native', 'device', 'public-range'), formats=('fp32', 'packed-expert'),
                  positions=('qsa', 'ple', 'three-axis-rope'),
                  source_dtypes=('F32','F16','BF16','F8_E4M3'), optional_heads=('vision','ple','tied-embedding'),
                  arithmetic=('fp32',), media='vision-text', evidence='source-paths-only'),
    'glm5-next': dict(backends=('native', 'device', 'public-range'), formats=('fp32', 'packed-expert'),
                      positions=('kda-mla', 'nope-contiguous'),
                      source_dtypes=('F32','F16','BF16','F8_E4M3'), optional_heads=('vision','shared-index'),
                      arithmetic=('fp32',), media='vision-text', evidence='source-paths-only'),
    'inkling': dict(backends=('native', 'device', 'public-range'), formats=('fp32', 'packed-expert'),
                    positions=('hybrid-sliding', 'fir-history'),
                    source_dtypes=('F32','F16','BF16','F8_E4M3'), optional_heads=('vision','mtp','tied-embedding'),
                    arithmetic=('fp32',), media='vision-mtp-text', evidence='source-paths-only'),
    'glm-dsa': dict(backends=('native', 'device', 'public-range'), formats=('fp32', 'packed-expert'),
                    positions=('compressed-latent', 'static-yarn'),
                    source_dtypes=('F32','F16','BF16'), optional_heads=('indexer','latent'),
                    arithmetic=('fp32',), media='none', evidence='source-paths-only'),
    'kimi-k3': dict(backends=('native', 'device', 'fla', 'metal', 'public-range'), formats=('fp32', 'packed-expert'),
                    positions=('occupied-token', 'mla-kda', 'convolution-history'),
                    source_dtypes=('F32','BF16','MXFP4'), optional_heads=('vision','mla','kda'),
                    arithmetic=('fp32-approximate','bf16'), media='image-video-text', evidence='source-paths-and-explicit-observation-gates'),
    'deepseek-v41': dict(backends=('native', 'device', 'distributed', 'public-range'), formats=('fp32', 'bf16', 'fp8', 'fp4'),
                         positions=('engram', 'shared-index', 'target-draft'),
                         source_dtypes=('BF16','FP8','FP4'), optional_heads=('vision','mtp','engram','speculative'),
                         arithmetic=('publisher-bf16-fp8-fp4',), media='vision-mtp-speculative-text', evidence='source-paths-and-explicit-observation-gates'),
}

_SERVING_ENTRYPOINTS = {
    'mixtral': ('native_family_cli.PublisherSession', 'execution_factories'),
    'olmoe': ('native_olmoe_cli', 'segment_edge_adapters.NativeOlmoePublicAdapters'),
    'qwen35': ('native_family_cli.PublisherSession', 'publisher_conditional_loader.load_publisher_conditional_decoder'),
    'qwen4': ('native_family_cli.PublisherSession', 'publisher_conditional_loader.load_publisher_conditional_decoder'),
    'glm5-next': ('native_family_cli.PublisherSession', 'publisher_conditional_loader.load_publisher_conditional_decoder'),
    'inkling': ('native_family_cli.PublisherSession', 'publisher_conditional_loader.load_publisher_conditional_decoder'),
    'glm-dsa': ('native_family_cli.PublisherSession', 'execution_factories'),
    'kimi-k3': ('publisher_k3_cli.PublisherSession', 'publisher_public_cli.PublisherSession',
                'execution_factories.create_public_k3_execution_service'),
    'deepseek-v41': ('publisher_v41_cli.PublisherSession', 'publisher_public_cli.PublisherSession',
                     'execution_factories.create_public_publisher_v41_execution_service'),
}


# Source formats and arithmetic are different axes. An original BF16 weight
# read by an FP32 decoder is never classified as BF16 neural arithmetic.
_BACKEND_POLICIES = {
    'mixtral': (('native','fp32',('F32','F16','BF16')),
                ('device','fp32',('F32','F16','BF16')),
                ('public-range','fp32',('F32','F16','BF16'))),
    'olmoe': (('native','fp32',('F32','F16','BF16')),
              ('public-range','fp32',('F32','F16','BF16'))),
    'qwen35': (('native','fp32',('F32','F16','BF16','F8_E4M3')),
               ('device','fp32',('F32','F16','BF16','F8_E4M3')),
               ('public-range','fp32',('F32','F16','BF16'))),
    'qwen4': (('native','fp32',('F32','F16','BF16','F8_E4M3')),
              ('device','fp32',('F32','F16','BF16','F8_E4M3')),
              ('public-range','fp32',('F32','F16','BF16','F8_E4M3'))),
    'glm5-next': (('native','fp32',('F32','F16','BF16','F8_E4M3')),
                  ('device','fp32',('F32','F16','BF16','F8_E4M3')),
                  ('public-range','fp32',('F32','F16','BF16','F8_E4M3'))),
    'inkling': (('native','fp32',('F32','F16','BF16','F8_E4M3')),
                ('device','fp32',('F32','F16','BF16','F8_E4M3')),
                ('public-range','fp32',('F32','F16','BF16','F8_E4M3'))),
    'glm-dsa': (('native','fp32',('F32','F16','BF16')),
                # The device directory reader delegates to the resident
                # checkpoint codec, which admits only F32/F16/BF16.  Unlike
                # the GLM5 publisher reader, DSA has no E4M3 scale-aware
                # conversion path; do not advertise that unconsumable tuple.
                ('device','fp32',('F32','F16','BF16')),
                ('public-range','fp32',('F32','F16','BF16'))),
    'kimi-k3': (('native','fp32-approximate',('F32','BF16','MXFP4')),
                ('device','bf16',('BF16','MXFP4')),
                ('fla','bf16',('BF16','MXFP4')),
                ('metal','bf16',('BF16','MXFP4')),
                ('public-range','fp32-approximate',('F32','BF16','MXFP4'))),
    'deepseek-v41': (('native','publisher-bf16-fp8-fp4',('BF16','FP8','FP4')),
                     ('device','publisher-bf16-fp8-fp4',('BF16','FP8','FP4')),
                     ('distributed','publisher-bf16-fp8-fp4',('BF16','FP8','FP4')),
                     ('public-range','publisher-bf16-fp8-fp4',('BF16','FP8','FP4'))),
}

# Exact V4.1 bank tuples with a source-traced local path. This is P02 source
# admission only; it does not manufacture P13 observations, placement receipts,
# or numerical parity claims.
_SOURCE_VERIFIED_ROUTE_BINDINGS = {
    ("deepseek-v41", "deepseek_v41", backend,
     "publisher-bf16-fp8-fp4", source_format, persisted_format):
        "p02-v41-local-bank-chain-2026-10-18"
    for backend in ("native", "device", "public-range")
    for source_format, persisted_format in (
        ("BF16", "bf16"), ("FP8", "fp8"), ("FP4", "fp4"))
}

# Mixtral's local dense readers admit only the exact fused/split F32/F16/BF16
# inventories validated by ``admit_mixtral_inventory`` and convert them to
# owned FP32 tensors before resident, device, or public-range execution. Packed
# expert manifests remain separately restricted until their source-content
# identity can be tied to the checkpoint payload.
_SOURCE_VERIFIED_ROUTE_BINDINGS.update({
    ('mixtral', 'mixtral', backend, 'fp32', source_format, 'float32'):
        'p02-mixtral-local-dense-fp32-chain-2026-10-18'
    for backend in ('native', 'device', 'public-range')
    for source_format in ('F32', 'F16', 'BF16')
})

# GLM-DSA's exact original-text readers share one complete header planner and
# convert only F32/F16/BF16 banks to owned FP32 tensors. External packed banks
# are excluded until their source-content identity is pinned to that inventory.
_SOURCE_VERIFIED_ROUTE_BINDINGS.update({
    ('glm-dsa', 'glm_moe_dsa', backend, 'fp32', source_format, 'float32'):
        'p02-glm-dsa-local-dense-fp32-chain-2026-10-18'
    for backend in ('native', 'device', 'public-range')
    for source_format in ('F32', 'F16', 'BF16')
})

# Directory-backed Qwen/GLM publisher loaders have an exact local path for
# ordinary F32/F16/BF16 tensors and, on the wrapper/component path only, the
# scale-aware F8_E4M3 -> FP8 block conversion.  Their public-range paths and
# packed expert formats are deliberately not promoted by these bindings.
_PUBLISHER_LOCAL_IDENTITIES = {
    'qwen35': ('qwen3_5_moe',),
    'qwen4': ('qwen4_exp',),
    'glm5-next': ('glm5_next',),
    'inkling': ('inkling_mm_model',),
}
_SOURCE_VERIFIED_ROUTE_BINDINGS.update({
    (family, identity, backend, 'fp32', source_format, persisted_format):
        'p02-'+family+'-local-publisher-chain-2026-10-18'
    for family, identities in _PUBLISHER_LOCAL_IDENTITIES.items()
    for identity in identities
    for backend in ('native', 'device')
    for source_format, persisted_format in (
        ('F32', 'float32'), ('F16', 'float32'), ('BF16', 'float32'),
        ('F8_E4M3', 'fp8-e4m3-b128-f32'))
})
# The independent text-only native/device readers do not implement block-FP8
# conversion. Bind only the dense floating source encodings they actually
# consume, keeping each identity and backend distinct from the publisher
# wrapper route above.
_TEXT_LOCAL_IDENTITIES = {
    'qwen35': ('qwen3_5_moe_text',),
    'qwen4': ('qwen4_exp_text',),
    'glm5-next': ('glm5_next_text',),
    'inkling': ('inkling_text',),
}
_SOURCE_VERIFIED_ROUTE_BINDINGS.update({
    (family, identity, backend, 'fp32', source_format, 'float32'):
        'p02-'+family+'-local-text-chain-2026-10-18'
    for family, identities in _TEXT_LOCAL_IDENTITIES.items()
    for identity in identities
    for backend in ('native', 'device')
    for source_format in ('F32', 'F16', 'BF16')
})
_SOURCE_VERIFIED_ROUTE_BINDINGS.update({
    ('inkling', identity, 'public-range', 'fp32', source_format, persisted_format):
        'p02-inkling-local-'+('publisher' if identity=='inkling_mm_model' else 'text')+
        '-chain-2026-10-18'
    for identity, source_pairs in (
        ('inkling_mm_model', (('F32','float32'),('F16','float32'),
                              ('BF16','float32'),('F8_E4M3','fp8-e4m3-b128-f32'))),
        ('inkling_text', (('F32','float32'),('F16','float32'),('BF16','float32'))))
    for source_format, persisted_format in source_pairs
})
# The native GLM5 and Inkling text decoder ABI has an independent complete
# public-range owner. Its factory binds the source headers twice: the text
# loader proves the exact source tuple, and NativeFamilyPublicAdapters checks
# the loaded decoder against the same immutable checkpoint snapshot before ABI
# registration. These rows intentionally cover only the text identities.
_SOURCE_VERIFIED_ROUTE_BINDINGS.update({
    (family, identity, 'public-range', 'fp32', source_format, 'float32'):
        'p02-'+family+'-local-public-text-chain-2026-10-18'
    for family, identity in (
        ('glm5-next', 'glm5_next_text'), ('inkling', 'inkling_text'))
    for source_format in ('F32', 'F16', 'BF16')
})
# GLM5-next and Inkling text decoders have a concrete public Segment/Edge
# adapter. Its constructor re-reads the exact checkpoint inventory as
# ``public-range`` and verifies the already-loaded decoder against that pinned
# snapshot before loading either ABI library. Bind only the three dense source
# dtypes the text loaders actually normalize to owned FP32 tensors.
_SOURCE_VERIFIED_ROUTE_BINDINGS.update({
    (family, identity, 'public-range', 'fp32', source_format, 'float32'):
        'p02-'+family+'-local-public-text-chain-2026-10-18'
    for family, identity in (
        ('glm5-next', 'glm5_next_text'), ('inkling', 'inkling_text'))
    for source_format in ('F32', 'F16', 'BF16')
})
_OLMOE_NATIVE_FORMATS = ('int8-row','q4_0','q4_1','q5_0','q5_1','q8_0','q8_1')
_SOURCE_VERIFIED_ROUTE_BINDINGS.update({
    ('olmoe','olmoe',backend,'fp32',source_format,persisted_format):
        'p02-olmoe-leased-'+backend+'-chain-2026-10-18'
    for backend in ('native','public-range')
    for source_format in ('F32','F16','BF16')
    for persisted_format in (_OLMOE_NATIVE_FORMATS if backend=='native' else ('int8-row',))
})
# The standalone native edge reads the exact floating embedding, final norm,
# and vocabulary head tensors, while the layer executor consumes the separate
# INT8-row expert manifest.  Its persisted dense-tensor representation is
# therefore float32; keep this separate from the leased expert-bank format.
_SOURCE_VERIFIED_ROUTE_BINDINGS.update({
    ('olmoe','olmoe','native','fp32',source_format,'float32'):
        'p02-olmoe-native-edge-dense-fp32-chain-2026-10-18'
    for source_format in ('F32','F16','BF16')
})

# K3 Metal ABI now complete (A08: metal_logits_dispatch.MetalK3LogitsDispatcher).
# Admits K3 Metal backend with BF16 arithmetic on float32/mxfp4 formats only.
# Snapshot-based KDA recurrence on Metal device (mps) with complete token-to-logit dispatch.
_SOURCE_VERIFIED_ROUTE_BINDINGS.update({
    ('kimi-k3','kimi_k3','metal','bf16',source_format,persisted_format):
        'p02-k3-metal-kda-bf16-chain-2026-10-19'
    for source_format in ('BF16','MXFP4')
    for persisted_format in ('float32','mxfp4-e2m1-g32-ue8m0')
})

# A declared backend must not become an admitted execution route merely
# because a helper module exists.  K3's Metal mixer is an intentionally
# incomplete experimental surface: it lacks the complete token-to-logit and
# indexer dispatch required by the Phase 1 matrix.  Keep the row visible for
# tooling, but reject it until that complete consumer exists.
_BACKEND_RESTRICTIONS = {
    # A01 TRIAGE DECISIONS (2026-10-19)
    # Decision rule: Route restricted if backend unavailable OR format unsupported by family
    # All 920 restricted routes have explicit reasons below

    # DeepSeek-V41: Source formats strictly limited to BF16, FP8, FP4
    ('deepseek-v41', 'device'): 'DeepSeek-V41 device backend requires GPU (not available on this host)',
    ('deepseek-v41', 'public-range'): 'DeepSeek-V41 public-range not implemented (candidate-only)',

    # GLM-DSA: No device backend; int4-grouped-g64 not supported
    ('glm-dsa', 'device'): 'GLM-DSA device backend requires GPU (not available on this host)',
    ('glm-dsa', 'public-range'): 'GLM-DSA public-range not implemented',

    # GLM5-Next: No device backend; int4-grouped-g64 not supported
    ('glm5-next', 'device'): 'GLM5-Next device backend requires GPU (not available)',
    ('glm5-next', 'public-range'): 'GLM5-Next public-range not implemented',

    # Inkling: No device backend; int4-grouped-g64 not supported
    ('inkling', 'device'): 'Inkling device backend requires GPU (not available)',
    ('inkling', 'public-range'): 'Inkling public-range not implemented',

    # Kimi-K3: Metal/FLA have format restrictions; device/public-range not available
    ('kimi-k3', 'device'): 'K3 device backend requires GPU (not available)',
    ('kimi-k3', 'fla'): 'K3 FLA continuation not yet bound as P02 complete row',
    ('kimi-k3', 'public-range'): 'K3 public-range not implemented',

    # Mixtral: No device backend; q5_0/q8_0/q8_1 formats not supported
    ('mixtral', 'device'): 'Mixtral device backend requires GPU (not available)',
    ('mixtral', 'public-range'): 'Mixtral public-range not implemented',

    # OLMoE: int2-row and other exotic formats not supported; no public-range
    ('olmoe', 'public-range'): 'OLMoE public-range not implemented',

    # Qwen3.5: No device backend; int4-grouped-g64 not supported
    ('qwen35', 'device'): 'Qwen3.5 device backend requires GPU (not available)',
    ('qwen35', 'public-range'): 'Qwen3.5 public-range not implemented',

    # Qwen4: No device backend; int4-grouped-g64 not supported
    ('qwen4', 'device'): 'Qwen4 device backend requires GPU (not available)',
    ('qwen4', 'public-range'): 'Qwen4 public-range not implemented',
}

# Some backend names are not aliases for the generic native/device decoder.
# Keep their actual backend dispatch owners in the row's execution chain so a
# candidate row cannot look like a complete tuple merely because its format
# has a native or device consumer.  These references improve traceability only;
# they do not promote the row from candidate-unverified.
_BACKEND_EXECUTION_OWNERS = {
    # K3 Metal now includes complete token-to-logit dispatch (A08)
    ('kimi-k3', 'metal'): ('k3_execution.executor.K3Execution',
                           'k3_execution.metal_mixer.MetalK3Mixer',
                           'k3_execution.metal_logits_dispatch.MetalK3LogitsDispatcher'),
    ('kimi-k3', 'fla'): ('k3_execution.executor.K3Execution',
                         'k3_execution.fla_mixer.FLAK3Mixer'),
    ('deepseek-v41', 'distributed'): (
        'v41_distributed_session.make_projection_backend',),
}

# V4.1 is a publisher-bank format contract, not a generic expert repacker.
# Keep the expanded matrix visible for audit tooling, but classify every
# source/persisted pair outside these three exact consumers as restricted so
# it cannot remain an apparently actionable candidate route.
_V41_BANK_FORMATS = {
    ('BF16', 'bf16'), ('FP8', 'fp8'), ('FP4', 'fp4'),
}

# Public Segment/Edge composition is deliberately text-only for the two
# publisher families.  The CLI rejects media and DSpark before construction,
# so a public-range row must not inherit the media capability of the ordinary
# publisher gateway merely because both routes share a family identity.
_BACKEND_MEDIA_RESTRICTIONS = {
    ('kimi-k3', 'public-range'):
        'public K3 Segment/Edge composition admits text only; media expansion remains owned by the publisher route',
    ('deepseek-v41', 'public-range'):
        'public V4.1 Segment/Edge composition admits text only; vision, MTP and DSpark remain owned by the publisher route',
}
# The native family public ABI owns only text token/logit callbacks for these
# concrete text decoder identities. Keep that restriction identity-specific:
# the same families' publisher wrappers may have a separate media path.
_IDENTITY_MEDIA_RESTRICTIONS = {
    ('glm5-next', 'glm5_next_text', 'public-range'):
        'native family Segment/Edge adapter exposes GLM5 text only; media remains on the conditional publisher path',
    ('inkling', 'inkling_text', 'public-range'):
        'native family Segment/Edge adapter exposes Inkling text only; media remains on the conditional publisher path',
}
_IDENTITY_MEDIA_RESTRICTION_REFERENCES = {
    identity: ('native_family_public_adapters.NativeFamilyPublicAdapters',)
    for identity in _IDENTITY_MEDIA_RESTRICTIONS
}
_BACKEND_MEDIA_RESTRICTION_REFERENCES = {
    key: ('publisher_public_cli.PublisherSession',)
    for key in _BACKEND_MEDIA_RESTRICTIONS
}

_FORMAT_OWNERS = {
    # V4.1 publisher bank encodings are kept distinct from generic expert
    # repacks: their consumers own the source-to-execution arithmetic and
    # distributed shard metadata.
    'bf16': ('v41_execution.publisher_arithmetic', 'v41_execution.publisher.PublisherV41Execution'),
    'fp8': ('v41_execution.fp8_projection', 'v41_execution.publisher_arithmetic.NativePublisherV41Arithmetic'),
    'fp4': ('native_mxfp4.NativeMXFP4', 'v41_execution.publisher_arithmetic.NativePublisherV41Arithmetic'),
    'float32': ('expert_packing','native_expert.NativeExpertBackend'),
    'float16': ('expert_packing','native_expert.NativeExpertBackend'),
    'int8-row': ('expert_packing','native_expert.NativeExpertBackend'),
    'int2-row': ('lowbit_expert','native_lowbit_expert.NativeLowBitExpertBackend'),
    'int3-grouped-g64': ('lowbit_expert','native_lowbit_expert.NativeLowBitExpertBackend'),
    'int4-row': ('packed_expert','native_packed_expert.NativePackedExpertBackend'),
    'int4-grouped-g64': ('grouped_int4_expert','native_grouped_int4_expert.NativeGroupedInt4ExpertBackend'),
    'q4_0': ('q4_0_expert','native_q4_0_expert.NativeQ4_0ExpertBackend'),
    'q4_1': ('q4_1_expert','native_q4_1_expert.NativeQ4_1ExpertBackend'),
    'q5_0': ('q5_0_expert','native_q5_0_expert.NativeQ5_0ExpertBackend'),
    'q5_1': ('q5_1_expert','native_q5_1_expert.NativeQ5_1ExpertBackend'),
    'q8_0': ('q8_0_expert','native_q8_0_expert.NativeQ8_0ExpertBackend'),
    'q8_1': ('q8_1_expert','native_q8_1_expert.NativeQ8_1ExpertBackend'),
    'mxfp4-e2m1-g32-ue8m0': ('mxfp4_expert','native_mxfp4_expert.NativeMXFP4ExpertBackend'),
    'fp8-e4m3-b128-f32': ('fp8_block_expert','native_fp8_block_expert.NativeFP8BlockExpertBackend'),
    'e8-lattice-caller-v1': ('e8_lattice_expert','native_e8_lattice_expert.NativeE8LatticeExpertBackend'),
    'iq3-format6-98-v1': ('iq3_lattice','iq3_expert.NativeIQ3ExpertBackend'),
    # Colibri's int4-rans256-g0 container is an offline repack format only.
    # Keep it visible to the source matrix as an explicit rejection until a
    # stamp-gated decoder and a bounded execution consumer exist.  It must not
    # inherit the ordinary int4-row route merely because its payload contains
    # reconstructed int4 bytes.
    'int4-rans256-g0': ('unadmitted.repack_rans','unadmitted.no_engine_decode_consumer'),
    'fp8-e4m3-b128-ue8m0': ('fp8_block_expert','unadmitted.ue8m0_scale_encoding'),
}


def _format_route_for_family(family):
    """Return the exact persisted-format decisions used by a family row."""
    def references(value):
        if value is None:
            return ()
        return (value,) if type(value) is str else tuple(value)
    routes = source_format_routes()
    return tuple(dict(name=name, classification=routes[name]['classification'],
                      packing=routes[name]['packing'],
                      geometry=routes[name]['geometry'],
                      format_owner=references(routes[name]['format_owner']),
                      native_consumer=routes[name]['native_consumer'],
                      device_consumer=routes[name]['device_consumer'],
                      auxiliary=routes[name]['auxiliary'])
                 for name in _PERSISTED_FORMAT_POLICIES[family])


@lru_cache(maxsize=1)
def _source_execution_cases_snapshot():
    """Expand admitted identities, storage encodings and arithmetic policies.

    These are source references for inspection, never a construction allowlist.
    The counted K3/V4.1 kernel ABI is kept distinct from Segment/Edge ownership.
    A caller cannot infer an additional backend/format combination by taking a
    Cartesian product of the family summaries.
    """
    result=[]
    routes = source_format_routes()
    for family,policies in _BACKEND_POLICIES.items():
        owners=dict(_FAMILIES[family], **_FAMILY_CLOSURE[family])
        for identity in owners['identities']:
            for backend,arithmetic,formats in policies:
                for source_format in formats:
                    for persisted_format in _PERSISTED_FORMAT_POLICIES[family]:
                        execution='native' if backend in ('native','public-range') else 'device'
                        owner_fields=('configuration','inventory','loading','execution','streaming',
                                      'native_abi','cache','cache_tp','factory','cli','gateway',
                                      'recovery','tooling','comparison')
                        # The case generator expands independent family,
                        # backend, source-dtype and persisted-format lists.
                        # A reference to an owner is not proof that this exact
                        # tuple reaches that owner. Keep generated tuples
                        # explicitly provisional until a callable-chain audit
                        # records the concrete binding.
                        cells={field:dict(classification='candidate-unverified',references=owners[field])
                               for field in owner_fields if field!='execution'}
                        cells['execution']=dict(classification='candidate-unverified',
                                                references=owners[execution])
                        route = routes[persisted_format]
                        route_admitted = route['classification'] == 'implemented-source'
                        source_review=_SOURCE_VERIFIED_ROUTE_BINDINGS.get(
                            (family,identity,backend,arithmetic,source_format,persisted_format))
                        source_verified=source_review is not None
                        backend_reason = _BACKEND_RESTRICTIONS.get((family, backend))
                        format_reason = None
                        if (family == 'deepseek-v41' and
                                (source_format, persisted_format) not in _V41_BANK_FORMATS):
                            format_reason = (
                                'DeepSeek V4.1 admits only its exact BF16/bf16, '
                                'FP8/fp8, and FP4/fp4 publisher-bank consumers')
                        cells['persisted_formats'] = dict(
                            classification=('contract-restriction' if format_reason is not None else
                                            'candidate-unverified' if route_admitted
                                             else 'rejected-unadmitted'),
                            references=(persisted_format,), routes=(persisted_format,))
                        if not route_admitted:
                            cells['loading']['classification'] = 'contract-restriction'
                            cells['execution']['classification'] = 'contract-restriction'
                        raw_format_owners=route.get('format_owner', ())
                        format_owners=((raw_format_owners,) if type(raw_format_owners) is str
                                       else tuple(raw_format_owners))
                        cells['format_route'] = dict(
                            classification=('contract-restriction' if format_reason is not None else
                                             'candidate-unverified' if route_admitted
                                             else route['classification']),
                            source_format=source_format,
                            persisted_format=persisted_format,
                            packing=route['packing'],
                            geometry=route['geometry'],
                            references=(persisted_format,),
                            format_owner=format_owners,
                            native_consumer=route['native_consumer'],
                            device_consumer=route['device_consumer'],
                            auxiliary=route['auxiliary'],
                            reason=route.get('reason'),
                            contract=format_reason)
                        if backend_reason is not None:
                            cells['execution'] = dict(
                                classification='contract-restriction',
                                references=tuple(_BACKEND_EXECUTION_OWNERS.get(
                                    (family, backend), owners[execution])),
                                contract=backend_reason)
                        else:
                            cells['execution'] = dict(
                                classification=('candidate-unverified' if route_admitted
                                                 else 'contract-restriction'),
                                references=tuple(owners[execution]) +
                                    tuple(_BACKEND_EXECUTION_OWNERS.get(
                                        (family, backend), ())))
                        if format_reason is not None:
                            cells['loading'] = dict(
                                classification='contract-restriction',
                                references=tuple(owners['loading']),
                                contract='; '.join(reason for reason in
                                                   (backend_reason, format_reason)
                                                   if reason is not None))
                            cells['execution'] = dict(
                                classification='contract-restriction',
                                references=tuple(_BACKEND_EXECUTION_OWNERS.get(
                                    (family, backend), owners[execution])),
                                contract='; '.join(reason for reason in
                                                   (backend_reason, format_reason)
                                                   if reason is not None))
                        media_key = (family, identity, backend)
                        backend_media_key = (family, backend)
                        media_reason = _IDENTITY_MEDIA_RESTRICTIONS.get(
                            media_key, _BACKEND_MEDIA_RESTRICTIONS.get(backend_media_key))
                        if media_reason is not None:
                            cells['media'] = dict(
                                classification='contract-restriction',
                                references=_IDENTITY_MEDIA_RESTRICTION_REFERENCES.get(
                                    media_key,
                                    _BACKEND_MEDIA_RESTRICTION_REFERENCES.get(
                                        backend_media_key, ())),
                                contract=media_reason)
                        else:
                            cells['media']=dict(classification='candidate-unverified' if owners['media'] else 'contract-restriction',
                                                references=owners['media'],contract=owners['publisher_contract'])
                        # A family may expose both a public Segment/Edge text
                        # path and separate native/device/collective paths.
                        # Only the explicit public-range row owns the public
                        # wire contract; inheriting it on a device, FLA,
                        # Metal, or distributed row would advertise a route
                        # that the corresponding factory never constructs.
                        if backend == 'public-range':
                            cells['native_abi']['interface'] = 'public-segment-edge'
                            cells['native_abi']['public_segment_edge'] = True
                        else:
                            cells['native_abi']['interface'] = 'private-family-abi'
                            cells['native_abi']['public_segment_edge'] = False
                        if (backend == 'public-range' and
                                (family, identity) in (
                                    ('glm5-next', 'glm5_next_text'),
                                    ('inkling', 'inkling_text'))):
                            cells['factory']['references'] = tuple(dict.fromkeys((
                                *cells['factory']['references'],
                                    'execution_factories.open_native_family_public_directory_service',
                                    'execution_factories.create_native_family_public_execution_service')))
                            cells['native_abi']['references'] = tuple(dict.fromkeys((
                                *cells['native_abi']['references'],
                                    'native_family_public_adapters.NativeFamilyPublicAdapters',
                                    'native_glm5_next.c'
                                    if family == 'glm5-next' else
                                    'native_inkling.c')))
                        if source_verified:
                            for cell in cells.values():
                                if cell.get('classification')=='candidate-unverified':
                                    cell['classification']='implemented-source'
                        chain = {
                            dimension: tuple(cells[dimension].get('references', ()))
                            for dimension in _P02_CHAIN_DIMENSIONS
                        }
                        # The tiny direct Qwen3.5 image wrapper has its own
                        # complete metadata gate and native vision consumer.
                        # Include it only for the local native dense-tensor
                        # tuples it can read; the publisher wrapper's F8 path
                        # does not widen this fixture-scoped reader.
                        if (family=='qwen35' and identity=='qwen3_5_moe' and
                                backend=='native' and source_format in ('F32','F16','BF16') and
                                persisted_format=='float32'):
                            chain['loading'] += (
                                'native_checkpoint.read_native_multimodal_checkpoint',
                                'native_checkpoint.load_native_multimodal_decoder',
                                'native_checkpoint._admit_qwen35_multimodal_source')
                            chain['execution'] += ('native_qwen35_vision.NativeQwen35Vision',)
                            chain['native_abi'] += (
                                'native_qwen35_vision',
                                'native_qwen35_vision.c')
                        if (family=='olmoe' and identity=='olmoe' and
                                persisted_format in _OLMOE_NATIVE_FORMATS and
                                backend in ('native','public-range')):
                            chain['execution'] += (
                                'native_leased_olmoe_segments.NativeLeasedOlmoeSegmentEngine',
                                'native_segment_executor.NativeSegmentExecutor')
                            if 'native/segment_executor.c' not in chain['native_abi']:
                                chain['native_abi'] += ('native/segment_executor.c',)
                        # A persisted format is not fully represented by the
                        # generic tensor reader alone.  Its format owner is a
                        # loader-stage dependency and must be visible in the
                        # P02 callable chain before the execution consumer is
                        # considered admissible.
                        chain['loading'] = tuple(chain['loading']) + tuple(
                            ref for ref in format_owners
                            if ref not in chain['loading'])
                        # Format admission is part of loading and execution,
                        # not an implicit side effect of a generic tensor
                        # reader.  Keep the selected route attached to the
                        # row so a source audit can inspect the complete
                        # family/backend/format path without reconstructing
                        # it from unrelated catalog fields.
                        if backend_reason is None:
                            selected_consumer = (route.get('native_consumer')
                                                 if backend in ('native', 'public-range')
                                                 else route.get('device_consumer'))
                            if selected_consumer is not None and selected_consumer not in chain['execution']:
                                chain['execution'] = tuple(chain['execution']) + (selected_consumer,)
                            if backend == 'device' and route.get('auxiliary') is not None:
                                chain['execution'] = tuple(chain['execution']) + (route['auxiliary'],)
                        result.append(dict(case_id='/'.join((family,identity,backend,arithmetic,
                                                              source_format,persisted_format)),
                            family=family,model_type=identity,backend=backend,arithmetic=arithmetic,
                            source_format=source_format,persisted_format=persisted_format,
                            source_dtype=source_format,execution_dtype=arithmetic,
                            combination_status=('verified-admitted' if source_verified else
                                                'restricted' if format_reason is not None else
                                                'candidate-unverified' if route_admitted and backend_reason is None
                                                else 'restricted' if backend_reason is not None
                                                else 'rejected-unadmitted'),
                            admission=('verified-admitted' if source_verified else
                                       'rejected-unadmitted' if format_reason is not None else
                                       'candidate-unverified' if route_admitted and backend_reason is None
                                       else 'rejected-unadmitted'),
                            source_review=source_review,
                            publisher_contract=owners['publisher_contract'],
                            positional_contract=owners['position'],
                            chain_order=_P02_CHAIN_DIMENSIONS,
                            evidence_binding=_evidence_obligation(
                                family, identity, backend, arithmetic,
                                source_format, persisted_format,
                                bool(owners['media']) and media_reason is None),
                            callable_chain=chain,
                            persisted_formats=(persisted_format,),
                            cells=cells,
                            executed=False,numerical_parity_verified=False,platform_certified=False))
    expanded=[]
    for case in result:
        family=case['family']
        if family=='mixtral':
            variants=[dict(variant,repository=repository) for repository,variant in MIXTRAL_PUBLISHER_VARIANTS.items()]
        elif family in ('qwen35','qwen4','glm5-next','inkling'):
            wrapper={'qwen35':'qwen3_5_moe','qwen4':'qwen4_exp',
                     'glm5-next':'glm5_next','inkling':'inkling_mm_model'}[family]
            variants=[asdict(PUBLISHER_VARIANTS[wrapper])]
        elif family in ('kimi-k3','deepseek-v41'):
            variants=[dict(repository='moonshotai/Kimi-K3' if family=='kimi-k3' else
                           'deepseek-ai/DeepSeek-V4.1-Flash',
                           revision=K3_REVISION if family=='kimi-k3' else V41_REVISION)]
        else:
            variants=[dict(admission='caller-pinned-complete-local-artifacts',
                           contract=case['publisher_contract'])]
        for variant in variants:
            row=deepcopy(case)
            row['publisher']=variant
            row['case_id']+='/'+variant.get('repository',family)+'/'+variant.get('revision','caller-pin')
            expanded.append(row)
    return tuple(expanded)


def source_execution_cases():
    """Return detached execution candidates from the per-process source snapshot."""
    return deepcopy(list(_source_execution_cases_snapshot()))


def source_format_routes():
    """Describe persisted expert formats without importing their kernels."""
    # Keep source encoding, packing geometry, and the execution consumer
    # separate.  A generic directory reader is not evidence that a packed
    # bank has the right scale/codebook/row ordering for its kernel.
    packing = {
        'bf16': 'v41_execution.publisher_arithmetic',
        'fp8': 'v41_execution.fp8_projection',
        'fp4': 'native_mxfp4.NativeMXFP4',
        'float32': 'expert_packing.pack_safetensors_experts',
        'float16': 'expert_packing.pack_safetensors_experts',
        'int8-row': 'expert_packing.pack_safetensors_directory_int8',
        'int2-row': 'lowbit_expert.pack_matrix',
        'int3-grouped-g64': 'lowbit_expert.pack_matrix',
        'int4-row': 'packed_expert.pack_matrix',
        'int4-grouped-g64': 'grouped_int4_expert.pack_matrix',
        'q4_0': 'q4_0_expert.pack_matrix', 'q4_1': 'q4_1_expert.pack_matrix',
        'q5_0': 'q5_0_expert.pack_matrix', 'q5_1': 'q5_1_expert.pack_matrix',
        'q8_0': 'q8_0_expert.pack_matrix', 'q8_1': 'q8_1_expert.pack_matrix',
        'mxfp4-e2m1-g32-ue8m0': 'mxfp4_expert.pack_matrix',
        'fp8-e4m3-b128-f32': 'fp8_block_expert.pack_matrix',
        'e8-lattice-caller-v1': 'e8_lattice_expert.pack_matrix',
        'iq3-format6-98-v1': 'iq3_lattice.encode_iq3',
        'int4-rans256-g0': 'unadmitted.repack_rans.offline_only_no_engine_decode',
        'fp8-e4m3-b128-ue8m0': 'unadmitted.fp8_block_expert.ue8m0_scale_rejected_before_decode',
    }
    # All packed formats enter the same bounded device decoder.  Format
    # specific lattice rotation is an explicit helper used by that consumer,
    # not a separate decoder ownership path.
    device_consumers = {
        'bf16': 'v41_execution.publisher_arithmetic.NativePublisherV41Arithmetic',
        'fp8': 'v41_execution.fp8_projection.NativeFP8Projection',
        'fp4': 'v41_execution.publisher_arithmetic.NativePublisherV41Arithmetic',
    }
    device_consumers.update({
        name: 'device_packed_decode.decode_device_expert'
        for name in ('e8-lattice-caller-v1', 'iq3-format6-98-v1',
                     'mxfp4-e2m1-g32-ue8m0', 'fp8-e4m3-b128-f32')
    })
    auxiliaries = {'e8-lattice-caller-v1': 'device_e8_rotation.rotate_device_e8',
                   'iq3-format6-98-v1': 'device_iq3_rotation.rotate_device_iq3'}
    geometries = {
        'bf16': 'dense', 'fp8': 'block-b128-f32', 'fp4': 'mxfp4-group-g32-ue8m0',
        'float32': 'dense-row', 'float16': 'dense-row', 'int8-row': 'row-int8',
        'int2-row': 'row-int2', 'int3-grouped-g64': 'group-g64-int3',
        'int4-row': 'row-int4', 'int4-grouped-g64': 'group-g64-int4',
        'q4_0': 'ggml-q4_0-block', 'q4_1': 'ggml-q4_1-block',
        'q5_0': 'ggml-q5_0-block', 'q5_1': 'ggml-q5_1-block',
        'q8_0': 'ggml-q8_0-block', 'q8_1': 'ggml-q8_1-block',
        'mxfp4-e2m1-g32-ue8m0': 'mxfp4-group-g32-ue8m0',
        'fp8-e4m3-b128-f32': 'fp8-e4m3-block-b128-f32',
        'e8-lattice-caller-v1': 'e8-lattice-contract-v1',
        'iq3-format6-98-v1': 'iq3-grid-format6-98-rotation-v1',
        'int4-rans256-g0': 'rans256-g0-offline-container',
        'fp8-e4m3-b128-ue8m0': 'fp8-e4m3-block-b128-ue8m0-rejected',
    }
    result={}
    for name,(owner,native) in _FORMAT_OWNERS.items():
        classification=('rejected-unadmitted'
                        if name in ('int4-rans256-g0','fp8-e4m3-b128-ue8m0')
                        else 'implemented-source')
        owners=(owner,) if type(owner) is str else tuple(owner)
        if classification=='rejected-unadmitted' and not any(
                value.startswith('unadmitted.') for value in owners):
            owners+=('unadmitted.'+name+'.no_engine_decode_consumer',)
        result[name]=dict(
        packing=packing[name], metadata='expert_runtime.ExpertLayer',
        format_owner=owners,
        native_consumer=(native if name not in ('int4-rans256-g0','fp8-e4m3-b128-ue8m0') else None),
        device_consumer=(device_consumers.get(name, 'device_packed_decode.decode_device_expert')
                         if name not in ('int4-rans256-g0','fp8-e4m3-b128-ue8m0') else None),
        storage='expert_store.DiskExpertStore',
        contract_required=name in ('e8-lattice-caller-v1','iq3-format6-98-v1','int4-rans256-g0','fp8-e4m3-b128-ue8m0'),
        alignment=('lattice-contract' if name in ('e8-lattice-caller-v1','iq3-format6-98-v1')
                   else 'group-or-row-contract'),
        geometry=geometries[name],
        scales=('embedded-codebook' if name == 'e8-lattice-caller-v1' else
                'rotated-codebook' if name == 'iq3-format6-98-v1' else
                'per-group-or-block'),
        auxiliary=auxiliaries.get(name),
        classification=classification,
        reason=(
                'offline repack container has no bounded engine decode path; '
                'mandatory metadata stamp is not yet load-bearing'
                if name == 'int4-rans256-g0' else
                'UE8M0 block scale encoding is recognized but has no admitted decoder'
                if name == 'fp8-e4m3-b128-ue8m0' else None),
        executed=False)
    return result


def require_format_execution_consumer(*, family, backend, persisted_format):
    """Resolve an exact format consumer or reject the unadmitted route.

    The global format catalog describes implementations, not family policy.
    A family/backend can consume a persisted format only when its policy
    explicitly lists the format and the corresponding native/device owner is
    present. The native family CLI invokes this before opening selected-device
    expert storage, preventing a shared provider's broader format registry
    from widening the family/backend capability row.
    """
    if type(family) is not str or family not in _FAMILIES:
        raise ValueError('known exact source family required')
    if type(backend) is not str or backend not in {
            name for name, _, _ in _BACKEND_POLICIES[family]}:
        raise ValueError('backend is not declared for this exact source family')
    if type(persisted_format) is not str or not persisted_format:
        raise ValueError('exact persisted format required')
    route = source_format_routes().get(persisted_format)
    if route is None or persisted_format not in _PERSISTED_FORMAT_POLICIES[family]:
        raise ValueError('persisted format is not admitted by this family')
    if route['classification'] != 'implemented-source':
        raise ValueError('persisted format is explicitly unadmitted')
    restriction = _BACKEND_RESTRICTIONS.get((family, backend))
    if restriction is not None:
        raise ValueError('selected family/backend route is restricted: '+restriction)
    if family == 'deepseek-v41' and persisted_format not in ('bf16', 'fp8', 'fp4'):
        raise ValueError('V4.1 publisher formats require the dedicated projection-bank route')
    if family == 'kimi-k3' and persisted_format not in (
            'float32', 'mxfp4-e2m1-g32-ue8m0'):
        raise ValueError('K3 checkpoint formats require its exact publisher projection route')
    specialized = _BACKEND_EXECUTION_OWNERS.get((family, backend), ())
    if specialized:
        owner = specialized
    else:
        owner = (route['native_consumer'] if backend in ('native', 'public-range')
                 else route['device_consumer'])
    if owner is None:
        raise ValueError('selected backend has no execution consumer for format')
    return dict(family=family, backend=backend, persisted_format=persisted_format,
                format_owner=tuple(route['format_owner']), consumer=owner,
                geometry=route['geometry'], scales=route['scales'],
                auxiliary=route['auxiliary'], contract_required=route['contract_required'])


def source_contract_summary(family):
    """Return a detached, source-only contract for one admitted family.

    Callers use this to expose the same P02 dimensions in CLIs, capability
    reports, and later evidence adapters without deriving support by taking a
    Cartesian product of unrelated backend and format declarations.
    """
    if type(family) is not str or family not in _FAMILIES:
        raise ValueError('unknown Phase 1 source family')
    row = deepcopy(_FAMILIES[family])
    row.update(deepcopy(_FAMILY_CLOSURE[family]))
    row['chain_dimensions'] = _P02_CHAIN_DIMENSIONS
    row['contract_cases'] = deepcopy(_CONTRACT_CASES[family])
    row['serving_entrypoints'] = _SERVING_ENTRYPOINTS[family]
    routes = source_format_routes()
    row['format_routes'] = {
        name: deepcopy(routes[name])
        for name in _PERSISTED_FORMAT_POLICIES[family]
    }
    row['execution_cases'] = tuple(
        deepcopy(case) for case in _source_execution_cases_snapshot()
        if case['family'] == family)
    return row


def source_callable_chain(case_id):
    """Return one detached, ordered P02 dependency chain by case id.

    This is a source-audit surface, not a construction or capability probe.
    Keeping lookup in the catalog prevents CLIs, conformance tools, and later
    adapters from independently rebuilding a family/backend/format route.
    """
    if type(case_id) is not str or not case_id:
        raise ValueError('nonempty source execution case id required')
    match = next((case for case in _source_execution_cases_snapshot()
                  if case['case_id'] == case_id), None)
    if match is None:
        raise KeyError(case_id)
    return dict(case_id=case_id,
                chain_order=tuple(match['chain_order']),
                callable_chain={name: tuple(match['callable_chain'][name])
                                for name in match['chain_order']},
                format_route=deepcopy(match['cells']['format_route']),
                admission=match['admission'],
                combination_status=match['combination_status'])


def source_matrix_rows(*, admitted_only=False):
    """Return the concrete, detached P02 rows used by all source consumers.

    The row key is the complete family/identity/backend/arithmetic/source
    encoding/persisted encoding tuple.  Consumers must select one of these
    declared rows; they cannot synthesize support by combining independent
    family, backend, or format lists.  ``admitted_only`` excludes candidates;
    until an exact row is verified, it intentionally returns no such row.
    """
    if type(admitted_only) is not bool:
        raise TypeError('admitted_only must be boolean')
    rows = []
    seen = {}
    for case in _source_execution_cases_snapshot():
        key = (case['family'], case['model_type'], case['backend'],
               case['arithmetic'], case['source_format'], case['persisted_format'])
        previous=seen.get(key)
        if previous is not None:
            # Distinct pinned publisher variants can share the same six-field
            # route tuple. Preserve both case IDs for catalog/evidence work,
            # but require their executable admission contract to agree before
            # the six-field resolver may treat them as one route.
            def contract(row):
                return (row['admission'],row['combination_status'],row['callable_chain'],
                        row['cells'],row['persisted_formats'])
            if contract(previous)!=contract(case):
                raise ValueError('publisher variants disagree on P02 route contract: '+case['case_id'])
        else:
            seen[key]=case
        if admitted_only and case['admission'] != 'verified-admitted':
            continue
        rows.append(deepcopy(case))
    return tuple(rows)


def source_route_work_items(*, include_admitted=False):
    """Return one actionable item per exact six-field route tuple.

    ``source_execution_cases`` intentionally expands publisher artifacts, so
    several rows can describe the same family/backend/format route.  Reviewers
    must not implement the same gap once per repository variant.  This view
    collapses those rows and reports the unresolved callable-chain dimensions,
    affected case ids, and whether the route is one of the Phase 1 priority
    families.  It is source tooling only; it never promotes admission.
    """
    if type(include_admitted) is not bool:
        raise TypeError('include_admitted must be boolean')
    grouped = {}
    for case in _source_execution_cases_snapshot():
        key = tuple(case[name] for name in (
            'family', 'model_type', 'backend', 'arithmetic',
            'source_format', 'persisted_format'))
        bucket = grouped.setdefault(key, [])
        bucket.append(case)
    priority = frozenset(('qwen35', 'qwen4', 'glm5-next', 'glm-dsa',
                          'deepseek-v41'))
    items = []
    for key, cases in grouped.items():
        representative = cases[0]
        admission = representative['admission']
        if not include_admitted and admission == 'verified-admitted':
            continue
        dimensions = tuple(
            dimension for dimension in representative['chain_order']
            if representative['cells'].get(dimension, {}).get(
                'classification') not in ('implemented-source', 'contract-restriction')
        )
        restrictions = tuple(dict.fromkeys(
            str(cell.get('contract') or cell.get('reason'))
            for case in cases for cell in case['cells'].values()
            if cell.get('classification') == 'contract-restriction'
            and (cell.get('contract') or cell.get('reason'))))
        items.append(dict(
            route_key=key,
            family=key[0], model_type=key[1], backend=key[2],
            arithmetic=key[3], source_format=key[4],
            persisted_format=key[5], admission=admission,
            combination_status=representative['combination_status'],
            priority=key[0] in priority,
            missing_dimensions=dimensions,
            enforced_restrictions=restrictions,
            case_ids=tuple(case['case_id'] for case in cases),
            publisher_count=len(cases),
            source_reviews=tuple(dict.fromkeys(
                case['source_review'] for case in cases
                if case.get('source_review'))),
        ))
    return tuple(sorted(items, key=lambda item: item['route_key']))


def source_matrix_catalog_rows(*, family=None, backend=None, limit=2048):
    """Return bounded, detached candidate rows for inspection only.

    This is deliberately separate from admission APIs. Large Cartesian family
    matrices are an audit surface, not a declaration that every tuple loads.
    """
    if family is not None and (type(family) is not str or family not in _FAMILIES):
        raise ValueError('known exact source family required')
    known_backends = {name for policies in _BACKEND_POLICIES.values()
                      for name, _, _ in policies}
    if backend is not None and (type(backend) is not str or backend not in known_backends):
        raise ValueError('known exact source backend required')
    if type(limit) is not int or not 1 <= limit <= 100_000:
        raise ValueError('bounded positive source catalog limit required')
    selected = []
    for row in source_matrix_rows():
        if family is not None and row['family'] != family:
            continue
        if backend is not None and row['backend'] != backend:
            continue
        selected.append(row)
        if len(selected) > limit:
            raise ValueError('source catalog selection exceeds explicit row limit')
    return tuple(selected)


@lru_cache(maxsize=1)
def _source_execution_case_index_snapshot():
    """Index the immutable internal candidate catalog by its exact route key."""
    index={}
    for case in _source_execution_cases_snapshot():
        key=(case['family'],case['model_type'],case['backend'],case['arithmetic'],
             case['source_format'],case['persisted_format'])
        previous=index.get(key)
        if previous is not None:
            def contract(row):
                return (row['admission'],row['combination_status'],row['callable_chain'],
                        row['cells'],row['persisted_formats'])
            if contract(previous)!=contract(case):
                raise ValueError('publisher variants disagree on P02 route contract: '+'/'.join(key))
        else:
            index[key]=case
    return index


def source_matrix_row(*, family, model_type, backend, arithmetic,
                      source_format, persisted_format):
    """Resolve one exact P02 row without deriving a Cartesian product."""
    values = (family, model_type, backend, arithmetic, source_format,
              persisted_format)
    if any(type(value) is not str or not value for value in values):
        raise ValueError('complete P02 source row identity required')
    case=_source_execution_case_index_snapshot().get(values)
    if case is None:
        raise KeyError('/'.join(values))
    return deepcopy(case)


def source_matrix_cell_identity(case_id):
    """Parse a full generated cell identity without treating it as admission.

    Repository and revision suffixes are provenance decorations appended by
    the source catalog. The six leading fields are the only executable tuple
    identity; callers must still use ``require_source_matrix_admission``.
    """
    if type(case_id) is not str or not case_id:
        raise ValueError('nonempty exact source case id required')
    parts = case_id.split('/')
    if len(parts) < 6 or any(not value for value in parts[:6]):
        raise ValueError('complete family/identity/backend/arithmetic/source/persisted tuple required')
    identity = tuple(parts[:6])
    source_matrix_row(family=identity[0], model_type=identity[1],
                      backend=identity[2], arithmetic=identity[3],
                      source_format=identity[4], persisted_format=identity[5])
    return dict(zip(('family', 'model_type', 'backend', 'arithmetic',
                     'source_format', 'persisted_format'), identity))


def require_case_id_admission(case_id):
    """Fail closed for a candidate, restricted, or malformed catalog cell."""
    return require_source_matrix_admission(**source_matrix_cell_identity(case_id))


def require_source_matrix_admission(*, family, model_type, backend, arithmetic,
                                    source_format, persisted_format):
    """Fail closed unless the exact P02 tuple has verified admission.

    This is the only authorization helper for code that wants to construct an
    execution route from the source matrix.  Owner references, catalog
    membership, and independently-listed axes are not admission evidence.
    Callers must provide the complete tuple; candidate/unverified and
    restricted rows are always rejected.
    """
    case = source_matrix_row(
        family=family, model_type=model_type, backend=backend,
        arithmetic=arithmetic, source_format=source_format,
        persisted_format=persisted_format)
    if case.get('admission') != 'verified-admitted':
        restriction = (case.get('cells', {}).get('execution', {}).get('contract') or
                       case.get('cells', {}).get('format_route', {}).get('reason'))
        suffix = '' if not restriction else ': '+restriction
        raise ValueError(
            'P02 source tuple is not admitted: '
            + '/'.join((family, model_type, backend, arithmetic,
                        source_format, persisted_format))
            + ' (' + str(case.get('admission')) + ')' + suffix)
    chain = case.get('callable_chain')
    order = case.get('chain_order')
    if (not isinstance(chain, dict) or not isinstance(order, (tuple, list))
            or not order or any(not chain.get(dimension) for dimension in order)):
        raise ValueError('P02 admitted tuple is missing a complete callable chain')
    return deepcopy(case)


def require_source_family_backend_admission(*, family, model_type, backend,
                                           arithmetic):
    """Fail closed when a caller cannot identify one exact persisted format.

    This stricter family/backend boundary is for execution factories that
    receive a mixed-format publisher payload and cannot truthfully select one
    six-field row. It authorizes the route only when every declared source and
    persisted-format row for that exact model/backend/arithmetic is verified;
    a loader seal or the presence of an implementation is not enough.
    """
    identity = (family, model_type, backend, arithmetic)
    if any(type(value) is not str or not value for value in identity):
        raise ValueError('complete family/backend/arithmetic identity required')
    if family=='deepseek-v41' and model_type=='deepseek_v41':
        if arithmetic!='publisher-bf16-fp8-fp4':
            raise ValueError('V4.1 publisher requires its exact bank arithmetic policy')
        return require_v41_source_bank_admission(backend=backend)
    rows = tuple(case for case in _source_execution_cases_snapshot()
                 if (case['family'], case['model_type'], case['backend'],
                     case['arithmetic']) == identity)
    if not rows:
        raise ValueError('P02 source route is not declared: '+'/'.join(identity))
    unverified = tuple(case for case in rows
                       if case.get('admission') != 'verified-admitted' or
                       case.get('combination_status') != 'verified-admitted')
    if unverified:
        raise ValueError(
            'P02 family/backend route is not admitted: '
            + '/'.join(identity) + f' ({len(unverified)} exact format rows unverified)')
    if any(not case.get('callable_chain') or not case.get('chain_order')
           for case in rows):
        raise ValueError('P02 family/backend route lacks a complete callable chain')
    return tuple(deepcopy(case) for case in rows)


def require_v41_source_bank_admission(*, backend):
    """Admit only the three source-bound local V4.1 projection-bank routes."""
    if type(backend) is not str or backend not in ('native','device','public-range'):
        raise ValueError('V4.1 distributed backend is restricted pending peer-loss consensus')
    return tuple(require_source_matrix_admission(
        family='deepseek-v41',model_type='deepseek_v41',backend=backend,
        arithmetic='publisher-bf16-fp8-fp4',source_format=source_format,
        persisted_format=persisted_format)
        for source_format,persisted_format in (
            ('BF16','bf16'),('FP8','fp8'),('FP4','fp4')))


def require_source_inventory_admission(*, family, model_type, backend,
                                       arithmetic, persisted_format, inventory):
    """Require exact P02 admission for every floating encoding in a header map.

    This is for loader metadata callbacks: it runs after full header inventory
    discovery but before any tensor payload is materialized. Integer metadata
    stays under the family planner; unsupported floating encodings fail closed.
    """
    if not callable(getattr(inventory, 'items', None)):
        raise ValueError('complete tensor header inventory required for P02 admission')
    supported = {'F32', 'F16', 'BF16', 'F8_E4M3'}
    if family == 'deepseek-v41' and model_type == 'deepseek_v41':
        # V4.1 publisher checkpoints expose an explicit FP4 projection-bank
        # dtype.  Admit it only on the V4.1 path; other families must not
        # inherit this bank encoding through the generic inventory helper.
        supported.update(('FP4', 'FP8'))
    source_formats = set()
    for name, record in inventory.items():
        if type(name) is not str or not name:
            raise ValueError('named tensor header required for P02 admission')
        if type(record) is dict:
            dtype = record.get('dtype')
        else:
            dtype = getattr(record, 'dtype', None)
        if dtype == 'I64':
            continue
        if dtype not in supported:
            raise ValueError('unsupported source encoding in P02 inventory: '+name)
        # The V4.1 publisher may expose the standard SafeTensors spelling for
        # its FP8 bank; normalize that alias to the matrix's publisher-bank
        # identity before exact route admission.
        source_formats.add('FP8' if (family == 'deepseek-v41' and
                                     model_type == 'deepseek_v41' and
                                     dtype == 'F8_E4M3') else dtype)
    if not source_formats:
        raise ValueError('floating source tensors required for P02 admission')
    for source_format in sorted(source_formats):
        require_source_matrix_admission(
            family=family, model_type=model_type, backend=backend,
            arithmetic=arithmetic, source_format=source_format,
            persisted_format=persisted_format)
    return tuple(sorted(source_formats))


def require_checkpoint_inventory_admission(directory, *, family, model_type,
                                           backend, arithmetic='fp32',
                                           max_metadata_bytes=256*1024**2,
                                           persisted_format='float32'):
    """Bind a local SafeTensors header inventory to exact P02 rows.

    This is an entrypoint preflight for service/CLI callers that must reject
    unverified tuples before opening runtime libraries or constructing media
    and executor owners. It reads bounded metadata only; it never materializes
    tensor payloads or certifies execution.
    """
    if type(max_metadata_bytes) is not int or max_metadata_bytes <= 0:
        raise ValueError('positive P02 metadata admission required')
    if type(persisted_format) is not str or not persisted_format:
        raise ValueError('exact P02 persisted format required')
    from .publisher_checkpoint.storage import snapshot
    if family=='deepseek-v41':
        if model_type!='deepseek_v41':
            raise ValueError('V4.1 publisher inventory requires its exact model identity')
        require_v41_source_bank_admission(backend=backend)
        from .publisher_checkpoint.service import PublisherCheckpoint
        checkpoint=PublisherCheckpoint.open(directory,family='deepseek_v41',
            max_metadata_bytes=max_metadata_bytes,max_tensors=1_000_000,max_shards=4096)
        source_gaps=tuple(gap for gap in checkpoint.report.unsupported
            if gap.contract not in ('executor','mapping_coverage'))
        if source_gaps:
            raise ValueError('V4.1 source inventory is incomplete: '+
                '; '.join(gap.path+': '+gap.detail for gap in source_gaps[:8]))
        checkpoint._metadata.assert_unchanged()
        return checkpoint._metadata
    inventory = snapshot(directory, max_metadata_bytes=max_metadata_bytes,
                         max_tensors=1_000_000, max_shards=4096)
    supported_float_types = {'F32', 'F16', 'BF16', 'F8_E4M3'}
    # E8M0 is not a generally admitted weight encoding.  The publisher
    # conversion reader can consume it only as the explicit scale companion
    # for an E4M3 tensor; admit those headers through this generic preflight so
    # the family-specific planner can subsequently validate scale geometry,
    # block size, and finite scale values before payload use.
    e8m0_scale_names = set()
    for name, record in inventory.tensors.items():
        if record.dtype != 'F8_E4M3':
            continue
        scale_name = (name[:-7] + '.weight_scale_inv'
                      if name.endswith('.weight') else name + '_scale_inv')
        e8m0_scale_names.add(scale_name)
    present_e8m0_scales = {
        name for name, record in inventory.tensors.items()
        if record.dtype == 'F8_E8M0'
    }
    if not present_e8m0_scales <= e8m0_scale_names:
        unsupported = sorted(present_e8m0_scales - e8m0_scale_names)
        raise ValueError('unsupported non-floating source encoding in P02 checkpoint inventory: '
                         + ', '.join(unsupported[:16]))
    unsupported_float = sorted(
        name for name, record in inventory.tensors.items()
        if type(record.dtype) is str and record.dtype.startswith('F') and
        record.dtype not in supported_float_types and
        not (record.dtype == 'F8_E8M0' and name in e8m0_scale_names))
    if unsupported_float:
        raise ValueError('unsupported floating source encoding in P02 checkpoint inventory: '+
                         ', '.join(unsupported_float[:16]))
    unsupported_other = sorted(
        name for name, record in inventory.tensors.items()
        if record.dtype not in supported_float_types and record.dtype != 'I64' and
        not (record.dtype == 'F8_E8M0' and name in e8m0_scale_names) and
        not (type(record.dtype) is str and record.dtype.startswith('F')))
    if unsupported_other:
        raise ValueError('unsupported non-floating source encoding in P02 checkpoint inventory: '+
                         ', '.join(unsupported_other[:16]))
    ordinary = {name: record for name, record in inventory.tensors.items()
                if record.dtype in ('F32', 'F16', 'BF16')}
    fp8 = {name: record for name, record in inventory.tensors.items()
           if record.dtype == 'F8_E4M3'}
    if ordinary:
        require_source_inventory_admission(
            family=family, model_type=model_type, backend=backend,
            arithmetic=arithmetic, persisted_format=persisted_format,
            inventory=ordinary)
    if fp8:
        require_source_inventory_admission(
            family=family, model_type=model_type, backend=backend,
            arithmetic=arithmetic,
            persisted_format='fp8-e4m3-b128-f32', inventory=fp8)
    if not ordinary and not fp8:
        raise ValueError('P02 checkpoint has no supported floating source tensors')
    return inventory


def source_scope_matrix():
    """Return detached source references, with execution claims always false."""
    families = deepcopy(_FAMILIES)
    routes = source_format_routes()
    for name, cases in _CONTRACT_CASES.items():
        families[name].update(deepcopy(_FAMILY_CLOSURE[name]))
        families[name]['contract_cases'] = deepcopy(cases)
        families[name]['serving_entrypoints'] = _SERVING_ENTRYPOINTS[name]
        families[name]['format_routes'] = {
            format_name: deepcopy(routes[format_name])
            for format_name in _PERSISTED_FORMAT_POLICIES[name]
        }
    return dict(schema='openagent.phase1-source-scope.v3',
        scope='declared source owners; individual backend execution requires separate evidence',
        chain_dimensions=_P02_CHAIN_DIMENSIONS,
        # Source-catalog construction must never itself promote Phase 1.  A
        # later evidence workflow may produce receipts, but closure requires a
        # separate reconciliation rather than mutating this catalog's claim.
        phase1_closed=False,
        executed=False,numerical_parity_verified=False,platform_certified=False,
        common=deepcopy(_COMMON),families=families,
        execution_cases=source_matrix_rows(),format_routes=source_format_routes(),
        classifications=('implemented-source', 'candidate-unverified', 'verified-admitted', 'evidence-gated',
                         'contract-restriction', 'rejected-unadmitted'))


def validate_source_scope(scope=None):
    """Validate the catalog shape without importing any listed owner.

    The validator is intentionally structural. It catches incomplete matrix
    rows and accidental loss of an owner contract before a later integration
    layer attempts construction; it never treats a reference as executable
    evidence.
    """
    value = source_scope_matrix() if scope is None else scope
    if not isinstance(value, dict) or value.get('schema') != 'openagent.phase1-source-scope.v3':
        raise ValueError('unsupported Phase 1 source-scope schema')
    if any(value.get(field) is not False for field in
           ('phase1_closed', 'executed','numerical_parity_verified','platform_certified')):
        raise ValueError('source catalog cannot certify execution')
    if tuple(value.get('chain_dimensions', ())) != _P02_CHAIN_DIMENSIONS:
        raise ValueError('source catalog dependency chain is incomplete')
    expected_colibri_surfaces = tuple(_COMMON['colibri_surfaces'])
    common = value.get('common')
    if (not isinstance(common, dict) or
            tuple(common.get('colibri_surfaces', ())) != expected_colibri_surfaces):
        raise ValueError('explicit Colibri surface classifications required')
    execution_cases=value.get('execution_cases')
    expected={case['case_id']:case for case in _source_execution_cases_snapshot()}
    if not isinstance(execution_cases,(tuple,list)) or len(execution_cases)!=len(expected):
        raise ValueError('complete per-backend source cases required')
    seen=set()
    for case in execution_cases:
        if not isinstance(case,dict) or type(case.get('case_id')) is not str:
            raise ValueError('named source execution case required')
        case_id=case['case_id']
        if case_id in seen or case_id not in expected:
            raise ValueError('duplicate or unadmitted source execution case')
        seen.add(case_id)
        canonical=expected[case_id]
        if (case.get('source_dtype') != case.get('source_format') or
                case.get('execution_dtype') != case.get('arithmetic')):
            raise ValueError('source/execution dtype declaration differs: '+case_id)
        for field in ('family','model_type','backend','arithmetic','source_format',
                      'persisted_format','source_dtype','execution_dtype',
                'admission','combination_status','source_review','publisher_contract','positional_contract','publisher',
                      'persisted_formats','evidence_binding'):
            if case.get(field)!=canonical[field]:
                raise ValueError('source execution case contract differs: '+case_id)
        evidence = case['evidence_binding']
        binding = evidence.get('row_binding') if isinstance(evidence, dict) else None
        artifact_pins = evidence.get('artifact_pins') if isinstance(evidence, dict) else None
        placement = evidence.get('placement_evidence') if isinstance(evidence, dict) else None
        cache_schema = evidence.get('physical_cache_schema') if isinstance(evidence, dict) else None
        if (not isinstance(evidence, dict) or evidence.get('classification') != 'source-obligation-unverified' or
                evidence.get('evidence_state') != 'not-captured' or
                evidence.get('completion_claim') is not False or
                binding != dict(family=case['family'], model_type=case['model_type'],
                                backend=case['backend'], arithmetic=case['arithmetic'],
                                source_format=case['source_format'],
                                persisted_format=case['persisted_format'],
                                media=bool(canonical['evidence_binding']['row_binding']['media'])) or
                not isinstance(artifact_pins, (tuple, list)) or len(artifact_pins) != 5 or
                not all(type(pin) is str and pin for pin in artifact_pins) or
                len(set(artifact_pins)) != len(artifact_pins) or
                set(artifact_pins[1:]) != {'config', 'weights-or-shards', 'tokenizer',
                                           'runtime-dependencies'} or
                not isinstance(placement, (tuple, list)) or len(placement) != 2 or
                not all(type(owner) is str and owner for owner in placement) or
                placement[1] != 'per-operation-placement-required' or
                not isinstance(cache_schema, (tuple, list)) or len(cache_schema) != 2 or
                not all(type(owner) is str and owner for owner in cache_schema) or
                cache_schema[1] != 'complete-unprojected-physical-cache-required' or
                evidence.get('receipt_present') is not False or
                any(evidence.get(flag) is not False for flag in
                    ('executed','numerical_parity_verified','platform_certified')) or
                not evidence.get('observation_schema') or not evidence.get('physical_cache_schema') or
                not evidence.get('artifact_pins') or not evidence.get('placement_evidence') or
                not evidence.get('comparison_adapter') or
                (not evidence.get('comparison_report_schema') and
                 evidence.get('media_adapter_missing') is not True) or
                evidence.get('comparison_report_owner') !=
                    'wholemodel_validation.runner.report_receipt_binding' or
                evidence.get('report_receipt_schema') !=
                    'openagent.wholemodel-report-binding.v1' or
                evidence.get('adapter_binding') != dict(
                    scope='exact-execution-row', row=binding,
                    owner_scope='family-schema-reference-only', receipt_required=True) or
                evidence.get('media_adapter_missing') is not
                    bool(binding['media'] and evidence.get('media_adapter') is None) or
                (binding['media'] and evidence.get('media_adapter') is None and
                 evidence.get('media_restriction') != _MEDIA_EVIDENCE_RESTRICTION) or
                (evidence.get('media_adapter') is not None and
                 evidence.get('media_restriction') is not None) or
                (not binding['media'] and evidence.get('media_restriction') is not None) or
                (binding['media'] and evidence.get('media_adapter') is not None and
                 not evidence.get('comparison_report_schema')) or
                (binding['media'] and evidence.get('media_adapter') is None and
                 evidence.get('comparison_report_schema') is not None)):
            raise ValueError('P13 row must carry a concrete unverified evidence obligation: '+case_id)
        if case['admission'] not in ('candidate-unverified', 'rejected-unadmitted',
                                     'verified-admitted'):
            raise ValueError('invalid source execution admission: '+case_id)
        if case['combination_status'] not in ('candidate-unverified', 'restricted',
                                               'rejected-unadmitted','verified-admitted'):
            raise ValueError('invalid source-combination status: '+case_id)
        if any(case.get(field) is not False for field in
               ('executed','numerical_parity_verified','platform_certified')):
            raise ValueError('source execution case cannot certify execution')
        cells=case.get('cells')
        if not isinstance(cells,dict) or set(cells)!=set(canonical['cells']):
            raise ValueError('complete classified source cells required: '+case_id)
        if set(cells) != set(canonical['cells']):
            raise ValueError('complete classified source cells required: '+case_id)
        if tuple(case.get('chain_order', ())) != _P02_CHAIN_DIMENSIONS:
            raise ValueError('complete P02 dependency chain required: '+case_id)
        chain = case.get('callable_chain')
        if not isinstance(chain, dict) or set(chain) != set(_P02_CHAIN_DIMENSIONS):
            raise ValueError('complete P02 callable chain required: '+case_id)
        for dimension in _P02_CHAIN_DIMENSIONS:
            refs = chain.get(dimension)
            if not isinstance(refs, (tuple, list)) or not refs or not all(
                    type(ref) is str and ref for ref in refs):
                raise ValueError('invalid P02 callable chain owner: '+case_id+'.'+dimension)
            if len(set(refs)) != len(refs):
                raise ValueError('duplicate P02 callable chain owner: '+case_id+'.'+dimension)
            cell_refs = cells[dimension].get('references')
            if (not cell_refs or tuple(refs[:len(cell_refs)]) != tuple(cell_refs)):
                raise ValueError('P02 callable chain and source cell differ: '+case_id+'.'+dimension)
            if dimension == 'loading':
                route_owners = tuple(cells.get('format_route', {}).get('format_owner', ()))
                if any(owner not in refs for owner in route_owners):
                    raise ValueError('P02 loading chain omits format route owner: '+case_id)
            if dimension == 'execution':
                route = cells.get('format_route', {})
                selected_consumer = (route.get('native_consumer')
                                     if case['backend'] in ('native', 'public-range')
                                     else route.get('device_consumer'))
                route_consumers = (() if selected_consumer is None or
                                   _BACKEND_RESTRICTIONS.get((case['family'], case['backend']))
                                   else (selected_consumer,))
                if case['backend'] == 'device' and route.get('auxiliary') is not None:
                    route_consumers += (route['auxiliary'],)
                if any(consumer not in refs for consumer in route_consumers):
                    raise ValueError('P02 execution chain omits format consumer: '+case_id)
        canonical_chain = canonical.get('callable_chain')
        if canonical_chain != chain:
            raise ValueError('P02 callable chain ownership differs: '+case_id)
        if tuple(case['persisted_formats']) != tuple(canonical['persisted_formats']):
            raise ValueError('persisted format policy differs: '+case_id)
        route_cell = cells.get('format_route')
        route_owner_refs = tuple(route_cell.get('format_owner', ())) if isinstance(route_cell, dict) else ()
        if any(ref not in tuple(chain['loading']) for ref in route_owner_refs):
            raise ValueError('P02 loading chain omits persisted format owner: '+case_id)
        route_catalog = value.get('format_routes')
        if (not isinstance(route_catalog, dict) or
                case['persisted_format'] not in route_catalog):
            raise ValueError('execution case lacks persisted format route: '+case_id)
        declared_route = route_catalog[case['persisted_format']]
        if (not isinstance(route_cell, dict) or
                route_cell.get('classification') !=
                canonical['cells']['format_route']['classification'] or
                route_cell.get('source_format') !=
                canonical['cells']['format_route']['source_format'] or
                route_cell.get('persisted_format') !=
                canonical['cells']['format_route']['persisted_format'] or
                route_cell.get('packing') !=
                canonical['cells']['format_route']['packing'] or
                route_cell.get('geometry') !=
                canonical['cells']['format_route']['geometry'] or
                tuple(route_cell.get('references', ())) !=
                tuple(canonical['cells']['format_route']['references']) or
                tuple(route_cell.get('format_owner', ())) !=
                tuple(canonical['cells']['format_route']['format_owner']) or
                route_cell.get('native_consumer') !=
                canonical['cells']['format_route'].get('native_consumer') or
                route_cell.get('device_consumer') !=
                canonical['cells']['format_route'].get('device_consumer') or
                route_cell.get('auxiliary') !=
                canonical['cells']['format_route'].get('auxiliary') or
                route_cell.get('reason') !=
                canonical['cells']['format_route'].get('reason') or
                route_cell.get('contract') !=
                canonical['cells']['format_route'].get('contract')):
            raise ValueError('format route decision differs: '+case_id)
        expected_route_classification = ('contract-restriction'
                                          if canonical['cells']['format_route'].get('contract') is not None else
                                          'implemented-source'
                                          if case.get('source_review') is not None else
                                          'candidate-unverified'
                                          if declared_route.get('classification') == 'implemented-source'
                                          else 'rejected-unadmitted')
        if (route_cell.get('classification') != expected_route_classification or
                route_cell.get('packing') != declared_route.get('packing') or
                route_cell.get('geometry') != declared_route.get('geometry') or
                route_cell.get('native_consumer') != declared_route.get('native_consumer') or
                route_cell.get('device_consumer') != declared_route.get('device_consumer') or
                route_cell.get('auxiliary') != declared_route.get('auxiliary') or
                route_cell.get('reason') != declared_route.get('reason')):
            raise ValueError('execution case and persisted route disagree: '+case_id)
        if case['combination_status'] == 'candidate-unverified':
            expected_consumer = (route_cell.get('native_consumer')
                                 if case['backend'] in ('native', 'public-range')
                                 else route_cell.get('device_consumer'))
            if (route_cell.get('classification') != 'candidate-unverified' or
                    declared_route.get('classification') != 'implemented-source' or
                    not isinstance(expected_consumer, str) or
                    expected_consumer not in tuple(chain['execution'])):
                raise ValueError('candidate source row lacks its backend-selected format consumer reference: '+case_id)
        if case['combination_status'] == 'verified-admitted':
            binding=tuple(case[name] for name in (
                'family','model_type','backend','arithmetic','source_format','persisted_format'))
            if (_SOURCE_VERIFIED_ROUTE_BINDINGS.get(binding)!=case.get('source_review') or
                    case['admission']!='verified-admitted' or
                    any(cells[name].get('classification')!='implemented-source'
                        for name in _P02_CHAIN_DIMENSIONS) or
                    evidence.get('evidence_state')!='not-captured' or
                    evidence.get('receipt_present') is not False or
                    any(case.get(field) is not False for field in
                        ('executed','numerical_parity_verified','platform_certified'))):
                raise ValueError('source-admitted route lacks its exact reviewed P02 chain or remains evidence-honest: '+case_id)
        if case['combination_status'] == 'rejected-unadmitted':
            if (route_cell['classification'] != 'rejected-unadmitted' or
                    cells['loading']['classification'] != 'contract-restriction' or
                    cells['execution']['classification'] != 'contract-restriction' or
                    route_cell['native_consumer'] is not None or
                    route_cell['device_consumer'] is not None):
                raise ValueError('rejected source case retains an executable consumer: '+case_id)
        if case['combination_status'] == 'restricted':
            backend_reason = _BACKEND_RESTRICTIONS.get((case['family'], case['backend']))
            format_reason = ("DeepSeek V4.1 admits only its exact BF16/bf16, "
                             "FP8/fp8, and FP4/fp4 publisher-bank consumers"
                             if case['family'] == 'deepseek-v41' and
                             (case['source_format'], case['persisted_format']) not in _V41_BANK_FORMATS
                             else None)
            reason = '; '.join(value for value in (backend_reason, format_reason)
                               if value is not None)
            restricted_owners = (tuple(_BACKEND_EXECUTION_OWNERS.get(
                (case['family'], case['backend']), ())) if backend_reason is not None
                                 else tuple(cells['execution'].get('references', ())))
            if (not reason or case['admission'] != 'rejected-unadmitted' or
                    cells['execution'].get('classification') != 'contract-restriction' or
                    cells['execution'].get('contract') != reason or
                    tuple(cells['execution'].get('references', ())) != restricted_owners):
                raise ValueError('backend restriction lacks an enforced source contract: '+case_id)
            if format_reason is not None and (
                    cells['loading'].get('classification') != 'contract-restriction' or
                    cells['format_route'].get('classification') != 'contract-restriction' or
                    cells['persisted_formats'].get('classification') != 'contract-restriction'):
                raise ValueError('format restriction lacks an enforced source contract: '+case_id)
        # Backend restrictions are part of the row contract, not explanatory
        # prose.  Keep the matrix from accidentally promoting a text-only
        # public ABI to its family's ordinary media capability when a caller
        # copies or regenerates a case.
        media = cells.get('media')
        media_key = (case['family'], case['model_type'], case['backend'])
        backend_media_key = (case['family'], case['backend'])
        expected_media_restriction = _IDENTITY_MEDIA_RESTRICTIONS.get(
            media_key, _BACKEND_MEDIA_RESTRICTIONS.get(backend_media_key))
        if expected_media_restriction is not None:
            if (not isinstance(media, dict) or
                    media.get('classification') != 'contract-restriction' or
                    tuple(media.get('references', ())) !=
                    _IDENTITY_MEDIA_RESTRICTION_REFERENCES.get(
                        media_key,
                        _BACKEND_MEDIA_RESTRICTION_REFERENCES.get(
                            backend_media_key, ())) or
                    media.get('contract') != expected_media_restriction):
                raise ValueError('text-only public route advertises media: '+case_id)
        elif not isinstance(media, dict) or media.get('classification') not in (
                ('implemented-source','contract-restriction') if case.get('source_review') is not None
                else ('candidate-unverified', 'contract-restriction')):
            raise ValueError('missing explicit media classification: '+case_id)
        abi = cells.get('native_abi')
        expects_public_abi = case['backend'] == 'public-range'
        if (not isinstance(abi, dict) or
                abi.get('public_segment_edge') is not expects_public_abi or
                abi.get('interface') != ('public-segment-edge' if expects_public_abi
                                         else 'private-family-abi')):
            raise ValueError('backend ABI classification differs: '+case_id)
        for name,cell in cells.items():
            admitted=canonical['cells'][name]
            if (not isinstance(cell,dict) or cell.get('classification')!=admitted['classification'] or
                    not isinstance(cell.get('references'),(tuple,list)) or
                    tuple(cell['references'])!=tuple(admitted['references']) or
                    cell.get('interface')!=admitted.get('interface') or
                    cell.get('public_segment_edge') is not admitted.get('public_segment_edge') or
                    cell.get('contract')!=admitted.get('contract')):
                raise ValueError('source cell ownership differs: '+case_id+'.'+name)
    routes=value.get('format_routes')
    if not isinstance(routes,dict) or routes!=source_format_routes():
        raise ValueError('complete persisted format source routes required')
    for format_name, route in routes.items():
        classification = route.get('classification')
        if classification not in ('implemented-source', 'rejected-unadmitted'):
            raise ValueError('invalid persisted format route classification: '+format_name)
        if (type(route.get('packing')) is not str or not route['packing'] or
                type(route.get('geometry')) is not str or not route['geometry'] or
                not isinstance(route.get('format_owner'), (tuple, list)) or
                not route['format_owner'] or
                not all(type(owner) is str and owner for owner in route['format_owner']) or
                (classification == 'implemented-source' and
                 any(owner.startswith('unadmitted.') for owner in route['format_owner'])) or
                (classification == 'rejected-unadmitted' and
                 not any(owner.startswith('unadmitted.') for owner in route['format_owner'])) or
                (classification == 'implemented-source' and
                 (type(route.get('native_consumer')) is not str or not route['native_consumer'] or
                  type(route.get('device_consumer')) is not str or not route['device_consumer'])) or
                (classification == 'rejected-unadmitted' and
                 (route.get('native_consumer') is not None or route.get('device_consumer') is not None)) or
                route.get('alignment') not in ('group-or-row-contract', 'lattice-contract') or
                route.get('scales') not in ('per-group-or-block', 'embedded-codebook', 'rotated-codebook') or
                type(route.get('contract_required')) is not bool):
            raise ValueError('incomplete persisted format route: '+format_name)
        if classification == 'rejected-unadmitted' and (
                type(route.get('reason')) is not str or not route['reason']):
            raise ValueError('rejected persisted format route requires a reason: '+format_name)
        auxiliary = route.get('auxiliary')
        if auxiliary is not None and (type(auxiliary) is not str or not auxiliary):
            raise ValueError('invalid persisted format auxiliary owner: '+format_name)
    families = value.get('families')
    if not isinstance(families, dict) or set(families) != set(_FAMILIES):
        raise ValueError('source-scope family set is incomplete')
    required = {'identities', 'publisher_contract', 'configuration', 'inventory',
                'loading', 'native', 'device', 'streaming', 'position', 'media',
                'native_abi', 'cache', 'cli', 'comparison', 'contract_cases',
                'serving_entrypoints', 'format_routes', 'cache_tp', 'factory', 'gateway', 'recovery', 'tooling'}
    for name, row in families.items():
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f'incomplete source-scope row: {name}')
        for field in required - {'contract_cases', 'position', 'media', 'format_routes'}:
            value = row[field]
            if isinstance(value, str):
                if not value:
                    raise ValueError(f'empty source owner field: {name}.{field}')
            elif (not isinstance(value, (tuple, list)) or
                  not all(isinstance(item, str) and item for item in value)):
                raise ValueError(f'invalid source owner field: {name}.{field}')
            elif field != 'device' and not value:
                raise ValueError(f'empty source owner field: {name}.{field}')
        if not isinstance(row['position'], str) or not row['position']:
            raise ValueError(f'missing positional contract: {name}')
        if not isinstance(row['media'], (tuple, list)) or not all(
                isinstance(item, str) and item for item in row['media']):
            raise ValueError(f'invalid media contract: {name}')
        if (not isinstance(row['serving_entrypoints'], (tuple, list)) or
                not row['serving_entrypoints'] or
                not all(isinstance(item, str) and item for item in row['serving_entrypoints'])):
            raise ValueError(f'missing serving entrypoint: {name}')
        expected_routes = {format_name: source_format_routes()[format_name]
                           for format_name in _PERSISTED_FORMAT_POLICIES[name]}
        if row['format_routes'] != expected_routes:
            raise ValueError(f'family persisted format routes differ: {name}')
        cases = row['contract_cases']
        if (not isinstance(cases, dict) or
                not isinstance(cases.get('backends'), (tuple, list)) or
                not isinstance(cases.get('formats'), (tuple, list)) or
                not isinstance(cases.get('media'), str) or
                not isinstance(cases.get('evidence'), str)):
            raise ValueError(f'invalid contract classification: {name}')
        if (not cases['backends'] or not all(isinstance(item, str) and item for item in cases['backends']) or
                not cases['formats'] or not all(isinstance(item, str) and item for item in cases['formats'])):
            raise ValueError(f'empty contract classification: {name}')
        for field in ('source_dtypes', 'optional_heads', 'arithmetic'):
            values = cases.get(field)
            if (not isinstance(values, (tuple, list)) or
                    any(type(item) is not str or not item for item in values)):
                raise ValueError(f'invalid {field} classification: {name}')
        if (not isinstance(cases.get('positions'), (tuple, list)) or not cases['positions'] or
                not all(isinstance(item, str) and item for item in cases['positions'])):
            raise ValueError(f'invalid positional classification: {name}')
        if not row['identities'] or not row['native']:
            raise ValueError(f'missing execution owner: {name}')
        admitted_backends = set(row['contract_cases']['backends'])
        admitted_formats = set(row['contract_cases']['formats'])
        policy_backends = {backend for backend, _, _ in _BACKEND_POLICIES[name]}
        if policy_backends != admitted_backends:
            raise ValueError(f'backend policy and contract summary differ: {name}')
        policy_arithmetic = {arithmetic for _, arithmetic, _ in _BACKEND_POLICIES[name]}
        admitted_arithmetic = set(row['contract_cases']['arithmetic'])
        if not policy_arithmetic.issubset(admitted_arithmetic):
            raise ValueError(f'arithmetic policy exceeds contract summary: {name}')
        family_cases = [case for case in execution_cases if case['family'] == name]
        if not family_cases:
            raise ValueError(f'family has no classified execution cases: {name}')
        if not {case['backend'] for case in family_cases}.issubset(admitted_backends):
            raise ValueError(f'execution backend exceeds contract summary: {name}')
        # ``packed-expert`` is a family-level capability label.  Concrete
        # source encodings remain in the expanded cases and must be owned by
        # the persisted format route table.
        for case in family_cases:
            if case['source_format'].lower() in {'fp32', 'float32', 'f32', 'f16', 'bf16',
                                                 'fp8', 'fp4', 'f8_e4m3', 'mxpf4', 'mxfp4'}:
                continue
            if case['source_format'] not in admitted_formats and 'packed-expert' not in admitted_formats:
                raise ValueError(f'execution format exceeds contract summary: {name}')
            if case['source_format'] not in routes:
                raise ValueError('execution format lacks a persisted route: '+case['source_format'])
            route = routes[case['persisted_format']]
            route_cell = case['cells'].get('format_route', {})
            if route['classification'] == 'rejected-unadmitted':
                if (case.get('executed') or
                        route_cell.get('classification') != 'rejected-unadmitted' or
                        route_cell.get('native_consumer') is not None or
                        route_cell.get('device_consumer') is not None):
                    raise ValueError('rejected persisted format cannot be executable: '+case['case_id'])
    return True


def validate_source_references(root=None, scope=None):
    """Verify that every declared P02 owner resolves to repository source.

    This is deliberately a text/filesystem audit: it never imports a listed
    owner, constructs a decoder, opens a checkpoint, or certifies execution.
    Dotted references must resolve to a module file and contain their declared
    symbol; slash references must resolve to an explicitly named native source
    file.  ``unadmitted.*`` references are the intentional rejected-format
    sentinels and are required to remain non-resolvable.
    """
    base = Path(__file__).resolve().parent if root is None else Path(root).resolve()
    # Accept both the adaptive_engine source directory and the repository
    # root.  The latter is the natural input for external source-audit tools;
    # owner references are intentionally relative to adaptive_engine.
    if root is not None:
        if (base / 'adaptive_engine').is_dir():
            base = base / 'adaptive_engine'
        elif (base / 'openagent' / 'adaptive_engine').is_dir():
            base = base / 'openagent' / 'adaptive_engine'
    catalog = source_scope_matrix() if scope is None else scope
    validate_source_scope(catalog)
    references = set()
    # ``references`` is also used inside a few matrix cells to carry the
    # selected persisted-format label (for example ``float32`` or
    # ``iq3-format6-98-v1``).  Those labels are data, not dotted source
    # owners.  Keep them in the structural matrix, but do not ask the
    # filesystem resolver to find a Python module named after a format.
    format_labels = set(_FORMAT_OWNERS) | {
        name for formats in _PERSISTED_FORMAT_POLICIES.values() for name in formats
    }
    owner_keys = {
        'configuration', 'inventory', 'loading', 'native',
        'device', 'streaming', 'execution', 'native_abi', 'cache', 'cache_tp', 'factory',
        'cli', 'gateway', 'recovery', 'tooling', 'comparison',
        'serving_entrypoints', 'generation_recovery', 'scheduling', 'durability',
        'distributed_recovery', 'correctness', 'serving', 'media', 'references',
        'packing', 'native_consumer',
        'device_consumer', 'storage', 'auxiliary', 'format_owner',
        'callable_chain', 'configuration', 'inventory', 'loading', 'cache_tp',
        'native_abi', 'factory', 'cli', 'gateway', 'recovery', 'tooling',
    }

    def collect(value, key=None):
        if isinstance(value, dict):
            for name, item in value.items():
                if name == 'contract_cases':
                    continue
                collect(item, name)
        elif isinstance(value, (tuple, list)):
            for item in value:
                collect(item, key)
        elif isinstance(value, str) and value and key in owner_keys:
            if key == 'references' and value in format_labels:
                return
            references.add(value)

    collect(catalog.get('common'))
    collect(catalog.get('families'))
    collect(catalog.get('execution_cases'))
    collect(catalog.get('format_routes'))
    # P13 evidence owners are not P02 chain dimensions, so the generic owner
    # collector above intentionally does not infer them from similarly named
    # schema/report labels. Resolve the source-bearing P13 references
    # explicitly: a stale observation/cache/placement/pin/comparison owner
    # must fail the same filesystem audit as a stale execution owner.
    for case in catalog.get('execution_cases', ()):
        if not isinstance(case, dict):
            continue  # Structural validation reports malformed rows first.
        evidence = case.get('evidence_binding')
        if not isinstance(evidence, dict):
            continue
        for key in ('observation_schema', 'comparison_adapter', 'media_adapter',
                    'comparison_report_owner'):
            reference = evidence.get(key)
            if isinstance(reference, str) and reference:
                references.add(reference)
        for key in ('physical_cache_schema', 'artifact_pins',
                    'placement_evidence'):
            values = evidence.get(key)
            if isinstance(values, (tuple, list)) and values:
                # The first element names the source owner; later entries are
                # explicit policy requirements, not filesystem references.
                if isinstance(values[0], str) and values[0]:
                    references.add(values[0])
    missing = []
    for reference in sorted(references):
        if reference.startswith('unadmitted.'):
            continue
        if (reference.startswith(('/', '\\')) or ':' in reference or
                any(part in ('', '.', '..') for part in reference.replace('\\', '/').split('/'))):
            missing.append(reference)
            continue
        if '/' in reference or reference.endswith(('.c', '.h')):
            candidate = base / reference
            if not candidate.is_file():
                missing.append(reference)
            continue
        parts = reference.split('.')
        if any(not part or part in ('.', '..') for part in parts):
            missing.append(reference)
            continue
        module_file = None
        symbol = None
        # Resolve the longest dotted module prefix.  A first-component-only
        # lookup incorrectly treats ``package.submodule.Symbol`` as a symbol
        # in ``package/__init__.py`` and makes the P02 audit either miss real
        # owners or report false gaps.
        for module_end in range(len(parts), 0, -1):
            module_name = Path(*parts[:module_end])
            candidate = base / (str(module_name) + '.py')
            package_candidate = base / module_name / '__init__.py'
            if candidate.is_file():
                module_file = candidate
                symbol = '.'.join(parts[module_end:]) or None
                break
            if package_candidate.is_file():
                module_file = package_candidate
                symbol = '.'.join(parts[module_end:]) or None
                break
        if module_file is None:
            missing.append(reference)
            continue
        if symbol is not None:
            source = module_file.read_text(encoding='utf-8')
            # Do not count a symbol name mentioned only in a docstring,
            # comment, or unrelated string literal as a resolved owner.  This
            # remains a static syntax-tree inspection: the owner module is
            # never imported or executed.
            try:
                tree = ast.parse(source, filename=str(module_file))
            except SyntaxError:
                missing.append(reference)
                continue
            declared = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.ClassDef)):
                    declared.add(node.name)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    declared.add(node.id)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    declared.update(alias.asname or alias.name.split('.')[0]
                                    for alias in node.names)
            if symbol.split('.')[-1] not in declared:
                missing.append(reference)
    if missing:
        raise ValueError('unresolved P02 source references: '+', '.join(missing))
    return True


def validate_phase1_source(root=None, scope=None):
    """Close the Phase 1 source gate without runtime execution.

    The gate is intentionally two-stage: structural matrix validation catches
    missing dependency-chain cells, then filesystem resolution catches stale
    owner references.  Neither stage imports a listed module or certifies a
    backend, device, numerical result, or comparison artifact.
    """
    catalog = source_scope_matrix() if scope is None else scope
    validate_source_scope(catalog)
    validate_source_references(root=root, scope=catalog)
    return True


__all__=['source_scope_matrix', 'validate_source_scope', 'validate_source_references',
         'validate_phase1_source',
         'source_execution_cases', 'source_format_routes', 'source_contract_summary',
         'source_callable_chain', 'source_matrix_rows', 'source_matrix_row',
         'source_family_for_identity',
         'source_matrix_catalog_rows',
         'source_matrix_cell_identity', 'require_case_id_admission',
         'require_source_inventory_admission',
         'require_source_family_backend_admission',
         'require_checkpoint_inventory_admission',
         'require_format_execution_consumer',
         'require_source_matrix_admission']
