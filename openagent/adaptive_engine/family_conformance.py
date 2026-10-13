"""Exact identity conformance data, not a decoder or architecture alias map.

Registry identities/sections are metadata only. No reference implementation
is imported or executed. Optional Transformers probing uses installed public
mapping metadata, never downloads weights or upgrades dependencies.
Run with --probe-transformers for a JSON report of the current environment.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy

from .hf_moe import SUPPORTED_MODEL_TYPES
from .text_decoder_registry import _DECODERS
from .mixtral_inventory import MIXTRAL_PUBLISHER_VARIANTS
from .source_scope import (source_scope_matrix, validate_phase1_source,
                           source_contract_summary, source_callable_chain)


# Identity and wrapper metadata only; similar attention names are not equations.
REGISTRY_IDENTITIES = (
    ('mixtral', ('mixtral',), 'root',
     'Exact fused/split inventory, native/device GQA, selected packed experts, public ranges and complete cache recovery have source paths; numerical and platform certification requires evidence'),
    ('glm53', ('glm5_next', 'glm5_next_text'), 'text_config',
     'Explicit publisher component mapping, vision, selected original experts and native/device text owners exist; publisher and platform comparisons remain unverified'),
    ('glm', ('glm_moe_dsa', 'glm5_moe', 'glm'), 'root',
     'Exact glm_moe_dsa native/device decoders, selected original device experts, public hidden ranges and durable latent/rotary/indexer recovery exist; other GLM identities remain separate contracts'),
    ('inkling', ('inkling_mm_model', 'inkling_text', 'inkling'), 'text_config',
     'Pinned Inkling wrapper mapping includes media, MTP, selected original experts and separate native/device text owners; the legacy inkling identity is not an alias'),
    ('kimi', ('kimi_k3', 'kimi_linear'), 'text_config',
     'K3 publisher contracts, image/video admission, FLA continuation, BF16/native/device arithmetic and exact evidence admission exist; full numerical and platform compatibility remains evidence-gated'),
    ('olmoe', ('olmoe',), 'root',
     'Leased packed expert ranges, native storage and public Segment/Edge adapters have source implementations; upstream numerical and ABI certification requires evidence'),
    ('qwen36', ('qwen3_5_moe', 'qwen3_5_moe_text'), 'text_config',
     'Qwen3.5 native/device hybrid text, interleaved MRoPE, publisher wrapper vision, selected original/packed experts and media recovery have source paths; the tiny vision fixture remains a distinct API'),
    ('qwen38', ('qwen4_exp', 'qwen4_exp_text'), 'text_config',
     'Qwen4 wrapper mapping, native/device QSA/PLE, selected original or packed stores, vision and public ranges exist; other rotary contracts require explicit admission'),
    ('deepseek_v4', ('deepseek_v4',), 'root',
     'Bounded resident FP32 C replay decoder exists; optimized KV ownership, quantized execution and production-scale support remain unsupported'),
    ('deepseek_v41', ('deepseek_v41', 'deepseek_v41_text'), 'text_config',
     'Pinned publisher mappings, packed attention/Engram state, vision, DSpark, streaming projections and distributed banks exist; exact numerical admission remains evidence-gated'),
)

# Public text entry points do not imply full registry-family or wrapper support.
RESIDENT_TEXT_CONTRACTS = {
    'mixtral': ('GQA-default-or-linear-RoPE-sliding-FP32-resident-or-explicit-compiled-streamed-experts-padding-positions', False),
    'glm_moe_dsa': ('compressed-latent-DSA-default-static-YaRN-RoPE-FP32-resident', False),
    'qwen3_5_moe_text': ('hybrid-gated-delta-text-default-or-interleaved-MRoPE-FP32-resident-or-selected-experts', False),
    'qwen4_exp_text': ('linear-QSA-PLE-default-or-linear-text-RoPE-FP32-resident', False),
    'glm5_next_text': ('bounded-KDA-MLA-KPool-MHC-NoPE-FP32-resident', False),
    'deepseek_v4': ('bounded-sliding-CSA-HCA-FP32-token-replay', False),
    'inkling_text': ('hybrid-sliding-FIR-text-FP32-resident', False),
    'kimi_linear': ('bounded-KDA-MLA-text-FP32-resident', True),
}


def conformance_matrix(*, probe_transformers=False):
    """Return JSON-serializable scope facts, not a model-load authorization.

    Library allowlist membership is only a necessary condition for loading;
    tensor geometry, checkpoint headers and module ownership still need checks.
    Native RMSNorm is family-neutral and proves no family decoder support.
    """
    metadata = {}
    version = None
    installed = None
    if probe_transformers:
        try:
            import transformers
        except ModuleNotFoundError as error:
            if error.name != 'transformers':
                raise
            installed = False
        else:
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
            from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
            installed = True
            version = transformers.__version__
            metadata = {name: (CONFIG_MAPPING_NAMES.get(name),
                               MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get(name))
                        for _, names, _, _ in REGISTRY_IDENTITIES for name in names}
    scope = source_scope_matrix()
    validate_phase1_source(scope=scope)
    scope_families = scope['families']
    rows = []
    for family, names, section, reason in REGISTRY_IDENTITIES:
        scope_name = {
            'glm53': 'glm5-next', 'glm': 'glm-dsa', 'kimi': 'kimi-k3',
            'qwen36': 'qwen35', 'qwen38': 'qwen4',
            'deepseek_v41': 'deepseek-v41',
        }.get(family, family)
        scope_row = scope_families.get(scope_name)
        if scope_row is None and family!='deepseek_v4':
            raise ValueError('registry family lacks Phase 1 source scope: '+family)
        for name in names:
            config, causal = metadata.get(name, (None, None))
            phase1_catalogued=scope_row is not None and name in scope_row['identities']
            phase1_admitted=any(case['family']==scope_name and
                                case['model_type']==name and
                                case.get('admission')=='verified-admitted'
                                for case in scope.get('execution_cases', ()))
            rows.append(dict(model_type=name, registry_family=family,
                config_section='root' if name in _DECODERS or name=='deepseek_v41_text' else section,
                library_allowlisted=name in SUPPORTED_MODEL_TYPES,
                independent_python_family=name in {'mixtral','olmoe', 'qwen3_5_moe_text', 'glm_moe_dsa', 'qwen4_exp_text'},
                independent_contract=('linear-QSA-PLE-default-text-RoPE-FP32-resident' if name == 'qwen4_exp_text' else None),
                native_resident_contract=RESIDENT_TEXT_CONTRACTS.get(name, (None, False))[0],
                native_resident_range_supported=name in {'mixtral','olmoe', 'qwen3_5_moe_text', 'glm_moe_dsa', 'qwen4_exp_text', 'glm5_next_text','inkling_text'},
                native_streamed_range_supported=name in {'mixtral','olmoe','glm_moe_dsa','qwen3_5_moe_text','qwen4_exp_text','glm5_next_text','inkling_text'},
                c_owned_storage_runtime=False,
                resident_text_loader_available=name in _DECODERS,
                resident_multimodal_loader_available=name in {'qwen3_5_moe','qwen4_exp','glm5_next','inkling_mm_model','kimi_k3','deepseek_v41'},
                native_resident_text_decoder_supported=name in RESIDENT_TEXT_CONTRACTS,
                resident_text_execution_contract=RESIDENT_TEXT_CONTRACTS.get(name, (None, False))[0],
                resident_text_experimental=RESIDENT_TEXT_CONTRACTS.get(name, (None, False))[1],
                native_decoder_supported=False, upstream_abi_compatible=False,
                phase1_identity_admitted=phase1_admitted,
                phase1_identity_catalogued=phase1_catalogued,
                source_execution_cases=[deepcopy(case) for case in scope['execution_cases']
                                        if case['family']==scope_name and
                                        case['model_type']==name]
                if phase1_catalogued else [],
                source_contract_summary=(source_contract_summary(scope_name)
                                        if phase1_catalogued else {}),
                source_contract_cases=dict(scope_row['contract_cases']) if phase1_catalogued else {},
                source_owner_modules=dict(
                    configuration=scope_row['configuration'],
                    inventory=scope_row['inventory'], loading=scope_row['loading'],
                    execution=scope_row['native'] + scope_row['device'],
                    streaming=scope_row['streaming'],
                    cache_tp=scope_row['cache_tp'], native_abi=scope_row['native_abi'],
                    factory=scope_row['factory'], cli=scope_row['cli'],
                    gateway=scope_row['gateway'], recovery=scope_row['recovery'],
                    tooling=scope_row['tooling'], comparison=scope_row['comparison'],
                    serving=scope_row['serving_entrypoints']) if phase1_catalogued else {},
                source_callable_chains=[
                    dict(source_callable_chain(case['case_id']),
                         admission=case['admission'],
                         persisted_format=case['persisted_format'],
                         callable_chain=deepcopy(case['callable_chain']))
                    for case in scope['execution_cases']
                    if phase1_catalogued and case['family']==scope_name and
                    case['model_type']==name],
                source_chain_dimensions=scope_row.get('chain_dimensions', ()) if phase1_catalogued else (),
                rejection_reason=reason, transformers_config_class=config,
                transformers_causal_class=causal,
                identity_collision=(name == 'glm' and causal == 'GlmForCausalLM')))
    return dict(schema='openagent.family-conformance.v1',
        source_scope=scope,
        evidence_scope='identity metadata and installed library mappings; not numerical conformance',
        native_decoder_supported_scope='full-registry-family',
        transformers_probed=probe_transformers, transformers_installed=installed,
        transformers_version=version, registry_models=rows,
        library_allowlist=sorted(SUPPORTED_MODEL_TYPES),
        mixtral_publisher_variants={repository:dict(variant) for repository,variant in MIXTRAL_PUBLISHER_VARIANTS.items()},
        independent_python_families=['mixtral', 'olmoe', 'qwen3_5_moe_text', 'glm_moe_dsa', 'qwen4_exp_text'],
        native_resident_range_families=['mixtral','olmoe', 'qwen3_5_moe_text', 'glm_moe_dsa', 'qwen4_exp_text', 'glm5_next_text', 'inkling_text'],
        native_resident_text_decoder_families=sorted(RESIDENT_TEXT_CONTRACTS),
        native_primitives=['rmsnorm_f32', 'gated_delta_f32',
                           'olmoe_resident_range_f32', 'embedding_f32', 'head_f32',
                           'qwen4_text_resident_f32', 'kimi_k3_projected_kda_f32',
                           'kimi_k3_normalized_latent_moe_f32',
                           'deepseek_v41_local_e2m1_e4m3_codec'],
        native_fixture_contracts=[dict(
            architecture='kimi_k3', module='native_kimi_k3_hybrid',
            constructor='NativeKimiK3Hybrid', configuration='KimiK3HybridConfig',
            contract='bounded-explicit-FP32-KDA-MLA-LatentMoE-AttnRes-token-to-logit',
            verification='independently evaluated equations; not unchanged full-hybrid vendor parity',
            published_checkpoint=False, packed_streaming=False,
            full_registry_family=False, upstream_abi_compatible=False)],
        resident_text_loader_families=sorted(_DECODERS),
        resident_text_contracts=[dict(model_type=name, module=_DECODERS[name][0],
            constructor=_DECODERS[name][1], contract=contract, experimental=experimental,
            packed_streaming=False, full_registry_family=False,
            upstream_abi_compatible=False)
            for name, (contract, experimental) in sorted(RESIDENT_TEXT_CONTRACTS.items())],
        experimental_text_families=['kimi_linear'],
        native_decoder_families=[])


def require_native_decoder(model_type):
    """Require full registry-family support, not a bounded text-only contract.

    The separately exposed resident text loader admits supported exact subsets.
    That does not authorize this broader full-family gate or upstream ABI use.
    """
    if type(model_type) is not str or not model_type:
        raise ValueError('expected exact nonempty model_type')
    raise ValueError(f'full registry-family native execution is not certified for {model_type!r}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-transformers', action='store_true')
    arguments = parser.parse_args()
    print(json.dumps(conformance_matrix(probe_transformers=arguments.probe_transformers), indent=2))
