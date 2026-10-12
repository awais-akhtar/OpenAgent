"""Prepare pinned GLM/Inkling full-model comparison cases for phase two."""
from copy import deepcopy

from .configuration import comparison_configuration
from .publisher_family_caches import schema, empty_shapes, PublisherFamilyCacheReader
from ..publisher_family_contracts import PUBLISHER_VARIANTS


def family_configuration(*, family, config, offline_root, metadata,
                         oracle_artifacts, candidate_artifacts, oracle_options,
                         decoder_factory, decoder_options, tokens,
                         max_cache_export_bytes, max_observation_bytes,
                         logits_tolerance, cache_tolerance, generation_steps=8,
                         seed=0, chunk_sizes=(1, 2, 7, 32),
                         routing_tolerance=None,max_routing_export_bytes=None):
    """Create source-only inputs with complete cache and continuation schedules.

    Artifact hashes must come from the caller's actual offline files. Numerical
    tolerances are explicit policy inputs, never assertions of certification.
    The oracle remains the unchanged pinned publisher implementation.
    """
    wrappers={'glm5_next_text':'glm5_next','inkling_text':'inkling_mm_model','qwen4_exp_text':'qwen4_exp',
              'qwen3_5_moe_text':'qwen3_5_moe'}
    if family not in wrappers:
        raise ValueError('explicit implemented GLM/Inkling observation family required')
    variant=PUBLISHER_VARIANTS[wrappers[family]]
    reader_type=PublisherFamilyCacheReader
    if family=='qwen3_5_moe_text':
        from .qwen35_publisher_caches import Qwen35PublisherCacheReader
        reader_type=Qwen35PublisherCacheReader
    if family=='qwen4_exp_text':
        from .qwen4_publisher_caches import Qwen4PublisherCacheReader
        reader_type=Qwen4PublisherCacheReader
    contract=reader_type(family=family,config=config,role='candidate',
                                      max_export_bytes=max_cache_export_bytes)
    if type(max_observation_bytes) is not int or max_observation_bytes<=0:
        raise ValueError('positive bounded token observation admission required')
    candidate_metadata=deepcopy(dict(metadata))
    candidate_metadata['cache_schema']=schema(family,config)
    candidate_metadata['empty_cache_shapes']=empty_shapes(family,config)
    candidate_metadata['num_layers']=len(contract.geometry or contract.config.layer_types)
    candidate_metadata['vocabulary_size']=(
        contract.config.get('unpadded_vocab_size') or contract.config['vocab_size']
        if family=='inkling_text' else contract.config.vocab_size)
    routing=[]
    if routing_tolerance is not None:
        if family not in ('glm5_next_text','qwen4_exp_text','qwen3_5_moe_text'):
            raise ValueError('implemented publisher routing observation family required')
        evidence=contract.enable_routing_observations(max_export_bytes=max_routing_export_bytes)
        candidate_metadata['evidence_schema']=deepcopy(evidence)
        candidate_metadata['empty_evidence_shapes']={}
        routing=[dict(name=name,oracle=name,candidate=name,tolerance=deepcopy(routing_tolerance))
                 for name in evidence]
    elif candidate_metadata.get('evidence_schema') or max_routing_export_bytes is not None:
        raise ValueError('explicit routing tolerance and export admission required together')
    candidate_metadata['publisher_repository']=variant.repository
    candidate_metadata['publisher_revision']=variant.revision
    candidate_metadata['publisher_source_version']=variant.source_version
    candidate_metadata['certified']=False
    oracle=deepcopy(dict(oracle_options))
    oracle_metadata=deepcopy(dict(oracle.get('metadata',candidate_metadata)))
    for name in ('cache_schema','empty_cache_shapes','num_layers','vocabulary_size'):
        oracle_metadata[name]=deepcopy(candidate_metadata[name])
    oracle['metadata']=oracle_metadata
    if 'cache_reader' in oracle or 'cache_reader_factory' in oracle:
        raise ValueError('family configuration owns the complete publisher cache reader')
    oracle['cache_reader_factory']='openagent.adaptive_engine.wholemodel_validation.publisher_family_caches:make_reader'
    oracle['cache_reader_options']=dict(family=family,config=deepcopy(dict(config)),
        role='oracle',max_export_bytes=max_cache_export_bytes,
        recurrent_layout=oracle.pop('recurrent_layout','key-value'))
    if routing:
        if 'forward_call' in oracle or (family=='glm5_next_text' and 'forward_call_factory' in oracle):
            raise ValueError('routing configuration owns the pinned observation forward call')
        oracle['metadata']['evidence_schema']=deepcopy(candidate_metadata['evidence_schema'])
        oracle['metadata']['empty_evidence_shapes']={}
        oracle['cache_reader_options']['routing_max_export_bytes']=max_routing_export_bytes
        if family=='glm5_next_text':
            oracle['forward_call_factory']='openagent.adaptive_engine.wholemodel_validation.glm5_routing_oracle:make_forward_call'
            oracle['forward_call_options']=dict(config=deepcopy(dict(config)),max_observation_bytes=max_routing_export_bytes)
        else:
            # Exact publisher module inventory is supplied by the pinned case.
            selected=deepcopy(dict(oracle.get('forward_call_options',{})))
            if not oracle.get('forward_call_factory'):
                module='qwen35_routing_oracle' if family=='qwen3_5_moe_text' else 'qwen4_routing_oracle'
                oracle['forward_call_factory']='openagent.adaptive_engine.wholemodel_validation.'+module+':make_forward_call'
                if any(name not in selected for name in ('router_paths','router_classes','output_indices')):
                    raise ValueError('pinned Qwen4 returned router inventory required')
            selected.update(config=deepcopy(dict(config)),max_observation_bytes=max_routing_export_bytes)
            oracle['forward_call_options']=selected
    candidate=dict(family=family,config=deepcopy(dict(config)),metadata=candidate_metadata,
        decoder_factory=decoder_factory,decoder_options=deepcopy(dict(decoder_options)),
        max_cache_export_bytes=max_cache_export_bytes,max_observation_bytes=max_observation_bytes)
    if routing:candidate['max_routing_export_bytes']=max_routing_export_bytes
    return comparison_configuration(offline_root=offline_root,
        oracle_factory='openagent.adaptive_engine.wholemodel_validation.local_transformers:create_local_transformers_adapter',
        candidate_factory='openagent.adaptive_engine.wholemodel_validation.publisher_family_candidate:create_candidate',
        oracle_artifacts=oracle_artifacts,candidate_artifacts=candidate_artifacts,
        oracle_options=oracle,candidate_options=candidate,tokens=tokens,
        logits_tolerance=logits_tolerance,cache_tolerance=cache_tolerance,routing_evidence=routing,
        generation_steps=generation_steps,seed=seed,chunk_sizes=chunk_sizes,continuation_restore=True)


__all__=['family_configuration']
