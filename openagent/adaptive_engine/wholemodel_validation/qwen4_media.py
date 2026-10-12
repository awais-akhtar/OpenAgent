"""Qwen4 media observation transport with its own complete QSA/PLE caches."""
from dataclasses import dataclass
import copy
from .qwen35_media import Qwen35MediaCase,Qwen35MediaSession,Qwen35MediaAdapter
from .qwen4_publisher_caches import Qwen4PublisherCacheReader


@dataclass(frozen=True)
class Qwen4MediaCase(Qwen35MediaCase):
    fixture_schema='openagent.qwen4-media-fixture.v1'


class Qwen4MediaSession(Qwen35MediaSession):
    reader_class=Qwen4PublisherCacheReader

    def _wrapper_class(self):
        from ..qwen4_multimodal import Qwen4MultimodalDecoder
        return Qwen4MultimodalDecoder

    def _extra_cache_elements(self,position):
        c=self.reader.config
        indexers=sum(position*c.indexer_head_dim for kind in c.layer_types if kind!='linear_attention')
        lexical=len(c.ple_layer_ids)*((c.ngram_size-1)*2+
            c.hc_count*c.hidden_size*(c.ple_conv_kernel_size-1)*c.ngram_size)
        return indexers+lexical+position*8


class Qwen4MediaAdapter(Qwen35MediaAdapter):
    reader_class=Qwen4PublisherCacheReader
    session_class=Qwen4MediaSession


def create_candidate(*,offline_root,options):
    from pathlib import Path
    import inspect
    from .runner import pinned_artifacts
    from .adapters import resolve_callable
    from . import qwen35_media
    root=Path(offline_root).resolve(strict=True)
    pins=pinned_artifacts({'artifacts':options['_artifact_pins']},root)
    loader=resolve_callable(options['decoder_factory']);source=inspect.getsourcefile(loader)
    if source is None:raise ValueError('inspectable original Qwen4 wrapper loader required')
    for item in (source,__file__,qwen35_media.__file__):
        path=Path(item).resolve(strict=True)
        if not any(row['kind']=='source' and Path(row['resolved_path'])==path for row in pins):
            raise ValueError('actual wrapper loader and complete media observation control sources must be pinned')
    meta=copy.deepcopy(options['metadata'])
    if meta.get('authoritative') is True:raise ValueError('candidate cannot declare authoritative oracle identity')
    reader=Qwen4PublisherCacheReader(family='qwen4_exp_text',config=options['config'],role='candidate',
        max_export_bytes=options['max_cache_export_bytes'],media=True)
    routing=bool(meta.get('evidence_schema'))
    if routing and reader.enable_routing_observations(max_export_bytes=options['max_routing_export_bytes'])!=meta['evidence_schema']:
        raise ValueError('complete executed Qwen4 routing schema required')
    if meta.get('num_layers')!=reader.config.num_hidden_layers or meta.get('vocabulary_size')!=reader.config.vocab_size:
        raise ValueError('media metadata differs from executed Qwen4 geometry')
    decoder_options=copy.deepcopy(dict(options.get('decoder_options',{})))
    def fresh(*,seed):return loader(seed=seed,offline_root=str(root),options=copy.deepcopy(decoder_options))
    return Qwen4MediaAdapter(fresh,meta,reader,max_bytes=options['max_observation_bytes'],capture_routing=routing)


def run_media_configuration(configuration,*,offline_root):
    from .qwen35_media import load_media_fixture,run_media_comparison
    from .runner import load_adapter,pinned_artifacts
    from .comparison import Tolerance
    fields={'schema','artifacts','candidate','oracle','fixture_config','max_observation_bytes','seed','generation_steps','tolerance'}
    if (type(configuration) is not dict or set(configuration)!=fields or
            configuration['schema']!='openagent.qwen4-media-comparison-configuration.v1'):
        raise ValueError('exact pinned Qwen4 media comparison configuration required')
    pins=configuration['artifacts'];pinned_artifacts({'artifacts':pins},offline_root)
    case=load_media_fixture(offline_root=offline_root,fixture_config=configuration['fixture_config'],
        artifact_pins=pins,max_bytes=configuration['max_observation_bytes'],case_class=Qwen4MediaCase)
    tolerance=configuration['tolerance']
    if type(tolerance) is not dict or set(tolerance)!={'atol','rtol'}:raise ValueError('explicit comparison tolerances required')
    adapters=[]
    for role in ('candidate','oracle'):
        spec=configuration[role]
        if type(spec) is not dict or set(spec)!={'factory','options'}:raise ValueError('explicit pinned media adapter required')
        adapters.append(load_adapter(dict(spec,artifacts=pins),offline_root))
    return run_media_comparison(*adapters,case,offline_root=offline_root,artifact_pins=pins,
        max_bytes=configuration['max_observation_bytes'],seed=configuration['seed'],
        generation_steps=configuration['generation_steps'],tolerance=Tolerance(**tolerance),
        fixture_config=configuration['fixture_config'])


__all__=['Qwen4MediaCase','Qwen4MediaSession','Qwen4MediaAdapter','create_candidate','run_media_configuration']
