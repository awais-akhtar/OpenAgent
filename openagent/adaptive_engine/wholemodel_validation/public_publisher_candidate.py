"""Explicit pinned publisher loader for native public-ABI comparison cases.

Configuration creation never imports a decoder or runs model arithmetic. The
runner invokes this factory only during the separately requested execution phase.
"""
import hashlib
from pathlib import Path


def _library(root,record):
    from ..native_library_identity import NativeLibraryIdentity
    if type(record) is not dict or set(record)!={'path','sha256'}:
        raise ValueError('explicit native artifact path and SHA-256 required')
    path=Path(record['path'])
    if path.is_absolute() or '..' in path.parts:raise ValueError('offline-root relative native artifact required')
    resolved=(root/path).resolve(strict=True)
    if not resolved.is_relative_to(root):raise ValueError('native artifact escapes admitted offline root')
    identity=NativeLibraryIdentity.read(resolved)
    if identity.sha256!=record['sha256']:raise ValueError('pinned native artifact identity differs')
    return identity


def load_public_publisher(*,seed,offline_root,options):
    """Own the full publisher wrapper and select its actual public native path."""
    if type(seed) is not int:raise ValueError('explicit integer comparison seed required')
    required={'family','checkpoint','config_sha256','media_library','text_library','runtime_library',
        'max_weight_bytes','max_bytes','context_tokens','max_conversion_bytes','segment_work_bytes'}
    optional={'segment_layer_span','max_snapshots','max_ple_bytes','max_expert_bytes',
        'native_expert_max_bytes','native_ple_max_bytes'}
    if type(options) is not dict or not required<=set(options) or set(options)-required-optional:
        raise ValueError('complete explicit public publisher loader options required')
    family=options['family']
    if family not in ('qwen4_exp','glm5_next','inkling_mm_model'):raise ValueError('implemented public publisher family required')
    root=Path(offline_root).resolve(strict=True);relative=Path(options['checkpoint'])
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('offline relative checkpoint directory required')
    checkpoint=(root/relative).resolve(strict=True)
    if not checkpoint.is_dir() or not checkpoint.is_relative_to(root):raise ValueError('checkpoint escapes offline admission')
    with (checkpoint/'config.json').open('rb') as source:raw=source.read(1024**2+1)
    if len(raw)>1024**2 or hashlib.sha256(raw).hexdigest()!=options['config_sha256']:
        raise ValueError('pinned publisher configuration differs')
    identities=[_library(root,options[name]) for name in ('media_library','text_library','runtime_library')]
    for name in ('max_weight_bytes','max_bytes','context_tokens','max_conversion_bytes','segment_work_bytes'):
        if type(options[name]) is not int or options[name]<=0:raise ValueError('explicit positive loader admissions required')
    extras={name:options[name] for name in optional if name in options}
    if family!='qwen4_exp' and any(extras.get(name,0) for name in ('native_expert_max_bytes','native_ple_max_bytes')):
        raise ValueError('Qwen4 native store admissions cannot be applied to another family')
    from ..publisher_conditional_loader import load_publisher_conditional_decoder
    decoder=None
    try:
        decoder=load_publisher_conditional_decoder(checkpoint,family=family,
            library=identities[0].path,text_library=identities[1].path,
            segment_runtime_library=identities[2].path,
            native_composition=family=='qwen4_exp',
            native_store_runtime_library=identities[2].path if family=='qwen4_exp' and
                (extras.get('native_expert_max_bytes',0) or extras.get('native_ple_max_bytes',0)) else None,
            **{name:options[name] for name in ('max_weight_bytes','max_bytes','context_tokens',
                'max_conversion_bytes','segment_work_bytes')},**extras)
        for identity in identities:identity.assert_unchanged()
        return decoder
    except BaseException as primary:
        if decoder is not None:
            try:decoder.close()
            except BaseException as cleanup:
                primary.cleanup=decoder.close;primary.add_note('public publisher factory cleanup retained: '+repr(cleanup))
        raise


__all__=['load_public_publisher']
