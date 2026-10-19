"""Bind materialized family decoders to the checkpoint snapshot they consumed."""
from pathlib import Path

_SEAL=object()
_ATTRIBUTE='_openagent_family_source_binding'


def bind_decoder_to_snapshot(decoder,snapshot,*,family,model_type):
    """Attach a non-public loader receipt after its immutable snapshot is checked."""
    if (type(family) is not str or not family or type(model_type) is not str or
            not model_type or not callable(getattr(snapshot,'assert_unchanged',None))):
        raise ValueError('exact family checkpoint snapshot required')
    snapshot.assert_unchanged()
    binding=(_SEAL,str(Path(snapshot.root).resolve(strict=True)),family,model_type,
             snapshot.config_sha256,tuple(sorted(snapshot.files.items())))
    setattr(decoder,_ATTRIBUTE,binding)
    return decoder


def require_decoder_snapshot(decoder,snapshot,*,family,model_type):
    """Reject ABI composition unless decoder and public checkpoint are identical."""
    binding=getattr(decoder,_ATTRIBUTE,None)
    expected=(_SEAL,str(Path(snapshot.root).resolve(strict=True)),family,model_type,
              snapshot.config_sha256,tuple(sorted(snapshot.files.items())))
    snapshot.assert_unchanged()
    if type(binding) is not tuple or len(binding)!=6 or binding!=expected:
        raise ValueError(
            'public family ABI requires a decoder loaded from this exact unchanged '
            'family checkpoint snapshot')
    return True


__all__=['bind_decoder_to_snapshot','require_decoder_snapshot']
