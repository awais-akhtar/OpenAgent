"""Local SafeTensors loading for explicitly selected native text decoders."""
from importlib import import_module
from contextlib import ExitStack
from dataclasses import dataclass
import json
import math
import os
import stat
from pathlib import Path
from threading import RLock

import numpy as np
from .session_recovery import _path

from .text_decoder_registry import _DECODERS, decoder_owner


@dataclass(frozen=True)
class CheckpointReadLimits:
    max_shards: int = 256
    max_tensors: int = 200000
    max_metadata_bytes: int = 256*1024*1024
    max_dimensions: int = 32
    chunk_bytes: int = 1024*1024

    def __post_init__(self):
        for name in ('max_shards', 'max_tensors', 'max_metadata_bytes',
                     'max_dimensions', 'chunk_bytes'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError('positive checkpoint read limit required: '+name)
        if self.chunk_bytes < 8 or self.chunk_bytes > 64*1024*1024:
            raise ValueError('checkpoint read chunks must be between 8 bytes and 64 MiB')


class CheckpointLoadCancelled(RuntimeError):
    pass


_PENDING_DECODERS = {}
_CLEANUP_LOCK = RLock()


def retry_checkpoint_cleanup():
    """Retry releasing native decoders retained after cancelled loading."""
    with _CLEANUP_LOCK:
        pending = tuple(_PENDING_DECODERS.items())
        failures = []
        for key, decoder in pending:
            try:
                decoder.close()
            except BaseException as error:
                failures.append(str(error))
            else:
                _PENDING_DECODERS.pop(key, None)
        return {'pending': len(_PENDING_DECODERS), 'errors': failures}


def _cancelled(cancel):
    if cancel is None:
        return
    if callable(cancel):
        active = bool(cancel())
    elif callable(getattr(cancel, 'is_set', None)):
        active = bool(cancel.is_set())
    else:
        raise TypeError('checkpoint cancellation must be callable or event-like')
    if active:
        raise CheckpointLoadCancelled('checkpoint loading cancelled')


def _finish_decoder(decoder, cancel):
    try:
        _cancelled(cancel)
    except BaseException as error:
        try:
            decoder.close()
        except BaseException as cleanup_error:
            with _CLEANUP_LOCK:
                _PENDING_DECODERS[id(decoder)] = decoder
            error.add_note(f'decoder cleanup failed: {cleanup_error}')
        raise
    return decoder


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON field: {key}')
        result[key] = value
    return result


def _stamp(source):
    value = os.fstat(source.fileno())
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _open_local(root, path, stack):
    resolved = _path(path).resolve(strict=True)
    if resolved.parent != root:
        raise ValueError('file escapes checkpoint directory')
    before = resolved.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError('regular checkpoint file required')
    descriptor=os.open(resolved,os.O_RDONLY|getattr(os,'O_BINARY',0)|
        getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0))
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError('regular opened checkpoint file required')
        source=os.fdopen(descriptor,'rb')
    except BaseException:
        os.close(descriptor)
        raise
    source=stack.enter_context(source)
    identity = _stamp(source)
    if identity != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns):
        raise ValueError('checkpoint file changed during opening')
    return source, identity


def _json(source, identity):
    if identity[2] > 16 * 1024 * 1024:
        raise ValueError('checkpoint JSON exceeds metadata budget')
    return json.loads(source.read(identity[2]), object_pairs_hook=_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _integer_table(name, model_type):
    if model_type=='qwen4_exp' and name.startswith('model.language_model.'):
        name='model.'+name[len('model.language_model.'):]
        model_type='qwen4_exp_text'
    parts = name.split('.')
    if (len(parts) != 6 or parts[:2] != ['model', 'layers']
            or not parts[2].isascii() or not parts[2].isdigit()):
        return False
    if model_type == 'deepseek_v4':
        return parts[3:] == ['mlp', 'gate', 'tid2eid']
    if model_type == 'qwen4_exp_text':
        return (parts[3:5] == ['ple', 'ple_embedding'] and parts[5] in (
            'layer_multipliers', 'ngram_heads_vocab_sizes', 'ngram_heads_offsets'))
    return False


def _header(source, identity, *, integer_tables=False, model_type=None,
            max_tensors=200000, max_dimensions=32):
    prefix = source.read(8)
    if len(prefix) != 8:
        raise ValueError('missing shard header length')
    length = int.from_bytes(prefix, 'little')
    if not 1 < length <= min(16 * 1024 * 1024, identity[2] - 8):
        raise ValueError('invalid shard header length or metadata budget')
    raw_header = source.read(length)
    if len(raw_header) != length:
        raise OSError('checkpoint header truncated during read')
    document = json.loads(raw_header, object_pairs_hook=_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if type(document) is not dict:
        raise ValueError('shard header must be an object')
    if len(document) > max_tensors+1:
        raise MemoryError('shard tensor inventory exceeds admission')
    base = 8 + length
    ranges = []
    for name, tensor in document.items():
        if name == '__metadata__':
            if (type(tensor) is not dict or any(type(key) is not str or
                    type(value) is not str for key, value in tensor.items())):
                raise ValueError('SafeTensors metadata must contain string pairs')
            continue
        if type(name) is not str or not name:
            raise ValueError('tensor name must be a nonempty string')
        if type(tensor) is not dict:
            raise ValueError('invalid tensor metadata')
        shape, offsets = tensor.get('shape'), tensor.get('data_offsets')
        width = {'F32': 4, 'F16': 2, 'BF16': 2}.get(tensor.get('dtype'))
        integer_family = model_type or ('deepseek_v4' if integer_tables else None)
        if tensor.get('dtype') == 'I64' and _integer_table(name, integer_family):
            width = 8
        if (width is None or type(shape) is not list
                or len(shape) > max_dimensions
                or any(type(n) is not int or n < 0 for n in shape)
                or type(offsets) is not list or len(offsets) != 2
                or any(type(n) is not int for n in offsets)):
            raise ValueError(f'unsupported weight metadata: {name}')
        start, end = offsets
        if not 0 <= start <= end <= identity[2] - base or end - start != math.prod(shape) * width:
            raise ValueError(f'invalid tensor byte range: {name}')
        ranges.append((start, end))
    position = 0
    for start, end in sorted(ranges):
        if start != position:
            raise ValueError('overlapping or noncontiguous shard payloads')
        position = end
    if position != identity[2] - base:
        raise ValueError('unaccounted shard payload bytes')
    return document, base


def _read_tensor(source, tensor, base, *, chunk_bytes=1024*1024, cancel=None):
    start, end = tensor['data_offsets']
    source.seek(base + start)
    if type(chunk_bytes) is not int or chunk_bytes < 8:
        raise ValueError('tensor conversion needs a chunk of at least 8 bytes')
    _cancelled(cancel)
    encoding = tensor['dtype']
    width = {'I64': 8, 'F32': 4, 'F16': 2, 'BF16': 2}[encoding]
    count = math.prod(tensor['shape'])
    value = np.empty(count, dtype=np.int64 if encoding == 'I64' else np.float32)
    chunk_elements = max(1, chunk_bytes//width)
    for offset in range(0, count, chunk_elements):
        _cancelled(cancel)
        size = min(chunk_elements, count-offset)
        raw = source.read(size*width)
        if len(raw) != size*width:
            raise OSError('checkpoint payload truncated during read')
        if encoding == 'BF16':
            converted = (np.frombuffer(raw, dtype='<u2').astype(np.uint32)
                         << np.uint32(16)).view('<f4')
        else:
            converted = np.frombuffer(raw, dtype={'I64':'<i8', 'F32':'<f4', 'F16':'<f2'}[encoding])
        value[offset:offset+size] = converted
        if encoding != 'I64' and not np.isfinite(value[offset:offset+size]).all():
            raise ValueError('nonfinite or FP32-overflow checkpoint tensor')
    _cancelled(cancel)
    return value.reshape(tensor['shape'])


def read_native_text_checkpoint(directory, *, model_type, max_weight_bytes,
                                read_limits=None,cancel=None,tensor_selection=None,context_tokens=None):
    """Read an exact registered text checkpoint, without wrapper reinterpretation."""
    decoder_owner(model_type)
    admission=None
    if tensor_selection is None:
        from .resident_family_inventory import FAMILIES,admit_resident_inventory
        if model_type in FAMILIES:
            def admission(config,inventory,resident):
                admit_resident_inventory(config,inventory,family=model_type,max_bytes=max_weight_bytes,context_tokens=context_tokens)
    return _read_native_checkpoint(directory,model_type=model_type,max_weight_bytes=max_weight_bytes,
                                   read_limits=read_limits, cancel=cancel,
                                   tensor_selection=tensor_selection,metadata_admission=admission)


def read_native_multimodal_checkpoint(directory, *, model_type, max_weight_bytes,
                                      read_limits=None, cancel=None):
    """Read the explicit Qwen3.5 wrapper; constructor validates vision geometry."""
    if model_type not in ('qwen3_5_moe','qwen4_exp','inkling_mm_model','glm5_next') or type(model_type) is not str:
        raise ValueError(f'unsupported exact native multimodal identity: {model_type!r}')
    return _read_native_checkpoint(directory, model_type=model_type, max_weight_bytes=max_weight_bytes,
                                   read_limits=read_limits, cancel=cancel)


def _read_native_checkpoint(directory, *, model_type, max_weight_bytes, runtime_budget=None,
                            metadata_admission=None, read_limits=None, cancel=None,
                            tensor_selection=None):
    """Admit resident FP32 weights before reading payloads; never unwrap models.

    The weight budget covers loaded arrays plus one decoder-owned copy and the
    largest source conversion. KV, inference scratch and Python metadata have
    separate limits. This is not a total process-RSS bound.
    """
    if type(max_weight_bytes) is not int or max_weight_bytes <= 0:
        raise ValueError('max_weight_bytes must be a positive integer')
    limits = CheckpointReadLimits() if read_limits is None else read_limits
    if type(limits) is not CheckpointReadLimits:
        raise TypeError('CheckpointReadLimits required')
    if cancel is not None and not callable(getattr(cancel, 'is_set', None)):
        raise ValueError('cancellation must provide is_set')
    _cancelled(cancel)
    root = _path(directory).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('checkpoint must be a local directory')
    with ExitStack() as stack:
        handles = []
        source, identity = _open_local(root, root / 'config.json', stack)
        handles.append((source, identity))
        config = _json(source, identity)
        metadata_bytes = identity[2]
        if metadata_bytes > limits.max_metadata_bytes:
            raise MemoryError('checkpoint metadata exceeds admission')
        if type(config) is not dict or config.get('model_type') != model_type:
            raise ValueError('checkpoint model_type differs from selected exact identity')
        if config.get('quantization_config') is not None:
            raise ValueError('quantized checkpoint is not a resident FP32 source')
        if runtime_budget is not None:
            from .native_qwen35_vision import NativeQwen35Vision
            NativeQwen35Vision.validate_config(config, max_bytes=runtime_budget)
        index = {}
        from itertools import islice
        shards = tuple(sorted(islice(root.glob('*.safetensors'), limits.max_shards+1)))
        if not shards:
            raise ValueError('checkpoint has no SafeTensors shards')
        if len(shards) > limits.max_shards:
            raise MemoryError('checkpoint shard inventory exceeds admission')
        for shard in shards:
            _cancelled(cancel)
            source, identity = _open_local(root, shard, stack)
            handles.append((source, identity))
            prefix = source.read(8)
            if len(prefix) != 8:
                raise ValueError('missing shard header length')
            header_bytes = int.from_bytes(prefix, 'little')
            source.seek(0)
            if header_bytes+8 > limits.max_metadata_bytes-metadata_bytes:
                raise MemoryError('aggregate checkpoint metadata exceeds admission')
            header, base = _header(source, identity, model_type=model_type,
                max_tensors=limits.max_tensors-len(index), max_dimensions=limits.max_dimensions)
            metadata_bytes += base
            for name, tensor in header.items():
                if name == '__metadata__':
                    continue
                if name in index:
                    raise ValueError(f'tensor appears in more than one shard: {name}')
                index[name] = (shard.name, source, tensor, base)
                if len(index) > limits.max_tensors:
                    raise MemoryError('checkpoint tensor inventory exceeds admission')
        index_path = root / 'model.safetensors.index.json'
        if index_path.exists():
            source, identity = _open_local(root, index_path, stack)
            handles.append((source, identity))
            if identity[2] > limits.max_metadata_bytes-metadata_bytes:
                raise MemoryError('aggregate checkpoint metadata exceeds admission')
            document = _json(source, identity)
            metadata_bytes += identity[2]
            mapping = document.get('weight_map') if type(document) is dict else None
            if type(mapping) is not dict or set(mapping) != set(index):
                raise ValueError('shard index does not match tensor inventory')
            for name, filename in mapping.items():
                if type(filename) is not str or filename != index[name][0]:
                    raise ValueError('shard index contains mismatched or nonlocal paths')
        selected = index
        if tensor_selection is not None:
            if not callable(tensor_selection):
                raise ValueError('explicit tensor selection callback required')
            names = tensor_selection(config, {name: item[2] for name, item in index.items()})
            if (type(names) not in (tuple, list) or not names or
                    any(type(name) is not str for name in names) or
                    len(set(names)) != len(names) or any(name not in index for name in names)):
                raise ValueError('selection must return unique existing checkpoint tensor names')
            selected = {name: index[name] for name in names}
        resident = largest = 0
        for _, _, tensor, _ in selected.values():
            _cancelled(cancel)
            count = math.prod(tensor['shape'])
            start, end = tensor['data_offsets']
            width = 8 if tensor['dtype'] == 'I64' else 4
            resident += count * width
            largest = max(largest, end - start + count * width)
        required = resident * 2 + largest
        if required > max_weight_bytes:
            raise MemoryError(f'resident checkpoint loading needs {required} weight bytes')
        if runtime_budget is not None and resident * 6 + 1024 * 1024 > runtime_budget:
            raise ValueError('wrapper memory budget exceeded')
        if metadata_admission is not None:
            metadata_admission(config, {name: item[2] for name, item in index.items()}, resident)
        weights = {}
        for name, (_, source, tensor, base) in selected.items():
            _cancelled(cancel)
            value = _read_tensor(source, tensor, base, chunk_bytes=limits.chunk_bytes, cancel=cancel)
            flat = value.reshape(-1)
            for offset in range(0, flat.size, 65536):
                _cancelled(cancel)
                if not np.isfinite(flat[offset:offset + 65536]).all():
                    raise ValueError(f'nonfinite checkpoint tensor: {name}')
            weights[name] = value
        if any(_stamp(source) != identity for source, identity in handles):
            raise ValueError('checkpoint file changed during loading')
        _cancelled(cancel)
        return config, weights


def load_native_text_decoder(directory, *, model_type, library,
                             max_weight_bytes, max_bytes=512 * 1024 * 1024,
                             context_tokens=None, read_limits=None, cancel=None,
                             expert_provider=None, max_snapshots=None,
                             retain_diagnostics=None):
    """Load an exact native text family without downloads or model fallback."""
    cleanup = retry_checkpoint_cleanup()
    if cleanup['pending']:
        raise RuntimeError('previous cancelled decoder cleanup is still pending: '
                           + '; '.join(cleanup['errors']))
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError('max_bytes must be a positive integer')
    decoder_owner(model_type)
    options = {}
    if max_snapshots is not None:
        if model_type not in ('glm_moe_dsa','qwen3_5_moe_text'):
            raise ValueError('explicit snapshot admission requires the DSA or Qwen3.5 owner')
        if model_type=='qwen3_5_moe_text' and (type(max_snapshots) is not int or not 1<=max_snapshots<=64):
            raise ValueError('bounded Qwen3.5 retained snapshot inventory required')
        options['max_snapshots']=max_snapshots
    if retain_diagnostics is not None:
        if model_type!='glm_moe_dsa':raise ValueError('native diagnostics policy requires the DSA owner')
        options['retain_diagnostics']=retain_diagnostics
    context_options = {'glm5_next_text': 'max_tokens', 'deepseek_v4': 'max_context',
                       'kimi_linear':'capacity','mixtral':'max_context','qwen3_5_moe_text':'context_tokens'}
    if context_tokens is not None:
        if type(context_tokens) is not int or context_tokens <= 0:
            raise ValueError('context_tokens must be a positive integer')
        if model_type not in context_options:
            raise ValueError('this family takes context capacity from its exact configuration')
        options[context_options[model_type]] = context_tokens
    elif model_type == 'kimi_linear':
        raise ValueError('Kimi Linear requires explicit context_tokens')
    library = Path(library).resolve(strict=True)
    if not library.is_file():
        raise ValueError('library must be an explicit local file')
    if expert_provider is not None:
        if model_type == 'glm_moe_dsa':
            from .native_glm_dsa import NativeGlmDsaDecoder
            return NativeGlmDsaDecoder.from_streamed_directory(directory,library=library,
                expert_provider=expert_provider,max_weight_bytes=max_weight_bytes,
                max_bytes=max_bytes,read_limits=read_limits,cancel=cancel,**options)
        if model_type != 'mixtral':
            raise ValueError('this native checkpoint family has no admitted streamed provider loader')
        from .native_mixtral_decoder import NativeMixtralDecoder
        return NativeMixtralDecoder.from_streamed_directory(directory,
            library=library, expert_provider=expert_provider,
            max_weight_bytes=max_weight_bytes, max_bytes=max_bytes,
            max_context=context_tokens, read_limits=read_limits, cancel=cancel)
    config, weights = read_native_text_checkpoint(
        directory, model_type=model_type, max_weight_bytes=max_weight_bytes,
        read_limits=read_limits,cancel=cancel,context_tokens=context_tokens)
    _cancelled(cancel)
    module, name = _DECODERS[model_type]
    factory = getattr(import_module(f'{__package__}.{module}'), name)
    decoder = factory(config, weights, library=library, max_bytes=max_bytes, **options)
    return _finish_decoder(decoder, cancel)


def load_native_multimodal_decoder(directory, *, model_type, library, text_library,
                                   max_weight_bytes, max_bytes=256 * 1024 * 1024,
                                   read_limits=None, cancel=None,arithmetic='fp32',context_tokens=None,
                                   max_snapshots=4):
    """Load the bounded direct Qwen3.5 wrapper, never a text-only fallback.

    Qwen4 and Inkling multimodal checkpoints are intentionally admitted only
    through ``publisher_conditional_loader``. Their component inventories need
    family-specific config and header planning before payload materialization;
    this simpler resident reader does not provide that preflight.
    """
    cleanup = retry_checkpoint_cleanup()
    if cleanup['pending']:
        raise RuntimeError('previous cancelled decoder cleanup is still pending: '
                           + '; '.join(cleanup['errors']))
    if type(model_type) is not str or model_type not in ('qwen3_5_moe','qwen4_exp','inkling_mm_model'):
        raise ValueError(f'unsupported exact native multimodal identity: {model_type!r}')
    if model_type in ('qwen4_exp', 'inkling_mm_model'):
        raise ValueError(
            f'direct native multimodal loading is not admitted for {model_type}; '
            'use the publisher conditional loader with its family inventory planner')
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError('max_bytes must be a positive integer')
    libraries = []
    for path in (library, text_library):
        resolved = Path(path).resolve(strict=True)
        if not resolved.is_file():
            raise ValueError('libraries must be explicit local files')
        libraries.append(resolved)
    metadata_admission = None
    if model_type == 'qwen3_5_moe':
        # The direct native Qwen3.5 wrapper is the explicitly bounded tiny
        # vision vertical slice. Validate its complete vision namespace from
        # SafeTensors headers before _read_native_checkpoint materializes any
        # payload. The general publisher conditional loader has its own full
        # producer/consumer planner and is intentionally unaffected.
        def metadata_admission(config, inventory, resident):
            from .native_qwen35_vision import NativeQwen35Vision
            NativeQwen35Vision.validate_config(config, max_bytes=max_bytes)
            vision = config['vision_config']
            expected = dict(NativeQwen35Vision._shapes)
            block_shapes = {
                'norm1.weight': (8,), 'norm1.bias': (8,),
                'norm2.weight': (8,), 'norm2.bias': (8,),
                'attn.qkv.weight': (24, 8), 'attn.qkv.bias': (24,),
                'attn.proj.weight': (8, 8), 'attn.proj.bias': (8,),
                'mlp.linear_fc1.weight': (16, 8), 'mlp.linear_fc1.bias': (16,),
                'mlp.linear_fc2.weight': (8, 16), 'mlp.linear_fc2.bias': (8,),
            }
            for layer in range(vision['depth']):
                expected.update({f'blocks.{layer}.{name}': shape
                                 for name, shape in block_shapes.items()})
            prefix = 'model.visual.'
            actual = {name[len(prefix):]: tuple(tensor['shape'])
                      for name, tensor in inventory.items() if name.startswith(prefix)}
            if actual != expected:
                raise ValueError('Qwen3.5 native wrapper vision header inventory differs')
    config, weights = _read_native_checkpoint(
        directory, model_type=model_type, max_weight_bytes=max_weight_bytes,
        runtime_budget=max_bytes if model_type=='qwen3_5_moe' else None,
        metadata_admission=metadata_admission, read_limits=read_limits, cancel=cancel)
    _cancelled(cancel)
    if model_type=='inkling_mm_model':
        from .inkling_multimodal import InklingMultimodalDecoder
        decoder=InklingMultimodalDecoder(config,weights,library=libraries[0],text_library=libraries[1],
            max_bytes=max_bytes,arithmetic=arithmetic,context_tokens=context_tokens,max_snapshots=max_snapshots)
    elif model_type=='qwen4_exp':
        from .qwen4_multimodal import Qwen4MultimodalDecoder
        decoder=Qwen4MultimodalDecoder(config,weights,library=libraries[0],text_library=libraries[1],
                                       max_bytes=max_bytes,arithmetic=arithmetic,max_snapshots=max_snapshots)
    else:
        from .native_qwen35_vision import NativeQwen35Vision
        decoder = NativeQwen35Vision(config, weights, library=libraries[0],
                                    text_library=libraries[1], max_bytes=max_bytes)
    return _finish_decoder(decoder, cancel)
