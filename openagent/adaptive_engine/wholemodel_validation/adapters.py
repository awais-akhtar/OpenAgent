"""Concrete adapters for unchanged Torch causal models and native executors.

Factories supply a fresh already-loaded model/executor per session. No source
patches, compilation, model equations, network calls or substitute kernels.
"""
from contextlib import contextmanager
import copy
import importlib
import random
import threading
import weakref

import numpy as np

from .cache_readers import CacheContext, StandardKVReader, detached, export_cache
from .contracts import Observation


_RNG_LOCK = threading.RLock()


def resolve_callable(value):
    if callable(value):
        return value
    if not isinstance(value, str) or ':' not in value:
        raise ValueError('expected callable or importable module:callable')
    module, name = value.split(':', 1)
    result = getattr(importlib.import_module(module), name)
    if not callable(result):
        raise TypeError('resolved object is not callable')
    return result


def checked_metadata(value):
    data = copy.deepcopy(dict(value))
    schema = data.get('cache_schema')
    layers = data.get('num_layers')
    if type(layers) is not int or layers < 1 or not isinstance(schema, dict) or not schema:
        raise ValueError('num_layers and explicit complete cache_schema required')
    if any(not isinstance(k, str) or not k for k in schema):
        raise ValueError('cache leaf names must be nonempty strings')
    if any(v != 'global' and (type(v) is not int or not 0 <= v < layers) for v in schema.values()):
        raise ValueError('invalid layer mapping')
    if not set(range(layers)) <= set(schema.values()):
        raise ValueError('cache_schema must explicitly cover every layer')
    return data


def reader_from_options(options):
    if 'cache_reader_factory' in options:
        if 'cache_reader' in options:
            raise ValueError('choose cache_reader OR cache_reader_factory')
        return resolve_callable(options['cache_reader_factory'])(
            schema=copy.deepcopy(options['metadata']['cache_schema']),
            options=options.get('cache_reader_options', {}))
    return options.get('cache_reader')


def forward_from_options(options):
    if 'forward_call_factory' in options:
        if 'forward_call' in options:raise ValueError('choose forward_call OR forward_call_factory')
        return resolve_callable(options['forward_call_factory'])(options=options.get('forward_call_options',{}))
    return options.get('forward_call')


class RandomState:
    """Keep session RNG streams independent and restore caller global state."""

    def __init__(self, seed, torch=None):
        self.seed = seed
        self.torch = torch
        self.states = None

    @contextmanager
    def activate(self):
        with _RNG_LOCK:
            torch = self.torch
            py_before, np_before = random.getstate(), np.random.get_state()
            cpu_before = torch.random.get_rng_state() if torch is not None else None
            cuda = torch is not None and torch.cuda.is_available()
            cuda_before = torch.cuda.get_rng_state_all() if cuda else None
            deterministic_before = torch.are_deterministic_algorithms_enabled() if torch is not None else None
            warn_before = torch.is_deterministic_algorithms_warn_only_enabled() if torch is not None else None
            backend_before = ((torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                               torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
                              if torch is not None else None)
            precision_before = ((torch.get_float32_matmul_precision(),
                                 torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
                                if torch is not None else None)
            try:
                if torch is not None:
                    torch.use_deterministic_algorithms(True, warn_only=False)
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.allow_tf32 = False
                    torch.backends.cuda.matmul.allow_tf32 = False
                    torch.set_float32_matmul_precision('highest')
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
                if self.states is None:
                    random.seed(self.seed)
                    np.random.seed(self.seed)
                    if torch is not None:
                        torch.manual_seed(self.seed)
                        if cuda:
                            torch.cuda.manual_seed_all(self.seed)
                else:
                    py_state, np_state, cpu_state, cuda_state = self.states
                    random.setstate(py_state)
                    np.random.set_state(np_state)
                    if torch is not None:
                        torch.random.set_rng_state(cpu_state)
                        if cuda:
                            torch.cuda.set_rng_state_all(cuda_state)
                yield
            finally:
                self.states = (random.getstate(), np.random.get_state(),
                               torch.random.get_rng_state() if torch is not None else None,
                               torch.cuda.get_rng_state_all() if cuda else None)
                random.setstate(py_before)
                np.random.set_state(np_before)
                if torch is not None:
                    torch.random.set_rng_state(cpu_before)
                    if cuda:
                        torch.cuda.set_rng_state_all(cuda_before)
                    torch.use_deterministic_algorithms(deterministic_before, warn_only=warn_before)
                    torch.set_float32_matmul_precision(precision_before[0])
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = precision_before[1]
                    (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                     torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32) = backend_before


class FreshOwner:
    def __init__(self):
        self._owners = []

    def claim(self, owner):
        if any(reference() is owner for reference in self._owners):
            raise ValueError('factory reused a model/executor: sessions require independent owners')
        try:
            self._owners.append(weakref.ref(owner))
        except TypeError:
            raise TypeError('model/executor must support weak references') from None
        self._owners = [reference for reference in self._owners if reference() is not None]


class TorchCausalAdapter(FreshOwner):
    """Real Torch inference using an injected factory(seed=...) -> loaded model.

    Supply cache_reader for recurrent/MLA/hybrid or nonstandard caches. The
    standard reader requires explicit kv_only=True and exact standard schema.
    Optional forward_call(model, kwargs) handles family-specific call signatures
    without changing the implementation. It must return logits/past_key_values.
    """

    def __init__(self, model_factory, metadata, *, cache_reader=None, kv_only=False,
                 input_device=None, position_ids=True, cache_position=False,
                 forward_call=None):
        super().__init__()
        import torch
        self.torch = torch
        self._metadata = checked_metadata(metadata)
        self._metadata['dependencies'] = dict(self._metadata.get('dependencies', {}),
                                             torch=torch.__version__, torch_cuda=torch.version.cuda)
        self.model_factory = resolve_callable(model_factory)
        self.reader = (resolve_callable(cache_reader) if cache_reader is not None else
                       StandardKVReader(self._metadata['num_layers'], kv_only=kv_only,
                                        schema=self._metadata['cache_schema']))
        if getattr(self.reader, 'requires_raw_evidence', False):
            if (self._metadata.get('evidence_schema') != self.reader.evidence_schema
                    or self._metadata.get('empty_evidence_shapes', {}) != self.reader.empty_evidence_shapes):
                raise ValueError('mapped Torch cache requires complete raw evidence metadata')
        self.input_device = input_device
        self.position_ids = position_ids
        self.cache_position = cache_position
        self.forward_call = resolve_callable(forward_call) if forward_call is not None else None
        self._runtime = []

    def metadata(self):
        return copy.deepcopy(self._metadata)

    def runtime_metadata(self):
        return copy.deepcopy(self._runtime)

    def new_session(self, *, seed):
        rng = RandomState(seed, self.torch)
        with rng.activate():
            model = self.model_factory(seed=seed)
            self.claim(model)
            model.eval()
            config = model.config
            text = getattr(config, 'text_config', None)
            if text is None:
                text = config
            if getattr(config, 'model_type', None) != self._metadata['model_type']:
                raise ValueError('loaded Torch outer model_type differs from exact model identity')
            if 'text_model_type' in self._metadata:
                expected_text = self._metadata['text_model_type']
                if not isinstance(expected_text, str) or not expected_text:
                    raise ValueError('text_model_type must be an explicit nonempty identity')
                if getattr(text, 'model_type', None) != expected_text:
                    raise ValueError('loaded Torch inner text_model_type differs from declared identity')
            for key, expected in (('vocab_size', self._metadata['vocabulary_size']),
                                  ('num_hidden_layers', self._metadata['num_layers'])):
                if getattr(text, key, None) != expected:
                    raise ValueError('loaded Torch model geometry/identity differs: ' + key)
            parameters = list(model.parameters())
            self._runtime.append(dict(seed=seed, model_class=type(model).__module__ + '.' + type(model).__qualname__,
                parameter_dtypes=sorted({str(p.dtype) for p in parameters}),
                parameter_devices=sorted({str(p.device) for p in parameters}),
                model_type=getattr(config, 'model_type', None),
                text_model_type=getattr(text, 'model_type', None),
                training=bool(model.training), deterministic_algorithms=True))
        return TorchSession(self, model, rng)


class TorchSession:
    def __init__(self, adapter, model, rng):
        self.adapter, self.model, self.rng = adapter, model, rng
        self.cache = None
        self.position = 0
        self.closed = False
        self.failed = False

    def forward(self, token_ids):
        if self.closed or self.failed:
            raise RuntimeError('closed or failed Torch session')
        tokens = tuple(token_ids)
        if not tokens or any(type(t) is not int or not 0 <= t < self.adapter._metadata['vocabulary_size'] for t in tokens):
            raise ValueError('nonempty in-vocabulary integer tokens required')
        adapter, torch = self.adapter, self.adapter.torch
        end = self.position + len(tokens)
        try:
            with self.rng.activate(), torch.inference_mode():
                device = adapter.input_device
                if device is None:
                    device = self.model.get_input_embeddings().weight.device
                if str(device) == 'meta':
                    raise ValueError('meta input device requires explicit materialized input_device')
                ids = torch.tensor([tokens], dtype=torch.long, device=device)
                kwargs = dict(input_ids=ids, past_key_values=self.cache,
                              attention_mask=torch.ones((1, end), dtype=torch.long, device=device),
                              use_cache=True, return_dict=True)
                positions = torch.arange(self.position, end, dtype=torch.long, device=device)
                if adapter.position_ids:
                    kwargs['position_ids'] = positions.unsqueeze(0)
                if adapter.cache_position:
                    kwargs['cache_position'] = positions
                output = (adapter.forward_call(self.model, kwargs) if adapter.forward_call else
                          self.model(**kwargs))
                logits = output['logits'] if isinstance(output, dict) else output.logits
                cache = output['past_key_values'] if isinstance(output, dict) else output.past_key_values
                if cache is None:
                    raise ValueError('model did not return continuation cache')
                logits = detached(logits)
                if logits.shape != (1, len(tokens), adapter._metadata['vocabulary_size']):
                    raise ValueError('model must return every input token logit, not last-token only')
                context = CacheContext(cache, self.model, output, end, len(tokens))
                leaves = export_cache(adapter.reader, context,
                    adapter._metadata['cache_schema'],
                    empty_shapes=adapter._metadata.get('empty_cache_shapes'))
                evidence = {}
                if adapter._metadata.get('evidence_schema'):
                    evidence_reader = getattr(adapter.reader, 'evidence', None)
                    if not callable(evidence_reader):
                        raise ValueError('declared Torch evidence requires an explicit evidence reader')
                    evidence = export_cache(evidence_reader, context, adapter._metadata['evidence_schema'],
                        empty_shapes=adapter._metadata.get('empty_evidence_shapes'))
            self.cache, self.position = cache, end
            return Observation(logits[0], leaves, evidence)
        except Exception:
            self.failed = True
            raise

    def close(self):
        self.closed = True
        self.cache = None
        self.model = None


class NativeExecutorAdapter(FreshOwner):
    """Fresh executor factory(seed=...) with mandatory complete cache reader.

    Default executor.forward(np.int64 tokens) must return [tokens, vocabulary].
    Optional forward_call(executor, tokens) and cache_getter(executor) map exact
    public APIs. Default cache_getter is executor.snapshot(), never inferred
    private fields. A reader can observe executor directly via context.model.
    """

    def __init__(self, executor_factory, metadata, *, cache_reader,
                 forward_call=None, cache_getter=None, close_call=None, torch=None):
        super().__init__()
        self._metadata = checked_metadata(metadata)
        self.factory = resolve_callable(executor_factory)
        self.reader = resolve_callable(cache_reader)
        if getattr(self.reader, 'requires_raw_evidence', False):
            if (self._metadata.get('evidence_schema') != self.reader.evidence_schema
                    or self._metadata.get('empty_evidence_shapes', {}) != self.reader.empty_evidence_shapes
                    or self._metadata.get('cache_projection', {}).get('certified') is not False):
                raise ValueError('projected reader requires its complete raw evidence schema and uncertified policy')
        self.forward_call = resolve_callable(forward_call) if forward_call else None
        self.cache_getter = resolve_callable(cache_getter) if cache_getter else None
        self.close_call = resolve_callable(close_call) if close_call else None
        self._runtime = []
        self.torch = torch

    def metadata(self):
        return copy.deepcopy(self._metadata)

    def runtime_metadata(self):
        return copy.deepcopy(self._runtime)

    def new_session(self, *, seed):
        rng = RandomState(seed, self.torch)
        with rng.activate():
            executor = self.factory(seed=seed)
            self.claim(executor)
            self._runtime.append(dict(seed=seed, executor_class=type(executor).__module__ + '.' + type(executor).__qualname__))
            execution_identity=getattr(executor,'_identity',getattr(executor,'identity',None))
            arithmetic=getattr(executor,'_arithmetic',None)
            if type(execution_identity) is str:
                self._runtime[-1]['execution_identity']=execution_identity
                self._runtime[-1]['arithmetic_identity']=getattr(arithmetic,'identity',None)
            if self.torch is not None:
                self._runtime[-1].update(torch_version=str(self.torch.__version__),
                    deterministic_algorithms=self.torch.are_deterministic_algorithms_enabled(),
                    float32_matmul_precision=self.torch.get_float32_matmul_precision(),
                    bf16_reduced_precision_reduction=self.torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        return NativeSession(self, executor, rng)


class NativeSession:
    def __init__(self, adapter, executor, rng):
        self.adapter, self.executor, self.rng = adapter, executor, rng
        self.position = 0
        self.closed = self.failed = False
        self._snapshots = {}

    def snapshot(self):
        if self.closed or self.failed:
            raise RuntimeError('unavailable native observation session')
        if len(self._snapshots)>=2:
            raise MemoryError('comparison snapshot inventory exhausted')
        capture, _, restore_name = _native_state_protocol(self.executor)
        rng=copy.deepcopy(self.rng.states)
        receipt=capture()
        handle=object()
        self._snapshots[handle]=(receipt,self.position,rng,restore_name)
        return handle

    def restore(self, handle):
        if self.closed or handle not in self._snapshots:
            raise ValueError('owned comparison snapshot required')
        receipt,position,rng,restore_name=self._snapshots[handle]
        restored_rng=copy.deepcopy(rng)
        getattr(self.executor,restore_name)(receipt)
        self.position,self.rng.states,self.failed=position,restored_rng,False

    def release_snapshot(self, handle):
        if self.closed or handle not in self._snapshots:
            raise ValueError('owned comparison snapshot required')
        receipt,_,_,_=self._snapshots[handle]
        release=getattr(self.executor,'release_snapshot',None)
        if callable(release):release(receipt)
        del self._snapshots[handle]

    def forward(self, token_ids):
        if self.closed or self.failed:
            raise RuntimeError('closed or failed native session')
        tokens = tuple(token_ids)
        if not tokens or any(type(t) is not int or not 0 <= t < self.adapter._metadata['vocabulary_size'] for t in tokens):
            raise ValueError('nonempty in-vocabulary integer tokens required')
        a, executor = self.adapter, self.executor
        try:
            with self.rng.activate():
                ids = np.asarray(tokens, dtype=np.int64)
                output = a.forward_call(executor, ids) if a.forward_call else executor.forward(ids)
                logits = detached(output)
                if logits.shape != (len(tokens), a._metadata['vocabulary_size']):
                    raise ValueError('native executor must return logits for every appended token')
                snapshot = (a.cache_getter(executor) if a.cache_getter else
                            _native_state_protocol(executor)[0]())
                end = self.position + len(tokens)
                context = CacheContext(snapshot, executor, output, end, len(tokens))
                leaves = export_cache(a.reader, context,
                                      a._metadata['cache_schema'],
                                      empty_shapes=a._metadata.get('empty_cache_shapes'))
                evidence_schema = a._metadata.get('evidence_schema', {})
                evidence = {}
                if evidence_schema:
                    evidence_reader = getattr(a.reader, 'evidence', None)
                    if not callable(evidence_reader):
                        raise ValueError('declared native evidence requires an explicit evidence reader')
                    evidence = export_cache(evidence_reader, context, evidence_schema,
                                            empty_shapes=a._metadata.get('empty_evidence_shapes'))
            self.position = end
            return Observation(logits, leaves, evidence)
        except Exception:
            self.failed = True
            raise

    def close(self):
        if not self.closed:
            for handle in tuple(self._snapshots):
                self.release_snapshot(handle)
            try:
                if self.adapter.close_call:
                    self.adapter.close_call(self.executor)
                elif callable(getattr(self.executor, 'close', None)):
                    self.executor.close()
            except BaseException:
                raise
            else:
                self.closed = True
                self.executor = None


def _native_state_protocol(executor):
    """Resolve one matched native snapshot/restore API without guessing state."""
    for capture_name, restore_name in (('snapshot', 'restore'),
                                       ('snapshot_state', 'restore_state')):
        capture = getattr(executor, capture_name, None)
        restore = getattr(executor, restore_name, None)
        if callable(capture) and callable(restore):
            return capture, restore, restore_name
    raise ValueError('complete candidate snapshot/restore protocol required')


def create_torch_adapter(*, offline_root, options):
    """CLI factory for injected already-loaded-model factories and readers."""
    factory = resolve_callable(options['model_factory'])
    factory_options = dict(options.get('model_options', {}))
    def loaded(*, seed):
        return factory(seed=seed, offline_root=offline_root, options=factory_options)
    return TorchCausalAdapter(loaded, options['metadata'],
        cache_reader=reader_from_options(options), kv_only=options.get('kv_only', False),
        input_device=options.get('input_device'), position_ids=options.get('position_ids', True),
        cache_position=options.get('cache_position', False), forward_call=forward_from_options(options))


def create_native_adapter(*, offline_root, options):
    """CLI factory for existing native executors with explicit cache mapping."""
    factory = resolve_callable(options['executor_factory'])
    factory_options = dict(options.get('executor_options', {}))
    def loaded(*, seed):
        return factory(seed=seed, offline_root=offline_root, options=factory_options)
    reader = reader_from_options(options)
    if reader is None:
        raise ValueError('native adapter requires explicit cache reader or cache reader factory')
    return NativeExecutorAdapter(loaded, options['metadata'], cache_reader=reader,
        forward_call=options.get('forward_call'), cache_getter=options.get('cache_getter'),
        close_call=options.get('close_call'))
