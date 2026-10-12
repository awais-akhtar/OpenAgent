"""Local K3 publisher-checkpoint text operator; numerical parity is unverified."""
import argparse
import copy
import hashlib
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
from .session_recovery import _path
from .publisher_media_files import read_regular


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('positive integer required')
    return number


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('checkpoint', type=Path)
    for name in ('kimi', 'attention', 'operators', 'mxfp4'):
        result.add_argument('--' + name + '-library', type=Path)
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument('--prompt', help='Raw text, without a chat template or automatic special tokens')
    source.add_argument('--token-ids', help='JSON array of vocabulary token IDs')
    result.add_argument('--tokenizer', type=Path, help='Local tokenizer JSON; defaults to checkpoint/tokenizer.json')
    result.add_argument('--context', type=positive, required=True)
    result.add_argument('--max-tokens', type=positive, default=32)
    result.add_argument('--prefill-chunk', type=positive, default=32)
    result.add_argument('--temperature', type=float, default=0.0)
    result.add_argument('--seed', type=int, default=0)
    result.add_argument('--stop-token-id', type=int, action='append', default=[])
    result.add_argument('--stream', action='store_true', help='Emit newline-delimited JSON events')
    for name in ('resident', 'runtime', 'expert', 'mapped', 'chunk', 'sampling', 'metadata', 'input'):
        result.add_argument('--' + name + '-mib', type=positive, required=True)
    result.add_argument('--max-tensors', type=positive, required=True)
    result.add_argument('--max-shards', type=positive, required=True)
    result.add_argument('--projection-epsilon', type=float, default=1e-6)
    result.add_argument('--l2-epsilon', type=float, default=1e-6)
    result.add_argument('--include-vision', action='store_true')
    result.add_argument('--vision-tower-epsilon', type=float)
    result.add_argument('--image', type=Path, action='append', default=[])
    result.add_argument('--image-token-id', type=int)
    result.add_argument('--image-mib', type=positive)
    result.add_argument('--arithmetic-device',help='Explicit BF16 Torch device; omitted selects native FP32')
    result.add_argument('--torch-version',help='Exact installed Torch version for BF16 execution')
    result.add_argument('--arithmetic-mib',type=positive)
    result.add_argument('--routing-mib',type=positive)
    result.add_argument('--fla-version',help='Exact installed fla-core version for BF16 KDA layers')
    result.add_argument('--fla-mib',type=positive,help='Admission per KDA mixer')
    result.add_argument('--dense-projection-mib',type=positive,help='Stream original dense checkpoint rows with BF16 arithmetic')
    result.add_argument('--attention-mode',choices=('eager','paged-online-fp32'),
                        help='Explicit BF16 attention reduction policy')
    result.add_argument('--original-projection-library',type=Path,help='Prebuilt Segment runtime with original dense row storage')
    result.add_argument('--native-dense-projection-mib',type=positive)
    result.add_argument('--native-projection-work-mib',type=positive)
    result.add_argument('--require-publisher-exact',action='store_true')
    result.add_argument('--exact-evidence',type=Path)
    result.add_argument('--exact-evidence-sha256')
    result.add_argument('--exact-evidence-mib',type=positive,default=16)
    return result


def _local_file(path):
    resolved = _path(path).resolve(strict=True)
    if not resolved.is_file():
        raise ValueError('local file required: ' + str(resolved))
    return resolved


def _sample(logits, temperature, rng, maximum):
    observed = np.asarray(logits)
    if observed.ndim != 1 or not observed.size or observed.dtype.kind != 'f':
        raise ValueError('floating vocabulary logits required')
    # FP64 row, probabilities and sampling scratch are separate from executor memory.
    if observed.size * 32 > maximum:
        raise MemoryError('vocabulary sampling exceeds explicit budget')
    row = np.array(observed, dtype=np.float64, copy=True)
    if not np.isfinite(row).all():
        raise ValueError('finite vocabulary logits required')
    if temperature == 0:
        return int(row.argmax())
    row -= row.max()
    with np.errstate(over='ignore', under='ignore', invalid='raise', divide='raise'):
        row /= temperature
        probabilities = np.exp(row)
    total = probabilities.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError('invalid sampling distribution')
    probabilities /= total
    return int(rng.choice(row.size, p=probabilities))


def _check_cancelled(cancelled):
    from .execution_service import ExecutionCancelled
    if cancelled is not None and cancelled():
        raise ExecutionCancelled('native generation cancelled')


class PublisherSession:
    """Own one warm local model; generate independent, cache-reset requests.

    Construction loads once. Use as a context manager or call close(). Requests
    are serialized by rejection, including callbacks that reenter this session.
    Cancellation is cooperative between load phases and native forward calls.
    """
    _loading_options = (
        'checkpoint', 'tokenizer', 'kimi_library', 'attention_library',
        'operators_library', 'mxfp4_library', 'context', 'resident_mib',
        'runtime_mib', 'expert_mib', 'mapped_mib', 'chunk_mib', 'sampling_mib',
        'metadata_mib', 'input_mib', 'max_tensors', 'max_shards',
        'projection_epsilon', 'l2_epsilon',
        'include_vision', 'vision_tower_epsilon', 'image_token_id', 'image_mib',
        'arithmetic_device','torch_version','arithmetic_mib','routing_mib','fla_version','fla_mib',
        'dense_projection_mib',
        'attention_mode',
        'original_projection_library','native_dense_projection_mib','native_projection_work_mib',
        'require_publisher_exact','exact_evidence','exact_evidence_sha256','exact_evidence_mib',
    )

    def __init__(self, args, *, cancelled=None):
        self._args = copy.deepcopy(args)
        for name, default in (('include_vision', False), ('vision_tower_epsilon', None),
                              ('image_token_id', None), ('image_mib', None), ('image', []),
                              ('arithmetic_device',None),('torch_version',None),('arithmetic_mib',None),
                              ('routing_mib',None),('fla_version',None),('fla_mib',None),('dense_projection_mib',None),
                              ('attention_mode',None),('original_projection_library',None),
                              ('native_dense_projection_mib',None),('native_projection_work_mib',None),
                              ('require_publisher_exact',False),('exact_evidence',None),
                              ('exact_evidence_sha256',None),('exact_evidence_mib',16)):
            if not hasattr(self._args, name):
                setattr(self._args, name, default)
        self._cancelled = cancelled
        self._lock = threading.Lock()
        self._closed = False
        self._requests = 0
        from .owned_resources import OwnedResources
        self._owned = OwnedResources()
        self._model = self._tokenizer = self._checkpoint = None
        started = time.perf_counter()
        try:
            self._load(cancelled)
            self._load_ms = (time.perf_counter() - started) * 1000
        except BaseException as error:
            try:self._close_unlocked()
            except BaseException as failure:
                error.cleanup=self.close;error.add_note('K3 dependent cleanup remains pending: '+str(failure))
            raise

    def _load(self, cancelled):
        _check_cancelled(cancelled)
        from tokenizers import Tokenizer
        from .publisher_checkpoint.service import PublisherCheckpoint
        from .publisher_checkpoint.k3_adapter import K3ExecutionPayload
        from .k3_execution import K3Execution

        args = self._args
        evidence=None
        if type(args.require_publisher_exact) is not bool:
            raise ValueError('explicit exact admission boolean required')
        if args.exact_evidence is not None or args.exact_evidence_sha256 is not None:
            if not args.require_publisher_exact or args.exact_evidence is None or args.exact_evidence_sha256 is None:
                raise ValueError('exact mode requires an evidence artifact and its explicit SHA256 pin')
            if type(args.exact_evidence_mib) is not int or not 1<=args.exact_evidence_mib<=256:
                raise ValueError('exact evidence admission must be one to 256 MiB')
            from .k3_execution.exact_admission import K3ExactEvidence
            evidence=K3ExactEvidence(args.exact_evidence,expected_sha256=args.exact_evidence_sha256,
                max_bytes=args.exact_evidence_mib*1024**2)
        elif args.require_publisher_exact:
            raise ValueError('exact mode requires explicitly pinned observed comparison evidence')
        bf16=args.arithmetic_device is not None
        native_dense=args.original_projection_library is not None
        if native_dense:
            if bf16 or any(type(value) is not int or value<=0 for value in
                           (args.native_dense_projection_mib,args.native_projection_work_mib)):
                raise ValueError('native original projections require native arithmetic and positive owner/work budgets')
        elif args.native_dense_projection_mib is not None or args.native_projection_work_mib is not None:
            raise ValueError('native dense budgets require the original projection runtime library')
        if bf16:
            if (type(args.arithmetic_device) is not str or not args.arithmetic_device or
                    type(args.torch_version) is not str or not args.torch_version or
                    any(type(value) is not int or value<=0 for value in (args.arithmetic_mib,args.routing_mib))):
                raise ValueError('BF16 serving requires explicit device, Torch version and arithmetic/routing budgets')
            metal=args.arithmetic_device.startswith('mps')
            if metal and args.fla_version is not None:
                raise ValueError('Metal KDA uses the explicit Torch recurrence and does not accept an FLA pin')
            if not metal and (args.fla_version is None)!=(args.fla_mib is None):
                raise ValueError('FLA version and per-mixer admission must be supplied together')
            if args.fla_mib is not None and (type(args.fla_mib) is not int or args.fla_mib<=0):
                raise ValueError('positive per-mixer KDA admission required')
            if args.fla_version is not None and (type(args.fla_version) is not str or not args.fla_version or
                    type(args.fla_mib) is not int or args.fla_mib<=0):
                raise ValueError('explicit FLA dependency pin and positive mixer admission required')
            if args.include_vision and args.vision_tower_epsilon!=2**-7:
                raise ValueError('BF16 vision requires --vision-tower-epsilon 0.0078125')
            if args.dense_projection_mib is not None and (type(args.dense_projection_mib) is not int or args.dense_projection_mib<=0):
                raise ValueError('positive original dense projection admission required')
            if args.attention_mode not in (None,'eager','paged-online-fp32'):
                raise ValueError('explicit supported BF16 attention policy required')
        elif any(value is not None for value in (args.torch_version,args.arithmetic_mib,args.routing_mib,
                                                 args.fla_version,args.fla_mib,args.dense_projection_mib,args.attention_mode)):
            raise ValueError('Torch/FLA options require --arithmetic-device')
        if type(args.include_vision) is not bool:
            raise ValueError('include_vision must be boolean')
        if args.include_vision:
            if (type(args.vision_tower_epsilon) not in (int, float) or
                    not np.isfinite(args.vision_tower_epsilon) or args.vision_tower_epsilon <= 0 or
                    type(args.image_mib) is not int or args.image_mib <= 0):
                raise ValueError('vision requires explicit positive tower epsilon and image budget')
        elif args.image or any(v is not None for v in
                              (args.image_token_id, args.image_mib, args.vision_tower_epsilon)):
            raise ValueError('image options require --include-vision')
        for name in ('context', 'max_tokens', 'prefill_chunk', 'max_tensors', 'max_shards'):
            if type(getattr(args, name)) is not int or getattr(args, name) <= 0:
                raise ValueError('positive integer required: ' + name)
        self._budgets = {}
        for name in ('resident', 'runtime', 'expert', 'mapped', 'chunk', 'sampling', 'metadata', 'input'):
            value = getattr(args, name + '_mib')
            if type(value) is not int or value <= 0:
                raise ValueError('explicit positive budget required: ' + name)
            self._budgets[name] = value * 1024**2
        budgets = self._budgets
        if native_dense and (args.native_dense_projection_mib+args.native_projection_work_mib)*1024**2>=budgets['runtime']:
            raise MemoryError('native dense owner/work admissions exhaust aggregate runtime budget')
        if args.prefill_chunk > 4096 or args.context > 1048576:
            raise ValueError('prefill/context exceeds K3Execution limits')
        self._sampling_options(args.max_tokens, args.prefill_chunk, args.temperature, args.seed)
        for name in ('projection_epsilon', 'l2_epsilon'):
            if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
                raise ValueError('positive finite epsilon required: ' + name)
        directory = _path(args.checkpoint).resolve(strict=True)
        if not directory.is_dir():
            raise ValueError('local checkpoint directory required')
        if not bf16 and any(getattr(args,name+'_library',None) is None for name in ('kimi','attention','operators','mxfp4')):
            raise ValueError('native K3 requires all four explicit prebuilt libraries')
        libraries = {name: None if getattr(args,name+'_library',None) is None else _local_file(getattr(args,name+'_library'))
                     for name in ('kimi','attention','operators','mxfp4')}
        tokenizer_path = _local_file(args.tokenizer or directory / 'tokenizer.json')
        if tokenizer_path.stat().st_size > budgets['input']:
            raise MemoryError('local tokenizer JSON exceeds input budget')
        tokenizer_bytes=read_regular(tokenizer_path,max_bytes=budgets['input'])
        if len(tokenizer_bytes)>budgets['input']:raise MemoryError('tokenizer changed beyond admission')
        self._tokenizer_sha256=hashlib.sha256(tokenizer_bytes).hexdigest()
        self._tokenizer = Tokenizer.from_str(tokenizer_bytes.decode('utf-8'))
        del tokenizer_bytes
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        _check_cancelled(cancelled)
        self._checkpoint = PublisherCheckpoint.open(directory, family='kimi_k3',
            max_metadata_bytes=budgets['metadata'], max_tensors=args.max_tensors, max_shards=args.max_shards)
        self._vocabulary = self._checkpoint.config['text_config']['vocab_size']
        if type(self._vocabulary) is not int or self._vocabulary <= 0:
            raise ValueError('positive publisher vocabulary required')
        if args.include_vision and (type(args.image_token_id) is not int or
                                    not 0 <= args.image_token_id < self._vocabulary):
            raise ValueError('vision requires an explicit vocabulary image placeholder ID')
        if args.include_vision:
            self._budgets['image'] = args.image_mib*1024**2
        if self._vocabulary * 32 > budgets['sampling']:
            raise MemoryError('publisher vocabulary exceeds sampling budget')
        eos = self._checkpoint.config['text_config'].get('eos_token_id')
        stops = [] if eos is None else [eos] if type(eos) is int else eos
        self._stops = self._validate_stops(stops)
        self._validate_stops(args.stop_token_id)
        _check_cancelled(cancelled)
        vision_options = dict(include_vision=args.include_vision)
        if args.include_vision:
            vision_options['vision_tower_epsilon'] = args.vision_tower_epsilon
        projection_bank=None
        native_projection_bank=None
        if args.dense_projection_mib is not None:
            from .k3_execution.publisher_projections import K3PublisherProjections
            projection_bank=K3PublisherProjections(self._checkpoint,device=args.arithmetic_device,
                torch_version=args.torch_version,max_bytes=args.dense_projection_mib*1024**2,
                include_vision=args.include_vision)
            self._owned.callback(projection_bank.close)
            vision_options['sharded_projections']=tuple(sorted(projection_bank.sources))
        if native_dense:
            from .segment_edge_runtime import _Functions
            from .native_original_projections import NativeOriginalProjections
            native_runtime=_Functions(_local_file(args.original_projection_library))
            self._owned.callback(native_runtime.close)
            native_projection_bank=NativeOriginalProjections.from_k3(native_runtime,self._checkpoint,
                max_bytes=args.native_dense_projection_mib*1024**2,include_vision=args.include_vision)
            self._owned.callback(native_projection_bank.close)
            vision_options['sharded_projections']=tuple(sorted(native_projection_bank.sources))
        payload = self._checkpoint.load_k3_execution_payload(capacity=args.context,
            max_weight_bytes=budgets['resident'], max_chunk_bytes=budgets['chunk'],
            expert_format='e2m1-low-first-e8m0-g32', use_expert_provider=True,
            max_expert_bytes=budgets['expert'], max_mapped_bytes=budgets['mapped'],
            projection_epsilon=args.projection_epsilon, l2_epsilon=args.l2_epsilon,
            rotary_mode='publisher-none', **vision_options)
        if type(payload) is not K3ExecutionPayload:
            raise ValueError('exact publisher K3ExecutionPayload required')
        if payload.expert_provider is not None:
            self._owned.callback(payload.expert_provider.close)
        if payload.unsupported or payload.expert_provider is None:
            raise ValueError('supported publisher payload with leased expert provider required')
        self._gaps = [asdict(gap) for gap in payload.execution_gaps]
        self._config_sha256 = payload.config_sha256
        _check_cancelled(cancelled)
        arithmetic=routing=fla_policy=None
        if bf16:
            from .k3_execution.bf16_arithmetic import K3BF16Arithmetic
            from .k3_execution.routing import TorchK3Router
            from .k3_execution.fla_mixer import FLAK3Policy
            has_kda='kda' in payload.config.layer_types
            metal=args.arithmetic_device.startswith('mps')
            if has_kda and (args.fla_mib is None or (not metal and args.fla_version is None)):
                raise ValueError('BF16 hybrid checkpoint requires explicit FLA dependency and mixer admission')
            if not has_kda and (args.fla_version is not None or args.fla_mib is not None):
                raise ValueError('FLA admission requires a checkpoint containing KDA layers')
            reserved=(args.arithmetic_mib+args.routing_mib+
                      (args.fla_mib or 0)*payload.config.layer_types.count('kda')+
                      (args.dense_projection_mib or 0))*1024**2
            if reserved>=budgets['runtime']:
                raise MemoryError('BF16 component admissions exhaust aggregate runtime budget')
            arithmetic=K3BF16Arithmetic(device=args.arithmetic_device,torch_version=args.torch_version,
                max_bytes=args.arithmetic_mib*1024**2,projection_backend=projection_bank,
                attention_mode=args.attention_mode or 'eager')
            if projection_bank is not None:projection_bank.bind(arithmetic)
            routing=TorchK3Router(device=args.arithmetic_device,expected_torch_version=args.torch_version,
                max_bytes=args.routing_mib*1024**2)
            if has_kda:
                if metal:
                    from .k3_execution.metal_mixer import MetalK3Policy
                    fla_policy=MetalK3Policy(device=args.arithmetic_device,torch_version=args.torch_version,
                        max_bytes=args.fla_mib*1024**2)
                else:
                    fla_policy=FLAK3Policy(device=args.arithmetic_device,torch_version=args.torch_version,
                        fla_version=args.fla_version,max_bytes=args.fla_mib*1024**2)
        self._model = K3Execution(payload.config, payload.tensors, library=libraries['kimi'],
            require_publisher_exact=args.require_publisher_exact,exact_evidence=evidence,
            attention_library=libraries['attention'], operators_library=libraries['operators'],
            mxfp4_library=libraries['mxfp4'], expert_provider=payload.expert_provider,
            max_bytes=budgets['runtime'],arithmetic=arithmetic,routing=routing,fla_policy=fla_policy,
            sharded_projections=payload.sharded_projections,native_projections=native_projection_bank,
            native_projection_work_bytes=(args.native_projection_work_mib or 64)*1024**2)
        self._owned.callback(self._model.close)
        _check_cancelled(cancelled)

    @staticmethod
    def _sampling_options(max_tokens, prefill_chunk, temperature, seed):
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError('positive max_tokens required')
        if type(prefill_chunk) is not int or not 1 <= prefill_chunk <= 4096:
            raise ValueError('prefill_chunk must be an integer in 1..4096')
        if (type(temperature) not in (float, int) or not np.isfinite(temperature) or temperature < 0
                or type(seed) is not int or seed < 0):
            raise ValueError('finite nonnegative temperature and integer seed required')

    def _validate_stops(self, stops):
        if (type(stops) not in (list, tuple)
                or len(stops) * 32 > self._budgets['input']
                or any(type(t) is not int or not 0 <= t < self._vocabulary for t in stops)):
            raise ValueError('bounded vocabulary stop token inventory required')
        return frozenset(stops)

    def __enter__(self):
        if self._closed:
            raise RuntimeError('publisher K3 session is closed')
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def tokenizer(self):
        if self._closed:
            raise RuntimeError('publisher K3 session is closed')
        return self._tokenizer

    def run(self, args, emit=None, cancelled=None):
        """Generate using request CLI args, without changing captured load options."""
        request = copy.deepcopy(args)
        return self.generate(emit=emit, cancelled=cancelled, _request_args=request)

    def _close_unlocked(self):
        self._closed = True
        self._owned.close()
        self._model = self._tokenizer = self._checkpoint = None

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('cannot close an active publisher K3 request')
        try:
            if not self._closed or self._owned.pending:
                self._close_unlocked()
        finally:
            self._lock.release()

    def generate(self, *, prompt=None, token_ids=None, max_tokens=None, prefill_chunk=None,
                 temperature=None, seed=None, stop_token_ids=None, emit=None, cancelled=None,
                 images=None, media_items=None, checkpoint_sink=None,checkpoint_max_bytes=None,
                 checkpoint_scratch_directory=None,_request_args=None):
        """Return the CLI result; token_ids accepts a list/tuple or JSON array.

        Omitted request options use the constructor's CLI namespace defaults.
        Exceptions and cancellation propagate after cache cleanup; cleanup
        failure closes the warm session so damaged state cannot be reused.
        """
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('publisher K3 session already has an active request')
        try:
            if self._closed:
                raise RuntimeError('publisher K3 session is closed')
            try:
                self._model.reset()
                if _request_args is not None:
                    for name in self._loading_options:
                        expected = getattr(self._args, name)
                        observed = getattr(_request_args, name, False if name == 'include_vision' else None)
                        if (type(observed) is not type(expected) or observed != expected):
                            raise ValueError('warm K3 loading option cannot change: ' + name)
                    if (_request_args.prompt is None) == (_request_args.token_ids is None):
                        raise ValueError('exactly one raw prompt or token-ID input required')
                    prompt, token_ids = _request_args.prompt, _request_args.token_ids
                    max_tokens, prefill_chunk = _request_args.max_tokens, _request_args.prefill_chunk
                    temperature, seed = _request_args.temperature, _request_args.seed
                    stop_token_ids = _request_args.stop_token_id
                    images = getattr(_request_args, 'image', [])
                    self._sampling_options(max_tokens, prefill_chunk, temperature, seed)
                    self._validate_stops(stop_token_ids)
                result = self._generate(prompt, token_ids, max_tokens, prefill_chunk,
                    temperature, seed, stop_token_ids, emit,
                    self._cancelled if cancelled is None else cancelled, images,media_items,
                    checkpoint_sink,checkpoint_max_bytes,checkpoint_scratch_directory)
            except BaseException as error:
                try:
                    self._model.reset()
                except BaseException as cleanup_error:
                    error.add_note('K3 cache reset failed: ' + str(cleanup_error))
                    try:
                        self._close_unlocked()
                    except BaseException as close_error:
                        error.add_note('K3 session close failed: ' + str(close_error))
                raise
            else:
                try:
                    self._model.reset()
                except BaseException:
                    self._close_unlocked()
                    raise
                return result
        finally:
            self._lock.release()

    def _generate(self, prompt, token_ids, max_tokens, prefill_chunk, temperature,
                  seed, stop_token_ids, emit, cancelled, images,media_items=None,
                  checkpoint_sink=None,checkpoint_max_bytes=None,checkpoint_scratch_directory=None):
        started = time.perf_counter()
        _check_cancelled(cancelled)
        args = self._args
        self._checkpoint_options(checkpoint_sink,checkpoint_max_bytes)
        max_tokens = args.max_tokens if max_tokens is None else max_tokens
        prefill_chunk = args.prefill_chunk if prefill_chunk is None else prefill_chunk
        temperature = args.temperature if temperature is None else temperature
        seed = args.seed if seed is None else seed
        self._sampling_options(max_tokens, prefill_chunk, temperature, seed)
        if prompt is None and token_ids is None:
            prompt, token_ids = args.prompt, args.token_ids
        if (prompt is None) == (token_ids is None):
            raise ValueError('exactly one raw prompt or token-ID input required')
        raw = prompt if prompt is not None else token_ids
        if isinstance(raw, str):
            if len(raw.encode('utf-8')) > self._budgets['input']:
                raise MemoryError('prompt/token-ID input exceeds input budget')
            ids = (self._tokenizer.encode(raw, add_special_tokens=False).ids
                   if prompt is not None else json.loads(raw))
        elif prompt is None and type(raw) in (list, tuple):
            if len(raw) * 32 > self._budgets['input']:
                raise MemoryError('token history exceeds input budget')
            ids = list(raw)
        else:
            raise ValueError('raw text or token-ID array required')
        if (type(ids) is not list or not ids
                or any(type(t) is not int or not 0 <= t < self._vocabulary for t in ids)):
            raise ValueError('nonempty vocabulary token-ID array required')
        if len(ids)+max_tokens > args.context:
            raise ValueError('prompt plus generation must fit selected context')
        if (len(ids)+max_tokens)*32 > self._budgets['input']:
            raise MemoryError('token history exceeds input budget')
        image_paths = args.image if images is None else images
        if media_items is not None:
            if image_paths:raise ValueError('decoded mixed media and encoded still images cannot be combined')
            if type(media_items) not in (tuple,list) or not 1<=len(media_items)<=4096:
                raise ValueError('bounded decoded media sequence required')
            image_paths=media_items
        if not isinstance(image_paths, (list, tuple)):
            raise ValueError('image inputs must be a sequence of local paths or encoded bytes')
        if image_paths and not args.include_vision:
            raise ValueError('warm session was loaded without vision capability')
        pixels = grids = None
        projected_lengths = ()
        media_metadata=()
        patch_offsets = [0]
        expanded_prompt = len(ids)
        if image_paths:
            if len(image_paths)*64 > self._budgets['image']:
                raise MemoryError('image inventory exceeds media budget')
            if ids.count(args.image_token_id) != len(image_paths):
                raise ValueError('one explicit prompt placeholder per image is required')
            if media_items is None:
                pixels, grids = self._preprocess_media(image_paths, cancelled)
                media_metadata=()
            else:
                from .k3_execution.video_input import prepare_media
                prepared=prepare_media(media_items,max_bytes=self._budgets['image'],cancelled=cancelled)
                pixels,grids=prepared.pixels,prepared.grids
                media_metadata=tuple(dict(timestamps=times,selection_identity=identity)
                    for times,identity in zip(prepared.timestamps,prepared.selection_identities))
                del prepared
            projected_lengths = self._model._vision.projected_lengths(grids)
            for t, h, w in grids:
                patch_offsets.append(patch_offsets[-1]+int(t)*int(h)*int(w))
            expanded_prompt += sum(projected_lengths)-len(image_paths)
        elif args.include_vision and args.image_token_id in ids:
            raise ValueError('image placeholder lacks an image input')
        if expanded_prompt + max_tokens > args.context:
            raise ValueError('prompt plus generation must fit selected context')
        if (expanded_prompt + max_tokens) * 32 > self._budgets['input']:
            raise MemoryError('token history exceeds input budget')
        additional_stops = self._validate_stops(args.stop_token_id if stop_token_ids is None else stop_token_ids)
        stops = self._stops | additional_stops
        rng = np.random.default_rng(seed)
        reused = self._requests > 0
        self._requests += 1

        def event(name, **fields):
            _check_cancelled(cancelled)
            if emit is not None:
                emit(dict(event=name, **fields))
            _check_cancelled(cancelled)

        scope = ('mixed-media-text' if media_items is not None else 'image-text') if image_paths else 'text-only'
        event('started', prompt_tokens=len(ids), expanded_prompt_tokens=expanded_prompt, context=args.context,
              max_tokens=max_tokens, seed=seed, temperature=temperature,
              execution_scope=scope, numerical_parity_verified=False)
        event('loaded', config_sha256=self._config_sha256, implementation=self._model.implementation,
              execution_gaps=copy.deepcopy(self._gaps), load_reused=reused,
              numerical_parity_verified=False, platform_certified=False)
        loaded = time.perf_counter()
        class RequestCancellation:
            def is_set(self):return bool(cancelled is not None and cancelled())
        request_cancel=RequestCancellation()

        def append(tokens, *, media=None, expanded_count=None):
            _check_cancelled(cancelled)
            start = self._model.position
            count = len(tokens) if expanded_count is None else expanded_count
            with self._model.cancellation_scope(request_cancel):
                output = np.asarray(self._model.forward(np.asarray(tokens, dtype=np.int64),
                    position_ids=np.arange(start, start + count, dtype=np.int64), **(media or {})))
            _check_cancelled(cancelled)
            if output.shape != (count, self._vocabulary) or self._model.position != start + count:
                raise ValueError('executor output/prefix differs from appended token inventory')
            row = output[-1].copy()
            if row.dtype.kind != 'f' or not np.isfinite(row).all():
                raise ValueError('invalid next-token logits')
            return row

        logits = None
        media_index = 0
        for offset in range(0, len(ids), prefill_chunk):
            chunk = ids[offset:offset + prefill_chunk]
            count = chunk.count(args.image_token_id) if image_paths else 0
            if count:
                end = media_index+count
                media = dict(pixel_values=pixels[patch_offsets[media_index]:patch_offsets[end]],
                             grid_thws=grids[media_index:end], media_token_id=args.image_token_id)
                logits = append(chunk, media=media,
                                expanded_count=len(chunk)-count+sum(projected_lengths[media_index:end]))
                media_index = end
            else:
                logits = append(chunk)
            event('prefill', position=self._model.position, prompt_tokens=expanded_prompt)
        pixels = grids = None
        generated, first_token_ms = [], None
        def save_boundary():
            if checkpoint_sink is not None:
                from .k3_generation_recovery import GenerationBoundary
                boundary=GenerationBoundary(self._config_sha256,self._model.position,tuple(ids),
                    tuple(generated),expanded_prompt,np.asarray(logits,dtype=np.float32),
                    copy.deepcopy(rng.bit_generator.state),temperature,tuple(sorted(stops)),scope,
                    tuple(copy.deepcopy(media_metadata)),self._tokenizer_sha256)
                self._publish_boundary(checkpoint_sink,boundary,checkpoint_max_bytes,checkpoint_scratch_directory)
                _check_cancelled(cancelled)
        save_boundary()
        finish_reason = 'length'
        for _ in range(max_tokens):
            _check_cancelled(cancelled)
            token = _sample(logits, temperature, rng, self._budgets['sampling'])
            logits = append([token])
            generated.append(token)
            save_boundary()
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - loaded) * 1000
            event('tokens', token_ids=[token], position=self._model.position,
                  text=self._tokenizer.decode(generated, skip_special_tokens=False))
            if token in stops:
                finish_reason = 'eos' if token in self._stops else 'stop_token'
                break
        _check_cancelled(cancelled)
        load_ms = 0.0 if reused else self._load_ms
        return dict(schema='openagent.publisher-k3-generation.v1',
            prompt_ids=ids, generated_ids=generated,
            text=self._tokenizer.decode(generated, skip_special_tokens=False),
            position=self._model.position, finish_reason=finish_reason,
            seed=seed, temperature=temperature, config_sha256=self._config_sha256,
            load_ms=load_ms, load_reused=reused, first_token_ms=first_token_ms,
            total_ms=(time.perf_counter() - started) * 1000 + load_ms,
            implementation=self._model.implementation, execution_scope=scope,
            expanded_prompt_tokens=expanded_prompt,
            image_count=len(image_paths)-sum(bool(row['timestamps']) for row in media_metadata) if image_paths else 0,
            video_count=sum(bool(row['timestamps']) for row in media_metadata) if image_paths else 0,
            media_metadata=list(media_metadata) if image_paths else [],
            execution_gaps=copy.deepcopy(self._gaps), numerical_parity_verified=False,
            platform_certified=False, published_checkpoint_supported=False)

    @staticmethod
    def _checkpoint_options(sink,maximum):
        if sink is None:
            if maximum is not None:raise ValueError('checkpoint budget requires a checkpoint sink')
        elif not callable(sink) or type(maximum) is not int or maximum<=0:
            raise ValueError('callable checkpoint sink and positive complete frame budget required')

    def _publish_boundary(self,sink,boundary,maximum,scratch):
        from .k3_generation_recovery import write_generation_boundary
        active=True;written=False;consumed=False
        owner_thread=threading.get_ident()
        def write_frame(output):
            nonlocal written,consumed
            if not active or consumed or threading.get_ident()!=owner_thread:
                raise RuntimeError('checkpoint writer expired, consumed or outside its request thread')
            consumed=True
            size=write_generation_boundary(self._model,boundary,output,max_bytes=maximum,scratch_directory=scratch)
            written=True
            return size
        try:
            sink(copy.deepcopy(boundary),write_frame)
            if not written:raise RuntimeError('checkpoint sink did not capture the complete frame')
        finally:active=False

    def resume_generation(self,source,*,checkpoint_max_bytes,max_tokens,emit=None,cancelled=None,
                          checkpoint_sink=None,checkpoint_scratch_directory=None):
        """Resume complete local generation state; sink publishes each boundary.

        A sink receives (boundary, write_frame). It must synchronously call
        write_frame with its staging stream and finish publication before
        returning. Streams, storage leases and retention remain caller-owned.
        """
        if not self._lock.acquire(blocking=False):raise RuntimeError('publisher K3 session already active')
        try:
            if self._closed:raise RuntimeError('publisher K3 session is closed')
            if type(max_tokens) is not int or max_tokens<=0:raise ValueError('positive continuation length required')
            if not callable(getattr(source,'read',None)):raise ValueError('caller-owned binary recovery source required')
            if checkpoint_sink is not None:self._checkpoint_options(checkpoint_sink,checkpoint_max_bytes)
            cancelled=self._cancelled if cancelled is None else cancelled
            _check_cancelled(cancelled)
            from .k3_generation_recovery import restore_generation_boundary,GenerationBoundary
            try:
                self._model.reset()
                class RecoverySource:
                    def read(self,count):
                        _check_cancelled(cancelled)
                        value=source.read(count)
                        _check_cancelled(cancelled)
                        return value
                boundary=restore_generation_boundary(self._model,RecoverySource(),config_sha256=self._config_sha256,
                    tokenizer_sha256=self._tokenizer_sha256,max_bytes=checkpoint_max_bytes,
                    scratch_directory=checkpoint_scratch_directory)
                if self._model.position+max_tokens>self._args.context:
                    raise ValueError('restored prefix plus continuation exceeds context')
                if (len(boundary.prompt_ids)+len(boundary.generated_ids)+max_tokens)*32>self._budgets['input']:
                    raise MemoryError('restored token history exceeds admission')
                rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(boundary.rng_state)
                logits=boundary.next_logits;generated=list(boundary.generated_ids)
                stops=set(boundary.stop_token_ids)
                if not self._stops<=stops:raise ValueError('recovery omitted configured EOS boundaries')
                finish='length';added=[]
                if emit is not None:emit(dict(event='recovered',position=self._model.position,
                    text=self._tokenizer.decode(generated,skip_special_tokens=False),recovered=True))
                class RequestCancellation:
                    def is_set(self):return bool(cancelled is not None and cancelled())
                for _ in range(max_tokens):
                    _check_cancelled(cancelled)
                    if generated and generated[-1] in stops:
                        finish='eos' if generated[-1] in self._stops else 'stop_token';break
                    token=_sample(logits,boundary.temperature,rng,self._budgets['sampling'])
                    position=self._model.position
                    with self._model.cancellation_scope(RequestCancellation()):
                        output=self._model.forward(np.asarray([token],dtype=np.int64),
                            position_ids=np.asarray([position],dtype=np.int64))
                    if output.shape!=(1,self._vocabulary) or self._model.position!=position+1:
                        raise ValueError('restored decoder continuation geometry differs')
                    logits=np.asarray(output[-1],dtype=np.float32).copy()
                    if not np.isfinite(logits).all():raise ValueError('nonfinite recovery logits')
                    generated.append(token);added.append(token)
                    current=GenerationBoundary(self._config_sha256,self._model.position,boundary.prompt_ids,
                        tuple(generated),boundary.expanded_prompt_tokens,logits,
                        copy.deepcopy(rng.bit_generator.state),boundary.temperature,boundary.stop_token_ids,
                        boundary.execution_scope,boundary.media_metadata,self._tokenizer_sha256)
                    if checkpoint_sink is not None:
                        self._publish_boundary(checkpoint_sink,current,checkpoint_max_bytes,checkpoint_scratch_directory)
                    _check_cancelled(cancelled)
                    if emit is not None:emit(dict(event='tokens',token_ids=[token],position=self._model.position,
                        text=self._tokenizer.decode(generated,skip_special_tokens=False),recovered=True))
                if generated and generated[-1] in stops:
                    finish='eos' if generated[-1] in self._stops else 'stop_token'
                result=dict(schema='openagent.publisher-k3-recovery.v1',generated_ids=generated,
                    resumed_ids=added,prompt_ids=list(boundary.prompt_ids),position=self._model.position,
                    text=self._tokenizer.decode(generated,skip_special_tokens=False),finish_reason=finish,
                    execution_scope=boundary.execution_scope,media_metadata=list(boundary.media_metadata),
                    recovered=True,config_sha256=self._config_sha256,numerical_parity_verified=False,
                    platform_certified=False)
            finally:
                primary=sys.exception()
                try:self._model.reset()
                except BaseException as cleanup:
                    try:self._close_unlocked()
                    except BaseException as release:cleanup.add_note('session release failed: '+repr(release))
                    if primary is not None:primary.add_note('recovery cache cleanup failed: '+repr(cleanup))
                    else:raise
            return result
        finally:self._lock.release()

    def _preprocess_media(self, paths, cancelled):
        import io
        from PIL import Image
        from .k3_execution import preprocess_images
        vision = self._model.config.vision
        if vision is None or vision.patch != 14 or vision.merge != (2, 2):
            raise ValueError('existing raw-image processor requires pinned patch14/merge2 configuration')
        maximum = self._budgets['image']
        with ExitStack() as owned:
            images, decoded, encoded = [], 0, 0
            for path in paths:
                _check_cancelled(cancelled)
                if isinstance(path, bytes):
                    if not path or len(path) > maximum-encoded:
                        raise MemoryError('encoded image inventory exceeds media budget')
                    source = owned.enter_context(io.BytesIO(path))
                else:
                    local = _local_file(path)
                    if local.stat().st_size > maximum-encoded:
                        raise MemoryError('encoded image inventory exceeds media budget')
                    source = owned.enter_context(local.open('rb'))
                source.seek(0, 2)
                encoded += source.tell()
                source.seek(0)
                if encoded > maximum:
                    raise MemoryError('encoded image inventory exceeds media budget')
                image = owned.enter_context(Image.open(source))
                if image.width <= 0 or image.height <= 0 or getattr(image, 'n_frames', 1) != 1:
                    raise ValueError('positive-sized still images only; no animation/video sampling')
                decoded += 16*image.width*image.height
                if decoded+2*encoded >= maximum:
                    raise MemoryError('decoded image inventory exceeds media budget')
                images.append(image)
            _check_cancelled(cancelled)
            pixels, grids = preprocess_images(images, max_bytes=maximum-decoded-2*encoded)
            _check_cancelled(cancelled)
            return pixels, grids


def run(args, emit=None, cancelled=None):
    with PublisherSession(args, cancelled=cancelled) as session:
        return session.run(args, emit=emit, cancelled=cancelled)


def main(argv=None):
    args = parser().parse_args(argv)

    def emit(event):
        print(json.dumps(event, allow_nan=False), flush=True)

    try:
        result = run(args, emit=emit if args.stream else None)
        emit(dict(event='completed', **result))
        return 0
    except KeyboardInterrupt:
        if args.stream:
            emit(dict(event='cancelled'))
        print('Generation cancelled', file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, ImportError, KeyError, MemoryError, ArithmeticError) as error:
        if args.stream:
            emit(dict(event='failed', error=str(error)))
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
