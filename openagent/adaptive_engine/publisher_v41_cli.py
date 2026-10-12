"""Explicit local publisher V4.1 generation with optional DSpark proposals."""
import argparse
import copy
import hashlib
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time
import threading

import numpy as np
from .session_recovery import _path
from .publisher_media_files import read_regular


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('positive integer required')
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('checkpoint', type=Path)
    result.add_argument('--execution-library', type=Path)
    result.add_argument('--mxfp4-library', type=Path)
    result.add_argument('--fp8-library', type=Path)
    result.add_argument('--device')
    result.add_argument('--torch-version')
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument('--prompt')
    source.add_argument('--token-ids')
    source.add_argument('--resume-generation', type=Path)
    result.add_argument('--save-generation', type=Path)
    result.add_argument('--checkpoint-mib', type=positive)
    result.add_argument('--tokenizer', type=Path)
    result.add_argument('--max-tokens', type=positive, default=32)
    result.add_argument('--context', type=positive, default=512)
    result.add_argument('--temperature', type=float, default=0.0)
    result.add_argument('--seed', type=int, default=0)
    result.add_argument('--dspark', action='store_true')
    result.add_argument('--stream', action='store_true')
    result.add_argument('--resident-mib', type=positive, default=4096)
    result.add_argument('--runtime-mib', type=positive, default=1024)
    result.add_argument('--arithmetic-mib', type=positive, default=256)
    result.add_argument('--expert-mib', type=positive, default=256)
    result.add_argument('--projection-mib', type=positive, default=64)
    result.add_argument('--bf16-projection-mib',type=int,default=64)
    result.add_argument('--metadata-mib',type=positive,default=256)
    result.add_argument('--total-mib',type=positive)
    result.add_argument('--draft-mib', type=positive, default=512)
    result.add_argument('--evidence-mib', type=positive, default=256)
    result.add_argument('--sampling-mib', type=positive, default=128)
    result.add_argument('--prefill-chunk', type=positive, default=32)
    result.add_argument('--image', type=Path, action='append', default=[])
    result.add_argument('--image-token-id', type=int)
    result.add_argument('--image-max-tokens', type=positive, default=256)
    result.add_argument('--image-min-pixels', type=int, default=0)
    result.add_argument('--image-max-aspect', type=float)
    result.add_argument('--image-mib', type=positive, default=256)
    result.add_argument('--vision-mib', type=positive, default=512)
    return result


def _sample(logits, temperature, rng, *, max_bytes=128*1024**2):
    observed = np.asarray(logits)
    if (observed.ndim != 1 or not observed.size or observed.dtype.kind != 'f' or
            type(max_bytes) is not int or max_bytes <= 0):
        raise ValueError('floating vocabulary logits and explicit sampling budget required')
    if observed.size*32 > max_bytes:
        raise MemoryError('vocabulary sampling exceeds byte budget')
    if (type(temperature) not in (int, float) or
            not np.isfinite(temperature) or temperature < 0):
        raise ValueError('finite nonnegative sampling temperature required')
    row = np.array(observed, dtype=np.float64, copy=True)
    if not np.isfinite(row).all():
        raise ValueError('finite target vocabulary logits required')
    if temperature == 0:
        return int(row.argmax())
    with np.errstate(over='ignore', under='ignore', invalid='raise', divide='raise'):
        row -= row.max()
        row /= temperature
        probabilities = np.exp(row)
    total = probabilities.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError('invalid target sampling distribution')
    probabilities /= total
    return int(rng.choice(len(row), p=probabilities))


def _run_request(session, args, *, emit=None, cancelled=None, checkpoint_sink=None,
                 max_checkpoint_bytes=0):
    from .execution_service import ExecutionCancelled
    def check_cancelled():
        if cancelled is not None and cancelled():
            raise ExecutionCancelled('native generation cancelled')
    check_cancelled()
    from .v41_execution.dspark import create_dspark_decoder

    if not np.isfinite(args.temperature) or args.temperature < 0 or args.seed < 0:
        raise ValueError('finite nonnegative temperature and seed required')
    tokenizer, checkpoint = session.tokenizer, session.checkpoint
    vocab = checkpoint.config['text_config']['vocab_size']
    raw = args.prompt if args.prompt is not None else args.token_ids
    if type(raw) is not str or len(raw.encode('utf-8')) > args.sampling_mib*1024**2:
        raise MemoryError('text/token input exceeds sampling admission budget')
    prompt = (tokenizer.encode(args.prompt).ids if args.prompt is not None else
        json.loads(args.token_ids))
    if (type(prompt) is not list or not prompt or
            any(type(t) is not int or not 0 <= t < vocab for t in prompt) or
            (not args.image and len(prompt)+args.max_tokens > args.context)):
        raise ValueError('valid prompt and generation within selected context required')
    eos = checkpoint.config['text_config'].get('eos_token_id', [])
    stops = [eos] if type(eos) is int else eos
    if type(stops) not in (list, tuple) or any(type(t) is not int or not 0 <= t < vocab for t in stops):
        raise ValueError('valid publisher EOS token inventory required')
    stop_tokens = frozenset(stops)
    mib = 1024**2
    started = time.perf_counter()
    generated, accepted, proposed = [], 0, 0
    if (len(prompt)+args.max_tokens)*32+vocab*32 > args.sampling_mib*mib:
        raise MemoryError('token history and vocabulary sampling exceed byte budget')
    target_rng, draft_rng = [np.random.default_rng(seed) for seed in
        np.random.SeedSequence(args.seed).spawn(2)]

    with ExitStack() as owned:
        check_cancelled()
        arithmetic, model = session.arithmetic, session.model
        image_options = {}
        if args.image:
            from .v41_execution.image_input import PublisherImageProcessor
            if (type(args.image_token_id) is not int or not 0 <= args.image_token_id < vocab):
                raise ValueError('explicit valid --image-token-id required for image inputs')
            vision = session._vision(args)
            processor = PublisherImageProcessor(patch_size=vision.config['patch_size'],
                downsample_ratio=vision.config['downsample_ratio'], max_tokens=args.image_max_tokens,
                min_pixels=args.image_min_pixels, max_width_height_ratio=args.image_max_aspect,
                max_bytes=args.image_mib*mib, cast=arithmetic.cast)
            prompt, types, images = processor.expand(prompt, args.image,
                image_token_id=args.image_token_id)
            check_cancelled()
            if len(prompt)+args.max_tokens > args.context:
                raise ValueError('expanded image prompt and generation exceed selected context')
            if (len(prompt)+args.max_tokens)*32+vocab*32 > args.sampling_mib*mib:
                raise MemoryError('expanded image history exceeds sampling admission budget')
            image_options = dict(images=images, vision=vision, token_types=types)
        loaded = time.perf_counter()
        first_token_ms = None

        def checkpoint_boundary():
            if checkpoint_sink is not None:
                from .v41_generation_recovery import publish
                current_logits = decoder.next_logits if args.dspark else logits
                header = dict(schema='openagent.v41-generation-boundary.v1',
                    identity=model.identity, tokenizer=session._tokenizer_identity,
                    position=model.length, prompt_ids=list(prompt), generated_ids=list(generated),
                    temperature=float(args.temperature), rng=copy.deepcopy(target_rng.bit_generator.state),
                    stops=sorted(stop_tokens), dspark=bool(args.dspark), accepted=accepted, proposed=proposed)
                publish(checkpoint_sink, header, current_logits, decoder if args.dspark else model,
                    model=model, tokenizer=session._tokenizer_identity, max_bytes=max_checkpoint_bytes,
                    host_bytes=args.sampling_mib*mib, cancelled=cancelled)

        def record(tokens, **details):
            nonlocal first_token_ms
            if first_token_ms is None:
                first_token_ms = (time.perf_counter()-loaded)*1000
            generated.extend(tokens)
            checkpoint_boundary()
            if emit is not None:
                emit(dict(event='tokens', token_ids=list(tokens), **details))

        if args.dspark:
            draft = session._draft(args, draft_rng)
            decoder = create_dspark_decoder(model, draft, temperature=args.temperature,
                max_bytes=args.sampling_mib*mib, evidence_max_bytes=args.evidence_mib*mib,
                target_rng=target_rng, proposal_rng=draft_rng)
            check_cancelled()
            decoder.prefill(prompt, **image_options)
            check_cancelled()
            checkpoint_boundary()
            for step in decoder.generate(args.max_tokens, stop_tokens=stop_tokens, cancelled=cancelled):
                accepted += step.accepted
                proposed += step.proposed
                record(step.tokens, accepted=step.accepted, proposed=step.proposed,
                    rejected=step.rejected)
        else:
            logits = None
            if image_options:
                check_cancelled()
                logits = model.forward(prompt, **image_options)[-1]
            else:
                for offset in range(0, len(prompt), args.prefill_chunk):
                    check_cancelled()
                    logits = model.forward(prompt[offset:offset+args.prefill_chunk])[-1]
            checkpoint_boundary()
            for _ in range(args.max_tokens):
                check_cancelled()
                token = _sample(logits, args.temperature, target_rng,
                    max_bytes=args.sampling_mib*mib-(len(prompt)+args.max_tokens)*32)
                # Commit the emitted token before reporting it to the caller.
                logits = model.forward([token])[-1]
                check_cancelled()
                record((token,))
                if token in stop_tokens:
                    break
        check_cancelled()
        return dict(schema='openagent.publisher-v41-generation.v1',
            prompt_ids=prompt, generated_ids=generated, text=tokenizer.decode(generated),
            dspark=args.dspark, accepted_draft_tokens=accepted, proposed_draft_tokens=proposed,
            load_ms=0.0, session_load_ms=session.load_ms,
            request_setup_ms=(loaded-started)*1000, first_token_ms=first_token_ms,
            total_ms=(time.perf_counter()-started)*1000,
            implementation=model.implementation, numerical_parity_verified=False,
            platform_certified=False)


class PublisherSession:
    """Owned serial warm session; request cancellation is checked between operations."""

    _fixed = ('checkpoint', 'execution_library', 'mxfp4_library', 'fp8_library',
              'tokenizer', 'context', 'resident_mib', 'runtime_mib', 'arithmetic_mib',
              'expert_mib', 'projection_mib', 'draft_mib', 'evidence_mib',
              'sampling_mib', 'image_mib', 'vision_mib', 'image_max_tokens','bf16_projection_mib',
              'metadata_mib','total_mib','device','torch_version')
    _budgets = ('resident_mib', 'runtime_mib', 'arithmetic_mib', 'expert_mib',
                'projection_mib', 'draft_mib', 'evidence_mib', 'sampling_mib',
                'image_mib', 'vision_mib','bf16_projection_mib')

    @staticmethod
    def _validate(args):
        for name in PublisherSession._budgets+('context', 'max_tokens', 'prefill_chunk', 'image_max_tokens'):
            value=getattr(args,name,0) if name=='bf16_projection_mib' else getattr(args,name)
            if type(value) is not int or (value<0 if name=='bf16_projection_mib' else value<=0):
                raise ValueError('positive integer request budget required: '+name)
        if (type(args.seed) is not int or args.seed < 0 or
                not np.isfinite(args.temperature) or args.temperature < 0):
            raise ValueError('finite nonnegative temperature and integer seed required')
        if sum(value is not None for value in (args.prompt,args.token_ids,getattr(args,'resume_generation',None)))!=1:
            raise ValueError('exactly one prompt or token-id input required')
        metadata=getattr(args,'metadata_mib',256);total=getattr(args,'total_mib',None)
        if type(metadata) is not int or metadata<=0 or (total is not None and (type(total) is not int or total<=0)):
            raise ValueError('positive metadata and optional aggregate admission required')

    @staticmethod
    def _signature(args):
        return tuple(str(Path(getattr(args, name)).resolve())
                     if name in PublisherSession._fixed[:5] and getattr(args,name,None) is not None else getattr(args, name,0)
                     for name in PublisherSession._fixed)

    def __init__(self, args, *, cancelled=None, distributed=None):
        from .execution_service import ExecutionCancelled
        def check_cancelled():
            if cancelled is not None and cancelled():
                raise ExecutionCancelled('native model loading cancelled')
        check_cancelled()
        from tokenizers import Tokenizer
        from .publisher_checkpoint.service import PublisherCheckpoint
        from .publisher_checkpoint.v41_adapter import load_v41
        from .v41_execution.publisher import PublisherV41Execution
        from .v41_execution.publisher_arithmetic import NativePublisherV41Arithmetic

        self._validate(args)
        args = copy.deepcopy(args)
        if distributed is not None:
            from .v41_distributed_session import V41DistributedOptions
            if type(distributed) is not V41DistributedOptions:
                raise TypeError('explicit caller-owned distributed publisher transport required')
            if distributed.bf16 and not getattr(args,'bf16_projection_mib',0):
                raise ValueError('distributed BF16 requires positive row-streaming admission')
        directory = _path(args.checkpoint).resolve(strict=True)
        if not directory.is_dir():
            raise ValueError('local checkpoint directory required')
        device=getattr(args,'device',None)
        if device is None:
            if any(path is None for path in (args.execution_library,args.mxfp4_library,args.fp8_library)):
                raise ValueError('native V4.1 requires all three explicit prebuilt arithmetic libraries')
            libraries = [_path(path).resolve(strict=True) for path in
                         (args.execution_library, args.mxfp4_library, args.fp8_library)]
            if any(not path.is_file() for path in libraries):
                raise ValueError('explicit prebuilt native libraries required')
        else:
            if not getattr(args,'torch_version',None):
                raise ValueError('device V4.1 requires an explicit Torch version pin')
            if distributed is not None and distributed.torch_version!=args.torch_version:
                raise ValueError('V4.1 collective and arithmetic Torch pins differ')
            libraries=[None,None,None]
        args.tokenizer = _path(args.tokenizer or directory/'tokenizer.json').resolve(strict=True)
        args.checkpoint = directory
        self.args = args
        self._admitted_args = copy.deepcopy(args)
        self._signature_value = self._signature(args)
        self._lock = threading.Lock()
        self._closed = False
        self._requests = 0
        self._vision_model = self._draft_model = None
        self._vision_budget = self._draft_budget = None
        from .owned_resources import OwnedResources
        self._owned = OwnedResources()
        self.distributed=distributed
        started, mib = time.perf_counter(), 1024**2
        try:
            from .v41_resource_admission import reserve_components
            components={name:getattr(args,name,0)*mib for name in self._budgets}
            components['metadata']=getattr(args,'metadata_mib',256)*mib
            components['token-and-engram-rows']=8*mib
            if distributed is not None:components['collective']=distributed.max_collective_bytes
            total=getattr(args,'total_mib',None)
            self.resource_ledger=reserve_components(components,
                max_total_bytes=None if total is None else total*mib,owned=self._owned)
            if args.tokenizer.stat().st_size>getattr(args,'metadata_mib',256)*mib:
                raise MemoryError('publisher tokenizer source exceeds metadata admission')
            from .v41_generation_recovery import tokenizer_identity
            self._tokenizer_identity = tokenizer_identity(args.tokenizer)
            tokenizer_bytes=read_regular(args.tokenizer,max_bytes=getattr(args,'metadata_mib',256)*mib)
            if hashlib.sha256(tokenizer_bytes).hexdigest()!=self._tokenizer_identity:
                raise ValueError('loaded tokenizer bytes differ from admitted generation identity')
            self.tokenizer = Tokenizer.from_str(tokenizer_bytes.decode('utf-8'))
            del tokenizer_bytes
            if tokenizer_identity(args.tokenizer) != self._tokenizer_identity:
                raise ValueError('publisher tokenizer changed while loading')
            self.tokenizer.no_truncation()
            self.tokenizer.no_padding()
            check_cancelled()
            self.checkpoint = PublisherCheckpoint.open(directory, family='deepseek_v41',
                max_metadata_bytes=getattr(args,'metadata_mib',256)*mib)
            checkpoint_close = getattr(self.checkpoint, 'close', None)
            if callable(checkpoint_close):
                self._owned.callback(checkpoint_close)
            self.bf16_projections=None
            bf16_budget=getattr(args,'bf16_projection_mib',0)*mib
            if bf16_budget:
                from .v41_execution.bf16_projections import PublisherBF16Projections
                self.bf16_projections=PublisherBF16Projections(self.checkpoint,max_bytes=bf16_budget)
                self._owned.callback(self.bf16_projections.close)
            self.payload = load_v41(self.checkpoint, capacity=args.context,
                max_weight_bytes=args.resident_mib*mib, max_expert_bytes=args.expert_mib*mib,
                max_projection_bytes=args.projection_mib*mib, use_providers=True,
                sharded_projections=() if self.bf16_projections is None else tuple(sorted(self.bf16_projections.sources)))
            for provider in (self.payload.engram_provider, self.payload.expert_provider,
                             self.payload.fp8_provider,self.payload.embedding_provider):
                if provider is not None:
                    self._owned.callback(provider.close)
            self.projection_backend=self.bf16_projections
            if distributed is not None:
                from .v41_distributed_session import make_projection_backend
                self.projection_backend=make_projection_backend(distributed,
                    checkpoint=self.checkpoint,payload=self.payload,
                    bf16_provider=self.bf16_projections,fp8_library=libraries[2],fp4_library=libraries[1],
                    max_arithmetic_bytes=args.arithmetic_mib*mib,owned=self._owned,arithmetic_device=device)
            check_cancelled()
            model_type=PublisherV41Execution
            if device is None:
                self.arithmetic = NativePublisherV41Arithmetic(self.payload,
                    execution_library=libraries[0], mxfp4_library=libraries[1],
                    fp8_library=libraries[2], tokenizer=self.tokenizer,
                    max_bytes=args.arithmetic_mib*mib,projection_backend=self.projection_backend)
            else:
                from .v41_execution.device_arithmetic import DevicePublisherV41Arithmetic
                from .v41_execution.device_execution import DevicePublisherV41Execution
                from .publisher_checkpoint.v41_provider import V41FP4ProjectionProvider
                fp4_rows=V41FP4ProjectionProvider(self.checkpoint,max_projection_bytes=args.projection_mib*mib)
                self._owned.callback(fp4_rows.close)
                self.arithmetic=DevicePublisherV41Arithmetic(self.payload,device=device,
                    torch_version=args.torch_version,max_bytes=args.arithmetic_mib*mib,
                    tokenizer=self.tokenizer,projection_backend=self.projection_backend,fp4_provider=fp4_rows)
                model_type=DevicePublisherV41Execution
            self._owned.callback(self.arithmetic.close)
            if self.bf16_projections is not None:self.bf16_projections.bind_arithmetic(self.arithmetic)
            check_cancelled()
            self.model = model_type.from_payload(self.payload, self.arithmetic,
                library=libraries[0], max_bytes=args.runtime_mib*mib)
            self._owned.callback(self.model.close)
            check_cancelled()
        except BaseException as error:
            self._closed = True
            try:self._owned.close()
            except BaseException as cleanup_error:
                error.cleanup=self._owned.close
                error.add_note('publisher session ownership release remains pending: '+str(cleanup_error))
            raise
        self.load_ms = (time.perf_counter()-started)*1000

    def _vision(self, args):
        from .v41_execution.publisher_auxiliary import PublisherV41Auxiliary
        if self._vision_model is None or self._vision_budget != args.vision_mib:
            self._vision_model = PublisherV41Auxiliary(self.arithmetic).vision(
                max_bytes=args.vision_mib*1024**2)
            self._vision_budget = args.vision_mib
        return self._vision_model

    def _draft(self, args, rng):
        from .v41_execution.publisher_auxiliary import PublisherV41Auxiliary
        if self._draft_model is None or self._draft_budget != args.draft_mib:
            self._draft_model = PublisherV41Auxiliary(self.arithmetic).mtp(
                max_bytes=args.draft_mib*1024**2, rng=rng)
            self._draft_budget = args.draft_mib
        self._draft_model.reset()
        self._draft_model.rng = rng
        return self._draft_model

    def _reset_request(self):
        try:
            self.model.reset()
            if self._draft_model is not None:
                self._draft_model.reset()
        except BaseException:
            # A failed reset must never leave a reusable contaminated session.
            self._closed = True
            self._owned.close()
            raise

    def run(self, args, emit=None, cancelled=None, *, checkpoint_sink=None, max_checkpoint_bytes=0):
        from .execution_service import ExecutionCancelled
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('publisher session is busy or reentrant')
        try:
            if self._closed:
                raise RuntimeError('publisher session is closed')
            if cancelled is not None and cancelled():
                raise ExecutionCancelled('native generation cancelled')
            args = copy.deepcopy(args)
            self._validate(args)
            self._checkpoint_options(checkpoint_sink, max_checkpoint_bytes)
            args.tokenizer = Path(args.tokenizer or Path(args.checkpoint)/'tokenizer.json').resolve()
            if self._signature(args) != self._signature_value:
                raise ValueError('warm session checkpoint, tokenizer, libraries, context and budgets cannot change')
            started = time.perf_counter()
            warm_reused = self._requests > 0
            self._requests += 1
            self._reset_request()
            try:
                class RequestCancellation:
                    def is_set(self):return cancelled is not None and bool(cancelled())
                request_cancel=RequestCancellation()
                with self.arithmetic.cancellation_scope(request_cancel),self.model.cancellation_scope(request_cancel):
                    result = _run_request(self, args, emit=emit, cancelled=cancelled,
                        checkpoint_sink=checkpoint_sink, max_checkpoint_bytes=max_checkpoint_bytes)
            finally:
                self._reset_request()
            result['warm_reused'] = warm_reused
            result['total_ms'] = (time.perf_counter()-started)*1000
            return result
        finally:
            self._lock.release()

    @staticmethod
    def _checkpoint_options(sink, maximum):
        if sink is None:
            if maximum not in (None, 0):
                raise ValueError('V4.1 checkpoint budget requires a sink')
        elif not callable(sink) or type(maximum) is not int or maximum <= 128:
            raise ValueError('synchronous checkpoint sink and complete frame budget required')

    def resume_generation(self, source, *, max_bytes, max_new_tokens=None, emit=None,
                          cancelled=None, checkpoint_sink=None, max_checkpoint_bytes=0):
        from .execution_service import ExecutionCancelled
        from .v41_generation_recovery import read, publish
        from .v41_execution.dspark import create_dspark_decoder
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('publisher session is busy or reentrant')
        def check():
            if cancelled is not None and cancelled():
                raise ExecutionCancelled('publisher recovery cancelled')
        try:
            if self._closed:
                raise RuntimeError('publisher session is closed')
            self._checkpoint_options(checkpoint_sink, max_checkpoint_bytes)
            args = copy.deepcopy(self._admitted_args)
            limit = args.max_tokens if max_new_tokens is None else max_new_tokens
            if type(limit) is not int or limit <= 0 or not callable(getattr(source, 'read', None)):
                raise ValueError('bounded continuation and binary checkpoint source required')
            check(); self._reset_request()
            class RequestCancellation:
                def is_set(self): return cancelled is not None and bool(cancelled())
            try:
                with self.arithmetic.cancellation_scope(RequestCancellation()), self.model.cancellation_scope(RequestCancellation()):
                    header, logits, cache, size = read(source, model=self.model,
                        tokenizer=self._tokenizer_identity, max_bytes=max_bytes,
                        host_bytes=args.sampling_mib*1024**2, cancelled=cancelled)
                    try:
                        if header['position'] + limit > args.context:
                            raise ValueError('recovered V4.1 continuation exceeds context')
                        if (header['position'] + limit)*128 + self.model.profile.vocabulary*32 > args.sampling_mib*1024**2:
                            raise MemoryError('recovered V4.1 history exceeds sampling admission')
                        history_reservation=(header['position']+limit)*128
                        target_rng, draft_rng = [np.random.default_rng(seed) for seed in
                            np.random.SeedSequence(args.seed).spawn(2)]
                        target_rng.bit_generator.state = copy.deepcopy(header['rng'])
                        owner = self.model
                        decoder = None
                        if header['dspark']:
                            draft = self._draft(args, draft_rng)
                            decoder = create_dspark_decoder(self.model, draft,
                                temperature=header['temperature'], max_bytes=args.sampling_mib*1024**2,
                                evidence_max_bytes=args.evidence_mib*1024**2,
                                target_rng=target_rng, proposal_rng=draft_rng)
                            owner = decoder
                        check(); owner.restore_cache(cache, max_bytes=size); check()
                        if cache.tell() != size or self.model.length != header['position']:
                            raise ValueError('recovered V4.1 physical cache extent differs')
                        if tuple(self.model.snapshot().tokens) != tuple(header['prompt_ids'] + header['generated_ids']):
                            raise ValueError('recovered V4.1 token history differs')
                        if decoder is not None and (decoder.history != tuple(header['prompt_ids'] + header['generated_ids'])
                                or not np.array_equal(decoder.next_logits, logits)
                                or decoder.rng.bit_generator.state != header['rng']):
                            raise ValueError('compound DSpark verifier boundary differs from envelope')
                    finally:
                        cache.close()
                    generated = list(header['generated_ids'])
                    stops = frozenset(header['stops'])
                    eos = self.checkpoint.config['text_config'].get('eos_token_id', [])
                    eos = [eos] if type(eos) is int else eos
                    if not set(eos) <= stops:
                        raise ValueError('V4.1 checkpoint omitted configured EOS boundaries')
                    if emit is not None:
                        emit(dict(event='recovered', text=self.tokenizer.decode(generated), position=self.model.length))
                    def record(tokens, accepted=0, proposed=0):
                        nonlocal logits
                        generated.extend(tokens)
                        header['generated_ids'] = list(generated)
                        header['position'] = self.model.length
                        header['accepted'] += accepted; header['proposed'] += proposed
                        header['rng'] = copy.deepcopy(target_rng.bit_generator.state)
                        if decoder is not None: logits = decoder.next_logits
                        if checkpoint_sink is not None:
                            publish(checkpoint_sink, header, logits, owner, model=self.model,
                                tokenizer=self._tokenizer_identity, max_bytes=max_checkpoint_bytes,
                                host_bytes=args.sampling_mib*1024**2, cancelled=cancelled)
                        if emit is not None:
                            emit(dict(event='tokens', token_ids=list(tokens), text=self.tokenizer.decode(generated)))
                    if not generated or generated[-1] not in stops:
                        if decoder is not None:
                            for step in decoder.generate(limit, stop_tokens=stops, cancelled=cancelled):
                                record(step.tokens, step.accepted, step.proposed)
                        else:
                            for _ in range(limit):
                                check()
                                token = _sample(logits, header['temperature'], target_rng,
                                    max_bytes=args.sampling_mib*1024**2-history_reservation)
                                logits = self.model.forward([token])[-1]
                                check(); record((token,))
                                if token in stops: break
                    check()
                    self._requests += 1
                    return dict(schema='openagent.publisher-v41-generation.v1',
                        prompt_ids=header['prompt_ids'], generated_ids=generated,
                        text=self.tokenizer.decode(generated), dspark=header['dspark'],
                        accepted_draft_tokens=header['accepted'], proposed_draft_tokens=header['proposed'],
                        recovered=True, implementation=self.model.implementation,
                        numerical_parity_verified=False, platform_certified=False)
            finally:
                self._reset_request()
        finally:
            self._lock.release()

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('cannot close an active publisher session')
        try:
            self._closed = True
            self._owned.close()
            self._vision_model = self._draft_model = None
        finally:
            self._lock.release()

    def __enter__(self):
        if self._closed:
            raise RuntimeError('publisher session is closed')
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def run(args, emit=None, cancelled=None):
    from .execution_service import ExecutionCancelled
    if cancelled is not None and cancelled():
        raise ExecutionCancelled('native generation cancelled')
    with PublisherSession(args, cancelled=cancelled) as session:
        save=getattr(args,'save_generation',None);resume=getattr(args,'resume_generation',None)
        checkpoint_mib=getattr(args,'checkpoint_mib',None)
        if (save is not None or resume is not None) and (type(checkpoint_mib) is not int or checkpoint_mib<=0):
            raise ValueError('generation files require an explicit complete checkpoint MiB budget')
        if checkpoint_mib is not None and save is None and resume is None:
            raise ValueError('checkpoint admission requires save or resume generation')
        maximum=0 if checkpoint_mib is None else checkpoint_mib*1024**2
        from .generation_checkpoint_files import AtomicGenerationCheckpoint
        with ExitStack() as owned:
            sink=None if save is None else AtomicGenerationCheckpoint(save,max_bytes=maximum)
            if sink is not None:owned.callback(sink.close)
            if resume is None:
                result=session.run(args,emit=emit,cancelled=cancelled,
                    checkpoint_sink=sink,max_checkpoint_bytes=maximum if sink is not None else 0)
            else:
                reader=AtomicGenerationCheckpoint(resume,max_bytes=maximum)
                owned.callback(reader.close)
                with reader.source() as stream:
                    result=session.resume_generation(stream,max_bytes=maximum,max_new_tokens=args.max_tokens,
                        emit=emit,cancelled=cancelled,checkpoint_sink=sink,
                        max_checkpoint_bytes=maximum if sink is not None else 0)
            if sink is not None:result['checkpoint_receipt']=sink.receipt
        result['load_ms'] = session.load_ms
        result['total_ms'] = result.get('total_ms',0.0)+session.load_ms
        return result


def main(argv=None):
    args = parser().parse_args(argv)
    def emit(event):
        print(json.dumps(event, allow_nan=False), flush=True)
    try:
        result = run(args, emit=emit if args.stream else None)
        print(json.dumps(dict(event='completed', **result), allow_nan=False), flush=True)
        return 0
    except KeyboardInterrupt:
        print('Generation cancelled', file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, ImportError, KeyError, MemoryError, ArithmeticError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
