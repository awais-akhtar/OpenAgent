"""Opt-in gateway bridge for explicitly configured local publisher decoders."""
import json
import math
import os
from pathlib import Path
import threading
import time
from contextlib import contextmanager
from queue import Empty, Full, Queue

from ..model_gateway import BaseProvider, ModelResult, StreamChunk, ProviderConfigurationError, ProviderUnavailableError
from .session_recovery import _path
from .native_library_identity import _stamp


def _metadata(path,max_bytes):
    from .publisher_media_files import read_regular
    try:return read_regular(path,max_bytes=max_bytes)
    except MemoryError as error:
        raise ProviderConfigurationError('native metadata exceeds configured admission') from error


def _fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate native provider configuration field: '+key)
        result[key] = value
    return result


class NativePublisherProvider(BaseProvider):
    id = 'native-publisher'
    label = 'Native Publisher Engine'
    local = True
    network = False
    default_model = 'configured-native-checkpoint'
    privacy_warning = 'Uses explicitly configured local checkpoint files and native libraries; no network requests.'

    def __init__(self):
        self.path = os.environ.get('OPENAGENT_NATIVE_PUBLISHER_CONFIG', '').strip()
        self._lock = threading.RLock()
        self._session = None
        self._session_key = None
        self._closed = False
        self._active = False
        self._loaded_at = self._last_used = None
        self._idle_seconds = self._lifetime_seconds = None
        self._cleanup_error = None
        self._construction_cleanup = None

    @staticmethod
    def _residency_key(config, args):
        root = _path(args.checkpoint).resolve(strict=True)
        paths = [root/'config.json', root/'model.safetensors.index.json',
                 args.tokenizer or root/'tokenizer.json']
        paths.extend(root.glob('*.safetensors'))
        if getattr(args,'family',None) in ('qwen3_5_moe','qwen4_exp','inkling_mm_model','glm5_next'):
            paths.extend(root.glob('*.json'))
            paths.extend(root.glob('*.jinja'))
            paths.extend(root/name for name in ('merges.txt','tokenizer.model') if (root/name).exists())
        ple_manifest = getattr(args,'ple_manifest',None)
        if ple_manifest is not None:
            manifest_path = _path(ple_manifest).resolve(strict=True)
            if not manifest_path.is_relative_to(root):
                raise ProviderConfigurationError('PLE manifest escapes admitted checkpoint root')
            paths.append(manifest_path)
        expert_manifest = getattr(args,'expert_manifest',None)
        if expert_manifest is not None:
            from .expert_runtime import SparseExpertManifest
            manifest_path = _path(expert_manifest).resolve(strict=True)
            if not manifest_path.is_relative_to(root):
                raise ProviderConfigurationError('expert manifest escapes admitted checkpoint root')
            manifest = SparseExpertManifest.read(manifest_path)
            paths.append(manifest_path)
            for layer in manifest.layers:
                for region in layer.regions.values():
                    path = _path(root/region.path).resolve(strict=True)
                    if region.storage!='' or not path.is_relative_to(root):
                        raise ProviderConfigurationError('expert container requires admitted local storage')
                    paths.append(path)
            for filename in ('openagent-iq3-grid.bin','openagent-e8-contract.bin'):
                path = root/filename
                if path.is_file():
                    paths.append(path)
        index_path = root/'model.safetensors.index.json'
        if index_path.is_file():
            maximum = getattr(args, 'metadata_mib', 16)*1024**2
            raw=_metadata(index_path,maximum)
            if len(raw) > maximum:
                raise ProviderConfigurationError('checkpoint index exceeds metadata budget')
            index = json.loads(raw, object_pairs_hook=_fields)
            mapping = index.get('weight_map') if type(index) is dict else None
            if type(mapping) is not dict or any(type(v) is not str for v in mapping.values()):
                raise ProviderConfigurationError('checkpoint index requires a shard map')
            for name in set(mapping.values()):
                shard = _path(root/name).resolve(strict=True)
                if not shard.is_relative_to(root):
                    raise ProviderConfigurationError('checkpoint shard escapes its directory')
                paths.append(shard)
        paths.extend(value for name, value in vars(args).items()
                     if name.endswith('_library') and value is not None)
        stamps = []
        for path in sorted(set(_path(p).resolve() for p in paths), key=str):
            if not path.is_file():
                if path.name == 'model.safetensors.index.json':
                    continue
                raise ProviderConfigurationError('native artifact missing: '+str(path))
            stamps.append((str(path), *_stamp(_path(path).stat())))
        return json.dumps(config, sort_keys=True), tuple(stamps)

    def _release_session(self):
        self._session_key = None
        if self._session is not None:
            try:
                self._session.close()
            except BaseException as error:
                self._cleanup_error = str(error)
                raise
            self._session = None
        if self._construction_cleanup is not None:
            try:self._construction_cleanup()
            except BaseException as error:
                self._cleanup_error=str(error)
                raise
            self._construction_cleanup=None
        self._loaded_at = self._last_used = None
        self._cleanup_error = None

    def _construct_session(self,operator,args,cancelled):
        if self._construction_cleanup is not None:
            raise ProviderUnavailableError('failed native construction cleanup remains pending')
        try:return operator.PublisherSession(args,cancelled=cancelled)
        except BaseException as primary:
            cleanup=getattr(primary,'cleanup',None)
            if callable(cleanup):
                try:cleanup()
                except BaseException as failure:
                    self._construction_cleanup=cleanup
                    self._cleanup_error=str(failure)
                    primary.add_note('gateway retains failed construction cleanup: '+repr(failure))
                else:primary.cleanup=None
            raise

    def close(self, cancelled=None):
        with self._lifecycle_lock(cancelled):
            self._closed = True
            self._release_session()

    def unload(self, cancelled=None):
        with self._lifecycle_lock(cancelled):
            self._release_session()

    def _evict_expired(self):
        if self._active:
            return
        if self._construction_cleanup is not None:self._release_session()
        if self._session is None:return
        now = time.monotonic()
        if (self._cleanup_error is not None or
                (self._idle_seconds is not None and self._last_used is not None and
                 now-self._last_used >= self._idle_seconds) or
                (self._lifetime_seconds is not None and self._loaded_at is not None and
                 now-self._loaded_at >= self._lifetime_seconds)):
            self._release_session()

    def status(self):
        if not self._lock.acquire(blocking=False):
            return dict(closed=self._closed, active=self._active, busy=True,
                        resident=self._session is not None,
                        eviction_policy='lazy', cleanup_pending=self._cleanup_error is not None)
        try:
            if not self._active:
                try:
                    self._evict_expired()
                except Exception:
                    pass
            now = time.monotonic()
            return dict(closed=self._closed, active=self._active, busy=self._active,
                        resident=self._session is not None,
                        reusable=not self._closed and self._session_key is not None and
                                 self._cleanup_error is None,
                        cleanup_pending=self._cleanup_error is not None,
                        cleanup_error=self._cleanup_error, eviction_policy='lazy',
                        idle_timeout_seconds=self._idle_seconds,
                        max_resident_seconds=self._lifetime_seconds,
                        resident_age_seconds=None if self._loaded_at is None else now-self._loaded_at,
                        idle_seconds=None if self._active or self._last_used is None else now-self._last_used)
        finally:
            self._lock.release()

    def __enter__(self):
        if self._closed:
            raise ProviderUnavailableError('native provider is closed')
        return self

    def __exit__(self, *_):
        self.close()

    def _configuration(self):
        if not self.path:
            raise ProviderConfigurationError('OPENAGENT_NATIVE_PUBLISHER_CONFIG is not set')
        raw=_metadata(self.path,128*1024)
        if len(raw) > 128*1024:
            raise ProviderConfigurationError('native provider configuration exceeds 128 KiB')
        config = json.loads(raw, object_pairs_hook=_fields,
            parse_constant=lambda value: (_ for _ in ()).throw(ProviderConfigurationError('nonfinite provider field: '+value)))
        if (type(config) is not dict or set(config)-
                {'family', 'model', 'argv', 'chat_template', 'template_variables',
                 'idle_timeout_seconds', 'max_resident_seconds', 'image_placeholder',
                 'video_placeholder','video_token_id'} or
                config.get('family') not in ('deepseek_v41', 'kimi_k3','mixtral','qwen3_5_moe_text','qwen3_5_moe','qwen4_exp_text','glm5_next_text','inkling_text','qwen4_exp','inkling_mm_model','glm5_next','glm_moe_dsa') or
                not isinstance(config.get('model'), str) or not config['model'].strip() or
                not isinstance(config.get('chat_template'), str) or not config['chat_template'].strip() or
                type(config.get('argv')) is not list or not config['argv'] or
                any(type(arg) is not str or not arg or '\0' in arg for arg in config['argv'])):
            raise ProviderConfigurationError('explicit family, model, argv and chat template required')
        placeholder = config.get('image_placeholder')
        if placeholder is not None and (type(placeholder) is not str or not placeholder or
                                         len(placeholder.encode('utf-8')) > 4096):
            raise ProviderConfigurationError('image_placeholder must be a bounded nonempty string')
        video_placeholder=config.get('video_placeholder');video_token=config.get('video_token_id')
        if ((video_placeholder is None)!=(video_token is None) or (video_placeholder is not None and
                (type(video_placeholder) is not str or not video_placeholder or len(video_placeholder.encode())>4096
                 or type(video_token) is not int or video_token<0))):
            raise ProviderConfigurationError('paired bounded video placeholder and explicit token ID required')
        for field in ('idle_timeout_seconds', 'max_resident_seconds'):
            value = config.get(field)
            if value is not None and (type(value) not in (int, float) or
                                      not math.isfinite(value) or value <= 0):
                raise ProviderConfigurationError(field+' must be null or positive finite seconds')
        variables = config.get('template_variables', {})
        if (type(variables) is not dict or
                any(type(k) is not str or type(v) not in (str, int, bool) for k, v in variables.items()) or
                {'messages', 'add_generation_prompt'} & variables.keys()):
            raise ProviderConfigurationError('invalid native chat template variables')
        self.default_model = config['model']
        return config

    @staticmethod
    def _operator(config):
        if any(arg in ('-h', '--help', '--prompt', '--prompt-text', '--token-ids') or
                arg.startswith(('--prompt=', '--prompt-text=', '--token-ids=')) for arg in config['argv']):
            raise ProviderConfigurationError('stored operator arguments cannot contain help or prompt inputs')
        if config['family'] == 'deepseek_v41':
            from . import publisher_v41_cli as operator
        elif config['family']=='kimi_k3':
            from . import publisher_k3_cli as operator
        elif config['family'] in ('qwen3_5_moe','qwen4_exp','inkling_mm_model') or (config['family']=='glm5_next' and
                any(arg=='--media-library' or arg.startswith('--media-library=') for arg in config['argv'])):
            from . import native_conditional_cli as operator
        else:
            from . import native_family_cli as operator
        parser = operator.parser()
        def reject(message):
            raise ProviderConfigurationError(message)
        parser.error = reject
        # Prompt tokens are supplied after rendering, never from stored argv.
        family_args = [] if config['family'] in ('deepseek_v41','kimi_k3') else ['--family',config['family']]
        args = parser.parse_args([*config['argv'],*family_args,'--token-ids','[]'])
        if any(getattr(args,name,None) for name in ('image','audio_wav','video_manifest')):
            raise ProviderConfigurationError('gateway requests cannot use stored media inputs')
        if getattr(args, 'max_tokens', 0) > getattr(args, 'context', 0):
            raise ProviderConfigurationError('generation exceeds configured native context')
        return operator, args

    def available(self):
        if self._closed or self._cleanup_error is not None:
            return False
        try:
            config = self._configuration()
            _, args = self._operator(config)
            root = _path(args.checkpoint).resolve(strict=True)
            if not root.is_dir() or not (root/'config.json').is_file():
                return False
            tokenizer = _path(args.tokenizer or root/'tokenizer.json').resolve(strict=True)
            if not tokenizer.is_file():
                return False
            if config['family']=='deepseek_v41':
                if getattr(args,'device',None) is None and any(getattr(args,name,None) is None for name in
                        ('execution_library','mxfp4_library','fp8_library')):
                    return False
                if getattr(args,'device',None) is not None and not getattr(args,'torch_version',None):
                    return False
            if config['family']=='kimi_k3':
                if getattr(args,'arithmetic_device',None) is None and any(getattr(args,name,None) is None for name in
                        ('kimi_library','attention_library','operators_library','mxfp4_library')):
                    return False
                if getattr(args,'arithmetic_device',None) is not None and not getattr(args,'torch_version',None):
                    return False
            if hasattr(args,'kernel_library') and getattr(args,'device',None) is None:
                if args.kernel_library is None:
                    return False
                if config['family'] not in ('glm5_next_text','inkling_text','glm5_next','glm_moe_dsa') and args.runtime_library is None:
                    return False
                if config['family']=='glm5_next' and getattr(args,'expert_library',None) is None:
                    return False
            for name, value in vars(args).items():
                # Required argparse library arguments have already been
                # admitted. Optional projection/store/segment libraries are
                # checked when supplied; absence must not disable other paths.
                if name.endswith('_library') and value is not None and not _path(value).is_file():
                    return False
            return True
        except (ValueError, OSError, TypeError, ImportError, RuntimeError):
            return False

    @contextmanager
    def _lifecycle_lock(self, cancelled):
        from .execution_service import ExecutionCancelled
        while not self._lock.acquire(timeout=0.1):
            if cancelled is not None and cancelled():
                raise ExecutionCancelled('cancelled while waiting for native model')
        try:
            if cancelled is not None and cancelled():
                raise ExecutionCancelled('native request cancelled')
            if self._active:
                raise ProviderUnavailableError('cannot reenter or unload an active native request')
            yield
        finally:
            self._lock.release()

    @contextmanager
    def _request_lock(self, cancelled):
        with self._lifecycle_lock(cancelled):
            self._evict_expired()
            self._active = True
            failed = False
            try:
                yield
            except BaseException as error:
                failed = True
                try:
                    self._release_session()
                except BaseException as cleanup_error:
                    if hasattr(error, 'add_note'):
                        error.add_note('native session cleanup remains pending: '+str(cleanup_error))
                raise
            finally:
                self._active = False
                if self._session is not None and self._cleanup_error is None:
                    self._last_used = time.monotonic()
                if not failed:
                    # Eviction errors retain ownership for a later explicit retry.
                    # They do not turn an already completed generation into failure.
                    try:
                        self._evict_expired()
                    except Exception:
                        pass

    def complete(self, messages, *, model=None):
        return self._run(messages, model=model)

    def supports_vision(self):
        try:
            config = self._configuration()
            _, args = self._operator(config)
            return (not self._closed and bool(config.get('image_placeholder')) and
                    type(getattr(args, 'image_token_id', None)) is int and
                    (config['family'] == 'deepseek_v41' or getattr(args,'include_vision',False)))
        except (ValueError, OSError, TypeError, ImportError, RuntimeError):
            return False

    def complete_vision(self, messages, image, *, media_type='image/png', model=None,
                        max_output_tokens=None):
        self._validate_vision_input(image, media_type, max_output_tokens)
        return self._run(messages, model=model, image=image,
                         max_output_tokens=max_output_tokens)

    @staticmethod
    def _validate_vision_input(image, media_type, max_output_tokens):
        if (type(image) is not bytes or not image or len(image) > 256*1024**2 or
                type(media_type) is not str or not media_type.startswith('image/')):
            raise ProviderConfigurationError('bounded encoded image bytes and image MIME type required')
        if max_output_tokens is not None and (type(max_output_tokens) is not int or max_output_tokens <= 0):
            raise ProviderConfigurationError('positive integer vision output limit required')

    def stream_complete_vision(self, messages, image, *, media_type='image/png', model=None,
                               max_output_tokens=None):
        self._validate_vision_input(image, media_type, max_output_tokens)
        return self._stream_run(messages, model=model, image=image,
                                max_output_tokens=max_output_tokens)

    @staticmethod
    def _validate_decoded_media(items,max_output_tokens):
        from .k3_execution.video_input import K3VideoInput,_dimensions
        from .publisher_media_inputs import PublisherVideoInput
        if type(items) not in (tuple,list) or not 1<=len(items)<=4096:
            raise ProviderConfigurationError('bounded decoded media sequence required')
        if max_output_tokens is not None and (type(max_output_tokens) is not int or max_output_tokens<=0):
            raise ProviderConfigurationError('positive media output limit required')
        total=0
        for item in items:
            frames=item.frames if type(item) in (K3VideoInput,PublisherVideoInput) else (item,)
            for frame in frames:
                if getattr(frame,'n_frames',1)!=1:
                    raise ProviderConfigurationError('animated media requires explicitly selected decoded frames')
                width,height=_dimensions(frame)
                if width<=0 or height<=0:raise ProviderConfigurationError('positive decoded media dimensions required')
                total+=width*height*16
                if total>256*1024**2:raise ProviderConfigurationError('decoded media request exceeds admission')
        return tuple(items),total

    def complete_media(self,messages,media_items,*,model=None,max_output_tokens=None):
        items,_=self._validate_decoded_media(media_items,max_output_tokens)
        items=self._freeze_decoded_media(items)
        return self._run(messages,model=model,media_items=items,max_output_tokens=max_output_tokens)

    def stream_complete_media(self,messages,media_items,*,model=None,max_output_tokens=None):
        items,_=self._validate_decoded_media(media_items,max_output_tokens)
        items=self._freeze_decoded_media(items)
        return self._stream_run(messages,model=model,media_items=items,max_output_tokens=max_output_tokens)

    @staticmethod
    def _freeze_decoded_media(items):
        """Detach asynchronous worker inputs from caller-owned frame buffers."""
        import numpy as np
        from .k3_execution.video_input import K3VideoInput
        from .publisher_media_inputs import PublisherVideoInput
        def frame_copy(frame):
            if isinstance(frame,np.ndarray):
                return np.frombuffer(frame.tobytes(order='C'),dtype=np.uint8).reshape(frame.shape)
            return frame.copy()
        result=[]
        for item in items:
            if type(item) is K3VideoInput:
                result.append(K3VideoInput(tuple(frame_copy(frame) for frame in item.frames),
                    item.timestamps,item.selection_identity))
            elif type(item) is PublisherVideoInput:
                result.append(PublisherVideoInput(tuple(frame_copy(frame) for frame in item.frames),
                    item.frames_indices,item.total_num_frames,item.fps))
            else:result.append(frame_copy(item))
        return tuple(result)

    def complete_checkpointed(self,messages,*,checkpoint_sink,checkpoint_max_bytes,
                              model=None,max_output_tokens=None,media_items=None):
        """Generate locally while a caller-owned store publishes recovery frames."""
        if not callable(checkpoint_sink) or type(checkpoint_max_bytes) is not int or checkpoint_max_bytes<=0:
            raise ProviderConfigurationError('checkpoint sink and positive complete frame budget required')
        if media_items is not None:
            self._validate_decoded_media(media_items,max_output_tokens)
            media_items=self._freeze_decoded_media(media_items)
        return self._run(messages,model=model,max_output_tokens=max_output_tokens,media_items=media_items,
            checkpoint_sink=checkpoint_sink,checkpoint_max_bytes=checkpoint_max_bytes)

    def stream_complete_checkpointed(self,messages,*,checkpoint_sink,checkpoint_max_bytes,
                                     model=None,max_output_tokens=None,media_items=None):
        if not callable(checkpoint_sink) or type(checkpoint_max_bytes) is not int or checkpoint_max_bytes<=0:
            raise ProviderConfigurationError('checkpoint sink and positive complete frame budget required')
        if media_items is not None:
            self._validate_decoded_media(media_items,max_output_tokens)
            media_items=self._freeze_decoded_media(media_items)
        yield from self._stream_run(messages,model=model,max_output_tokens=max_output_tokens,
            media_items=media_items,checkpoint_sink=checkpoint_sink,checkpoint_max_bytes=checkpoint_max_bytes)

    def complete_checkpoint_file(self,messages,destination,*,checkpoint_max_bytes,
                                 model=None,max_output_tokens=None,media_items=None):
        """Publish complete recovery frames to one caller-selected local file."""
        from .generation_checkpoint_files import AtomicGenerationCheckpoint
        store=AtomicGenerationCheckpoint(destination,max_bytes=checkpoint_max_bytes)
        try:
            result=self.complete_checkpointed(messages,checkpoint_sink=store,
                checkpoint_max_bytes=checkpoint_max_bytes,model=model,
                max_output_tokens=max_output_tokens,media_items=media_items)
            metadata=dict(result.metadata,checkpoint_receipt=store.receipt)
            return ModelResult(result.provider,result.model,result.content,metadata)
        finally:store.close()

    def resume_checkpoint_file(self,source,*,checkpoint_max_bytes,max_output_tokens,
                               receipt=None,destination=None,model=None,emit=None,cancelled=None):
        """Resume a stable local frame and optionally publish later boundaries.

        Source and destination can name the same file: capture releases its
        read lease before the request publishes new frames.
        """
        from .generation_checkpoint_files import AtomicGenerationCheckpoint
        reader=AtomicGenerationCheckpoint(source,max_bytes=checkpoint_max_bytes)
        writer=None
        try:
            if destination is not None:
                writer=AtomicGenerationCheckpoint(destination,max_bytes=checkpoint_max_bytes)
            with reader.source(receipt=receipt) as stream:
                result=self.resume_checkpoint(stream,checkpoint_max_bytes=checkpoint_max_bytes,
                    max_output_tokens=max_output_tokens,checkpoint_sink=writer,model=model,
                    emit=emit,cancelled=cancelled)
            return ModelResult(result.provider,result.model,result.content,
                dict(result.metadata,checkpoint_receipt=None if writer is None else writer.receipt))
        finally:
            try:
                if writer is not None:writer.close()
            finally:reader.close()

    def stream_resume_checkpoint(self,source,*,checkpoint_max_bytes,max_output_tokens,
                                 model=None,checkpoint_sink=None):
        """Stream recovered tokens using the same cancellation and bounded queue."""
        yield from self._stream_run(None,model=model,max_output_tokens=max_output_tokens,
            recovery_source=source,checkpoint_sink=checkpoint_sink,checkpoint_max_bytes=checkpoint_max_bytes)

    def resume_checkpoint(self,source,*,checkpoint_max_bytes,max_output_tokens,model=None,
                          checkpoint_sink=None,emit=None,cancelled=None):
        """Restore a complete supported decoder and sampler boundary under gateway ownership."""
        if (type(checkpoint_max_bytes) is not int or checkpoint_max_bytes<=0
                or type(max_output_tokens) is not int or not 1<=max_output_tokens<=100000
                or (checkpoint_sink is not None and not callable(checkpoint_sink))):
            raise ProviderConfigurationError('bounded recovery and continuation options required')
        with self._request_lock(cancelled):
            if self._closed:raise ProviderUnavailableError('native provider is closed')
            config=self._configuration()
            if model is not None and model!=config['model']:
                raise ProviderConfigurationError('requested model differs from configured checkpoint')
            self._idle_seconds=config.get('idle_timeout_seconds')
            self._lifetime_seconds=config.get('max_resident_seconds')
            operator,args=self._operator(config)
            key=self._residency_key(config,args)
            reused=self._session is not None and self._session_key==key
            if not reused:
                self._release_session()
                started=time.monotonic()
                try:self._session=self._construct_session(operator,args,cancelled)
                except (ValueError,OSError,MemoryError,RuntimeError,ImportError) as error:
                    raise ProviderUnavailableError('recovery session could not load: '+str(error)) from error
                self._session_key=key;self._loaded_at=started;self._last_used=time.monotonic()
            prior='';emitted=''
            def event(row):
                nonlocal prior,emitted
                if row.get('event')=='recovered':prior=row['text'];emitted=prior;return
                if row.get('event')!='tokens' or emit is None:return
                text=row['text']
                if text.endswith('\ufffd'):return
                if not text.startswith(emitted):
                    raise ProviderUnavailableError('recovery tokenizer changed streamed prefix')
                delta=text[len(emitted):]
                if delta:emit(StreamChunk(self.id,config['model'],delta,False,
                    dict(network=False,offline=True,recovered=True,token_ids=list(row['token_ids']))))
                emitted=text
            try:
                if config['family']=='kimi_k3':
                    output=self._session.resume_generation(source,checkpoint_max_bytes=checkpoint_max_bytes,
                        max_tokens=min(max_output_tokens,args.max_tokens),checkpoint_sink=checkpoint_sink,
                        emit=event,cancelled=cancelled)
                else:
                    output=self._session.resume_generation(source,max_bytes=checkpoint_max_bytes,
                        max_new_tokens=min(max_output_tokens,args.max_tokens),checkpoint_sink=checkpoint_sink,
                        max_checkpoint_bytes=checkpoint_max_bytes if checkpoint_sink is not None else 0,
                        emit=event,cancelled=cancelled)
                content=output['text']
                if not content.startswith(prior) or not content.startswith(emitted):
                    raise ProviderUnavailableError('final recovery output changed its captured prefix')
                if emit is not None and len(content)>len(emitted):
                    emit(StreamChunk(self.id,config['model'],content[len(emitted):],False,
                        dict(network=False,offline=True,recovered=True)))
                return ModelResult(self.id,config['model'],content[len(prior):],
                    dict(network=False,offline=True,recovered=True,family=config['family'],runtime=output,
                        resident_session_reused=reused,numerical_parity_verified=False,platform_certified=False))
            except (ValueError,OSError,MemoryError,RuntimeError,ArithmeticError) as error:
                try:self._release_session()
                except Exception as cleanup:error.add_note('recovery session release failed: '+repr(cleanup))
                raise ProviderUnavailableError('native recovery failed: '+str(error)) from error

    def _run(self, messages, *, model=None, emit=None, cancelled=None, image=None,
             max_output_tokens=None,media_items=None,checkpoint_sink=None,checkpoint_max_bytes=None):
        if image is not None and media_items is not None:
            raise ProviderConfigurationError('one encoded image or decoded media sequence required')
        try:
            from jinja2 import StrictUndefined, TemplateError
            from jinja2.sandbox import SandboxedEnvironment
        except ImportError as error:
            raise ProviderConfigurationError('native chat requires local Jinja2') from error
        if (type(messages) is not list or not messages or len(messages) > 10000 or
                any(type(item) is not dict or item.get('role') not in
                    ('system', 'user', 'assistant', 'tool') or type(item.get('content')) is not str
                    for item in messages)):
            raise ProviderConfigurationError('validated textual chat messages required')
        if sum(len(item['content'].encode('utf-8')) for item in messages) > 1024**2:
            raise ProviderConfigurationError('native chat input exceeds 1 MiB')
        with self._request_lock(cancelled):
            if self._closed:
                raise ProviderUnavailableError('native provider is closed')
            config = self._configuration()
            self._idle_seconds = config.get('idle_timeout_seconds')
            self._lifetime_seconds = config.get('max_resident_seconds')
            if model is not None and model != config['model']:
                raise ProviderConfigurationError('requested model differs from configured native checkpoint')
            operator, args = self._operator(config)
            rendered_messages = [{'role': item['role'], 'content': item['content']} for item in messages]
            if image is not None or media_items is not None:
                from .publisher_media_inputs import PublisherVideoInput
                videos=[] if media_items is None else [item for item in media_items if type(item) is PublisherVideoInput]
                if videos and config['family'] not in ('qwen3_5_moe','qwen4_exp','glm5_next'):
                    raise ProviderConfigurationError('publisher frame-index video requires Qwen4 or GLM5 vision')
                if videos and config.get('video_placeholder') is None:
                    raise ProviderConfigurationError('publisher video requires configured video placeholder and token ID')
                if media_items is not None and config['family']!='kimi_k3':
                    from .k3_execution.video_input import K3VideoInput
                    if config['family'] not in ('deepseek_v41','qwen3_5_moe','qwen4_exp','inkling_mm_model','glm5_next'):
                        raise ProviderConfigurationError('decoded media requires an admitted vision family')
                    if any(type(item) is K3VideoInput for item in media_items):
                        raise ProviderConfigurationError('K3 temporal layouts require the K3 processor; other video families need publisher frame metadata')
                placeholder = config.get('image_placeholder')
                if (not placeholder or type(getattr(args, 'image_token_id', None)) is not int or
                        (config['family'] == 'kimi_k3' and not args.include_vision)):
                    raise ProviderConfigurationError('native vision requires configured image placeholder, token ID and vision weights')
                maximum = getattr(args, 'image_mib', None)
                media_bytes=(len(image) if image is not None else self._validate_decoded_media(media_items,max_output_tokens)[1])
                if type(maximum) is not int or maximum <= 0:
                    raise ProviderConfigurationError('encoded image exceeds configured native media budget')
                if image is not None:
                    if media_bytes>maximum*1024**2:
                        raise ProviderConfigurationError('encoded image exceeds configured native media budget')
                else:
                    from .k3_execution.video_input import K3VideoInput,_dimensions
                    for item in media_items:
                        frames=item.frames if type(item) in (K3VideoInput,PublisherVideoInput) else (item,)
                        if any(_dimensions(frame)[0]*_dimensions(frame)[1]*16>maximum*1024**2 for frame in frames):
                            raise ProviderConfigurationError('decoded frame exceeds configured native image budget')
                    aggregate=getattr(args,'media_mib',None)
                    if aggregate is not None and media_bytes>aggregate*1024**2:
                        raise ProviderConfigurationError('decoded media exceeds configured aggregate budget')
                if any(placeholder in item['content'] or
                       (videos and config['video_placeholder'] in item['content'])
                       for item in rendered_messages):
                    raise ProviderConfigurationError('vision placeholder must be inserted by the gateway')
                index = next((i for i in range(len(rendered_messages)-1, -1, -1)
                              if rendered_messages[i]['role'] == 'user'), None)
                if index is None:
                    raise ProviderConfigurationError('native vision requires a user message')
                markers=([placeholder] if image is not None else
                    [config['video_placeholder'] if type(item) is PublisherVideoInput else placeholder for item in media_items])
                rendered_messages[index]['content'] += '\n'+'\n'.join(markers)
            if max_output_tokens is not None:
                args.max_tokens = min(args.max_tokens, max_output_tokens)
            env = SandboxedEnvironment(undefined=StrictUndefined, autoescape=False)
            def raise_exception(message):
                raise ProviderConfigurationError(str(message))
            env.globals['raise_exception'] = raise_exception
            try:
                prompt = env.from_string(config['chat_template']).render(
                    messages=rendered_messages,
                    add_generation_prompt=True, **config.get('template_variables', {}))
            except TemplateError as error:
                raise ProviderConfigurationError('native chat template could not render: '+str(error)) from error
            if len(prompt.encode('utf-8')) > 2*1024**2:
                raise ProviderConfigurationError('rendered native prompt exceeds 2 MiB')
            key = self._residency_key(config, args)
            reused = self._session is not None and self._session_key == key
            if not reused:
                self._release_session()
                loading_started = time.monotonic()
                try:
                    self._session = self._construct_session(operator,args,cancelled)
                except (ValueError, OSError, MemoryError, RuntimeError, ImportError) as error:
                    raise ProviderUnavailableError('native session could not load: '+str(error)) from error
                self._session_key = key
                self._loaded_at = loading_started
                self._last_used = time.monotonic()
            tokenizer = self._session.tokenizer
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
            if image is not None or media_items is not None:
                marker_ids = tokenizer.encode(config['image_placeholder'], add_special_tokens=False).ids
                expected_media=1 if image is not None else sum(type(item) is not PublisherVideoInput for item in media_items)
                if marker_ids != [args.image_token_id] or prompt_ids.count(args.image_token_id) != expected_media:
                    raise ProviderConfigurationError('rendered template must preserve one configured token per media item')
                if videos:
                    video_ids=tokenizer.encode(config['video_placeholder'],add_special_tokens=False).ids
                    if video_ids!=[config['video_token_id']] or prompt_ids.count(config['video_token_id'])!=len(videos):
                        raise ProviderConfigurationError('rendered template must preserve publisher video token inventory')
                args.image = ([image] if image is not None else list(media_items)
                    if media_items is not None and config['family']=='deepseek_v41' else [])
                if hasattr(args,'prompt_text'):
                    args.prompt_text = prompt
            args.token_ids = json.dumps(prompt_ids)
            generated, emitted = [], ''
            def on_event(event):
                nonlocal emitted
                if emit is None or event.get('event') != 'tokens':
                    return
                generated.extend(event['token_ids'])
                text = tokenizer.decode(generated, skip_special_tokens=config['family'] != 'kimi_k3')
                # Incomplete UTF-8 tokens may temporarily decode to replacement
                # characters. Hold them until a later token completes the bytes.
                if text.endswith('\ufffd'):
                    return
                if not text.startswith(emitted):
                    raise ProviderUnavailableError('tokenizer changed previously streamed text')
                delta = text[len(emitted):]
                if delta:
                    emit(StreamChunk(self.id, config['model'], delta, False,
                        dict(network=False, offline=True, token_ids=list(event['token_ids']))))
                    emitted = text
            try:
                if media_items is None and checkpoint_sink is None:
                    output = self._session.run(args, emit=on_event if emit is not None else None,cancelled=cancelled)
                elif config['family']!='kimi_k3':
                    options=dict(checkpoint_sink=checkpoint_sink,
                        max_checkpoint_bytes=checkpoint_max_bytes if checkpoint_sink is not None else 0)
                    if media_items is not None and config['family']!='deepseek_v41':options['media_items']=media_items
                    output=self._session.run(args,emit=on_event if emit is not None else None,cancelled=cancelled,**options)
                else:
                    output=self._session.generate(_request_args=args,media_items=media_items,
                        emit=on_event if emit is not None else None,cancelled=cancelled,
                        checkpoint_sink=checkpoint_sink,checkpoint_max_bytes=checkpoint_max_bytes)
            except (ValueError, OSError, MemoryError, RuntimeError, ArithmeticError) as error:
                try:
                    self._release_session()
                except Exception as cleanup_error:
                    raise ProviderUnavailableError(
                        'native decoder failed: '+str(error)+'; cleanup failed: '+str(cleanup_error)) from error
                raise ProviderUnavailableError('native decoder failed: '+str(error)) from error
            content = output.get('text')
            if type(content) is not str:
                content = tokenizer.decode(output['generated_ids'])
            if not content.strip():
                raise ProviderUnavailableError('native decoder returned no text')
            if emit is not None:
                if not content.startswith(emitted):
                    raise ProviderUnavailableError('final output differs from streamed tokenizer text')
                if len(content) > len(emitted):
                    emit(StreamChunk(self.id, config['model'], content[len(emitted):], False,
                        dict(network=False, offline=True)))
            return ModelResult(self.id, config['model'], content,
                dict(network=False, offline=True, family=config['family'],
                     engine='openagent-native-publisher',
                     numerical_parity_verified=False, platform_certified=False,
                     runtime=output, model_residency='lazy-expiry' if
                         self._idle_seconds is not None or self._lifetime_seconds is not None
                         else 'resident-until-unload',
                     residency_idle_timeout_seconds=self._idle_seconds,
                     residency_max_seconds=self._lifetime_seconds,
                     residency_eviction_policy='lazy-between-requests-or-status',
                     resident_session_reused=reused))

    def stream_complete(self, messages, *, model=None):
        return self._stream_run(messages, model=model)

    def _stream_run(self, messages, *, model=None, image=None, max_output_tokens=None,media_items=None,
                    checkpoint_sink=None,checkpoint_max_bytes=None,recovery_source=None):
        from .execution_service import ExecutionCancelled
        queue = Queue(maxsize=8)
        stopped = threading.Event()
        def put(item):
            while not stopped.is_set():
                try:
                    queue.put(item, timeout=0.1)
                    return
                except Full:
                    pass
            raise ExecutionCancelled('native stream consumer closed')
        def worker():
            try:
                if recovery_source is not None:
                    result=self.resume_checkpoint(recovery_source,model=model,emit=put,cancelled=stopped.is_set,
                        checkpoint_max_bytes=checkpoint_max_bytes,max_output_tokens=max_output_tokens,
                        checkpoint_sink=checkpoint_sink)
                else:
                    result = self._run(messages, model=model, emit=put, cancelled=stopped.is_set,
                        image=image, max_output_tokens=max_output_tokens,media_items=media_items,
                        checkpoint_sink=checkpoint_sink,checkpoint_max_bytes=checkpoint_max_bytes)
                put(StreamChunk(result.provider, result.model, '', True, result.metadata))
            except Exception as error:
                if not stopped.is_set():
                    try:
                        put(error)
                    except ExecutionCancelled:
                        pass
        thread = threading.Thread(target=worker, name='native-publisher-stream', daemon=True)
        thread.start()
        try:
            while True:
                try:
                    item = queue.get(timeout=0.1)
                except Empty:
                    if not thread.is_alive():
                        raise ProviderUnavailableError('native stream ended without a completion event')
                    continue
                if isinstance(item, Exception):
                    raise item
                yield item
                if item.done:
                    return
        finally:
            stopped.set()
            # Native calls cannot be interrupted safely. The worker releases
            # owned resources after the current decoding operation returns.
            thread.join(timeout=1.0)
