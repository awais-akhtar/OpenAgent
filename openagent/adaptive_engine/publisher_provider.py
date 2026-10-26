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
from .admission_scheduler import AdmissionScheduler, AdmissionRejected, AdmissionTimedOut
from .session_recovery import _path
from .native_library_identity import _stamp
from .failure_reporting import add_failure_note, failure_detail, failure_summary


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


def parse_publisher_operator(family, argv):
    """Parse stored operator arguments exactly as the gateway admits them.

    Shared by ``NativePublisherProvider`` and the universal production
    adapters so both entrypoints select the same operator module and apply
    the same help/prompt/media/context rejections.
    """
    if type(argv) is not list or any(type(arg) is not str or not arg or '\0' in arg for arg in argv):
        raise ProviderConfigurationError('stored operator arguments must be nonempty strings')
    if any(arg in ('-h', '--help', '--prompt', '--prompt-text', '--token-ids') or
            arg.startswith(('--prompt=', '--prompt-text=', '--token-ids=')) for arg in argv):
        raise ProviderConfigurationError('stored operator arguments cannot contain help or prompt inputs')
    if family == 'deepseek_v41':
        from . import publisher_v41_cli as operator
    elif family=='kimi_k3':
        from . import publisher_k3_cli as operator
    elif family in ('qwen3_5_moe','qwen4_exp','inkling_mm_model') or (family=='glm5_next' and
            any(arg=='--media-library' or arg.startswith('--media-library=') for arg in argv)):
        from . import native_conditional_cli as operator
    else:
        from . import native_family_cli as operator
    parser = operator.parser()
    def reject(message):
        raise ProviderConfigurationError(message)
    parser.error = reject
    # Prompt tokens are supplied after rendering, never from stored argv.
    family_args = [] if family in ('deepseek_v41','kimi_k3') else ['--family',family]
    args = parser.parse_args([*argv,*family_args,'--token-ids','[]'])
    if any(getattr(args,name,None) for name in ('image','audio_wav','video_manifest')):
        raise ProviderConfigurationError('gateway requests cannot use stored media inputs')
    if getattr(args, 'max_tokens', 0) > getattr(args, 'context', 0):
        raise ProviderConfigurationError('generation exceeds configured native context')
    return operator, args


def publisher_operator_backend(family, args):
    """Return the exact P02 backend selected by parsed operator arguments."""
    if family == 'deepseek_v41':
        return 'device' if getattr(args, 'device', None) is not None else 'native'
    if family == 'kimi_k3':
        bf16 = getattr(args, 'arithmetic_device', None) is not None
        return 'fla' if bf16 and getattr(args, 'fla_version', None) is not None else 'device' if bf16 else 'native'
    if (getattr(args,'segment_runtime_library',None) is not None or
            getattr(args,'public_runtime_library',None) is not None):
        return 'public-range'
    if getattr(args,'device',None) is not None or getattr(args,'text_device',None) is not None:
        return 'device'
    return 'native'


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
        self._stream_lock = threading.Lock()
        self._stream_thread = None
        # The HTTP compatibility surface has one native session owner. Bound
        # both active and pending work so overload is observable and finite.
        self._request_admission = AdmissionScheduler(max_active=1, max_bytes=1, max_waiting=64)
        self._request_local = threading.local()
        from .phase1_a04_media_lifecycle import FamilyMediaLifecycleRegistry
        self._media_lifecycle_registry = FamilyMediaLifecycleRegistry(
            max_frames_per_family=8)

    @staticmethod
    def _require_source_route(config, args, root):
        family=getattr(args,'family',None) or config.get('family')
        if family == 'kimi_k3':
            # The K3 CLI has no --family option, so resolve its route from the
            # provider's validated configuration and gate it here as well as
            # in PublisherSession. Gateway construction must not infer tuple
            # admission from reaching the CLI later in the request.
            from .source_scope import require_source_family_backend_admission
            bf16 = getattr(args, 'arithmetic_device', None) is not None
            fla = bf16 and getattr(args, 'fla_version', None) is not None
            require_source_family_backend_admission(
                family='kimi-k3', model_type='kimi_k3',
                backend='fla' if fla else 'device' if bf16 else 'native',
                arithmetic='bf16' if bf16 else 'fp32-approximate')
        elif family == 'deepseek_v41':
            # V4.1 consumes three independently encoded local banks; all are
            # required by its ordinary service factory. Gate each row before
            # artifact fingerprinting or warm-session reuse.
            from .source_scope import require_source_matrix_admission
            backend = 'device' if getattr(args, 'device', None) is not None else 'native'
            for source_format, persisted_format in (
                    ('BF16', 'bf16'), ('FP8', 'fp8'), ('FP4', 'fp4')):
                require_source_matrix_admission(
                    family='deepseek-v41', model_type='deepseek_v41',
                    backend=backend, arithmetic='publisher-bf16-fp8-fp4',
                    source_format=source_format,
                    persisted_format=persisted_format)
        source_family={'mixtral':'mixtral','qwen3_5_moe_text':'qwen35',
            'qwen4_exp_text':'qwen4','glm5_next_text':'glm5-next',
            'glm5_next':'glm5-next','inkling_text':'inkling',
            'glm_moe_dsa':'glm-dsa','qwen3_5_moe':'qwen35',
            'qwen4_exp':'qwen4','inkling_mm_model':'inkling'}.get(family)
        if source_family is not None:
            from .source_scope import require_checkpoint_inventory_admission
            model_type=family
            # Shared with the universal adapters: native_family_cli selects
            # public-range with --public-runtime-library and the conditional
            # CLI with --segment-runtime-library; both must gate that row.
            backend=publisher_operator_backend(family, args)
            maximum=getattr(args,'metadata_mib',None)
            if type(maximum) is not int or maximum<=0:
                maximum=getattr(args,'staging_mib',None)
            if type(maximum) is not int or maximum<=0:
                maximum=256
            maximum*=1024**2
            require_checkpoint_inventory_admission(root,family=source_family,
                model_type=model_type,backend=backend,max_metadata_bytes=maximum)

    @staticmethod
    def _residency_key(config, args):
        root = _path(args.checkpoint).resolve(strict=True)
        NativePublisherProvider._require_source_route(config, args, root)
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
        # Artifact identity alone is insufficient for warm reuse.  A decoder
        # admitted under one bounded execution policy must not be reused after
        # context, cache, projection, expert, media or continuation settings
        # change.  Exclude prompt/request fields; include only construction and
        # state-schema controls that affect resident ownership or token layout.
        # Keep every parser-owned construction/state option in the reuse key.
        # The gateway appends request data to the same namespace, so only fields
        # that are inherently per-request are excluded.  This prevents a new
        # backend option from silently reusing a resident session with a
        # different execution contract.
        request_fields = {
            'prompt', 'prompt_text', 'token_ids', 'messages', 'image', 'images',
            'audio_wav', 'video_manifest', 'media_library', 'media_items',
            # ``checkpoint`` selects the resident model artifact and is a
            # construction input, not request data.  Recovery destinations
            # and sinks remain request-scoped below.
            'checkpoint_file', 'checkpoint_sink', 'emit',
            'cancelled', 'max_output_tokens', 'model',
            'stop_prefix', 'stop_prefixes', 'stop_token_id', 'stop_ids',
        }
        policy = tuple((name, repr(value)) for name, value in sorted(vars(args).items())
                       if name not in request_fields and not name.startswith('_'))
        return json.dumps(config, sort_keys=True), policy, tuple(stamps)

    def _release_session(self):
        self._session_key = None
        if self._session is not None:
            try:
                self._session.close()
            except BaseException as error:
                self._cleanup_error = failure_detail(error)
                raise
            self._session = None
        if self._construction_cleanup is not None:
            try:self._construction_cleanup()
            except BaseException as error:
                self._cleanup_error=failure_detail(error)
                raise
            self._construction_cleanup=None
        self._loaded_at = self._last_used = None
        self._cleanup_error = None
    @staticmethod
    def _state_key(state):
        import hashlib
        return hashlib.sha256(state.encode('utf-8')).hexdigest()

    @staticmethod
    def _serialize_jev_text(value, label):
        if type(value) is str:
            return value
        if type(value) in (dict, list):
            try:
                rendered = json.dumps(value, ensure_ascii=False, separators=(',', ':'),
                                      allow_nan=False)
            except (TypeError, ValueError) as error:
                raise ProviderConfigurationError(label+' must be JSON-serializable') from error
            if NativePublisherProvider._utf8_size(rendered, label=label) > 1024**2:
                raise ProviderConfigurationError(label+' exceeds the 1 MiB serialized limit')
            return rendered
        raise ProviderConfigurationError(label+' must be text, object or array')

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
                    self._cleanup_error=failure_detail(failure)
                    add_failure_note(primary, 'gateway retains failed construction cleanup: '+failure_summary(failure))
                else:primary.cleanup=None
            raise

    def close(self, cancelled=None):
        # Wake bounded-queue waiters before waiting for the live session owner.
        self._request_admission.close()
        with self._lifecycle_lock(cancelled):
            primary = None
            try:
                self._release_session()
            except BaseException as error:
                primary = error
            finally:
                self._closed = True
                try:
                    self._media_lifecycle_registry.close()
                except BaseException as cleanup:
                    if primary is None:
                        primary = cleanup
                    else:
                        add_failure_note(primary,
                            'publisher media registry close failed: '+failure_summary(cleanup))
            if primary is not None:
                raise primary

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
        request_queue = self._request_admission.status()
        if not self._lock.acquire(blocking=False):
            return dict(closed=self._closed, active=self._active, busy=True,
                        stream_worker_active=self._stream_lock.locked(),
                        execution_runtime=self._execution_runtime_status(),
                        request_queue=request_queue,
                        resident=self._session is not None,
                        eviction_policy='lazy', cleanup_pending=self._cleanup_error is not None)
        try:
            if not self._active:
                try:
                    self._evict_expired()
                except Exception:
                    pass
            now = time.monotonic()
            execution_runtime = self._execution_runtime_status()
            return dict(closed=self._closed, active=self._active, busy=self._active,
                        stream_worker_active=self._stream_lock.locked(),
                        execution_runtime=execution_runtime,
                        request_queue=request_queue,
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

    def _execution_runtime_status(self):
        """Return an actual scheduler snapshot, or None when no owner exists."""
        try:
            session = self._session
            service = getattr(session, 'service', None)
            runtime_status = getattr(service, 'runtime_status', None)
            if not callable(runtime_status):
                return None
            snapshot = runtime_status()
        except Exception:
            # Status reporting must not mutate or tear down a live inference
            # service. Callers represent an unavailable snapshot as unknown.
            return None
        return snapshot if type(snapshot) is dict else None

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
        return parse_publisher_operator(config['family'], config['argv'])

    def available(self):
        if self._closed or self._cleanup_error is not None:
            return False
        try:
            config = self._configuration()
            _, args = self._operator(config)
            root = _path(args.checkpoint).resolve(strict=True)
            self._require_source_route(config, args, root)
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
            stream_thread = self._stream_thread
            # Treat either marker as ownership evidence.  The worker clears
            # its thread reference and releases the stream lock in separate
            # operations; checking only their conjunction creates a teardown
            # window in which a warm session can be unloaded while the worker
            # is still unwinding.
            current_thread = threading.current_thread()
            owns_stream = (stream_thread is current_thread and self._stream_lock.locked())
            if ((stream_thread is not None and stream_thread is not current_thread) or
                    (self._stream_lock.locked() and not owns_stream)):
                raise ProviderUnavailableError('cannot unload while a native stream worker owns the session')
            yield
        finally:
            self._lock.release()

    @contextmanager
    def _request_lock(self, cancelled, queue_timeout=None):
        lease = None
        previous_queue_wait = getattr(self._request_local, 'queue_wait_seconds', None)
        previous_queue_position = getattr(self._request_local, 'queue_position', None)
        try:
            try:
                if queue_timeout == 0:
                    lease = self._request_admission.try_acquire(1)
                else:
                    lease = self._request_admission.acquire(1, cancelled=cancelled,
                                                             timeout=queue_timeout)
            except InterruptedError as error:
                from .execution_service import ExecutionCancelled
                raise ExecutionCancelled('native request cancelled during admission') from error
            self._request_local.queue_wait_seconds = lease.queue_wait_seconds
            self._request_local.queue_position = lease.queue_position
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
                            add_failure_note(error, 'native session cleanup remains pending: '+failure_summary(cleanup_error))
                    raise
                finally:
                    self._active = False
                    if self._session is not None and self._cleanup_error is None:
                        self._last_used = time.monotonic()
                    if not failed:
                        # Eviction errors retain ownership for a later explicit retry.
                        try:
                            self._evict_expired()
                        except Exception:
                            pass
        finally:
            if lease is not None:
                self._request_admission.release(lease)
            if previous_queue_wait is None:
                self._request_local.__dict__.pop('queue_wait_seconds', None)
                self._request_local.__dict__.pop('queue_position', None)
            else:
                self._request_local.queue_wait_seconds = previous_queue_wait
                self._request_local.queue_position = previous_queue_position

    def complete(self, messages, *, model=None, max_output_tokens=None, stop_prefixes=(),
                 stop_sequences=(), cancelled=None, queue_timeout=None):
        return self._run(messages, model=model, max_output_tokens=max_output_tokens,
                         stop_prefixes=stop_prefixes, stop_sequences=stop_sequences,
                         cancelled=cancelled,
                         queue_timeout=queue_timeout)

    def score_options(self, prompt, options, *, model=None, normalize='mean', cancelled=None):
        return self.score_option_groups([prompt], [options], [normalize],
                                        model=model, cancelled=cancelled)[0]

    def score_option_groups(self, prompts, option_groups, normalizations, *, model=None,
                            cancelled=None, option_limit=64, cache_slot=None, state_key=None):
        """Score several closed sets from one admitted shared-prefix snapshot."""
        if (type(prompts) not in (list, tuple) or not prompts or len(prompts) > 64 or
                type(option_groups) not in (list, tuple) or len(option_groups) != len(prompts) or
                type(normalizations) not in (list, tuple) or len(normalizations) != len(prompts) or
                any(normalize not in ('mean', 'sum') for normalize in normalizations) or
                type(option_limit) is not int or option_limit not in (64, 255)):
            raise ProviderConfigurationError('bounded matching Brio prompt and option groups required')
        if any(type(prompt) is not str or not prompt or len(prompt) > 2 * 1024**2
               for prompt in prompts):
            raise ProviderConfigurationError('bounded nonempty Brio prompts required')
        prompt_sizes = [self._utf8_size(prompt, label='Brio prompt') for prompt in prompts]
        if any(size > 2 * 1024**2 for size in prompt_sizes):
            raise ProviderConfigurationError('individual Brio prompt exceeds the 2 MiB UTF-8 budget')
        if sum(prompt_sizes) > 8 * 1024**2:
            raise ProviderConfigurationError('Brio prompt groups exceed the 8 MiB budget')
        if cache_slot is not None and (type(cache_slot) is not int or not 0 <= cache_slot < 16):
            raise ProviderConfigurationError('cache_slot must be an integer from 0 through 15')
        if state_key is not None and (type(state_key) is not str or len(state_key) != 64 or
                                      any(char not in '0123456789abcdef' for char in state_key)):
            raise ProviderConfigurationError('state cache identity must be a lowercase SHA256 digest')
        for options in option_groups:
            if (type(options) not in (list, tuple) or not 2 <= len(options) <= option_limit or
                    any(type(option) is not str or not option.strip() or len(option) > 1024**2
                        for option in options) or len(set(options)) != len(options)):
                raise ProviderConfigurationError('each Brio group exceeds its option limit or contains invalid choices')
            if sum(self._utf8_size(option, label='Brio option') for option in options) > 8 * 1024**2:
                raise ProviderConfigurationError('Brio option group exceeds the 8 MiB UTF-8 budget')
        if cancelled is not None and not callable(cancelled):
            raise ProviderConfigurationError('cancellation callback must be callable')
        if cache_slot is not None:
            raise ProviderConfigurationError('persistent Brio cache_slot snapshots are not supported by this session owner')
        with self._request_lock(cancelled):
            if self._closed:
                raise ProviderUnavailableError('native provider is closed')
            config = self._configuration()
            if model is not None and model != config['model']:
                raise ProviderConfigurationError('requested model differs from configured native checkpoint')
            operator, args = self._operator(config)
            if config['family'] in ('kimi_k3', 'deepseek_v41'):
                raise ProviderConfigurationError('closed-set scoring is not composed for direct K3/V4.1 sessions')
            key = self._residency_key(config, args)
            reused = self._session is not None and self._session_key == key
            if not reused:
                self._release_session()
                started = time.monotonic()
                try:
                    self._session = self._construct_session(operator, args, cancelled)
                except (ValueError, OSError, MemoryError, RuntimeError, ImportError) as error:
                    raise ProviderUnavailableError('native scoring session could not load: '+failure_detail(error)) from error
                self._session_key = key
                self._loaded_at = started
                self._last_used = time.monotonic()
            return self._score_option_groups_in_session(
                config, prompts, option_groups, normalizations, cancelled, reused,
                option_limit=option_limit, cache_slot=cache_slot)

    def _score_option_groups_in_session(self, config, prompts, option_groups,
                                        normalizations, cancelled, reused, *,
                                        option_limit=64, cache_slot=None):
        scorer = getattr(self._session, 'score_option_groups', None)
        if not callable(scorer):
            raise ProviderConfigurationError('selected native family has no closed-set scoring owner')
        results = scorer(prompts, option_groups, normalizations, cancelled=cancelled,
                         option_limit=option_limit)
        for result in results:
            result['model'] = config['model']
            result['resident_session_reused'] = reused
        return tuple(results)

    @staticmethod
    def _render_brio_prompt(config, state, question):
        if type(question) is not str or len(question) > 1024**2:
            raise ProviderConfigurationError('bounded textual Brio question required')
        if type(state) is not str:
            raise ProviderConfigurationError('Brio state must be text')
        if not state and not question.strip():
            raise ProviderConfigurationError('Brio requires state or a question')
        if NativePublisherProvider._utf8_size(state, label='Brio state') > 1024**2:
            raise ProviderConfigurationError('Brio state exceeds the 1 MiB input limit')
        content = ('Context:\n' + state + '\n\n') if state else ''
        if question:
            content += 'Question: ' + question + '\n'
        content += 'Answer:'
        try:
            from jinja2 import StrictUndefined, TemplateError
            from jinja2.sandbox import SandboxedEnvironment
        except ImportError as error:
            raise ProviderConfigurationError('native Brio scoring requires local Jinja2') from error
        env = SandboxedEnvironment(undefined=StrictUndefined, autoescape=False)
        try:
            prompt = env.from_string(config['chat_template']).render(
                messages=[{'role': 'user', 'content': content}],
                add_generation_prompt=True, **config.get('template_variables', {}))
        except TemplateError as error:
            raise ProviderConfigurationError('native Brio prompt template could not render: '+failure_detail(error)) from error
        if NativePublisherProvider._utf8_size(prompt, label='rendered Brio prompt') > 2 * 1024**2:
            raise ProviderConfigurationError('rendered Brio prompt exceeds 2 MiB')
        return prompt

    def score_brio(self, state, options, *, question='', model=None,
                   normalize='mean', cancelled=None, cache_slot=None):
        config = self._configuration()
        prompt = self._render_brio_prompt(config, state, question)
        result = self.score_options(prompt, options, model=model, normalize=normalize,
                                    cancelled=cancelled, cache_slot=cache_slot,
                                    state_key=self._state_key(state))
        if question:
            result['question'] = question
        result['object'] = 'brio.choice'
        return result

    def score_brio_questions(self, state, questions, *, model=None,
                             normalize='mean', cancelled=None, option_limit=64,
                             cache_slot=None, jev_model_alias=False):
        state = self._serialize_jev_text(state, 'state')
        if (not state or self._utf8_size(state, label='Brio state') > 1024**2 or
                type(questions) is not list or not questions or len(questions) > 64):
            raise ProviderConfigurationError('Brio questions must be a bounded nonempty array')
        if normalize not in ('mean', 'sum'):
            raise ProviderConfigurationError('Brio normalization must be mean or sum')
        if type(jev_model_alias) is not bool:
            raise ProviderConfigurationError('explicit Jev model alias policy required')
        if jev_model_alias and (type(model) is not str or not model.strip()):
            raise ProviderConfigurationError('Jev model name must be a nonempty string')
        config = self._configuration()
        prompts = []
        option_groups = []
        normalizations = []
        questions_text = []
        question_bytes = prompt_bytes = 0
        for entry in questions:
            if (type(entry) is not dict or set(entry) - {'question', 'options', 'normalize'} or
                    type(entry.get('question')) is not str or not entry['question'].strip()):
                raise ProviderConfigurationError('each Brio question requires question and options')
            question_bytes += self._utf8_size(entry['question'], label='Brio question')
            if question_bytes > 1024**2:
                raise ProviderConfigurationError('Brio batch questions exceed the 1 MiB text budget')
            per_normalize = entry.get('normalize', normalize)
            if per_normalize not in ('mean', 'sum'):
                raise ProviderConfigurationError('Brio question normalization must be mean or sum')
            options = entry.get('options')
            if (type(options) not in (list, tuple) or not 2 <= len(options) <= option_limit or
                    any(type(option) is not str or not option.strip() for option in options) or
                    len(set(options)) != len(options)):
                raise ProviderConfigurationError('each Brio question requires distinct nonempty bounded options')
            rendered = self._render_brio_prompt(config, state, entry['question'])
            prompt_bytes += self._utf8_size(rendered, label='rendered Brio prompt')
            if prompt_bytes > 8 * 1024**2:
                raise ProviderConfigurationError('Brio rendered prompt groups exceed the 8 MiB budget')
            prompts.append(rendered)
            option_groups.append(options)
            normalizations.append(per_normalize)
            questions_text.append(entry['question'])
        if type(option_limit) is not int or option_limit not in (64, 255):
            raise ProviderConfigurationError('Brio option limit must be the public 64 or Jev 255 contract')
        if any(len(group) > option_limit for group in option_groups):
            raise ProviderConfigurationError('Brio option group exceeds its admitted limit')
        scored = self.score_option_groups(prompts, option_groups, normalizations,
            model=None if jev_model_alias else model, cancelled=cancelled, option_limit=option_limit,
            cache_slot=cache_slot, state_key=self._state_key(state))
        answers = []
        prompt_tokens = read_tokens = 0
        for question, result in zip(questions_text, scored):
            answers.append(dict(question=question, answer=result['answer'],
                entropy=result['entropy'], normalize=result['normalize'],
                choices=result['choices']))
            prompt_tokens = max(prompt_tokens, result['usage']['prompt_tokens'])
            read_tokens += result['usage']['read_tokens']
        return dict(object='brio.answers', model=config['model'], answers=answers,
            usage=dict(prompt_tokens=prompt_tokens, completion_tokens=0,
                       read_tokens=read_tokens, total_tokens=prompt_tokens+read_tokens))

    def score_brio_schema(self, state, schema, *, task='', model=None,
                          normalize='mean', cancelled=None, cache_slot=None):
        state = self._serialize_jev_text(state, 'state')
        if (not state or self._utf8_size(state, label='Brio state') > 1024**2 or
                type(schema) is not dict or not schema or len(schema) > 64):
            raise ProviderConfigurationError('Brio schema must be a nonempty bounded object')
        if type(task) is not str or len(task) > 1024**2:
            raise ProviderConfigurationError('Brio schema task must be bounded text')
        if normalize not in ('mean', 'sum'):
            raise ProviderConfigurationError('Brio normalization must be mean or sum')
        if cache_slot is not None:
            if type(cache_slot) is not int or not 0 <= cache_slot < 16:
                raise ProviderConfigurationError('cache_slot must be an integer from 0 through 15')
            raise ProviderConfigurationError('persistent Brio cache_slot snapshots are not supported by this session owner')
        answers = {}
        fields = []
        prompt_tokens = read_tokens = 0
        model_config = self._configuration()
        config = model_config
        if model is not None and model != model_config['model']:
            raise ProviderConfigurationError('requested model differs from configured native checkpoint')
        prepared = []
        option_groups = []
        normalizations = []
        fields_order = []
        field_normalizations = []
        option_bytes = 0
        prompt_bytes = 0
        for field, options in schema.items():
            if (type(field) is not str or not field.strip() or
                    any(character in field for character in '"\\\n\r')):
                raise ProviderConfigurationError('Brio schema field names cannot contain quotes or newlines')
            if (type(options) not in (list, tuple) or not 2 <= len(options) <= 64 or
                    any(type(option) is not str or not option.strip() for option in options) or
                    len(set(options)) != len(options)):
                raise ProviderConfigurationError('each Brio schema field requires 2 to 64 distinct choices')
            option_bytes += sum(self._utf8_size(option, label='Brio schema option') for option in options)
            if option_bytes > 512 * 1024:
                raise ProviderConfigurationError('Brio schema choices exceed the 512 KiB aggregate limit')
            rendered = self._render_brio_prompt(
                config, state, (task+'\n' if task else '')+
                'Choose a value for '+field+'.\nValues already chosen: ')
            prompt_bytes += self._utf8_size(rendered, label='rendered Brio schema prompt')
            if prompt_bytes > 8 * 1024**2:
                raise ProviderConfigurationError('Brio schema prompts exceed the 8 MiB aggregate limit')
            prepared.append(rendered)
            option_groups.append(options)
            normalizations.append(normalize)
            field_normalizations.append(normalize)
            fields_order.append(field)
        # Each field includes prior selected values. The provider lifecycle lock
        # pins one resident model for the complete dependent sequence; scoring
        # restores the service session after every field.
        # Keep the resident provider lifecycle lock for the entire schema so
        # eviction or another request cannot replace the model between fields.
        with self._request_lock(cancelled):
            if self._closed:
                raise ProviderUnavailableError('native provider is closed')
            active_config = self._configuration()
            if active_config != model_config:
                raise ProviderConfigurationError('native scoring configuration changed during schema admission')
            if model is not None and model != active_config['model']:
                raise ProviderConfigurationError('requested model differs from configured native checkpoint')
            operator, args = self._operator(active_config)
            if active_config['family'] in ('kimi_k3', 'deepseek_v41'):
                raise ProviderConfigurationError('closed-set scoring is not composed for direct K3/V4.1 sessions')
            key = self._residency_key(active_config, args)
            reused = self._session is not None and self._session_key == key
            if not reused:
                self._release_session()
                started = time.monotonic()
                try:
                    self._session = self._construct_session(operator, args, cancelled)
                except (ValueError, OSError, MemoryError, RuntimeError, ImportError) as error:
                    raise ProviderUnavailableError('native scoring session could not load: '+failure_detail(error)) from error
                self._session_key = key
                self._loaded_at = started
                self._last_used = time.monotonic()
            tokenizer = self._session.tokenizer
            prompt_ids = [tuple(tokenizer.encode(prompt, add_special_tokens=False).ids)
                          for prompt in prepared]
            common = list(prompt_ids[0])
            for encoded in prompt_ids[1:]:
                limit = min(len(common), len(encoded))
                index = 0
                while index < limit and common[index] == encoded[index]:
                    index += 1
                del common[index:]
            if not common:
                raise ProviderConfigurationError('schema prompts lack a reusable shared token prefix')
            option_ids = [[tuple(tokenizer.encode(' '+option,
                add_special_tokens=False).ids) for option in options]
                for options in option_groups]
            if any(not ids for group in option_ids for ids in group):
                raise ProviderConfigurationError('schema choices must encode to nonempty token sequences')
            if sum(len(ids) for group in option_ids for ids in group) > 65536:
                raise ProviderConfigurationError('schema choices exceed the 65,536 token read budget')
            prompt_suffix_tokens = 0
            with self._session.scoring_prefix_tokens(common, cancelled=cancelled):
                for field_index, field in enumerate(fields_order):
                    context_suffix = json.dumps(answers, ensure_ascii=False,
                        separators=(',', ':'), allow_nan=False)
                    full_prompt = prepared[field_index] + context_suffix
                    if self._utf8_size(full_prompt, label='Brio schema field prompt') > 2 * 1024**2:
                        raise ProviderConfigurationError('Brio schema field prompt exceeds the 2 MiB limit')
                    full_ids = tuple(tokenizer.encode(full_prompt,
                        add_special_tokens=False).ids)
                    if full_ids[:len(common)] != tuple(common):
                        raise ProviderConfigurationError('schema prompt changed its admitted shared token prefix')
                    suffix = full_ids[len(common):]
                    prompt_suffix_tokens += len(suffix)
                    if prompt_suffix_tokens > 65536:
                        raise ProviderConfigurationError('schema prompt suffixes exceed the 65,536 token budget')
                    result = self._session.score_option_groups_token_ids(
                        [suffix], [option_ids[field_index]], [field_normalizations[field_index]],
                        option_labels=[option_groups[field_index]], cancelled=cancelled)[0]
                    value = result['answer']
                    answers[field] = value
                    fields.append(dict(field=field, value=value, p=result['choices'][0]['p'],
                        entropy=result['entropy'], choices=result['choices']))
                    prompt_tokens = max(prompt_tokens, len(full_ids))
                    read_tokens += result['usage']['read_tokens']
        return dict(object='brio.schema', model=model_config['model'],
            json=answers, fields=fields,
            usage=dict(prompt_tokens=prompt_tokens, completion_tokens=0,
                       read_tokens=read_tokens, total_tokens=prompt_tokens+read_tokens))

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
                        max_output_tokens=None, stop_prefixes=()):
        self._validate_vision_input(image, media_type, max_output_tokens)
        return self._run(messages, model=model, image=image,
                         max_output_tokens=max_output_tokens, stop_prefixes=stop_prefixes)

    @staticmethod
    def _validate_vision_input(image, media_type, max_output_tokens):
        if (type(image) is not bytes or not image or len(image) > 256*1024**2 or
                type(media_type) is not str or not media_type.startswith('image/') or
                any(ord(char) < 33 or ord(char) > 126 for char in media_type) or
                media_type.count('/') != 1 or not media_type[6:].strip()):
            raise ProviderConfigurationError('bounded encoded image bytes and image MIME type required')
        if max_output_tokens is not None and (type(max_output_tokens) is not int or max_output_tokens <= 0):
            raise ProviderConfigurationError('positive integer vision output limit required')

    def stream_complete_vision(self, messages, image, *, media_type='image/png', model=None,
                               max_output_tokens=None, stop_prefixes=(), cancelled=None):
        self._validate_vision_input(image, media_type, max_output_tokens)
        return self._stream_run(messages, model=model, image=image,
                                max_output_tokens=max_output_tokens, stop_prefixes=stop_prefixes,
                                cancelled=cancelled)

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
                dtype = getattr(frame, 'dtype', None)
                if (getattr(frame, 'ndim', None) == 1 and
                        getattr(dtype, 'kind', None) == 'f'):
                    raise ProviderConfigurationError(
                        'waveform media is not admitted by the decoded-media gateway: '
                        'family-specific sampling-rate and placeholder-order contracts are required')
                try:
                    width,height=_dimensions(frame)
                except (TypeError, ValueError, AttributeError, OverflowError) as error:
                    raise ProviderConfigurationError('decoded RGB/RGBA media frame required') from error
                if width<=0 or height<=0:raise ProviderConfigurationError('positive decoded media dimensions required')
                total+=width*height*16
                if total>256*1024**2:raise ProviderConfigurationError('decoded media request exceeds admission')
        return tuple(items),total

    @staticmethod
    def _validate_media_modality_order(family, items):
        """Reject cross-modality publisher batches before model construction.

        The pinned publisher processor has no cross-modality placeholder
        ordering contract. K3 consumes one typed, ordered media sequence and
        can safely mix stills with its explicitly timestamped video inputs;
        the other processor path accepts one modality per request. Enforce
        that restriction at the serving boundary, before a warm session is
        loaded or any media is dispatched.
        """
        if family == 'kimi_k3' or items is None:
            return
        from .publisher_media_inputs import PublisherVideoInput
        modalities = {('video' if type(item) is PublisherVideoInput else 'image')
                      for item in items}
        if len(modalities) > 1:
            raise ProviderConfigurationError(
                'publisher processors require one media modality per request; '
                'mixed image/video placeholder ordering is not admitted')

    @staticmethod
    def _validate_stop_prefixes(value):
        if (type(value) not in (tuple, list) or len(value)>64 or
                any(type(prefix) not in (tuple, list) or not prefix or len(prefix)>128 or
                    any(type(token) is not int or not 0 <= token < 2**31 for token in prefix)
                    for prefix in value) or
                len({tuple(prefix) for prefix in value}) != len(value)):
            raise ProviderConfigurationError('bounded unique stop-prefix inventory required')
        return tuple(tuple(prefix) for prefix in value)

    @classmethod
    def _encode_stop_sequences(cls, tokenizer, value):
        """Convert bounded API text stops with the active checkpoint tokenizer."""
        if (type(value) not in (tuple, list) or len(value) > 4 or
                any(type(item) is not str or not item or
                    cls._utf8_size(item, label='stop sequence') > 4096
                    for item in value)):
            raise ProviderConfigurationError('bounded nonempty UTF-8 stop sequence list required')
        encoded = []
        for item in value:
            ids = tokenizer.encode(item, add_special_tokens=False).ids
            if (type(ids) not in (list, tuple) or not ids or len(ids) > 128 or
                    any(type(token) is not int or not 0 <= token < 2**31 for token in ids)):
                raise ProviderConfigurationError('stop sequence must encode to 1..128 valid token IDs')
            prefix = tuple(ids)
            if prefix not in encoded:
                encoded.append(prefix)
        return tuple(encoded)

    @staticmethod
    def _parse_stop_prefix_arguments(value):
        if type(value) not in (tuple, list) or len(value) > 64:
            raise ProviderConfigurationError('bounded stop-prefix arguments required')
        prefixes = []
        for raw in value:
            if type(raw) is not str or not raw:
                raise ProviderConfigurationError('stop prefixes require comma-separated token IDs')
            try:
                parts = raw.split(',')
                if any(not part.strip() for part in parts):
                    raise ValueError('empty stop token')
                prefix = tuple(int(part.strip()) for part in parts)
            except ValueError as error:
                raise ProviderConfigurationError(
                    'stop prefixes require comma-separated token IDs') from error
            if (not prefix or len(prefix) > 128 or
                    any(token < 0 or token >= 2**31 for token in prefix)):
                raise ProviderConfigurationError('stop prefix token inventory exceeds admission')
            prefixes.append(prefix)
        if len(set(prefixes)) != len(prefixes):
            raise ProviderConfigurationError('unique stop-prefix arguments required')
        return tuple(prefixes)

    @staticmethod
    def _strip_terminal_stop_prefix(token_ids, prefixes):
        ids = tuple(token_ids)
        for prefix in sorted(prefixes, key=len, reverse=True):
            if len(ids) >= len(prefix) and ids[-len(prefix):] == prefix:
                return ids[:-len(prefix)], True
        return ids, False

    @classmethod
    def _matched_stop_sequence(cls, token_ids, tokenizer, stop_sequences):
        """Return the request text whose encoded tokens terminate generation."""
        ids = tuple(token_ids)
        for text in stop_sequences:
            encoded = tokenizer.encode(text, add_special_tokens=False).ids
            prefix = tuple(encoded)
            if prefix and len(ids) >= len(prefix) and ids[-len(prefix):] == prefix:
                return text
        return None

    @staticmethod
    def _utf8_size(value, *, label):
        try:
            return len(value.encode('utf-8'))
        except UnicodeEncodeError as error:
            raise ProviderConfigurationError(label+' must be valid UTF-8 text') from error

    def complete_media(self,messages,media_items,*,model=None,max_output_tokens=None,stop_prefixes=(),cancelled=None):
        items,_=self._validate_decoded_media(media_items,max_output_tokens)
        items=self._freeze_decoded_media(items,cancelled=cancelled)
        return self._run(messages,model=model,media_items=items,max_output_tokens=max_output_tokens,
                         stop_prefixes=stop_prefixes,cancelled=cancelled)

    def stream_complete_media(self,messages,media_items,*,model=None,max_output_tokens=None,stop_prefixes=(),cancelled=None):
        # Detach borrowed caller buffers before returning the iterator. Deferring
        # this copy until first iteration lets a caller mutate/reuse its frames
        # after stream creation and silently changes the media bound to the
        # request. Eager staging may cost work for an abandoned stream, but it
        # preserves stable request ownership; cancellation is checked during
        # the bounded copy and the generation/worker still receives the same
        # cancellation predicate.
        items,_=self._validate_decoded_media(media_items,max_output_tokens)
        items=self._freeze_decoded_media(items,cancelled=cancelled)
        return self._stream_run(messages,model=model,media_items=items,
            max_output_tokens=max_output_tokens,stop_prefixes=stop_prefixes,cancelled=cancelled)

    @staticmethod
    def _freeze_decoded_media(items, *, cancelled=None):
        """Detach asynchronous worker inputs, checking cancellation between copy chunks."""
        import numpy as np
        from .k3_execution.video_input import K3VideoInput
        from .publisher_media_inputs import PublisherVideoInput
        if cancelled is not None and not callable(cancelled):
            raise ProviderConfigurationError('media-freeze cancellation must be callable')
        def boundary():
            if cancelled is not None and cancelled():
                from .execution_service import ExecutionCancelled
                raise ExecutionCancelled('decoded media staging cancelled')
        def frame_copy(frame):
            boundary()
            if isinstance(frame,np.ndarray):
                source = frame[..., :3] if frame.ndim == 3 and frame.shape[2] == 4 else frame
                if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
                    raise ProviderConfigurationError('decoded media frame changed after admission')
                copied=np.empty(source.shape,dtype=np.uint8,order='C')
                # Keep large frame staging cancellable instead of hiding one
                # full-frame bytes conversion between request cancellation
                # checkpoints.  Row slices preserve the same contiguous RGB
                # bytes and stable detached ownership.
                rows=max(1,min(source.shape[0],(1024**2)//max(1,source.shape[1]*3)))
                for start in range(0,source.shape[0],rows):
                    boundary()
                    end=min(source.shape[0],start+rows)
                    copied[start:end]=source[start:end]
                boundary()
                copied.flags.writeable = False
                return copied
            converted = frame.convert('RGB')
            try:
                boundary()
                copied = np.asarray(converted, dtype=np.uint8, order='C').copy()
                boundary()
            except BaseException as primary:
                close = getattr(converted, 'close', None)
                if callable(close):
                    try:
                        close()
                    except BaseException as cleanup:
                        add_failure_note(primary, 'temporary publisher RGB conversion cleanup failed: '+failure_summary(cleanup))
                raise
            close = getattr(converted, 'close', None)
            if callable(close):
                close()
            copied.flags.writeable = False
            return copied
        result=[]
        for item in items:
            boundary()
            if type(item) is K3VideoInput:
                # frame_copy returns immutable detached RGB arrays. Adopt
                # them directly rather than passing through the public
                # borrowed-input constructor, which would perform a second
                # uninterruptible full-frame copy.
                result.append(K3VideoInput._from_owned_frames(
                    tuple(frame_copy(frame) for frame in item.frames),
                    item.timestamps,item.selection_identity))
            elif type(item) is PublisherVideoInput:
                # frame_copy already creates immutable, private RGB arrays
                # with cancellation checks per chunk. Re-entering the public
                # constructor here would duplicate each full frame in one
                # uncancellable copy, so transfer those private arrays through
                # its invariant-checking adoption path instead.
                result.append(PublisherVideoInput._from_owned_frames(
                    tuple(frame_copy(frame) for frame in item.frames),
                    item.frames_indices,item.total_num_frames,item.fps))
            else:result.append(frame_copy(item))
        boundary()
        return tuple(result)

    def complete_checkpointed(self,messages,*,checkpoint_sink,checkpoint_max_bytes,
                              model=None,max_output_tokens=None,media_items=None,stop_prefixes=(),cancelled=None):
        """Generate locally while a caller-owned store publishes recovery frames."""
        if not callable(checkpoint_sink) or type(checkpoint_max_bytes) is not int or checkpoint_max_bytes<=0:
            raise ProviderConfigurationError('checkpoint sink and positive complete frame budget required')
        if media_items is not None:
            self._validate_decoded_media(media_items,max_output_tokens)
            media_items=self._freeze_decoded_media(media_items,cancelled=cancelled)
        return self._run(messages,model=model,max_output_tokens=max_output_tokens,media_items=media_items,
            stop_prefixes=stop_prefixes,cancelled=cancelled,
            checkpoint_sink=checkpoint_sink,checkpoint_max_bytes=checkpoint_max_bytes)

    def stream_complete_checkpointed(self,messages,*,checkpoint_sink,checkpoint_max_bytes,
                                     model=None,max_output_tokens=None,media_items=None,stop_prefixes=(),cancelled=None):
        if not callable(checkpoint_sink) or type(checkpoint_max_bytes) is not int or checkpoint_max_bytes<=0:
            raise ProviderConfigurationError('checkpoint sink and positive complete frame budget required')
        if media_items is not None:
            self._validate_decoded_media(media_items,max_output_tokens)
            media_items=self._freeze_decoded_media(media_items,cancelled=cancelled)
        # Keep stream construction a normal function (not a generator body) so
        # decoded media is detached before control returns to the caller. This
        # prevents mutation/reuse between iterator creation and its first pull.
        return self._stream_run(messages,model=model,max_output_tokens=max_output_tokens,
            media_items=media_items,stop_prefixes=stop_prefixes,cancelled=cancelled,
            checkpoint_sink=checkpoint_sink,checkpoint_max_bytes=checkpoint_max_bytes)

    def complete_checkpoint_file(self,messages,destination,*,checkpoint_max_bytes,
                                 model=None,max_output_tokens=None,media_items=None,stop_prefixes=(),cancelled=None):
        """Publish complete recovery frames to one caller-selected local file."""
        from .generation_checkpoint_files import AtomicGenerationCheckpoint
        store=AtomicGenerationCheckpoint(destination,max_bytes=checkpoint_max_bytes)
        try:
            result=self.complete_checkpointed(messages,checkpoint_sink=store,
                checkpoint_max_bytes=checkpoint_max_bytes,model=model,
                max_output_tokens=max_output_tokens,media_items=media_items,
                stop_prefixes=stop_prefixes,cancelled=cancelled)
            metadata=dict(result.metadata,checkpoint_receipt=store.receipt)
            return ModelResult(result.provider,result.model,result.content,metadata)
        finally:store.close()

    def resume_checkpoint_file(self,source,*,checkpoint_max_bytes,max_output_tokens,
                               receipt=None,destination=None,model=None,emit=None,cancelled=None,
                               stop_prefixes=()):
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
                    emit=emit,cancelled=cancelled,stop_prefixes=stop_prefixes)
            return ModelResult(result.provider,result.model,result.content,
                dict(result.metadata,checkpoint_receipt=None if writer is None else writer.receipt))
        finally:
            try:
                if writer is not None:writer.close()
            finally:reader.close()

    def stream_resume_checkpoint(self,source,*,checkpoint_max_bytes,max_output_tokens,
                                 model=None,checkpoint_sink=None,stop_prefixes=(),cancelled=None):
        """Stream recovered tokens using the same cancellation and bounded queue."""
        yield from self._stream_run(None,model=model,max_output_tokens=max_output_tokens,
            recovery_source=source,checkpoint_sink=checkpoint_sink,checkpoint_max_bytes=checkpoint_max_bytes,
            stop_prefixes=stop_prefixes,cancelled=cancelled)

    def resume_checkpoint(self,source,*,checkpoint_max_bytes,max_output_tokens,model=None,
                          checkpoint_sink=None,emit=None,cancelled=None,stop_prefixes=()):
        """Restore a complete supported decoder and sampler boundary under gateway ownership."""
        from .execution_service import ExecutionCancelled
        if (type(checkpoint_max_bytes) is not int or checkpoint_max_bytes<=0
                or type(max_output_tokens) is not int or not 1<=max_output_tokens<=100000
                or (checkpoint_sink is not None and not callable(checkpoint_sink))):
            raise ProviderConfigurationError('bounded recovery and continuation options required')
        stop_prefixes = self._validate_stop_prefixes(stop_prefixes)
        with self._request_lock(cancelled):
            if self._closed:raise ProviderUnavailableError('native provider is closed')
            config=self._configuration()
            if model is not None and model!=config['model']:
                raise ProviderConfigurationError('requested model differs from configured checkpoint')
            self._idle_seconds=config.get('idle_timeout_seconds')
            self._lifetime_seconds=config.get('max_resident_seconds')
            operator,args=self._operator(config)
            if stop_prefixes:
                args.stop_prefix = [','.join(str(token) for token in prefix)
                                    for prefix in stop_prefixes]
            key=self._residency_key(config,args)
            reused=self._session is not None and self._session_key==key
            if not reused:
                self._release_session()
                started=time.monotonic()
                try:self._session=self._construct_session(operator,args,cancelled)
                except ExecutionCancelled:
                    raise
                except (ValueError,OSError,MemoryError,RuntimeError,ImportError) as error:
                    raise ProviderUnavailableError('recovery session could not load: '+failure_detail(error)) from error
                self._session_key=key;self._loaded_at=started;self._last_used=time.monotonic()
            tokenizer = self._session.tokenizer
            emitted='';stopped=False;stop_start_index=None;token_history=[]
            visible_ids=[];pending_stop=[];recovered_token_count=0
            skip_special=config['family']!='kimi_k3'

            def consume_token(token):
                """Hold possible stop prefixes in token space, including across restore."""
                nonlocal stopped,stop_start_index
                index=len(token_history)
                token_history.append(token)
                if stopped:
                    return
                pending_stop.append((token,index))
                while pending_stop:
                    candidate=tuple(value for value,_ in pending_stop)
                    if candidate in stop_prefixes:
                        stop_start_index=len(token_history)-len(candidate)
                        stopped=True;pending_stop.clear()
                        return
                    if any(prefix[:len(candidate)]==candidate for prefix in stop_prefixes):
                        return
                    value,value_index=pending_stop.pop(0)
                    if value_index>=recovered_token_count:
                        visible_ids.append(value)

            def publish_visible(row_tokens):
                nonlocal emitted
                if emit is None:
                    return
                text=tokenizer.decode(visible_ids,skip_special_tokens=skip_special)
                if text.endswith('\ufffd'):
                    return
                if not text.startswith(emitted):
                    raise ProviderUnavailableError('recovery tokenizer changed streamed prefix')
                delta=text[len(emitted):]
                if delta:
                    emit(StreamChunk(self.id,config['model'],delta,False,
                        dict(network=False,offline=True,recovered=True,committed=False,
                             token_ids=list(row_tokens))))
                    emitted=text

            def recovery_cancelled():
                # The decoder already owns and enforces the persisted stop
                # policy. Treating a matched stop as cancellation turns a
                # normal finish into ExecutionCancelled on its next poll.
                return cancelled is not None and cancelled()
            def event(row):
                nonlocal emitted,recovered_token_count,stop_prefixes
                if row.get('event')=='recovered':
                    recovered_text=row.get('text','')
                    recovered_ids=row.get('token_ids', ())
                    if (type(recovered_text) is not str or type(recovered_ids) not in (list,tuple) or
                            any(type(token) is not int or token<0 for token in recovered_ids)):
                        raise ProviderUnavailableError('recovery boundary lacks its token history')
                    if 'stop_prefixes' in row:
                        persisted=self._validate_stop_prefixes(row['stop_prefixes'])
                        if stop_prefixes and stop_prefixes!=persisted:
                            raise ProviderConfigurationError('recovery stop-prefix policy differs from the checkpoint')
                        stop_prefixes=persisted
                    recovered_token_count=len(recovered_ids)
                    for token in recovered_ids:
                        consume_token(token)
                    return
                if row.get('event')!='tokens':return
                token_ids=row.get('token_ids', ())
                if type(token_ids) not in (list, tuple) or any(type(token) is not int for token in token_ids):
                    raise ProviderUnavailableError('recovery token event lacks an integer token inventory')
                for token in token_ids:
                    consume_token(token)
                publish_visible(token_ids)
            try:
                if config['family']=='kimi_k3':
                    output=self._session.resume_generation(source,checkpoint_max_bytes=checkpoint_max_bytes,
                        max_tokens=min(max_output_tokens,args.max_tokens),checkpoint_sink=checkpoint_sink,
                        emit=event if emit is not None else None,cancelled=recovery_cancelled)
                else:
                    output=self._session.resume_generation(source,max_bytes=checkpoint_max_bytes,
                        max_new_tokens=min(max_output_tokens,args.max_tokens),checkpoint_sink=checkpoint_sink,
                        max_checkpoint_bytes=checkpoint_max_bytes if checkpoint_sink is not None else 0,
                        emit=event if emit is not None else None,cancelled=recovery_cancelled)
                generated_ids=output.get('generated_ids')
                if (type(generated_ids) not in (list,tuple) or
                        any(type(token) is not int or token<0 for token in generated_ids)):
                    raise ProviderUnavailableError('final recovery output lacks generated token IDs')
                if emit is not None and tuple(generated_ids)!=tuple(token_history):
                    raise ProviderUnavailableError('final recovery token inventory differs from captured events')
                if 'stop_prefixes' in output:
                    persisted=self._validate_stop_prefixes(output['stop_prefixes'])
                    if stop_prefixes and stop_prefixes!=persisted:
                        raise ProviderUnavailableError('final recovery stop-prefix policy changed')
                    stop_prefixes=persisted
                if config['family']=='kimi_k3':
                    output_base=output.get('recovered_token_count',recovered_token_count)
                    if (type(output_base) is not int or output_base<0 or
                            output_base>len(generated_ids) or
                            (emit is not None and output_base!=recovered_token_count)):
                        raise ProviderUnavailableError('K3 recovery boundary token count differs')
                    recovered_token_count=output_base
                    if stop_start_index is None:
                        visible_history,stopped_at_boundary=self._strip_terminal_stop_prefix(
                            generated_ids,stop_prefixes)
                    else:
                        visible_history=tuple(generated_ids[:stop_start_index])
                        stopped_at_boundary=True
                    resumed_ids=output.get('resumed_ids')
                    if (type(resumed_ids) not in (list,tuple) or
                            any(type(token) is not int or token<0 for token in resumed_ids)):
                        raise ProviderUnavailableError('K3 recovery output lacks continuation token IDs')
                    if tuple(generated_ids[recovered_token_count:])!=tuple(resumed_ids):
                        raise ProviderUnavailableError('K3 continuation IDs differ from restored token history')
                    content_ids=(visible_history[recovered_token_count:]
                        if len(visible_history)>=recovered_token_count else ())
                    stopped=stopped or stopped_at_boundary
                elif config['family']=='deepseek_v41':
                    recovered_token_count=output.get('recovered_token_count')
                    resumed_ids=output.get('resumed_ids')
                    terminal_start=output.get('terminal_stop_start')
                    if (type(recovered_token_count) is not int or recovered_token_count<0 or
                            recovered_token_count>len(generated_ids) or
                            type(resumed_ids) not in (list,tuple) or
                            any(type(token) is not int or token<0 for token in resumed_ids) or
                            tuple(generated_ids[recovered_token_count:])!=tuple(resumed_ids) or
                            (terminal_start is not None and type(terminal_start) is not int)):
                        raise ProviderUnavailableError('V4.1 recovery continuation inventory differs')
                    if stop_start_index is not None:
                        terminal_start=stop_start_index
                    if terminal_start is None:
                        visible_history,stopped=self._strip_terminal_stop_prefix(generated_ids,stop_prefixes)
                    else:
                        visible_history=tuple(generated_ids[:max(0,terminal_start)])
                        stopped=True
                    content_ids=(visible_history[recovered_token_count:]
                        if len(visible_history)>=recovered_token_count else ())
                else:
                    if stop_start_index is None:
                        content_ids,stopped=self._strip_terminal_stop_prefix(generated_ids,stop_prefixes)
                    else:
                        content_ids=tuple(generated_ids[:stop_start_index]);stopped=True
                # A stop sequence can begin in the restored history. In that
                # case the visible history is shorter than its old boundary;
                # no token from that sequence belongs in this continuation.
                pending_stop.clear()
                content=tokenizer.decode(list(content_ids),skip_special_tokens=skip_special)
                if emit is not None and not content.startswith(emitted):
                    raise ProviderUnavailableError('final recovery output differs from token-level stream')
                if emit is not None and len(content)>len(emitted):
                    emit(StreamChunk(self.id,config['model'],content[len(emitted):],False,
                        dict(network=False,offline=True,recovered=True,committed=True)))
                return ModelResult(self.id,config['model'],content,
                    dict(network=False,offline=True,recovered=True,family=config['family'],runtime=output,
                        resident_session_reused=reused,numerical_parity_verified=False,platform_certified=False))
            except ExecutionCancelled:
                raise
            except (ValueError,OSError,MemoryError,RuntimeError,ArithmeticError) as error:
                try:self._release_session()
                except Exception as cleanup:add_failure_note(error, 'recovery session release failed: '+failure_summary(cleanup))
                raise ProviderUnavailableError('native recovery failed: '+failure_detail(error)) from error

    def _run(self, messages, *, model=None, emit=None, cancelled=None, image=None,
             max_output_tokens=None,media_items=None,stop_prefixes=(),stop_sequences=(),
             checkpoint_sink=None,checkpoint_max_bytes=None,queue_timeout=None):
        """Track media through the production publisher request boundary.

        Conditional decoder executors have their own processor-array lifecycle,
        but K3 and V4.1 use different owners. Keeping this wrapper at the shared
        serving boundary gives every publisher family the same terminal
        cleanup contract without imposing one family's media representation on
        another family's decoder.
        """
        if image is None and media_items is None:
            return self._run_impl(
                messages, model=model, emit=emit, cancelled=cancelled,
                image=image, max_output_tokens=max_output_tokens,
                media_items=media_items, stop_prefixes=stop_prefixes,
                stop_sequences=stop_sequences, checkpoint_sink=checkpoint_sink,
                checkpoint_max_bytes=checkpoint_max_bytes,
                queue_timeout=queue_timeout)

        from .execution_service import ExecutionCancelled
        config = self._configuration()
        family = {
            'qwen3_5_moe': 'qwen35',
            'qwen4_exp': 'qwen4',
            'glm5_next': 'glm5-next',
            'inkling_mm_model': 'inkling',
            'kimi_k3': 'kimi-k3',
            'deepseek_v41': 'deepseek-v41',
        }.get(config.get('family'))
        if family is None:
            raise ProviderConfigurationError(
                'selected publisher family has no admitted media lifecycle')
        if image is not None:
            if type(image) is not bytes or not image:
                raise ProviderConfigurationError('encoded image bytes required for media lifecycle')
            media_types = ['vision' if family == 'deepseek-v41' else 'image']
            input_bytes = len(image)
        else:
            items, input_bytes = self._validate_decoded_media(
                media_items, max_output_tokens)
            # Reject unsupported cross-modality combinations before creating
            # lifecycle state. K3 is the only publisher consumer with an
            # explicit shared image/video ordering contract.
            self._validate_media_modality_order(config['family'], items)
            from .k3_execution.video_input import K3VideoInput
            from .publisher_media_inputs import PublisherVideoInput
            media_types = []
            for item in items:
                media_type = ('video' if type(item) in
                              (K3VideoInput, PublisherVideoInput) else
                              'vision' if family == 'deepseek-v41' else 'image')
                if media_type not in media_types:
                    media_types.append(media_type)
        lifecycle = self._media_lifecycle_registry.enable(family)
        frame_id = lifecycle.admit_processed(
            tuple(media_types), input_byte_count=input_bytes)
        primary = None
        def record_media_dispatch():
            # The lifecycle counts consumed media chunks, not generated text
            # events. Video containers can contain several independently
            # dispatched frames; preserve that count at the native call edge.
            count = 1
            if media_items is not None:
                from .k3_execution.video_input import K3VideoInput
                from .publisher_media_inputs import PublisherVideoInput
                count = sum(
                    len(item.frames) if type(item) in
                    (K3VideoInput, PublisherVideoInput) else 1
                    for item in media_items)
            for _ in range(count):
                lifecycle.on_stream_chunk(frame_id)

        try:
            result = self._run_impl(
                messages, model=model, emit=emit,
                cancelled=cancelled, image=image,
                max_output_tokens=max_output_tokens, media_items=media_items,
                stop_prefixes=stop_prefixes, stop_sequences=stop_sequences,
                checkpoint_sink=checkpoint_sink,
                checkpoint_max_bytes=checkpoint_max_bytes,
                queue_timeout=queue_timeout,
                _media_chunk_callback=record_media_dispatch)
        except BaseException as error:
            primary = error
            try:
                is_cancelled = isinstance(error, ExecutionCancelled)
                if not is_cancelled and callable(cancelled):
                    try:
                        is_cancelled = bool(cancelled())
                    except BaseException as cancellation_error:
                        add_failure_note(error,
                            'media cancellation observer failed: '+failure_summary(cancellation_error))
                if is_cancelled:
                    lifecycle.cancel(frame_id, str(error) or 'cancelled')
                else:
                    lifecycle.error(frame_id, error)
            except BaseException as cleanup:
                add_failure_note(error,
                    'publisher media lifecycle terminal cleanup failed: '+failure_summary(cleanup))
            try:
                lifecycle.release(frame_id)
            except BaseException as cleanup:
                add_failure_note(error,
                    'publisher media lifecycle release failed: '+failure_summary(cleanup))
            raise
        try:
            lifecycle.complete(frame_id)
        except BaseException as error:
            primary = error
            try:
                lifecycle.error(frame_id, error)
            except BaseException as cleanup:
                add_failure_note(error,
                    'publisher media lifecycle completion cleanup failed: '+failure_summary(cleanup))
            raise
        finally:
            try:
                lifecycle.release(frame_id)
            except BaseException as cleanup:
                if primary is None:
                    raise
                add_failure_note(primary,
                    'publisher media lifecycle release failed: '+failure_summary(cleanup))
        return result

    def _run_impl(self, messages, *, model=None, emit=None, cancelled=None, image=None,
             max_output_tokens=None,media_items=None,stop_prefixes=(),stop_sequences=(),
             checkpoint_sink=None,checkpoint_max_bytes=None,queue_timeout=None,
             _media_chunk_callback=None):
        from .execution_service import ExecutionCancelled
        if image is not None and media_items is not None:
            raise ProviderConfigurationError('one encoded image or decoded media sequence required')
        if media_items is not None:
            if type(media_items) not in (list, tuple):
                raise ProviderConfigurationError('decoded media sequence must be an explicit list or tuple')
            if not media_items:
                raise ProviderConfigurationError('decoded media sequence must be nonempty')
            # Preserve caller order and membership for the complete request;
            # the media input objects own their detached pixel/frame buffers.
            media_items = tuple(media_items)
        stop_prefixes = self._validate_stop_prefixes(stop_prefixes)
        if (type(stop_sequences) not in (tuple, list) or len(stop_sequences) > 4 or
                any(type(item) is not str or not item or
                    self._utf8_size(item, label='stop sequence') > 4096
                    for item in stop_sequences)):
            raise ProviderConfigurationError('bounded nonempty UTF-8 stop sequence list required')
        if (max_output_tokens is not None and
                (type(max_output_tokens) is not int or max_output_tokens <= 0)):
            raise ProviderConfigurationError('positive integer output limit required')
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
        message_bytes = sum(self._utf8_size(item['content'], label='native chat message')
                            for item in messages)
        if message_bytes > 1024**2:
            raise ProviderConfigurationError('native chat input exceeds 1 MiB')
        if queue_timeout is not None and (type(queue_timeout) not in (int, float) or
                not math.isfinite(queue_timeout) or queue_timeout < 0 or queue_timeout > 300):
            raise ProviderConfigurationError('queue timeout must be between 0 and 300 seconds')
        with self._request_lock(cancelled, queue_timeout=queue_timeout):
            if self._closed:
                raise ProviderUnavailableError('native provider is closed')
            config = self._configuration()
            self._idle_seconds = config.get('idle_timeout_seconds')
            self._lifetime_seconds = config.get('max_resident_seconds')
            if model is not None and model != config['model']:
                raise ProviderConfigurationError('requested model differs from configured native checkpoint')
            self._validate_media_modality_order(config['family'], media_items)
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
                video_placeholder=config.get('video_placeholder')
                if any(placeholder in item['content'] or
                       (video_placeholder is not None and video_placeholder in item['content'])
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
                raise ProviderConfigurationError('native chat template could not render: '+failure_detail(error)) from error
            if self._utf8_size(prompt, label='rendered native prompt') > 2*1024**2:
                raise ProviderConfigurationError('rendered native prompt exceeds 2 MiB')
            key = self._residency_key(config, args)
            reused = self._session is not None and self._session_key == key
            if not reused:
                self._release_session()
                loading_started = time.monotonic()
                try:
                    self._session = self._construct_session(operator,args,cancelled)
                except ExecutionCancelled:
                    raise
                except (ValueError, OSError, MemoryError, RuntimeError, ImportError) as error:
                    raise ProviderUnavailableError('native session could not load: '+failure_detail(error)) from error
                self._session_key = key
                self._loaded_at = loading_started
                self._last_used = time.monotonic()
            tokenizer = self._session.tokenizer
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
            encoded_stops = self._encode_stop_sequences(tokenizer, stop_sequences)
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
            configured_prefixes = (stop_prefixes if stop_prefixes else
                self._parse_stop_prefix_arguments(getattr(args, 'stop_prefix', ()) or ()))
            stop_prefixes = tuple(dict.fromkeys((*configured_prefixes, *encoded_stops)))
            if len(stop_prefixes) > 64:
                raise ProviderConfigurationError('combined stop-prefix inventory exceeds admission')
            if stop_prefixes:
                args.stop_prefix = [','.join(str(token) for token in prefix)
                                    for prefix in stop_prefixes]
            emitted = ''
            generated_visible = []
            pending_stop = []
            stopped_by_prefix = False
            def on_event(event):
                nonlocal emitted, stopped_by_prefix
                if emit is None or event.get('event') != 'tokens':
                    return
                token_ids = event.get('token_ids')
                if type(token_ids) not in (list, tuple):
                    raise ProviderUnavailableError('native token event has no token-ID sequence')
                for token in token_ids:
                    if stopped_by_prefix:
                        break
                    if type(token) is not int or token < 0:
                        raise ProviderUnavailableError('native token event contains an invalid token ID')
                    pending_stop.append(token)
                    while pending_stop:
                        candidate = tuple(pending_stop)
                        if candidate in stop_prefixes:
                            pending_stop.clear()
                            stopped_by_prefix = True
                            break
                        if any(prefix[:len(candidate)] == candidate for prefix in stop_prefixes):
                            break
                        generated_visible.append(pending_stop.pop(0))
                text = tokenizer.decode(generated_visible,
                    skip_special_tokens=config['family'] != 'kimi_k3')
                # Incomplete UTF-8 tokens may temporarily decode to replacement
                # characters. Hold them until a later token completes the bytes.
                if text.endswith('\ufffd'):
                    return
                if not text.startswith(emitted):
                    raise ProviderUnavailableError('tokenizer changed previously streamed text')
                delta = text[len(emitted):]
                if delta:
                    emit(StreamChunk(self.id, config['model'], delta, False,
                        dict(network=False, offline=True, committed=False,
                             token_ids=list(event['token_ids']), prompt_tokens=len(prompt_ids),
                             queue_wait_ms=round(getattr(self._request_local,
                                 'queue_wait_seconds', 0.0)*1000))))
                    emitted = text
            try:
                if _media_chunk_callback is not None and (
                        image is not None or media_items is not None):
                    _media_chunk_callback()
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
            except ExecutionCancelled:
                raise
            except (ValueError, OSError, MemoryError, RuntimeError, ArithmeticError) as error:
                try:
                    self._release_session()
                except Exception as cleanup_error:
                    raise ProviderUnavailableError(
                        'native decoder failed: '+failure_detail(error)+'; cleanup failed: '+failure_detail(cleanup_error)) from error
                raise ProviderUnavailableError('native decoder failed: '+failure_detail(error)) from error
            content = output.get('text')
            stopped_by_prefix = False
            if stop_prefixes:
                generated_ids = output.get('generated_ids')
                if (type(generated_ids) not in (list, tuple) or
                        any(type(token) is not int or token < 0 for token in generated_ids)):
                    raise ProviderUnavailableError('native output has no valid generated-token sequence')
                visible_ids, stopped_by_prefix = self._strip_terminal_stop_prefix(
                    generated_ids, stop_prefixes)
                content = tokenizer.decode(visible_ids,
                    skip_special_tokens=config['family'] != 'kimi_k3')
            elif type(content) is not str:
                content = tokenizer.decode(output['generated_ids'])
            matched_stop_sequence = (self._matched_stop_sequence(
                output.get('generated_ids', ()), tokenizer, stop_sequences)
                if stopped_by_prefix else None)
            if matched_stop_sequence is not None:
                # Preserve the exact user-visible stop cause for API adapters.
                # The generation engine's generic "stop" reason also covers
                # EOS and checkpoint-configured token stops.
                output = dict(output)
                output['finish_reason'] = 'stop_sequence'
                output['stop_sequence'] = matched_stop_sequence
            if not content.strip() and not stopped_by_prefix:
                raise ProviderUnavailableError('native decoder returned no text')
            if emit is not None:
                if not content.startswith(emitted):
                    raise ProviderUnavailableError('final output differs from streamed tokenizer text')
                if len(content) > len(emitted):
                    emit(StreamChunk(self.id, config['model'], content[len(emitted):], False,
                        dict(network=False, offline=True, committed=True,
                             prompt_tokens=len(prompt_ids),
                             queue_wait_ms=round(getattr(self._request_local,
                                 'queue_wait_seconds', 0.0)*1000))))
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
                     resident_session_reused=reused,
                     queue_wait_ms=round(getattr(self._request_local, 'queue_wait_seconds', 0.0)*1000),
                     queue_position=getattr(self._request_local, 'queue_position', 0)))

    def stream_complete(self, messages, *, model=None, max_output_tokens=None, stop_prefixes=(),
                        stop_sequences=(), cancelled=None, queue_timeout=None):
        return self._stream_run(messages, model=model, max_output_tokens=max_output_tokens,
                                stop_prefixes=stop_prefixes, stop_sequences=stop_sequences,
                                cancelled=cancelled,
                                queue_timeout=queue_timeout)

    def _stream_run(self, messages, *, model=None, image=None, max_output_tokens=None,media_items=None,
                    stop_prefixes=(), stop_sequences=(),
                    checkpoint_sink=None,checkpoint_max_bytes=None,recovery_source=None,
                    cancelled=None, queue_timeout=None):
        from .execution_service import ExecutionCancelled
        if cancelled is not None and not callable(cancelled):
            raise ProviderConfigurationError('stream cancellation must be callable')
        if (max_output_tokens is not None and
                (type(max_output_tokens) is not int or max_output_tokens <= 0)):
            raise ProviderConfigurationError('positive integer output limit required')
        stop_prefixes = self._validate_stop_prefixes(stop_prefixes)
        if messages is not None:
            if (type(messages) is not list or not messages or len(messages)>10000 or
                    any(type(item) is not dict or item.get('role') not in
                        ('system','user','assistant','tool') or type(item.get('content')) is not str
                        for item in messages)):
                raise ProviderConfigurationError('validated textual chat messages required')
            message_bytes = sum(self._utf8_size(item['content'], label='native chat message')
                                for item in messages)
            if message_bytes>1024**2:
                raise ProviderConfigurationError('native chat input exceeds 1 MiB')
            messages=[{'role':item['role'],'content':item['content']} for item in messages]
        queue = Queue(maxsize=8)
        stopped = threading.Event()
        def request_cancelled():
            if cancelled is None:return False
            # User callbacks belong to the iterator consumer thread. The
            # worker observes only the event set by that consumer, avoiding
            # cross-thread callback reentry and thread-affinity surprises.
            return bool(cancelled())
        # A transport-owned Event is different from an arbitrary predicate:
        # it is safe for the backend worker to observe directly. This matters
        # while the consumer is suspended at ``yield`` (for example, blocked
        # writing an SSE frame), when it cannot poll the callback and set the
        # local stop flag itself. Never invoke a user callback on the worker.
        worker_cancel_event = getattr(cancelled, 'worker_event', None)
        if not isinstance(worker_cancel_event, threading.Event):
            worker_cancel_event = None
        def is_cancelled():
            return stopped.is_set() or (worker_cancel_event is not None and
                                        worker_cancel_event.is_set())
        # Do not allocate a worker or enter model loading when cancellation
        # was already signalled before the stream was first advanced. This
        # check stays on the consumer thread; later cancellations continue to
        # flow through `stopped` to the worker at its normal safe points.
        if request_cancelled():
            raise ExecutionCancelled('native stream request cancelled before worker startup')
        def put(item):
            while not stopped.is_set():
                try:
                    queue.put(item, timeout=0.1)
                    return
                except Full:
                    pass
            raise ExecutionCancelled('native stream consumer closed or request cancelled')
        def worker():
            try:
                if recovery_source is not None:
                    result=self.resume_checkpoint(recovery_source,model=model,emit=put,cancelled=is_cancelled,
                        checkpoint_max_bytes=checkpoint_max_bytes,max_output_tokens=max_output_tokens,
                        checkpoint_sink=checkpoint_sink,stop_prefixes=stop_prefixes)
                else:
                    result = self._run(messages, model=model, emit=put, cancelled=is_cancelled,
                        image=image, max_output_tokens=max_output_tokens,media_items=media_items,
                        stop_prefixes=stop_prefixes, stop_sequences=stop_sequences,
                        checkpoint_sink=checkpoint_sink,checkpoint_max_bytes=checkpoint_max_bytes,
                        queue_timeout=queue_timeout)
                # This terminal event is delivered only after the request
                # transaction returns successfully, unlike provisional token
                # deltas sent from the callback during generation.
                put(StreamChunk(result.provider, result.model, '', True,
                    {**result.metadata, 'committed': True}))
            except BaseException as error:
                if not stopped.is_set():
                    try:
                        put(error)
                    except ExecutionCancelled:
                        pass
            finally:
                # Publish the terminal worker state while the ownership lock
                # is still held.  Lifecycle/status readers must never see a
                # cleared thread marker before the stream ownership token is
                # released, or vice versa.
                with self._lock:
                    if self._stream_thread is threading.current_thread():
                        self._stream_thread = None
                self._stream_lock.release()
        thread = threading.Thread(target=worker, name='native-publisher-stream', daemon=True)
        if not self._stream_lock.acquire(blocking=False):
            raise ProviderUnavailableError('native stream worker is still active')
        with self._lock:
            if self._closed:
                self._stream_lock.release()
                raise ProviderUnavailableError('native provider is closed')
            self._stream_thread = thread
        try:
            thread.start()
        except BaseException:
            with self._lock:
                if self._stream_thread is thread:
                    self._stream_thread = None
            self._stream_lock.release()
            raise
        try:
            while True:
                try:
                    item = queue.get(timeout=0.1)
                except Empty:
                    if request_cancelled():
                        stopped.set()
                        raise ExecutionCancelled('native stream request cancelled')
                    if not thread.is_alive():
                        raise ProviderUnavailableError('native stream ended without a completion event')
                    continue
                # Cancellation can arrive while events are already buffered;
                # checking only on Queue.Empty would still publish those stale
                # token events after the caller had stopped the request.
                if request_cancelled():
                    stopped.set()
                    raise ExecutionCancelled('native stream request cancelled before event dispatch')
                if isinstance(item, BaseException):
                    raise item
                yield item
                if item.done:
                    return
        finally:
            stopped.set()
            # Native calls cannot be interrupted safely. The worker releases
            # owned resources after the current decoding operation returns.
            thread.join(timeout=1.0)
