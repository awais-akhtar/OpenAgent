"""Local conditional-generation operator with owned native media sessions."""
import argparse
import hashlib
import json
from pathlib import Path
import threading
import os

from .native_family_cli import _Cancellation,_pairs
from .execution_service import ExecutionCancelled,_admit_checkpoint_publication
from .session_recovery import _path
from .native_library_identity import _stamp
from .publisher_media_files import read_regular

def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('--family',required=True,choices=('qwen3_5_moe','qwen4_exp','inkling_mm_model','glm5_next'))
    result.add_argument('--checkpoint',required=True,type=Path)
    result.add_argument('--media-library',required=True,type=Path)
    result.add_argument('--text-library',required=True,type=Path)
    result.add_argument('--tokenizer',type=Path)
    result.add_argument('--context',required=True,type=int)
    result.add_argument('--runtime-mib',required=True,type=int)
    result.add_argument('--aggregate-runtime-mib',required=True,type=int)
    result.add_argument('--checkpoint-buffer-mib',type=int,default=16)
    result.add_argument('--staging-mib',required=True,type=int)
    result.add_argument('--media-mib',required=True,type=int)
    result.add_argument('--image-mib',required=True,type=int)
    result.add_argument('--max-pixels',required=True,type=int)
    result.add_argument('--max-audio-samples',required=True,type=int)
    result.add_argument('--mtp-mib',type=int,default=0)
    result.add_argument('--mtp-work-mib',type=int,default=0)
    result.add_argument('--publisher-expert-mib',type=int,default=0)
    result.add_argument('--text-device')
    result.add_argument('--projection-mib',type=int,default=0)
    result.add_argument('--segment-runtime-library',type=Path)
    result.add_argument('--segment-work-mib',type=int,default=0)
    result.add_argument('--segment-layer-span',type=int,default=0)
    result.add_argument('--native-composition',action='store_true',help='Execute the complete Qwen4 composition in C')
    result.add_argument('--native-store-runtime-library',type=Path)
    result.add_argument('--native-expert-mib',type=int,default=0)
    result.add_argument('--native-ple-mib',type=int,default=0)
    result.add_argument('--prefill-chunk',required=True,type=int)
    result.add_argument('--max-tokens',type=int,default=128)
    result.add_argument('--token-ids',required=True)
    result.add_argument('--prompt-text')
    result.add_argument('--image',type=Path,action='append',default=[])
    result.add_argument('--audio-wav',type=Path,action='append',default=[])
    result.add_argument('--video-manifest',type=Path,action='append',default=[])
    result.add_argument('--image-token-id',required=True,type=int)
    result.add_argument('--include-vision',action='store_true')
    result.add_argument('--stop-token-id',type=int,action='append',default=[])
    result.add_argument('--temperature',type=float,default=0.0)
    result.add_argument('--top-k',type=int,default=0)
    result.add_argument('--top-p',type=float,default=1.0)
    result.add_argument('--min-p',type=float,default=0.0)
    result.add_argument('--repetition-penalty',type=float,default=1.0)
    result.add_argument('--frequency-penalty',type=float,default=0.0)
    result.add_argument('--presence-penalty',type=float,default=0.0)
    result.add_argument('--seed',type=int,default=0)
    result.add_argument('--continue-session',action='store_true')
    result.add_argument('--restore-cache',type=Path)
    result.add_argument('--save-cache',type=Path)
    result.add_argument('--cache-mib',type=int)
    return result

class PublisherSession:
    def __init__(self,args,*,cancelled=None):
        self._lock=threading.RLock();self._closed=False
        self.service=self.processor=self.tokenizer=self.session_id=None
        self._key=tuple((name,str(getattr(args,name,None))) for name in
            ('family','checkpoint','media_library','text_library','tokenizer','context','runtime_mib',
             'aggregate_runtime_mib','checkpoint_buffer_mib','staging_mib','media_mib','image_mib','max_pixels','max_audio_samples',
             'prefill_chunk','image_token_id','include_vision','mtp_mib','mtp_work_mib','publisher_expert_mib','text_device','projection_mib',
             'segment_runtime_library','segment_work_mib','segment_layer_span','native_composition',
             'native_store_runtime_library','native_expert_mib','native_ple_mib'))
        cancel=_Cancellation(cancelled)
        if type(args.publisher_expert_mib) is not int or args.publisher_expert_mib<0:
            raise ValueError('nonnegative original expert streaming budget required')
        if type(args.projection_mib) is not int or args.projection_mib<0:
            raise ValueError('nonnegative original device projection budget required')
        if (type(args.mtp_mib) is not int or type(args.mtp_work_mib) is not int or
                args.mtp_mib<0 or args.mtp_work_mib<0 or
                (args.mtp_mib==0 and args.mtp_work_mib!=0) or
                (args.mtp_mib>0 and not 0<args.mtp_work_mib<args.mtp_mib)):
            raise ValueError('MTP work reservation must fit explicit MTP admission')
        for name in ('context','runtime_mib','aggregate_runtime_mib','staging_mib','media_mib',
            'image_mib','max_pixels','max_audio_samples','prefill_chunk','max_tokens','checkpoint_buffer_mib'):
            if type(getattr(args,name)) is not int or getattr(args,name)<=0:
                raise ValueError('positive conditional admission required: '+name)
        root=_path(args.checkpoint).resolve(strict=True)
        self._artifact_root=root
        self.family=args.family
        raw=read_regular(root/'config.json',max_bytes=1024**2)
        if len(raw)>1024**2: raise MemoryError('conditional metadata exceeds admission')
        config=json.loads(raw,object_pairs_hook=_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        config_sha256=hashlib.sha256(raw).hexdigest()
        if type(config) is not dict or config.get('model_type')!=args.family:
            raise ValueError('conditional checkpoint identity differs from configured family')
        image_default={'inkling_mm_model':200054,'qwen3_5_moe':248056,'qwen4_exp':248056,'glm5_next':154854}[args.family]
        if args.image_token_id!=config.get('image_token_id',image_default):
            raise ValueError('configured image token differs from publisher checkpoint')
        from .publisher_family_contracts import component_contract
        c=component_contract(config,arithmetic='fp32').text_config
        native_composition=getattr(args,'native_composition',False)
        native_store_library=getattr(args,'native_store_runtime_library',None)
        native_expert_mib=getattr(args,'native_expert_mib',0);native_ple_mib=getattr(args,'native_ple_mib',0)
        if type(native_composition) is not bool or any(type(v) is not int or v<0 for v in (native_expert_mib,native_ple_mib)):
            raise ValueError('explicit bounded native Qwen4 composition options required')
        if native_composition:
            if args.family!='qwen4_exp' or args.text_device is not None:
                raise ValueError('native composition requires the Qwen4 native text path')
            if bool(native_expert_mib)!=bool(args.publisher_expert_mib) or bool(native_ple_mib)!=bool(c.get('ple_layer_ids')):
                raise ValueError('native bank budgets must match original streamed expert and PLE inventories')
            if bool(native_store_library is not None)!=bool(native_expert_mib or native_ple_mib):
                raise ValueError('native original store runtime and owner budgets must be supplied together')
        elif native_store_library is not None or native_expert_mib or native_ple_mib:
            raise ValueError('native original stores require complete native composition')
        self.vocabulary=c.get('unpadded_vocab_size') or c['vocab_size']
        eos=c.get('eos_token_id');self.stops=() if eos is None else (eos,) if type(eos) is int else tuple(eos)
        # Hash complete source artifacts before opening executable model state.
        paths=set(root.glob('*.safetensors'))|set(root.glob('*.json'))|set(root.glob('*.jinja'))
        paths.update(path for name in ('merges.txt','tokenizer.model') if (path:=root/name).exists())
        paths.update((_path(args.media_library).resolve(strict=True),_path(args.text_library).resolve(strict=True)))
        if native_store_library is not None:paths.add(_path(native_store_library).resolve(strict=True))
        segment_library=getattr(args,'segment_runtime_library',None)
        segment_work=getattr(args,'segment_work_mib',0)
        segment_span=getattr(args,'segment_layer_span',0)
        if type(segment_span) is not int or segment_span<0 or (segment_span and
                (segment_library is None or args.family not in ('inkling_mm_model','glm5_next','qwen4_exp'))):
            raise ValueError('native layer span requires a supported publisher Segment runtime')
        if type(segment_work) is not int or segment_work<0 or (segment_library is None)!=(segment_work==0):
            raise ValueError('native Segment runtime and positive work admission must be supplied together')
        if segment_library is not None:paths.add(_path(segment_library).resolve(strict=True))
        digest=hashlib.sha256();self._artifacts={}
        self._checkpoint_artifacts={path for path in paths if path.parent==root}
        for path in sorted(paths,key=str):
            path=_path(path)
            if path.is_symlink() or not path.is_file(): raise ValueError('regular conditional artifact required')
            before=path.stat();digest.update(str(path).encode());content=hashlib.sha256();count=0
            with path.open('rb') as stream:
                if _stamp(os.fstat(stream.fileno()))!=_stamp(before):
                    raise ValueError('conditional artifact changed during opening')
                for chunk in iter(lambda:stream.read(1024**2),b''):
                    if cancel.is_set(): raise ExecutionCancelled('conditional admission cancelled')
                    count+=len(chunk)
                    if count>before.st_size:raise ValueError('conditional artifact grew during admission')
                    digest.update(chunk);content.update(chunk)
                if count!=before.st_size or _stamp(os.fstat(stream.fileno()))!=_stamp(before):
                    raise ValueError('conditional artifact changed during reading')
            after=_path(path).stat()
            if _stamp(before)!=_stamp(after):
                raise ValueError('conditional artifact changed during admission')
            if path==root/'config.json' and content.hexdigest()!=config_sha256:
                raise ValueError('conditional configuration changed before artifact admission')
            self._artifacts[path]=_stamp(after)
        self.checkpoint_identity=digest.hexdigest()
        try:
            from .publisher_media_processor import PublisherMediaProcessor
            from .execution_factories import open_native_conditional_directory_service
            self.processor=PublisherMediaProcessor(root,family=args.family,max_bytes=args.media_mib*1024**2,
                max_pixels=args.max_pixels,max_audio_samples=args.max_audio_samples)
            self.tokenizer=self.processor.tokenizer
            if args.tokenizer is not None and _path(args.tokenizer).resolve(strict=True)!=root/'tokenizer.json':
                raise ValueError('conditional tokenization must use the owned publisher processor')
            self.service=open_native_conditional_directory_service(root,family=args.family,
                library=args.media_library,text_library=args.text_library,context_tokens=args.context,
                max_weight_bytes=args.staging_mib*1024**2,runtime_max_bytes=args.runtime_mib*1024**2,
                max_total_model_bytes=args.aggregate_runtime_mib*1024**2,max_media_bytes=args.media_mib*1024**2,
                max_checkpoint_buffer_bytes=args.checkpoint_buffer_mib*1024**2,
                max_new_tokens=args.context,max_sessions=1,prefill_chunk_tokens=args.prefill_chunk,
                mtp_max_bytes=args.mtp_mib*1024**2,mtp_work_bytes=args.mtp_work_mib*1024**2,
                max_expert_bytes=args.publisher_expert_mib*1024**2,text_device=args.text_device,
                max_projection_bytes=args.projection_mib*1024**2,
                segment_runtime_library=segment_library,segment_work_bytes=segment_work*1024**2,
                segment_layer_span=segment_span,native_composition=native_composition,
                native_store_runtime_library=native_store_library,native_expert_max_bytes=native_expert_mib*1024**2,
                native_ple_max_bytes=native_ple_mib*1024**2)
            self.session_id=self.service.open_session()
            self._artifacts_unchanged()
            if cancel.is_set(): raise ExecutionCancelled('conditional model load cancelled')
        except BaseException as error:
            previous=getattr(error,'cleanup',None)
            pending=([previous] if callable(previous) and previous!=self.close else [])+[self.close]
            def cleanup_session():
                while pending:
                    pending[0]();pending.pop(0)
            try:cleanup_session()
            except BaseException as failure:
                error.cleanup=cleanup_session;error.add_note('conditional session cleanup remains pending: '+str(failure))
            raise

    def _artifacts_unchanged(self):
        root=_path(self._artifact_root)
        current=set(root.glob('*.safetensors'))|set(root.glob('*.json'))|set(root.glob('*.jinja'))
        current.update(path for name in ('merges.txt','tokenizer.model') if (path:=root/name).exists())
        # Explicit libraries can also live in the checkpoint directory.
        current.update(path for path in self._checkpoint_artifacts
            if path.suffix not in ('.safetensors','.json','.jinja') and path.name not in ('merges.txt','tokenizer.model'))
        if current!=self._checkpoint_artifacts:
            raise ValueError('conditional checkpoint artifact inventory changed after admission')
        for path,stamp in self._artifacts.items():
            if _stamp(_path(path).stat())!=stamp:
                raise ValueError('conditional artifact changed after admission: '+str(path))

    def run(self,args,*,emit=None,cancelled=None,checkpoint_sink=None,max_checkpoint_bytes=0,media_items=None):
        with self._lock:
            _admit_checkpoint_publication(checkpoint_sink,max_checkpoint_bytes)
            if self._closed: raise RuntimeError('conditional session is closed')
            self._artifacts_unchanged()
            if self._key!=tuple((name,str(getattr(args,name,None))) for name,_ in self._key):
                raise ValueError('conditional residency options changed after admission')
            tokens=json.loads(args.token_ids);media=None
            images=[];encoded_extent=0
            if len(args.image)>64: raise ValueError('bounded conditional image inventory required')
            for value in args.image:
                import numpy as np
                from PIL import Image
                if isinstance(value,(np.ndarray,Image.Image)):
                    if isinstance(value,np.ndarray):
                        if value.dtype!=np.uint8 or value.ndim!=3 or value.shape[2]!=3:
                            raise ValueError('decoded conditional image must be uint8 RGB HWC')
                        height,width=value.shape[:2]
                    else:width,height=value.size
                    size=width*height*16
                    if width<=0 or height<=0 or size>args.image_mib*1024**2:
                        raise MemoryError('decoded conditional image exceeds input admission')
                    encoded_extent+=size
                    if encoded_extent>args.media_mib*1024**2:
                        raise MemoryError('aggregate decoded images exceed media admission')
                    images.append(value)
                    continue
                if type(value) is bytes: raw=value
                else:
                    raw=read_regular(value,max_bytes=args.image_mib*1024**2)
                if len(raw)>args.image_mib*1024**2: raise MemoryError('encoded image exceeds configured admission')
                encoded_extent+=len(raw)
                if encoded_extent>args.media_mib*1024**2: raise MemoryError('aggregate encoded images exceed media admission')
                images.append(raw)
            from .publisher_media_files import read_pcm_audio,read_selected_video
            audio=[];videos=[];metadata=[];rate=None;extent=encoded_extent
            if media_items is not None:
                from .publisher_media_inputs import PublisherVideoInput
                if type(media_items) not in (tuple,list) or not 1<=len(media_items)<=64 or images:
                    raise ValueError('one bounded decoded media sequence without encoded image inputs required')
                for item in media_items:
                    if type(item) is PublisherVideoInput:
                        if args.family not in ('qwen3_5_moe','qwen4_exp','glm5_next'):
                            raise ValueError('selected publisher video requires Qwen4 or GLM5 vision')
                        videos.append(item.frames);metadata.append(item.metadata())
                        frames=item.frames
                    else:
                        images.append(item);frames=(item,)
                    from .k3_execution.video_input import _dimensions
                    for frame in frames:
                        width,height=_dimensions(frame)
                        size=width*height*16
                        if width<=0 or height<=0 or size>args.image_mib*1024**2:
                            raise MemoryError('decoded publisher frame exceeds input admission')
                        extent+=size
                        if extent>args.media_mib*1024**2:
                            raise MemoryError('decoded publisher media exceeds aggregate admission')
                if getattr(args,'audio_wav',[]) or getattr(args,'video_manifest',[]):
                    raise ValueError('decoded media cannot be combined with file media')
            audio_paths=getattr(args,'audio_wav',[]);video_paths=getattr(args,'video_manifest',[])
            if len(images)+len(videos)+len(audio_paths)+len(video_paths)>64: raise ValueError('bounded media item count required')
            if audio_paths and args.family!='inkling_mm_model': raise ValueError('audio input requires Inkling')
            if video_paths and args.family not in ('qwen3_5_moe','qwen4_exp','glm5_next'): raise ValueError('selected video input requires a vision/video publisher')
            for path in audio_paths:
                waveform,current_rate=read_pcm_audio(path,max_bytes=args.media_mib*1024**2-extent,
                    max_samples=args.max_audio_samples)
                if rate is not None and rate!=current_rate: raise ValueError('audio items must share their explicit sampling rate')
                rate=current_rate;extent+=waveform.nbytes*16;audio.append(waveform)
            for path in video_paths:
                frames,info=read_selected_video(path,max_bytes=args.media_mib*1024**2-extent,
                    max_frame_bytes=args.image_mib*1024**2,cancelled=cancelled)
                extent+=sum(len(value) for value in frames);videos.append(frames);metadata.append(info)
            if images or audio or videos:
                if not args.include_vision or not args.prompt_text:
                    raise ValueError('raw media requires explicit media admission and rendered prompt text')
                tokens,media=self.processor.prepare(args.prompt_text,images=images,audio=audio,
                    sampling_rate=rate,videos=videos,video_metadata=metadata or None,cancelled=cancelled)
            elif type(tokens) is not list or not tokens:
                raise ValueError('nonempty conditional token inventory required')
            from .sampling import SamplingPolicy
            sampling=SamplingPolicy(temperature=args.temperature,top_k=args.top_k,top_p=args.top_p,
                min_p=args.min_p,repetition_penalty=args.repetition_penalty,
                frequency_penalty=args.frequency_penalty,presence_penalty=args.presence_penalty,seed=args.seed)
            if args.restore_cache is not None or args.save_cache is not None:
                if type(args.cache_mib) is not int or args.cache_mib<=0: raise ValueError('explicit durable cache budget required')
            if args.restore_cache is not None:
                self.service.load_session_cache(self.session_id,args.restore_cache,max_bytes=args.cache_mib*1024**2)
            elif not args.continue_session: self.service.reset(self.session_id)
            def event(value):
                if emit is not None and value.get('event')=='generation.token':
                    emit(dict(event='tokens',token_ids=[value['token_id']]))
            result=self.service.generate(self.session_id,tokens,max_new_tokens=args.max_tokens,
                stop_ids=(*self.stops,*args.stop_token_id),cancel=_Cancellation(cancelled),
                on_event=event,sampling=sampling,media=media or None,checkpoint_sink=checkpoint_sink,
                max_checkpoint_bytes=max_checkpoint_bytes,application_identity=self.checkpoint_identity)
            cache=None
            if args.save_cache is not None:
                cache=self.service.save_session_cache(self.session_id,args.save_cache,max_bytes=args.cache_mib*1024**2)
            generated=list(result.token_ids)
            return dict(generated_ids=generated,text=self.tokenizer.decode(generated,skip_special_tokens=True),
                family=args.family,checkpoint_identity=self.checkpoint_identity,cache=cache,
                finish_reason=result.finish_reason,context_tokens=result.context_tokens,
                numerical_parity_verified=False,platform_certified=False)

    def resume_generation(self,source,*,max_bytes,max_new_tokens=None,emit=None,cancelled=None,
                          checkpoint_sink=None,max_checkpoint_bytes=0):
        with self._lock:
            _admit_checkpoint_publication(checkpoint_sink,max_checkpoint_bytes)
            if self._closed:raise RuntimeError('conditional session is closed')
            self._artifacts_unchanged()
            if max_new_tokens is not None and (type(max_new_tokens) is not int or
                    not 1<=max_new_tokens<=self.service.limits.max_new_tokens):
                raise ValueError('admitted positive generation continuation required')
            cancel=_Cancellation(cancelled)
            boundary=self.service.restore_generation_checkpoint(self.session_id,source,max_bytes=max_bytes,
                cancel=cancel,application_identity=self.checkpoint_identity)
            generated_events=[]
            def event(value):
                if emit is not None and value.get('event')=='generation.token':
                    generated_events.append(value['token_id'])
                    emit(dict(event='tokens',token_ids=[value['token_id']],
                        text=self.tokenizer.decode(generated_events,skip_special_tokens=True)))
            result=self.service.resume_generation(self.session_id,boundary,max_new_tokens=max_new_tokens,
                cancel=cancel,on_event=event,checkpoint_sink=checkpoint_sink,
                max_checkpoint_bytes=max_checkpoint_bytes,application_identity=self.checkpoint_identity)
            generated=list(result.token_ids)
            return dict(generated_ids=generated,text=self.tokenizer.decode(generated,skip_special_tokens=True),
                family=self.family,checkpoint_identity=self.checkpoint_identity,finish_reason=result.finish_reason,
                context_tokens=result.context_tokens,numerical_parity_verified=False,platform_certified=False)

    def run_checkpoint_file(self,args,destination,*,max_bytes,emit=None,cancelled=None):
        from .generation_checkpoint_files import AtomicGenerationCheckpoint
        store=AtomicGenerationCheckpoint(destination,max_bytes=max_bytes)
        try:
            result=self.run(args,emit=emit,cancelled=cancelled,checkpoint_sink=store,max_checkpoint_bytes=max_bytes)
            result['generation_checkpoint']=store.receipt;return result
        finally:store.close()

    def resume_checkpoint_file(self,source,*,max_bytes,max_new_tokens=None,emit=None,cancelled=None,
                               destination=None,receipt=None):
        from .generation_checkpoint_files import AtomicGenerationCheckpoint
        incoming=AtomicGenerationCheckpoint(source,max_bytes=max_bytes)
        outgoing=incoming if destination is None else AtomicGenerationCheckpoint(destination,max_bytes=max_bytes)
        try:
            with incoming.source(receipt=receipt) as stream:
                result=self.resume_generation(stream,max_bytes=max_bytes,max_new_tokens=max_new_tokens,emit=emit,
                    cancelled=cancelled,checkpoint_sink=outgoing,max_checkpoint_bytes=max_bytes)
            result['generation_checkpoint']=outgoing.receipt;return result
        finally:
            outgoing.close()
            if incoming is not outgoing:incoming.close()

    def continuation_store(self,directory,*,max_entries,max_total_bytes,max_cache_bytes,
                           expiry_seconds=None,indexed=True):
        from .continuation_store import DurableContinuationStore
        from .indexed_continuation_store import IndexedContinuationStore
        with self._lock:
            if self._closed: raise RuntimeError('conditional session is closed')
            self._artifacts_unchanged()
            if type(indexed) is not bool:
                raise ValueError('explicit boolean continuation index policy required')
            factory=IndexedContinuationStore if indexed else DurableContinuationStore
            return factory(directory,model_identity=self.checkpoint_identity,max_entries=max_entries,
                max_total_bytes=max_total_bytes,max_cache_bytes=max_cache_bytes,expiry_seconds=expiry_seconds)

    def save_continuation(self,store,key):
        with self._lock:
            if self._closed or store.identity!=self.checkpoint_identity:
                raise ValueError('open matching conditional continuation owner required')
            self._artifacts_unchanged()
            return store.save(self.service,self.session_id,key)

    def recover_continuation(self,store,key):
        with self._lock:
            if self._closed or store.identity!=self.checkpoint_identity:
                raise ValueError('open matching conditional continuation owner required')
            self._artifacts_unchanged()
            previous=self.session_id
            # Single-resident services recover in place under their session lock;
            # opening a second decoder would exceed the admitted model count.
            return store.recover_into(self.service,previous,key)

    def close(self):
        with self._lock:
            if self.service is not None: self.service.close();self.service=None
            if self.processor is not None: self.processor.close();self.processor=None
            self.tokenizer=self.session_id=None;self._closed=True

def main(argv=None):
    args=parser().parse_args(argv);session=PublisherSession(args)
    try: print(json.dumps(session.run(args),sort_keys=True,allow_nan=False))
    finally: session.close()
    return 0

__all__=['parser','PublisherSession','main']

if __name__=='__main__': raise SystemExit(main())
