"""Source-only engine artifact receipts; no imports, builds or execution probes.

This operator hashes source files for later pinned comparison manifests. It
does not import engine implementations, resolve binary dependencies, inspect
devices, execute tests or turn file presence into a support certificate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile


EXTENSIONS=frozenset(('.py','.c','.h','.cpp','.cu','.hip','.metal','.swift'))
AREAS={
    'model-families':('publisher_family_contracts.py','publisher_weight_mapping.py','family_conformance.py',
        'mixtral_inventory.py','resident_family_inventory.py','native_mixtral_decoder.py','native_qwen35.py','native_qwen4.py',
        'native_glm5_next.py','native_inkling.py','native_glm_dsa.py','qwen35_multimodal.py','qwen35_publisher_experts.py',
        'publisher_conditional_loader.py','publisher_glm5_loader.py','family_source_binding.py',
        'inkling_media.py','inkling_multimodal.py','qwen4_vision.py','qwen4_multimodal.py',
        'glm5_multimodal.py','multimodal_session.py'),
    'v41':('v41_execution/publisher.py','v41_execution/publisher_vision.py',
        'v41_execution/publisher_mtp.py','v41_execution/speculative.py','v41_distributed_session.py',
        'v41_execution/device_arithmetic.py','v41_execution/device_execution.py',
        'v41_execution/device_vision.py','v41_execution/device_projection.py',
        'bf16_tensor_parallel.py','packed_tensor_parallel.py','distributed_execution.py'),
    'k3':('k3_execution/executor.py','k3_execution/fla_contract.py',
        'k3_execution/exact_admission.py','k3_execution/exact_evidence_files.py','k3_execution/video_input.py','k3_execution/batch.py',
        'k3_execution/metal_mixer.py','k3_execution/device_policy.py',
        'k3_execution/observations.py','k3_execution/bf16_arithmetic.py'),
    'checkpoint-streaming':('publisher_checkpoint/service.py','publisher_tensor_reader.py',
        'expert_store.py','device_packed_decode.py','device_iq3_rotation.py',
        'packed_tensor_parallel.py','device_glm_dsa_experts.py','glm_dsa_streamed_experts.py',
        'device_publisher_projections.py','qwen35_publisher_experts.py','qwen4_ple_store.py',
        'torch_glm_dsa_experts.py','mixtral_streamed_experts.py','expert_packing.py'),
    'accelerator-decoders':('torch_mixtral_decoder.py','torch_qwen35_decoder.py',
        'torch_qwen4_decoder.py','torch_glm5_decoder.py','torch_glm_dsa_decoder.py','torch_inkling_decoder.py',
        'device_qwen35_layers.py','device_execution_policy.py','device_logits.py',
        'torch_mixtral_experts.py','device_qwen4_publisher_experts.py',
        'v41_execution/device_execution.py','k3_execution/device_policy.py'),
    'segment-edge':('native/segment_edge_compat.c','native/segment_family_models.c',
        'segment_edge_runtime.py','native_qwen4_public_adapters.py','mixtral_public_adapters.py',
        'qwen35_public_adapters.py','native_qwen35_ranges.py',
        'glm_dsa_public_adapters.py','native_glm_dsa_ranges.py',
        'native_family_public_adapters.py','native_family_public_text.py','family_source_binding.py',
        'native/segment_edge_compat.h','native/qwen4_public_adapters.c','native/qwen4_public_session.c'),
    'serving':('execution_factories.py','execution_service.py','publisher_provider.py',
        'publisher_conditional_loader.py',
        'publisher_media_inputs.py','publisher_media_processor.py','multimodal_session.py',
        'qwen35_multimodal.py','qwen4_multimodal.py','glm5_multimodal.py','inkling_multimodal.py',
        'native_conditional_cli.py','publisher_k3_cli.py',
        'v41_generation_recovery.py','service_generation_checkpoint.py','generation_checkpoint_files.py'),
    'scale-recovery':('admission_scheduler.py','executor_budget.py','indexed_continuation_store.py',
        'cache_file_recovery.py','native_glm_dsa_cache.py',
        'scalable_execution/cache.py','scalable_execution/scheduler.py','scalable_execution/k3_compressed.py',
        'v41_generation_recovery.py','service_generation_checkpoint.py','k3_generation_recovery.py',
        'distributed_recovery.py','distributed_generation_checkpoint.py'),
    'correctness-tooling':('wholemodel_validation/runner.py','platform_certification/decoder_matrix.py',
        'platform_certification/source_inventory.py',
        'wholemodel_validation/adapters.py','wholemodel_validation/cache_readers.py',
        'wholemodel_validation/olmoe_native.py',
        'wholemodel_validation/glm_dsa_caches.py','wholemodel_validation/k3_media.py',
        'wholemodel_validation/k3.py','wholemodel_validation/k3_fla.py',
        'wholemodel_validation/qwen35_publisher_caches.py','wholemodel_validation/qwen35_routing_oracle.py',
        'wholemodel_validation/qwen35_media.py','wholemodel_validation/qwen35_media_cli.py',
        'wholemodel_validation/qwen4_media.py','wholemodel_validation/qwen4_media_cli.py',
        'wholemodel_validation/qwen4_publisher_caches.py','wholemodel_validation/native_qwen4_caches.py',
        'wholemodel_validation/publisher_family_configuration.py',
        'wholemodel_validation/publisher_family_candidate.py',
        'platform_certification/command_adapter.py'),
}


def _stamp(info):
    # Windows exposes st_ctime as creation time on path stats but as the
    # underlying last-write timestamp on descriptor stats. Prefer the explicit
    # birth-time field there so path and fstat stamps describe the same facts;
    # POSIX platforms continue using ctime to detect metadata/content changes.
    changed=getattr(info,'st_birthtime_ns',info.st_ctime_ns)
    return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,changed)


def _unlinked_path(value):
    path=Path(value).absolute()
    for component in (*reversed(path.parents),path):
        info=component.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',0x400):
            raise ValueError('source inventory rejects links and reparse points')
    return path


def source_inventory(root,*,max_files=4096,max_file_bytes=64*1024**2,max_total_bytes=512*1024**2):
    """Hash a stable bounded engine source tree without importing its modules."""
    for value in (max_files,max_file_bytes,max_total_bytes):
        if type(value) is not int or value<=0:raise ValueError('positive source inventory bounds required')
    root=_unlinked_path(root).resolve(strict=True)
    if not root.is_dir():raise ValueError('engine source directory required')
    paths=[]
    for directory,children,files in os.walk(root,followlinks=False):
        parent=Path(directory)
        children[:]=sorted(name for name in children if name not in ('__pycache__','.git','.venv'))
        for name in children:_unlinked_path(parent/name)
        for name in sorted(files):
            path=parent/name
            if path.suffix not in EXTENSIONS and name!='CMakeLists.txt':continue
            if len(paths)>=max_files:raise MemoryError('engine source file inventory exceeds admission')
            _unlinked_path(path)
            paths.append(path)
    records=[];total=0;stamps=[]
    for path in sorted(paths,key=lambda p:p.relative_to(root).as_posix()):
        resolved=path.resolve(strict=True)
        if not resolved.is_relative_to(root):raise ValueError('source path escaped engine root')
        descriptor=os.open(path,os.O_RDONLY|getattr(os,'O_BINARY',0)|getattr(os,'O_NOFOLLOW',0)|
            getattr(os,'O_NONBLOCK',0))
        try:
            before=os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):raise ValueError('regular source file required')
            if before.st_size>max_file_bytes or before.st_size>max_total_bytes-total:
                raise MemoryError('engine source byte inventory exceeds admission')
        except BaseException as primary:
            from ..owned_file_descriptor import close_after_failure
            close_after_failure(descriptor,primary,owner='source inventory input')
            raise
        from ..owned_file_descriptor import fdopen_owned
        stream=fdopen_owned(descriptor,'rb',owner='source inventory input')
        digest=hashlib.sha256();size=0
        with stream:
            while True:
                block=stream.read(min(1024**2,before.st_size-size+1))
                if not block:break
                size+=len(block)
                if size>before.st_size:raise ValueError('source grew during inventory')
                digest.update(block)
            if _stamp(before)!=_stamp(os.fstat(stream.fileno())) or _stamp(before)!=_stamp(path.stat()):
                raise ValueError('source changed during inventory: '+path.relative_to(root).as_posix())
        if size!=before.st_size:raise ValueError('source was truncated during inventory')
        total+=size
        stamps.append((path,_stamp(before)))
        records.append(dict(path=path.relative_to(root).as_posix(),bytes=size,sha256=digest.hexdigest()))
    for path,stamp in stamps:
        _unlinked_path(path)
        if _stamp(path.stat())!=stamp:
            raise ValueError('source tree changed before inventory publication')
    final=set()
    for directory,children,files in os.walk(root,followlinks=False):
        parent=Path(directory)
        children[:]=sorted(name for name in children if name not in ('__pycache__','.git','.venv'))
        for name in children:_unlinked_path(parent/name)
        for name in files:
            path=parent/name
            if path.suffix in EXTENSIONS or name=='CMakeLists.txt':
                _unlinked_path(path)
                if len(final)>=max_files:raise MemoryError('source inventory grew beyond admission')
                final.add(path)
    if final!=set(paths):raise ValueError('source file inventory changed before publication')
    indexed={row['path'] for row in records}
    areas={name:dict(sources=list(paths),missing=[path for path in paths if path not in indexed])
           for name,paths in AREAS.items()}
    body=dict(schema='openagent.engine-source-inventory.v1',files=records,bytes=total,areas=areas,
        source_only=True,executed=False,numerical_parity_verified=False,platform_certified=False)
    canonical=json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    body['inventory_sha256']=hashlib.sha256(canonical).hexdigest()
    return body


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--max-files',type=int,default=4096)
    parser.add_argument('--max-file-bytes',type=int,default=64*1024**2)
    parser.add_argument('--max-total-bytes',type=int,default=512*1024**2)
    args=parser.parse_args(argv)
    report=source_inventory(args.engine_root,max_files=args.max_files,
        max_file_bytes=args.max_file_bytes,max_total_bytes=args.max_total_bytes)
    encoded=json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if args.output is None:
        print(encoded,end='')
    else:
        path=args.output.absolute()
        descriptor,temporary=tempfile.mkstemp(prefix='.engine-sources-',dir=path.parent)
        try:
            from ..owned_file_descriptor import fdopen_owned
            stream=fdopen_owned(descriptor,'w',encoding='utf-8',newline='\n',
                                owner='source inventory report')
            with stream:
                stream.write(encoded);stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,path)
        finally:
            try:os.unlink(temporary)
            except FileNotFoundError:pass
    return 1 if any(area['missing'] for area in report['areas'].values()) else 0


if __name__=='__main__':raise SystemExit(main())
