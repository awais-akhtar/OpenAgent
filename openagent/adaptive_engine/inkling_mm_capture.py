"""Measured Inkling multimodal/MTP candidate path for Phase 1 evidence.

This module deliberately returns measured values rather than writing canonical
receipts. Receipt publication remains the responsibility of the canonical
Phase 1 runner after all route checks accept the measurements.
"""
from __future__ import annotations

import hashlib
import json
import copy
from pathlib import Path
from typing import Any

from .device_capture import CaptureError, _pin, canonical_row
from .device_observations import (coverage_applicability, reference_implementation,
                                  reference_source_pins)
from .universal_engine import ModelRoute


INKLING_MM_BACKENDS = ("native", "device", "public-range")
INKLING_MM_SOURCE_FORMATS = ("F32", "F16", "BF16", "F8_E4M3")
INKLING_MM_IDENTITY = ("inkling", "inkling_mm_model")


def admitted_inkling_mm_rows() -> tuple[dict[str, Any], ...]:
    """Return the exact canonical Inkling multimodal rows, not generated guesses."""
    from .source_scope import source_scope_matrix
    values=[]
    for row in source_scope_matrix()["execution_cases"]:
        if row.get("family")!="inkling" or row.get("model_type")!="inkling_mm_model":
            continue
        if row.get("backend") not in INKLING_MM_BACKENDS:
            continue
        if row.get("combination_status")!="verified-admitted":
            continue
        values.append(row)
    return tuple(values)


def inkling_mm_capture_accounting() -> dict[str, Any]:
    """Summarize supported route tuples and why successful receipts remain external."""
    rows=admitted_inkling_mm_rows()
    by_backend={backend:sum(row["backend"]==backend for row in rows)
                for backend in INKLING_MM_BACKENDS}
    return {"identity":list(INKLING_MM_IDENTITY),"admitted_rows":len(rows),
            "rows_by_backend":by_backend,"capture_implementation":True,
            "receipt_count":0,
            "receipt_prerequisites":["exact local checkpoint and case_id",
                "pinned processor artifacts and real image or waveform input",
                "native media/text libraries and optional Segment/device runtime",
                "checkpoint-declared, admitted MTP weights and budgets",
                "canonical gate acceptance; this module never fabricates receipts"]}


def _numpy(value):
    import numpy as np
    detach=getattr(value,"detach",None)
    if callable(detach):
        value=detach();cpu=getattr(value,"cpu",None)
        if callable(cpu):value=cpu()
        convert=getattr(value,"numpy",None)
        if callable(convert):value=convert()
    return np.asarray(value)


def _rows(output):
    import numpy as np
    value=output.get("logits") if isinstance(output,dict) else getattr(output,"logits",None)
    result=_numpy(value).astype(np.float32,copy=False)
    if result.ndim==3 and result.shape[0]==1:return result[0]
    if result.ndim==2:return result
    raise CaptureError("Inkling conditional decoder returned malformed logits")


def _route_ids(routing,config):
    import numpy as np
    from .device_observations import routed_layer_mask
    values=_numpy(routing)
    experts=int(config["n_routed_experts"]);top_k=int(config["num_experts_per_tok"])
    layers=int(config["num_hidden_layers"]);routed=routed_layer_mask(config,layers)
    if (values.ndim!=4 or values.shape[0]!=1 or values.shape[-1]<experts or
            values.shape[2]!=layers or not np.isfinite(values).all()):
        raise CaptureError("Inkling execution did not expose finite per-layer route weights")
    result=[]
    for layer in range(layers):
        scores=values[0,:,layer,:experts]
        if not routed[layer]:
            if np.any(scores!=0):
                raise CaptureError("Inkling dense MLP layer emitted nonzero expert route weights")
            result.append([[] for _ in range(values.shape[1])])
            continue
        if np.any(np.count_nonzero(scores>0,axis=-1)<top_k):
            raise CaptureError("Inkling route weights cannot prove the selected expert IDs")
        result.append(np.argsort(-scores,axis=-1,kind="stable")[:,:top_k].astype(int).tolist())
    return result


def _generator_digest(target_rng, draft_rng):
    payload=json.dumps({"target_state":target_rng.bit_generator.state,
        "draft_state":draft_rng.bit_generator.state},sort_keys=True,
        separators=(",",":" )).encode()
    return hashlib.sha256(payload).hexdigest()


def _probabilities(logits,temperature=1.0):
    import numpy as np
    if (type(temperature) not in (int,float) or isinstance(temperature,bool) or
            not np.isfinite(temperature) or temperature<=0):
        raise CaptureError('positive finite sampling temperature required')
    value=np.asarray(logits,dtype=np.float64)/float(temperature)
    if value.ndim!=1 or not value.size or not np.isfinite(value).all():
        raise CaptureError("speculative verifier requires one finite target logit row")
    value-=np.max(value);result=np.exp(value);total=float(result.sum())
    if not np.isfinite(total) or total<=0:raise CaptureError("invalid target sampling distribution")
    return result/total


def _sample_verify_impl(candidate, prompt_ids, target_output, target_rng, draft_rng, *, cancelled=None):
    """Sample real MTP proposals and apply exact p/q acceptance correction."""
    import numpy as np
    draft_tokens,draft_logits,draft_probabilities=candidate.draft(
        np.asarray(prompt_ids,dtype=np.int64)[None,:],
        _numpy(target_output.get('last_hidden_state') if isinstance(target_output,dict)
               else target_output.last_hidden_state),rng=draft_rng,
        return_distributions=True,cancelled=cancelled)
    if (not draft_tokens or len(draft_tokens)!=candidate.mtp.depth or
            draft_logits.shape[0]!=len(draft_tokens) or
            draft_probabilities.shape!=(len(draft_tokens),candidate.components.text_config["unpadded_vocab_size"] or candidate.components.text_config["vocab_size"])):
        raise CaptureError("Inkling MTP did not return sampled full-vocabulary proposal distributions")
    result={"draft_tokens":tuple(map(int,draft_tokens)),"draft_logits":np.asarray(draft_logits).copy(),
        "draft_probabilities":np.asarray(draft_probabilities).copy(),"accepted":0,"rejected":0,
        "bonus":0,"verified_tokens":[]}
    current=target_output
    config=candidate.components.text_config
    eos=config.get('eos_token_id',getattr(candidate,'config',{}).get('eos_token_id'))
    capacity=getattr(candidate,'context_tokens',None)
    result['finish_reason']='depth'
    for token,q in zip(draft_tokens,draft_probabilities):
        if cancelled is not None and cancelled():raise InterruptedError("Inkling speculative target verification cancelled")
        if capacity is not None and candidate.position>=capacity:
            result['finish_reason']='context';break
        p=_probabilities(_rows(current)[-1])
        if q.shape!=p.shape or not np.isfinite(q).all() or np.any(q<0) or not np.isclose(q.sum(),1.0,rtol=1e-6,atol=1e-8):
            raise CaptureError("Inkling MTP q distribution differs from target vocabulary")
        q_token=float(q[int(token)])
        if q_token<=0:raise CaptureError("sampled Inkling proposal has zero q probability")
        if target_rng.random()<min(1.0,float(p[int(token)])/q_token):
            result["accepted"]+=1;result["verified_tokens"].append(int(token))
            if eos is not None and int(token)==eos:
                result['finish_reason']='eos';break
            current=candidate.forward(np.asarray([token],dtype=np.int64),cancelled=cancelled)
        else:
            result["rejected"]+=1
            residual=np.maximum(0.0,p-q);mass=float(residual.sum())
            correction=int(target_rng.choice(len(p),p=(residual/mass if mass>1e-12 else p)))
            result["verified_tokens"].append(correction)
            result['finish_reason']='eos' if correction==eos else 'rejected'
            break
    if result['rejected']==0 and result['finish_reason']=='depth' and result['accepted']==len(draft_tokens):
        if capacity is not None and candidate.position>=capacity:
            result['finish_reason']='context'
            return result
        if cancelled is not None and cancelled():raise InterruptedError("Inkling speculative bonus sampling cancelled")
        bonus=int(target_rng.choice(len(_probabilities(_rows(current)[-1])),
                                    p=_probabilities(_rows(current)[-1])))
        result["verified_tokens"].append(bonus);result["bonus"]=1
        if bonus==eos:result['finish_reason']='eos'
    return result


def _sample_verify(candidate, prompt_ids, target_output, target_rng, draft_rng, *, cancelled=None):
    """Run p/q sampling inside an isolated target/MTP physical-state transaction."""
    target_state=copy.deepcopy(target_rng.bit_generator.state)
    draft_state=copy.deepcopy(draft_rng.bit_generator.state)
    snapshot=candidate.snapshot();primary=None
    try:
        result=_sample_verify_impl(candidate,prompt_ids,target_output,target_rng,
                                   draft_rng,cancelled=cancelled)
        candidate.restore(snapshot)
        return result
    except BaseException as error:
        primary=error
        from .failure_reporting import add_failure_note, failure_summary
        try:candidate.restore(snapshot)
        except BaseException as recovery:
            add_failure_note(primary,"Inkling speculative physical rollback failed: "+failure_summary(recovery))
        try:target_rng.bit_generator.state=copy.deepcopy(target_state)
        except BaseException as recovery:
            add_failure_note(primary,"Inkling target RNG rollback failed: "+failure_summary(recovery))
        try:draft_rng.bit_generator.state=copy.deepcopy(draft_state)
        except BaseException as recovery:
            add_failure_note(primary,"Inkling proposal RNG rollback failed: "+failure_summary(recovery))
        raise
    finally:
        try:candidate.release_snapshot(snapshot)
        except BaseException as cleanup:
            if primary is None:
                # A failed transaction must not silently consume either stream.
                target_rng.bit_generator.state=copy.deepcopy(target_state)
                draft_rng.bit_generator.state=copy.deepcopy(draft_state)
                raise RuntimeError("Inkling speculative snapshot release remains pending") from cleanup
            from .failure_reporting import add_failure_note, failure_summary
            add_failure_note(primary,"Inkling speculative snapshot release failed: "+failure_summary(cleanup))


def _cache_measurement(candidate):
    import numpy as np
    text=candidate.text
    if type(text).__name__=="NativeFamilyPublicTextDecoder":
        from .native_family_capture import _public_range_cache_digests
        return _public_range_cache_digests(text,int(candidate.components.text_config["num_hidden_layers"]))
    snapshot=getattr(text,"cache_snapshot",None)
    if callable(snapshot):states=snapshot()
    elif type(text).__name__=="TorchInklingDecoder":states=tuple(text._cache)
    else:raise CaptureError("Inkling candidate exposes no physical cache snapshot")
    total=0;digests=[]
    for state in states:
        value=_numpy(state)
        if not np.isfinite(value).all():raise CaptureError("Inkling cache contains nonfinite state")
        raw=np.ascontiguousarray(value).tobytes();total+=len(raw);digests.append(hashlib.sha256(raw).hexdigest())
    if len(digests)!=int(candidate.components.text_config["num_hidden_layers"]) or total<=0:
        raise CaptureError("Inkling physical cache does not cover every configured layer")
    return total,digests


def _component_weights(owner,*,max_bytes):
    import numpy as np
    source=getattr(owner,"weights",None)
    if not hasattr(source,"items"):
        raise CaptureError("exact Inkling reference loader exposed no owned text tensors")
    result={};total=0
    for name,value in source.items():
        array=_numpy(value)
        if array.dtype.kind!="f" or not np.isfinite(array).all():
            raise CaptureError("Inkling reference tensor is not a finite floating array: "+str(name))
        total+=array.size*4
        if total>max_bytes:raise CaptureError("independent Inkling reference tensor budget exceeded")
        result[name]=np.array(array,dtype=np.float32,order="C",copy=True)
    return result


def _independent_reference(source_decoder,*,max_bytes):
    from .independent_inkling import (IndependentInklingDecoder,
                                      IndependentInklingMediaEncoder)
    text_config=dict(source_decoder.components.text_config)
    text_weights=_component_weights(source_decoder.text,max_bytes=max_bytes)
    reference=IndependentInklingDecoder(text_config,text_weights,max_bytes=max_bytes)
    media_config=source_decoder.components
    media_weights=_component_weights(source_decoder.media,max_bytes=max_bytes)
    media=IndependentInklingMediaEncoder(media_config.vision_config,media_config.audio_config,
        media_weights,text_hidden=text_config["hidden_size"],max_bytes=max_bytes)
    return reference,media


def _media_embeddings(media_reference,media):
    import numpy as np
    result={}
    if "pixel_values" in media:
        result["image"]=media_reference.vision_features(media["pixel_values"])
    if "audio_input_ids" in media:
        result["audio"]=media_reference.audio_features(
            media["audio_input_ids"],media.get("audio_input_ids_mask"))
    if len(result)!=1:
        raise CaptureError("Inkling capture requires exactly one processor-backed modality per measured prompt")
    return result


def _reference_forward(reference,token_ids,media,embeddings,*,image_token_id,audio_token_id):
    import numpy as np
    ids=np.asarray(token_ids,dtype=np.int64).reshape(1,-1)
    config=reference.config
    kind=next(iter(embeddings));token_id=image_token_id if kind=="image" else audio_token_id
    positions=np.flatnonzero(ids[0]==token_id)
    features=embeddings[kind]
    if len(positions)!=len(features):
        raise CaptureError("independent Inkling media features and processor placeholders differ")
    embedded=reference.weights["model.embed_tokens.weight"][ids].copy()
    embedded=reference.weights["model.embed_norm.weight"] * (
        embedded/np.sqrt(np.mean(embedded*embedded,axis=-1,keepdims=True,dtype=np.float32)+
                         np.float32(config["rms_norm_eps"])))
    embedded[0,positions]=features
    return reference.forward(ids,inputs_embeds=np.asarray(embedded,dtype=np.float32))


def _candidate_forward(candidate,token_ids,media=None):
    """Call the real family wrapper, retaining public-range router telemetry."""
    return candidate.forward(token_ids,**(media or {}),capture_routing=True)


def capture_inkling_mm_measurements(
        checkpoint: str | Path, *, case_id: str, source_format: str, backend: str,
        prompt_text: str, image: str | Path | None = None,
        audio: str | Path | None = None, sampling_rate: int | None = None,
        media_library: str | Path, text_library: str | Path,
        runtime_library: str | Path | None = None, device: str | None = None,
        max_tokens: int = 4, context_tokens: int = 512,
        max_weight_bytes: int = 128 * 1024**3, max_runtime_bytes: int = 64 * 1024**3,
        max_conversion_bytes: int = 16 * 1024**3, max_mtp_bytes: int = 8 * 1024**3,
        max_mtp_work_bytes: int = 512 * 1024**2,
        max_processor_bytes: int = 512 * 1024**2,
        tolerance: float = 1e-3) -> dict[str, Any]:
    """Execute real Inkling media + target/MTP operations and return measurements.

    This function does not publish or claim canonical receipts. It fails closed
    if the exact route, independent reference, media lifecycle, MTP, or source
    ownership cannot be measured. The canonical Phase 1 runner must bind these
    measurements to observations and call ``verify_route_receipt``.
    """
    import numpy as np
    from .device_observations import encode_matrix, sha256_file
    from .publisher_media_processor import PublisherMediaProcessor
    from .publisher_conditional_loader import load_publisher_conditional_decoder
    from .execution_service import ExecutionCancelled

    for name,value in (('max_tokens',max_tokens),('context_tokens',context_tokens),
                       ('max_weight_bytes',max_weight_bytes),('max_runtime_bytes',max_runtime_bytes),
                       ('max_conversion_bytes',max_conversion_bytes),('max_mtp_bytes',max_mtp_bytes),
                       ('max_mtp_work_bytes',max_mtp_work_bytes),('max_processor_bytes',max_processor_bytes)):
        if type(value) is not int or value<=0:
            raise CaptureError('positive bounded Inkling capture budget required: '+name)
    if (type(tolerance) not in (float,int) or isinstance(tolerance,bool) or
            not np.isfinite(tolerance) or not 0<=tolerance<=1e-2):
        raise CaptureError('finite Inkling comparison tolerance in [0, 0.01] required')

    if backend not in INKLING_MM_BACKENDS or source_format not in INKLING_MM_SOURCE_FORMATS:
        raise CaptureError("unsupported exact Inkling multimodal route tuple")
    if (bool(image)==bool(audio) or type(prompt_text) is not str or not prompt_text.strip() or
            type(case_id) is not str or not case_id):
        raise CaptureError("exact case ID, prompt, and exactly one real image/audio file are required")
    if (backend=="device") != (type(device) is str and bool(device)):
        raise CaptureError("device route requires --device; host routes reject it")
    if (backend=="public-range") != (runtime_library is not None):
        raise CaptureError("public-range routes require exactly one Segment runtime library")
    route=ModelRoute("inkling","inkling_mm_model",backend,"fp32",source_format,
                     {"F32":"float32","F16":"float32","BF16":"float32",
                      "F8_E4M3":"fp8-e4m3-b128-f32"}[source_format])
    row=canonical_row(route,case_id=case_id)
    applicability=coverage_applicability(row)
    if not applicability.get("media_lifecycle") or not applicability.get("speculative_state"):
        raise CaptureError("canonical Inkling multimodal row does not bind required media/MTP coverage")
    comparison_name=reference_implementation("inkling",backend,"inkling_mm_model")
    comparison_sources=reference_source_pins("inkling",backend,"inkling_mm_model")
    candidate_source_pins=_candidate_source_pins()

    root=Path(checkpoint).resolve(strict=True)
    media_library=Path(media_library).resolve(strict=True)
    text_library=Path(text_library).resolve(strict=True)
    for path,label in ((root,"checkpoint"),(media_library,"media library"),(text_library,"text library")):
        if not path.exists():raise CaptureError("missing exact Inkling "+label)
    if not root.is_dir():raise CaptureError("Inkling checkpoint must be a local directory")
    from .source_scope import require_checkpoint_inventory_admission
    inventory=require_checkpoint_inventory_admission(root,family='inkling',
        model_type='inkling_mm_model',backend=backend,arithmetic='fp32',
        persisted_format='float32',max_metadata_bytes=min(max_weight_bytes,256*1024**2))
    floating={record.dtype for name,record in inventory.tensors.items()
              if record.dtype in INKLING_MM_SOURCE_FORMATS and
              not name.endswith(('.weight_scale_inv','_scale_inv'))}
    if source_format not in floating or not floating <= {source_format,'F32'}:
        raise CaptureError('Inkling checkpoint storage encodings differ from the selected source format')
    artifact_pins_before=[_pin(root/'config.json','config'),
        *[_pin(path,'weights') for path in sorted(root.glob('*.safetensors'))],
        _pin(media_library,'runtime'),_pin(text_library,'runtime')]
    if runtime_library is not None:
        runtime_library=Path(runtime_library).resolve(strict=True)
        if not runtime_library.is_file():raise CaptureError("Segment runtime library is not a regular file")
    if image is not None:
        media_path=Path(image).resolve(strict=True);kind="image"
        from .publisher_media_files import read_regular
        media_value=read_regular(media_path,max_bytes=max_processor_bytes//8)
        if not media_value or len(media_value)>max_processor_bytes//8:
            raise CaptureError("encoded Inkling image exceeds processor admission")
    else:
        media_path=Path(audio).resolve(strict=True);kind="audio"
        if type(sampling_rate) is not int or sampling_rate<=0:
            raise CaptureError("Inkling audio capture requires a positive explicit sampling rate")
        from io import BytesIO
        from .publisher_media_files import read_regular
        audio_file=np.load(BytesIO(read_regular(media_path,max_bytes=max_processor_bytes//8)),allow_pickle=False)
        if audio_file.dtype.kind!="f" or audio_file.ndim!=1 or not audio_file.size:
            raise CaptureError("audio input must be a nonempty mono floating NumPy waveform")
        media_value=audio_file

    if type(max_tokens) is not int or max_tokens<=0 or type(context_tokens) is not int or context_tokens<=0:
        raise CaptureError("positive bounded Inkling generation and context limits required")
    media_pin_before=_pin(media_path,'media_fixture')
    if runtime_library is not None:artifact_pins_before.append(_pin(runtime_library,'runtime'))
    processor=PublisherMediaProcessor(root,family="inkling_mm_model",max_bytes=max_processor_bytes,
        max_pixels=40_000_000,max_audio_samples=30*60*48000)
    candidate=reference_source=reference=media_reference=None
    held_snapshots=[]
    try:
        prompt_ids,media=processor.prepare(prompt_text,
            images=(media_value,) if kind=="image" else (),
            audio=(media_value,) if kind=="audio" else (),
            sampling_rate=sampling_rate if kind=="audio" else None)
        required_field="pixel_values" if kind=="image" else "audio_input_ids"
        if required_field not in media:
            raise CaptureError("pinned Inkling processor produced no admitted "+kind+" tensors")
        common=dict(family="inkling_mm_model",library=media_library,text_library=text_library,
            max_weight_bytes=max_weight_bytes,max_bytes=max_runtime_bytes,
            context_tokens=context_tokens,max_conversion_bytes=max_conversion_bytes,
            mtp_max_bytes=max_mtp_bytes,mtp_work_bytes=max_mtp_work_bytes,
            max_expert_bytes=0,max_snapshots=8)
        candidate=load_publisher_conditional_decoder(root,**common,
            text_device=device,segment_runtime_library=runtime_library,
            segment_layer_span=1 if runtime_library else 0,
            segment_work_bytes=max(4*1024**2,max_runtime_bytes//8) if runtime_library else 0)
        if candidate.mtp is None:
            raise CaptureError("checkpoint has no admitted Inkling MTP tensor bank; speculative route is not capturable")
        recovery_tokens=max(1,min(max_tokens,4))
        if len(prompt_ids)+max(max_tokens+recovery_tokens,candidate.mtp.depth+1)>context_tokens:
            raise CaptureError('Inkling prompt, generation, recovery and MTP exceed context admission')
        # Load a second exact native owner solely to materialize the reviewed
        # NumPy reference tensors. Reference execution is independent; no
        # candidate logits or routed outputs are reused by the oracle.
        reference_source=load_publisher_conditional_decoder(root,**common)
        reference,media_reference=_independent_reference(reference_source,max_bytes=max_runtime_bytes)
        reference_source.close();reference_source=None
        embeddings=_media_embeddings(media_reference,media)
        candidate_result=_candidate_forward(candidate,np.asarray(prompt_ids,dtype=np.int64),media)
        image_token_id=int(candidate.config.get("image_token_id",200054))
        audio_token_id=int(candidate.config.get("audio_token_id",200053))
        reference_result=_reference_forward(reference,prompt_ids,media,embeddings,
            image_token_id=image_token_id,audio_token_id=audio_token_id)
        candidate_prefill=_rows(candidate_result);reference_prefill=_rows(reference_result)
        if candidate_prefill.shape!=reference_prefill.shape:
            raise CaptureError("Inkling multimodal candidate/reference logit geometry differs")
        prefill_error=float(np.max(np.abs(candidate_prefill.astype(np.float64)-
                                          reference_prefill.astype(np.float64))))
        if prefill_error>tolerance:
            raise CaptureError(f"Inkling independent media/text logit error {prefill_error:.6g} exceeds tolerance")

        if candidate.position!=len(prompt_ids) or reference.position!=len(prompt_ids):
            raise CaptureError("Inkling target/reference cache positions differ from processed media prompt")
        route_rows=_route_ids(getattr(candidate_result,"routing_weights",None),candidate.components.text_config)
        target_snapshot=candidate.snapshot();held_snapshots.append(target_snapshot);start_position=candidate.position
        seed=int.from_bytes(hashlib.sha256((candidate.identity+row["case_id"]).encode()).digest()[:8],"little")
        draft_seed,target_seed=np.random.SeedSequence(seed).spawn(2)
        draft_rng=np.random.default_rng(draft_seed);target_rng=np.random.default_rng(target_seed)
        initial_draft=copy.deepcopy(draft_rng.bit_generator.state)
        initial_target=copy.deepcopy(target_rng.bit_generator.state)
        rng_digest=_generator_digest(target_rng,draft_rng)
        sampled=_sample_verify(candidate,prompt_ids,candidate_result,target_rng,draft_rng)
        post_sample_digest=_generator_digest(target_rng,draft_rng)
        candidate.restore(target_snapshot)
        draft_rng.bit_generator.state=copy.deepcopy(initial_draft)
        target_rng.bit_generator.state=copy.deepcopy(initial_target)
        replayed=_sample_verify(candidate,prompt_ids,candidate_result,target_rng,draft_rng)
        rng_after_digest=_generator_digest(target_rng,draft_rng)
        if (sampled["draft_tokens"]!=replayed["draft_tokens"] or
                not np.array_equal(sampled["draft_logits"],replayed["draft_logits"]) or
                not np.array_equal(sampled["draft_probabilities"],replayed["draft_probabilities"]) or
                sampled["verified_tokens"]!=replayed["verified_tokens"] or
                post_sample_digest!=rng_after_digest):
            raise CaptureError("Inkling proposal/target RNG and MTP recovery replay differ")
        draft_tokens=sampled["draft_tokens"];draft_logits=sampled["draft_logits"]
        accepted=sampled["accepted"];rejected=sampled["rejected"];bonus=sampled["bonus"]
        candidate.restore(target_snapshot)
        # The measurement is an isolated speculative transaction. Restore the
        # physical target/MTP owners and both owned random streams before the
        # ordinary generation path continues.
        draft_rng.bit_generator.state=copy.deepcopy(initial_draft)
        target_rng.bit_generator.state=copy.deepcopy(initial_target)
        rng_after_digest=_generator_digest(target_rng,draft_rng)
        if rng_digest!=rng_after_digest:
            raise CaptureError("Inkling speculative transaction did not roll both RNG streams back")
        cancel_draft_rng=np.random.default_rng(draft_seed);cancel_target_rng=np.random.default_rng(target_seed)
        cancel_before=_generator_digest(cancel_target_rng,cancel_draft_rng)
        cancel_position=candidate.position;cancel_mtp_position=candidate.mtp.position
        class CancelAfterProposal:
            def __init__(self):self.calls=0;self.observed_proposal_rng_advance=False
            def __call__(self):
                self.calls+=1
                self.observed_proposal_rng_advance=(
                    _generator_digest(cancel_target_rng,cancel_draft_rng)!=cancel_before)
                return self.observed_proposal_rng_advance
        # Interrupt only once the actual proposal RNG has advanced. This makes
        # the cancellation exercise a partially executed MTP transaction,
        # independent of backend-specific cancellation callback counts.
        spec_cancel=CancelAfterProposal()
        try:
            _sample_verify(candidate,prompt_ids,candidate_result,cancel_target_rng,
                cancel_draft_rng,cancelled=spec_cancel)
        except InterruptedError:
            pass
        else:
            raise CaptureError("Inkling MTP did not interrupt its speculative stage sequence")
        spec_cancel_after=_generator_digest(cancel_target_rng,cancel_draft_rng)
        cancel_position_after=candidate.position;cancel_mtp_after=candidate.mtp.position
        if (not spec_cancel.observed_proposal_rng_advance or spec_cancel_after!=cancel_before or candidate.position!=cancel_position or
                candidate.mtp.position!=cancel_mtp_position):
            raise CaptureError("cancelled Inkling speculation did not restore target/draft positions and both RNGs")
        candidate.restore(target_snapshot)

        # Native caller lifecycle cancellation: force cancellation only after
        # processor inputs have been detached and a media frame is admitted.
        from .multimodal_session import MultimodalSessionExecutor
        class CancelAfter:
            def __init__(self,threshold):self.threshold=threshold;self.calls=0
            def is_set(self):
                self.calls+=1;return self.calls>=self.threshold
        lifecycle_payload=None
        lifecycle_snapshot=candidate.snapshot();held_snapshots.append(lifecycle_snapshot)
        candidate.reset()
        lifecycle_start=candidate.position
        lifecycle_clean=candidate.snapshot();held_snapshots.append(lifecycle_clean)
        for threshold in range(4,65):
            candidate.restore(lifecycle_clean)
            executor=MultimodalSessionExecutor(candidate,max_media_bytes=max_processor_bytes)
            cancel=CancelAfter(threshold)
            try:
                executor.forward_media(np.asarray(prompt_ids,dtype=np.int64),media,cancel)
            except (InterruptedError,ExecutionCancelled):
                lifecycle=executor._media_lifecycle
                if candidate.position==lifecycle_start and lifecycle.active_count()==0:
                    lifecycle_payload={"items":[{"kind":kind,"sha256":sha256_file(media_path),
                        "processor_output_sha256":hashlib.sha256(np.ascontiguousarray(media[required_field]).tobytes()).hexdigest()}],
                        "position_before_cancel":lifecycle_start,"position_after_cancel":candidate.position,
                        "released_items":1,"cancel_callback_checks":cancel.calls}
                    break
            else:
                candidate.restore(lifecycle_clean)
            finally:
                executor._media_lifecycle.close()
        if lifecycle_payload is None:
            raise CaptureError("could not observe post-admission Inkling media cancellation with position rollback")
        candidate.restore(lifecycle_snapshot)
        candidate.release_snapshot(lifecycle_clean);held_snapshots.remove(lifecycle_clean)
        candidate.release_snapshot(lifecycle_snapshot);held_snapshots.remove(lifecycle_snapshot)

        generated=[];candidate_rows=[];reference_rows=[]
        candidate_output=candidate_result;reference_output=reference_result
        route_rows=_route_ids(getattr(candidate_result,"routing_weights",None),candidate.components.text_config)
        finish_reason="length";eos=candidate.components.text_config.get('eos_token_id',candidate.config.get('eos_token_id'))
        for index in range(max_tokens):
            candidate_logit=_rows(candidate_output)[-1].copy()
            reference_logit=_rows(reference_output)[-1].copy()
            next_id=int(np.argmax(candidate_logit))
            if next_id!=int(np.argmax(reference_logit)):
                raise CaptureError("Inkling target/reference greedy token differs")
            generated.append(next_id);candidate_rows.append(candidate_logit);reference_rows.append(reference_logit)
            if eos is not None and next_id==int(eos):
                finish_reason="eos";break
            if index+1==max_tokens:break
            candidate_output=_candidate_forward(candidate,np.asarray([next_id],dtype=np.int64))
            next_routes=_route_ids(getattr(candidate_output,"routing_weights",None),candidate.components.text_config)
            for layer_index,rows in enumerate(next_routes):route_rows[layer_index].extend(rows)
            reference_output=reference.forward(np.asarray([next_id],dtype=np.int64))
            if candidate.position>=context_tokens:
                finish_reason="context";break
        if not generated:raise CaptureError("Inkling capture generated no target tokens")
        processed_positions=len(prompt_ids)+len(generated)-1
        if candidate.position!=processed_positions or reference.position!=processed_positions:
            raise CaptureError("Inkling cache position differs from processed prompt/output tokens")
        cache_bytes,cache_digests=_cache_measurement(candidate)
        # Capture physical state and target recovery with the production owner's
        # native snapshot API. The returned source runner can turn these values
        # into the strict kind-specific observation documents.
        snapshot=candidate.snapshot();held_snapshots.append(snapshot);snapshot_position=candidate.position
        continuation=[]
        for _ in range(recovery_tokens):
            value=_candidate_forward(candidate,np.asarray([generated[-1] if not continuation else continuation[-1]],dtype=np.int64))
            token=int(np.argmax(_rows(value)[-1]));continuation.append(token)
            if token==eos:break
        candidate.restore(snapshot)
        restored_position=candidate.position
        if restored_position!=snapshot_position:
            raise CaptureError("Inkling conditional restore did not return to its saved position")
        replay=[]
        for _ in range(len(continuation)):
            value=_candidate_forward(candidate,np.asarray([generated[-1] if not replay else replay[-1]],dtype=np.int64))
            replay.append(int(np.argmax(_rows(value)[-1])))
        if continuation!=replay:
            raise CaptureError("Inkling conditional recovery replay differs")
        candidate.release_snapshot(snapshot)
        held_snapshots.remove(snapshot)
        maximum_error=float(np.max(np.abs(np.stack(candidate_rows).astype(np.float64)-
                                           np.stack(reference_rows).astype(np.float64))))
        if maximum_error>tolerance:raise CaptureError("Inkling generated-logit reference error exceeds tolerance")
        candidate.release_snapshot(target_snapshot);held_snapshots.remove(target_snapshot);target_snapshot=None
        processor_artifacts=[_pin(path,"processor") for path in sorted(processor._files)]
        inventory.assert_unchanged()
        for pin in artifact_pins_before+candidate_source_pins+[media_pin_before]:
            if _pin(Path(pin['path']),pin['role'])!=pin:
                raise CaptureError('Inkling checkpoint, runtime or source changed during capture')
        if reference_source_pins('inkling',backend,'inkling_mm_model')!=comparison_sources:
            raise CaptureError('Inkling independent reference source changed during capture')
        consumer="inkling_multimodal.InklingMultimodalDecoder"
        text_owner=candidate.text
        abi_owner=type(text_owner).__module__.rsplit(".",1)[-1]+"."+type(text_owner).__name__
        measured_device=str(candidate.text.device) if backend=="device" else "cpu"
        result={"case_id":row["case_id"],"route":list(route.as_tuple()),
            "consumer":consumer,"abi_owner":abi_owner,
            "implementation":str(candidate.implementation),
            "reference":comparison_name,
            "prompt_ids":[int(value) for value in prompt_ids],
            "generated_ids":generated,"reference_generated_ids":[int(np.argmax(value)) for value in reference_rows],
            "finish_reason":finish_reason,
            "candidate_logits":np.stack(candidate_rows),
            "reference_logits":np.stack(reference_rows),"tolerance":float(tolerance),
            "max_abs_logit_error":maximum_error,"processed_positions":processed_positions,
            "cache":{"position":processed_positions,"layers":len(cache_digests),
                "layer_digests":cache_digests,"bytes":cache_bytes},
            "cache_schema":{"format":"openagent.inkling-kv-fir.v1",
                "layers":int(candidate.components.text_config["num_hidden_layers"])},
            "routing_layers":{f"model.layers.{index}":rows for index,rows in enumerate(route_rows)},
            "placement":{"backend":backend,"device":measured_device,
                "host_fallback_operations":0,"implementation":str(candidate.implementation),
                "components":[{'component':'vision' if kind=='image' else 'audio','device':'cpu',
                                'implementation':type(candidate.media).__name__},
                               {'component':'text','device':measured_device,
                                'implementation':candidate.text.implementation},
                               {'component':'mtp','device':measured_device,
                                'implementation':type(candidate.mtp).__name__}]},
            "comparison_reference":{"implementation":comparison_name,
                "sources":comparison_sources},
            "media_lifecycle":lifecycle_payload,
            "speculative_state":{"proposals":accepted+rejected,"drafted_proposals":len(draft_tokens),
                "discarded_after_rejection":len(draft_tokens)-accepted-rejected,
                "accepted":accepted,"rejected":rejected,"bonus":bonus,
                "verified_token_ids":sampled["verified_tokens"],
                "rng_digest_before":rng_digest,
                "rng_digest_after_restore":rng_after_digest,
                "rng_digest_after_sample":post_sample_digest,
                "cancelled":{"callback_checks":spec_cancel.calls,
                    "rng_digest_before":cancel_before,"rng_digest_after":spec_cancel_after,
                    "target_position_before":cancel_position,"target_position_after":cancel_position_after,
                    "mtp_position_before":cancel_mtp_position,"mtp_position_after":cancel_mtp_after},
                "target_snapshot_position":start_position,"draft_replay_ids":list(replayed['draft_tokens'])},
            "recovery":{"snapshot_position":snapshot_position,"restored_position":restored_position,
                "continuation_ids":continuation,"replay_ids":replay},
            "artifacts":[_pin(root/"config.json","config"),
                _pin(media_path,"media_fixture"),*candidate_source_pins,*processor_artifacts,
                _pin(media_library,"runtime"),_pin(text_library,"runtime"),
                *([_pin(runtime_library,"runtime")] if runtime_library else []),
                *[_pin(root/path,"weights") for path in sorted(p.name for p in root.glob("*.safetensors"))]],
            "receipt_published":False,
            "receipt_note":"Measurements are returned to the canonical runner; no receipt is fabricated here."}
        result["comparison_reference"]["sources"]=comparison_sources
        return result
    finally:
        primary_error=__import__("sys").exc_info()[1]
        cleanup_errors=[]
        if candidate is not None:
            for receipt in tuple(held_snapshots):
                try:
                    candidate.release_snapshot(receipt)
                    held_snapshots.remove(receipt)
                except BaseException as error:cleanup_errors.append(error)
        for owner in (candidate,reference_source,reference,media_reference,processor):
            if owner is not None:
                close=getattr(owner,"close",None)
                if callable(close):
                    try:close()
                    except BaseException as error:cleanup_errors.append(error)
        if cleanup_errors:
            from .failure_reporting import add_failure_note, failure_summary
            if primary_error is not None:
                for error in cleanup_errors:
                    add_failure_note(primary_error,"Inkling capture cleanup failed: "+failure_summary(error))
            else:
                raise RuntimeError("Inkling capture owner cleanup remains pending") from cleanup_errors[0]


def _candidate_source_pins():
    root=Path(__file__).resolve().parent
    names=("inkling_mm_capture.py","inkling_multimodal.py","inkling_mtp.py",
        "inkling_media.py","native_media_ops.py","native_media_ops.c","native_inkling.py",
        "native_inkling.c","torch_inkling_decoder.py","device_inkling_forward.py",
        "publisher_conditional_loader.py","publisher_media_processor.py","multimodal_session.py",
        "publisher_family_contracts.py","publisher_tensor_reader.py","publisher_weight_mapping.py",
        "native_family_public_text.py","segment_edge_runtime.py","segment_snapshot.py",
        "qwen4_segment_wire.py","native_family_public_adapters.py",
        "independent_inkling.py","independent_media_math.py","grouped_continuation.py",
        "source_scope.py","device_capture.py","device_observations.py",
        "media_route_capture.py","failure_reporting.py","device_logits.py",
        "device_publisher_projections.py","torch_inkling_cache.py","flat_native_session.py")
    names+=tuple("native/"+name for name in (
        "segment_edge_compat.h","segment_edge_compat.c","segment_family_models.h",
        "segment_family_models.c","original_experts.h","original_experts.c",
        "segment_executor.h","segment_executor.c","segment_bridge.h","segment_bridge.c",
        "segment_math.h","segment_math.c","sha256.h","qwen4_public_model.h",
        "qwen4_public_model.c","qwen4_public_session.h","qwen4_public_session.c",
        "qwen4_public_internal.h","qwen4_public_adapters.c"))
    return [_pin((root/name).resolve(strict=True),"runtime") for name in names]


def add_inkling_mm_subparser(subparsers):
    parser=subparsers.add_parser("inkling-mm",help="measure exact Inkling multimodal/MTP routes")
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--case-id",required=True)
    parser.add_argument("--source-format",choices=INKLING_MM_SOURCE_FORMATS,required=True)
    parser.add_argument("--backend",choices=INKLING_MM_BACKENDS,required=True)
    parser.add_argument("--prompt",required=True)
    media=parser.add_mutually_exclusive_group(required=True)
    media.add_argument("--image",type=Path)
    media.add_argument("--audio",type=Path)
    parser.add_argument("--sampling-rate",type=int)
    parser.add_argument("--media-library",type=Path,required=True)
    parser.add_argument("--text-library",type=Path,required=True)
    parser.add_argument("--runtime-library",type=Path)
    parser.add_argument("--device")
    parser.add_argument("--max-tokens",type=int,default=4)
    parser.add_argument("--context-tokens",type=int,default=512)
    from .device_capture import DEFAULT_EVIDENCE, DEFAULT_RECEIPTS
    parser.add_argument('--evidence-root',type=Path,default=DEFAULT_EVIDENCE)
    parser.add_argument('--receipts',type=Path,default=DEFAULT_RECEIPTS)


def run_inkling_mm_subcommand(args):
    try:
        result=capture_inkling_mm_measurements(args.checkpoint,case_id=args.case_id,
            source_format=args.source_format,backend=args.backend,prompt_text=args.prompt,
            image=args.image,audio=args.audio,sampling_rate=args.sampling_rate,
            media_library=args.media_library,text_library=args.text_library,
            runtime_library=args.runtime_library,device=args.device,
            max_tokens=args.max_tokens,context_tokens=args.context_tokens)
    except (CaptureError,ValueError,OSError,ImportError,MemoryError,RuntimeError) as error:
        print("capture rejected: "+str(error))
        return 1
    try:
        from .media_route_capture import publish_media_measurements
        receipt=publish_media_measurements(result,evidence_root=args.evidence_root,receipts_path=args.receipts)
    except (CaptureError,ValueError,OSError,ImportError,MemoryError,RuntimeError) as error:
        print("receipt rejected: "+str(error))
        return 1
    print(json.dumps(receipt,indent=1,sort_keys=True))
    return 0


__all__=["INKLING_MM_BACKENDS","INKLING_MM_IDENTITY","INKLING_MM_SOURCE_FORMATS",
    "admitted_inkling_mm_rows","inkling_mm_capture_accounting",
    "capture_inkling_mm_measurements","add_inkling_mm_subparser",
    "run_inkling_mm_subcommand"]
