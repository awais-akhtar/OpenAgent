"""Publish measured publisher media execution through the canonical receipt gate."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

from .device_capture import (
    CaptureError, DEFAULT_EVIDENCE, DEFAULT_RECEIPTS, _pin, _write_observation,
    _write_receipts, canonical_row,
)
from .device_observations import coverage_applicability, encode_matrix, verify_route_receipt
from .universal_engine import ModelRoute


def publish_media_measurements(measured: dict, *, evidence_root=DEFAULT_EVIDENCE,
                               receipts_path=DEFAULT_RECEIPTS) -> dict:
    """Write only route-specific observations accepted by the existing gate.

    Runners must finish numerical execution, cancellation/replay checks and
    owner cleanup before calling this function. No lifecycle counter, RNG
    digest, route observation or reference token is synthesized here.
    """
    required = ('case_id', 'route', 'consumer', 'abi_owner', 'implementation',
                'prompt_ids', 'generated_ids', 'reference_generated_ids', 'finish_reason',
                'candidate_logits', 'reference_logits', 'tolerance', 'cache',
                'routing_layers', 'recovery', 'placement', 'comparison_reference', 'artifacts',
                'cache_schema')
    if type(measured) is not dict or any(name not in measured for name in required):
        raise CaptureError('complete measured media execution is required before receipt publication')
    route_values = measured['route']
    if type(route_values) not in (list, tuple) or len(route_values) != 6:
        raise CaptureError('exact six-field media route required')
    route = ModelRoute(*route_values)
    if (route.family, route.model_type) not in (
            ('inkling', 'inkling_mm_model'), ('qwen4', 'qwen4_exp'),
            ('glm5-next', 'glm5_next')):
        raise CaptureError('publisher media receipt requires an implemented family identity')
    row = canonical_row(route, case_id=measured['case_id'])
    artifacts = measured['artifacts']
    if type(artifacts) is not list or any(type(pin) is not dict for pin in artifacts):
        raise CaptureError('media execution artifacts must be a flat list of source/runtime pins')
    configs = [pin for pin in artifacts if pin.get('role') == 'config']
    if len(configs) != 1:
        raise CaptureError('one pinned outer checkpoint config required')
    # Rehash after execution to reject replaced checkpoints/libraries/sources.
    for pin in artifacts:
        if _pin(Path(pin['path']), pin['role']) != pin:
            raise CaptureError('media artifact changed during execution')
    artifacts = [*artifacts, _pin(Path(__file__), 'runtime')]
    applicable = coverage_applicability(row)
    for name in ('media_lifecycle', 'speculative_state', 'distributed_recovery'):
        if applicable[name] and type(measured.get(name)) is not dict:
            raise CaptureError(name + ' requires actual execution measurements')
    root = Path(evidence_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.phase1-media-', dir=root))
    final = None
    publication_started = False
    case_id, config_sha = row['case_id'], configs[0]['sha256']
    observations, checks = {}, {}

    def write(kind, payload):
        return _write_observation(staging, kind, payload, case_id=case_id, config_sha=config_sha)

    try:
        payloads = {
            'tokens': {'prompt_ids': measured['prompt_ids'], 'generated_ids': measured['generated_ids'],
                       'finish_reason': measured['finish_reason'],
                       'max_tokens': len(measured['generated_ids']), 'decoding': 'greedy',
                       'implementation': measured['implementation']},
            'logits': {'matrix': encode_matrix(measured['candidate_logits'])},
            'cache': measured['cache'], 'routing': {'layers': measured['routing_layers']},
            'recovery': measured['recovery'], 'placement': measured['placement'],
        }
        for kind, payload in payloads.items():
            observations[kind] = write(kind, payload)
        reference = dict(measured['comparison_reference'])
        reference.update(generated_ids=measured['reference_generated_ids'],
                         logits=encode_matrix(measured['reference_logits']))
        observations['comparison'] = write('comparison', {
            'candidate_logits_sha256': observations['logits']['sha256'],
            'reference': reference, 'tolerance': {'max_abs_logit': float(measured['tolerance'])},
        })
        for name, applies in applicable.items():
            if not applies:
                checks[name] = {'status': 'not-applicable',
                                'reason': 'canonical route does not require ' + name}
                continue
            if name == 'format_device':
                payload = {'persisted_format': route.persisted_format, 'consumer': measured['consumer'],
                           'placement_sha256': observations['placement']['sha256']}
            elif name == 'abi_recovery':
                payload = {'abi_owner': measured['abi_owner'],
                           'recovery_sha256': observations['recovery']['sha256']}
            else:
                payload = measured[name]
            checks[name] = {'status': 'verified', 'evidence': write(name, payload)}
        receipt = {
            'case_id': case_id, 'route': list(route_values), 'artifacts': artifacts,
            'cache_schema': measured['cache_schema'],
            'placement_constraints': {'backend': route.backend, 'device': measured['placement']['device']},
            'comparison_adapter': reference['implementation'],
            'observations': [observations[kind] for kind in sorted(observations)], 'checks': checks,
        }
        result = verify_route_receipt(row, receipt)
        final = root / (case_id.replace('/', '__') + '--' + uuid4().hex)
        os.replace(staging, final)
        for pin in receipt['observations']:
            pin['path'] = str(final / Path(pin['path']).name)
        for check in checks.values():
            if 'evidence' in check:
                check['evidence']['path'] = str(final / Path(check['evidence']['path']).name)
        result = verify_route_receipt(row, receipt)
        # Publication can commit the manifest and then fail while releasing its
        # lock. Once attempted, retain the validated directory even on error:
        # an orphan is recoverable, a committed receipt with deleted pins is not.
        publication_started = True
        _write_receipts(Path(receipts_path), receipt)
        return result
    except BaseException:
        if not publication_started:
            shutil.rmtree(final if final is not None and final.exists() else staging, ignore_errors=True)
        raise


__all__ = ['publish_media_measurements']
