"""Trusted local command implementation of wholemodel Adapter/Session.

Each invocation reads OPENAGENT_DECODER_REQUEST and writes
OPENAGENT_DECODER_RESPONSE. Responses echo request_id and action. Forward
responses export Observation arrays in a local NPZ and actual runtime evidence.
The session directory belongs to one fresh session; runners maintain their own
continuation state there. Runner declarations are not signed attestations.
"""
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time
import uuid

import numpy as np

from ..wholemodel_validation.contracts import Observation
from ..wholemodel_validation.runner import digest_file


class DecoderUnavailable(RuntimeError):
    """The supplied runner explicitly could not execute the requested backend."""


class CommandAdapter:
    def __init__(self, root, options):
        self.root = Path(root).resolve()
        self.options = options
        self.records = []
        self.evidence = []
        self.audit_directory = Path(options['audit_directory'])
        self.audit_directory.mkdir(parents=True, exist_ok=False)
        self.command = options['command']
        if (not isinstance(self.command, list) or not self.command or
                any(not isinstance(x, str) or not x or '\0' in x for x in self.command)):
            raise ValueError('runner command must be an explicit nonempty argv list')
        if type(options['timeout']) is not int or options['timeout'] <= 0:
            raise ValueError('positive command timeout required')
        pins = {str((self.root / row['path']).resolve()): row
                for row in options['_artifact_pins']}
        executable = shutil.which(self.command[0])
        if executable is None:
            candidate = (self.root / self.command[0]).resolve()
            if not candidate.is_file():
                raise DecoderUnavailable('runner executable unavailable')
            executable = str(candidate)
        self.command = [str(Path(executable).resolve()), *self.command[1:]]
        command_files = [Path(self.command[0])]
        command_files.extend((self.root / arg).resolve() for arg in self.command[1:]
                             if (self.root / arg).is_file())
        for path in command_files:
            row = pins.get(str(path.resolve()))
            if row is None or row['kind'] not in ('source', 'dependency'):
                raise ValueError('executable/runner file must be a pinned local source or dependency: ' + str(path))
            if digest_file(path) != row['sha256']:
                raise ValueError('runner file digest mismatch')
        self.command_files = [(path, pins[str(path.resolve())]['sha256']) for path in command_files]
        with tempfile.TemporaryDirectory(prefix='oa-decoder-metadata-') as directory:
            response = self.invoke('metadata', Path(directory), {})
        self.loading = response['loading_evidence']
        if self.loading.get('decoder') != options['decoder']:
            raise ValueError('loading evidence decoder family mismatch')
        for key in ('complete_inventory', 'exact_shapes', 'exact_dtypes', 'quantization_validated',
                    'all_shards_loaded', 'capacity_admitted', 'executable_decoder'):
            if self.loading.get(key) is not True:
                raise ValueError('complete decoder loading evidence required: ' + key)
        for key in ('missing_tensors', 'unsupported_tensors', 'subset_loading', 'fallback'):
            expected = False if key in ('subset_loading', 'fallback') else []
            if type(self.loading.get(key)) is not type(expected) or self.loading[key] != expected:
                raise ValueError('decoder loading gap: ' + key)
        if self.loading.get('artifact_pins') != options['_artifact_pins']:
            raise ValueError('loading evidence must bind the complete runner-supplied artifact pins')
        self.meta = response['metadata']
        if not isinstance(self.meta, dict):
            raise ValueError('runner metadata must be an object')
        if self.meta.get('cache_projection') is not None:
            raise ValueError('projected cache cannot certify actual full-model cache parity')

    def invoke(self, action, directory, payload):
        request_id = uuid.uuid4().hex
        request = dict(schema='openagent.decoder-command.v1', action=action,
                       request_id=request_id, offline_root=str(self.root),
                       session_directory=str(directory), backend=self.options['backend'],
                       decoder=self.options['decoder'],
                       device_index=self.options['device'],
                       artifact_pins=self.options['_artifact_pins'], **payload)
        request_path = directory / (request_id + '.request.json')
        response_path = directory / (request_id + '.response.json')
        request_path.write_text(json.dumps(request, allow_nan=False), encoding='utf-8')
        env = dict(os.environ, OPENAGENT_DECODER_REQUEST=str(request_path),
                   OPENAGENT_DECODER_RESPONSE=str(response_path),
                   PYTORCH_ENABLE_MPS_FALLBACK='0', HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1')
        record = dict(action=action, request_id=request_id, argv=list(self.command),
                      returncode=None)
        self.records.append(record)
        started = time.monotonic()
        try:
            for path, digest in self.command_files:
                if digest_file(path) != digest:
                    raise ValueError('pinned runner changed before invocation')
            result = subprocess.run(self.command, cwd=self.root, env=env, shell=False,
                                    capture_output=True, text=True, timeout=self.options['timeout'])
            record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
            if not response_path.is_file():
                raise RuntimeError('runner response absent')
            response = json.loads(response_path.read_text(encoding='utf-8'))
            record['response'] = response
            if (response.get('schema') != request['schema'] or
                    response.get('request_id') != request_id or response.get('action') != action):
                raise ValueError('stale or mismatched runner response')
            if result.returncode == 2 and response.get('status') == 'unavailable':
                raise DecoderUnavailable(str(response.get('reason', 'runner unavailable')))
            if result.returncode or response.get('status') != 'passed':
                raise RuntimeError('decoder command failed or did not report success')
            return response
        except Exception as exc:
            record['error'] = str(exc)
            raise
        finally:
            record['duration_seconds'] = time.monotonic() - started
            (self.audit_directory / (request_id + '.json')).write_text(
                json.dumps(dict(request=request, command=record), allow_nan=False), encoding='utf-8')

    def metadata(self):
        return dict(self.meta)

    def runtime_metadata(self):
        return dict(commands=self.records, forward_evidence=self.evidence,
                    loading_evidence=self.loading,
                    trust='local operator-supplied runner declarations, not signed attestation')

    def new_session(self, *, seed):
        return CommandSession(self, seed)

    def placement(self, response, position):
        evidence = response['runtime_evidence']
        backend = self.options['backend']
        device = ('cpu' if backend == 'linux' else 'mps' if backend == 'metal'
                  else 'cuda:' + str(self.options['device']))
        required = dict(backend=backend, runtime='cpu' if backend == 'linux' else backend,
                        device=device, device_index=self.options['device'],
                        host_system=platform.system(), executed=True, synchronized=True,
                        fallback=False, position=position, complete_decoder=True)
        if any(type(evidence.get(k)) is not type(v) or evidence[k] != v for k, v in required.items()):
            raise ValueError('actual decoder runtime/device evidence mismatch')
        if self.meta['device'] != device:
            raise ValueError('adapter metadata device differs from observed execution')
        if (type(evidence.get('memory_bytes')) is not int or
                evidence['memory_bytes'] < self.options['min_memory_bytes']):
            raise ValueError('decoder memory evidence absent or insufficient')
        if self.options.get('device_name') and self.options['device_name'] not in evidence.get('name', ''):
            raise ValueError('decoder device name mismatch')
        operations = evidence.get('operations')
        expected = {'embedding', 'final_norm', 'lm_head'} | {
            f'layer.{i}.{op}' for i in range(self.meta['num_layers'])
            for op in ('attention', 'ffn', 'cache')}
        if not isinstance(operations, dict) or set(operations) != expected:
            raise ValueError('complete decoder operation placement inventory required')
        for operation in operations.values():
            if (not isinstance(operation, dict) or operation.get('device') != device or
                    operation.get('runtime') != required['runtime'] or
                    operation.get('executed') is not True or operation.get('fallback') is not False):
                raise ValueError('decoder operation placement or fallback mismatch')
        for key in ('skipped', 'failures', 'errors', 'expected_failures', 'unexpected_successes'):
            if type(evidence.get(key)) is not int or evidence[key] != 0:
                raise ValueError('missing or unsuccessful decoder execution counters')
        self.evidence.append(evidence)


class CommandSession:
    def __init__(self, adapter, seed):
        self.adapter = adapter
        self.seed = seed
        self.position = 0
        self.directory = tempfile.TemporaryDirectory(prefix='oa-decoder-session-')
        self.closed = False

    def forward(self, token_ids):
        if self.closed:
            raise RuntimeError('session already closed')
        response = self.adapter.invoke('forward', Path(self.directory.name),
                                       dict(seed=self.seed, tokens=list(token_ids), position=self.position))
        self.adapter.placement(response, self.position + len(token_ids))
        path = (Path(self.directory.name) / response['observation_artifact']).resolve()
        if not path.is_relative_to(Path(self.directory.name).resolve()) or not path.is_file():
            raise ValueError('observation NPZ must be inside the fresh session directory')
        if digest_file(path) != response.get('observation_sha256'):
            raise ValueError('observation artifact digest absent or mismatched')
        captured = self.adapter.audit_directory / (response['request_id'] + '.npz')
        shutil.copyfile(path, captured)
        if digest_file(captured) != response['observation_sha256']:
            raise ValueError('observation changed during capture')
        self.adapter.evidence[-1] = dict(self.adapter.evidence[-1],
            observation_artifact=dict(path=str(captured), sha256=response['observation_sha256']))
        with np.load(path, allow_pickle=False) as arrays:
            expected = {'logits'} | {'cache.' + key for key in self.adapter.meta['cache_schema']} | {
                'evidence.' + key for key in self.adapter.meta.get('evidence_schema', {})}
            if set(arrays.files) != expected:
                raise ValueError('missing or extra observation arrays')
            observation = Observation(np.array(arrays['logits'], copy=True),
                {key: np.array(arrays['cache.' + key], copy=True) for key in self.adapter.meta['cache_schema']},
                {key: np.array(arrays['evidence.' + key], copy=True)
                 for key in self.adapter.meta.get('evidence_schema', {})})
        self.position += len(token_ids)
        return observation

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                response = self.adapter.invoke('close', Path(self.directory.name), dict(position=self.position))
                if response.get('released') is not True:
                    raise RuntimeError('runner did not confirm session resource release')
            finally:
                self.directory.cleanup()


def create_command_adapter(*, offline_root, options):
    return CommandAdapter(offline_root, options)
