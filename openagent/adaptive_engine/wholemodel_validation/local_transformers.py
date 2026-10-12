"""Explicitly authorized, hash-pinned local Transformers causal-model loader.

No download/provision/build/patch paths. Local custom code remains trusted code,
not sandboxed: enforce a process-level network policy for hostile implementations.
"""
from contextlib import contextmanager
import copy
import inspect
import json
import os
from pathlib import Path

from .adapters import TorchCausalAdapter, _RNG_LOCK, reader_from_options,forward_from_options
from .runner import digest_file, pinned_artifacts


@contextmanager
def offline_environment():
    names = ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE')
    with _RNG_LOCK:
        previous = {name: os.environ.get(name) for name in names}
        try:
            for name in names:
                os.environ[name] = '1'
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def contained(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError('local model path escapes offline root')
    return path


class LocalTransformersFactory:
    def __init__(self, root, options):
        if options.get('authorize_authoritative_local_files') is not True:
            raise ValueError('caller must explicitly authorize authoritative local files')
        self.root = Path(root).resolve()
        self.options = dict(options)
        self.pins = options['_artifact_pins']
        self.directory = contained(self.root, options['model_path'])
        self.trust_code = options.get('trust_local_code', False)
        if type(self.trust_code) is not bool:
            raise ValueError('trust_local_code must be boolean')
        if self.trust_code and options.get('authorize_local_code') is not True:
            raise ValueError('local custom-code execution needs explicit caller authorization')
        self.source_pin = contained(self.root, options['implementation_source_pin'])
        self.verify_snapshot()

    def verify_snapshot(self):
        rows = pinned_artifacts({'artifacts': self.pins}, self.root)
        by_path = {Path(row['resolved_path']): row for row in rows}
        if self.source_pin not in by_path or by_path[self.source_pin]['kind'] != 'source':
            raise ValueError('implementation_source_pin must identify a pinned source artifact')
        if not self.directory.is_dir():
            raise ValueError('model_path must be an existing offline directory')
        # Require every snapshot file, not just one selected weight shard.
        for file in self.directory.rglob('*'):
            if file.is_file():
                resolved = file.resolve()
                if not resolved.is_relative_to(self.directory) or resolved not in by_path:
                    raise ValueError('unpinned or escaping snapshot file: ' + str(file))
        config_path = self.directory / 'config.json'
        if config_path not in by_path or by_path[config_path]['kind'] != 'config':
            raise ValueError('snapshot config.json must be pinned as config')
        if self.options['metadata']['config_sha256'] != by_path[config_path]['sha256']:
            raise ValueError('metadata config identity differs from actual snapshot config')
        config = json.loads(config_path.read_text(encoding='utf-8'))
        def inspect_auto_maps(value):
            if isinstance(value, dict):
                for key, entry in value.items():
                    if key == 'auto_map':
                        if not isinstance(entry, dict):
                            raise ValueError('invalid auto_map')
                        for target in entry.values():
                            targets = target if isinstance(target, list) else [target]
                            for name in targets:
                                if name is None:
                                    continue
                                if not isinstance(name, str) or '--' in name:
                                    raise ValueError('remote code repository references are prohibited')
                                parts = name.split('.')
                                if len(parts) < 2 or any(not p.isidentifier() for p in parts):
                                    raise ValueError('invalid local code module reference')
                                source = self.directory.joinpath(*parts[:-1]).with_suffix('.py')
                                if source not in by_path or by_path[source]['kind'] != 'source':
                                    raise ValueError('auto_map must resolve to pinned local source')
                    else:
                        inspect_auto_maps(entry)
            elif isinstance(value, list):
                for entry in value:
                    inspect_auto_maps(entry)
        inspect_auto_maps(config)
        weights = [file for file in self.directory.rglob('*.safetensors')]
        if not weights:
            raise ValueError('local loader requires safetensors weights, not executable pickle checkpoints')
        for file in weights:
            if by_path[file.resolve()]['kind'] != 'weights':
                raise ValueError('every safetensors shard must be classified as weights')
        for index in self.directory.glob('*.safetensors.index.json'):
            if by_path[index.resolve()]['kind'] != 'weights':
                raise ValueError('weight index must be pinned as weights')
            index_data = json.loads(index.read_text(encoding='utf-8'))
            weight_map = index_data['weight_map']
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError('weight index must contain a nonempty weight_map')
            for name in set(weight_map.values()):
                file = contained(self.directory, name)
                if file not in by_path or by_path[file]['kind'] != 'weights' or file.suffix != '.safetensors':
                    raise ValueError('weight index references missing or unpinned shard')

    def __call__(self, *, seed):
        self.verify_snapshot()
        with offline_environment():
            import torch
            from transformers import AutoModelForCausalLM
            dtype_name = self.options.get('torch_dtype', 'auto')
            dtypes = {'float32': torch.float32, 'float64': torch.float64,
                      'float16': torch.float16, 'bfloat16': torch.bfloat16, 'auto': 'auto'}
            if dtype_name not in dtypes:
                raise ValueError('unsupported torch_dtype')
            model = AutoModelForCausalLM.from_pretrained(
                str(self.directory), local_files_only=True,
                trust_remote_code=self.trust_code, use_safetensors=True,
                torch_dtype=dtypes[dtype_name],
                attn_implementation=self.options.get('attn_implementation', 'eager'))
            # Match the code actually loaded, including built-in installed code.
            source = inspect.getsourcefile(type(model))
            if source is None or digest_file(source) != digest_file(self.source_pin):
                raise ValueError('loaded model implementation source differs from authoritative source pin')
            device = self.options.get('device', 'cpu')
            model = model.to(device)
            self.verify_snapshot()
            return model


def create_local_transformers_adapter(*, offline_root, options):
    """CLI factory; runner supplies _artifact_pins from the role's manifest."""
    factory = LocalTransformersFactory(offline_root, options)
    with offline_environment():
        import transformers
    metadata = copy.deepcopy(options['metadata'])
    metadata['dependencies'] = dict(metadata.get('dependencies', {}),
                                    transformers=transformers.__version__)
    return TorchCausalAdapter(factory, metadata,
        cache_reader=reader_from_options(options), kv_only=options.get('kv_only', False),
        input_device=options.get('input_device'), position_ids=options.get('position_ids', True),
        cache_position=options.get('cache_position', False), forward_call=forward_from_options(options))
