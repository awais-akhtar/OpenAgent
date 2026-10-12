"""Explicit network probe of unchanged pinned vendor text sources; not inference."""
import hashlib
import importlib
from pathlib import Path
import sys
import subprocess
import tempfile
import traceback
import urllib.request


REVISION = 'f831ab66814297da540d832a5235f8e904f29d06'
HASHES = {
    'configuration_kimi_k3.py': '735eb9ebe593e17d231e08e1df7f7be9b5ee0e079f511aa201f9572077b416ae',
    'modeling_kimi_linear.py': '9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a',
}


def main():
    if '--worker' in sys.argv:
        directory = sys.argv[sys.argv.index('--worker') + 1]
        dependencies = Path(directory) / 'dependencies'
        sys.path.insert(0, directory)
        if dependencies.exists():
            sys.path.insert(0, str(dependencies))
        import transformers
        print('transformers=', transformers.__version__, flush=True)
        vendor = importlib.import_module('kimi_k3_vendor_probe.modeling_kimi_linear')
        print('official unchanged text module import succeeded', flush=True)
        if '--verify-decoder' in sys.argv:
            from tests.test_native_kimi_k3_decoder import run_vendor_suite
            return run_vendor_suite(vendor)
        print('forward NOT verified', flush=True)
        return 0
    with tempfile.TemporaryDirectory(prefix='kimi_k3_oracle_') as directory:
        package = Path(directory) / 'kimi_k3_vendor_probe'
        package.mkdir()
        dependencies = None
        if '--provision' in sys.argv or '--provision-windows' in sys.argv:
            dependencies = Path(directory) / 'dependencies'
            requirements = ['einops==0.8.1', 'fla-core==0.5.2']
            if '--provision-windows' in sys.argv:
                requirements.append('triton-windows==3.8.0.post29')
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps',
                            '--target', str(dependencies), *requirements], check=True, timeout=120)
        # A namespace package avoids modifying or supplying vendor implementations.
        for name in ('configuration_kimi_k3.py', 'modeling_kimi_linear.py'):
            url = f'https://huggingface.co/moonshotai/Kimi-K3/resolve/{REVISION}/{name}'
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read(2 * 1024 * 1024)
            print(name, 'sha256=', hashlib.sha256(data).hexdigest(), 'bytes=', len(data), flush=True)
            if hashlib.sha256(data).hexdigest() != HASHES[name]:
                raise ValueError('Pinned vendor SHA-256 mismatch: ' + name)
            (package / name).write_bytes(data)
        # Import native extensions in a child so Windows releases DLLs before cleanup.
        args = [sys.executable, str(Path(__file__).resolve()), '--worker', directory]
        if '--verify-decoder' in sys.argv:
            args.append('--verify-decoder')
        return subprocess.run(args, timeout=180).returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
