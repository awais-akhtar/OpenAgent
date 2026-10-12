"""Explicitly fetch hashed vendor sources for the isolated Kimi test oracle."""
import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request

from tests.test_native_kimi import ORACLE_HASHES, ORACLE_REVISION


def prepare(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    base = ('https://raw.githubusercontent.com/huggingface/transformers/'
            f'{ORACLE_REVISION}/src/transformers/models/kimi_linear/')
    for name, digest in ORACLE_HASHES.items():
        destination = directory / name
        if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == digest:
            continue
        with urllib.request.urlopen(base + name, timeout=30) as response:
            data = response.read(2 * 1024**2 + 1)
        if len(data) > 2 * 1024**2 or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f'official oracle size/hash mismatch: {name}')
        descriptor, temporary = tempfile.mkstemp(dir=directory, prefix='.verified-')
        try:
            with os.fdopen(descriptor, 'wb') as output:
                output.write(data)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    print(prepare(parser.parse_args().directory))
