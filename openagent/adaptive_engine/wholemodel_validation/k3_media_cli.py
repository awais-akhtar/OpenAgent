"""Explicit offline phase-two entry point for pinned K3 media comparisons.

Importing this module does not create sessions, import a vendor model, build
libraries or execute comparison. CLI invocation is reserved for phase two.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def _read_configuration(path,sha256):
    if (type(sha256) is not str or len(sha256)!=64
            or any(c not in '0123456789abcdef' for c in sha256)):
        raise ValueError('explicit lowercase comparison configuration SHA-256 required')
    path=Path(path).resolve(strict=True)
    if not path.is_file() or path.stat().st_size>4*1024*1024:
        raise ValueError('bounded complete comparison configuration required')
    with path.open('rb') as stream:
        raw=stream.read(4*1024*1024+1)
    if len(raw)>4*1024*1024 or hashlib.sha256(raw).hexdigest()!=sha256:
        raise ValueError('comparison configuration differs from explicit pin')
    from ..native_checkpoint import _pairs
    return json.loads(raw.decode('utf-8'),object_pairs_hook=_pairs)


def _publish_report(path,report):
    from .runner import safe_json
    from ..session_recovery import _path
    path=_path(path)
    if not path.parent.is_dir():raise ValueError('existing local report directory required')
    descriptor,name=tempfile.mkstemp(prefix='.media-comparison-',dir=path.parent)
    try:
        try:output=os.fdopen(descriptor,'w',encoding='utf-8')
        except BaseException:os.close(descriptor);raise
        with output:
            json.dump(safe_json(report),output,allow_nan=False,indent=2)
            output.write('\n');output.flush();os.fsync(output.fileno())
        _path(path);os.replace(name,path)
        from ..checkpoint_recovery import _sync_directory
        _sync_directory(path.parent)
    finally:Path(name).unlink(missing_ok=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description='Pinned complete K3 processor-media comparison')
    parser.add_argument('configuration',type=Path)
    parser.add_argument('--configuration-sha256',required=True)
    parser.add_argument('--offline-root',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args(argv)
    configuration=_read_configuration(args.configuration,args.configuration_sha256)
    from .k3_media import run_media_configuration
    report=run_media_configuration(configuration,offline_root=args.offline_root)
    report['configuration_sha256']=args.configuration_sha256
    _publish_report(args.report,report)
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
