"""Explicit later-phase offline Qwen4 media comparison command."""
import argparse
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(description='Pinned complete Qwen4 processor-media comparison')
    parser.add_argument('configuration',type=Path)
    parser.add_argument('--configuration-sha256',required=True)
    parser.add_argument('--offline-root',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args(argv)
    from .k3_media_cli import _read_configuration,_publish_report
    from .qwen4_media import run_media_configuration
    configuration=_read_configuration(args.configuration,args.configuration_sha256)
    report=run_media_configuration(configuration,offline_root=args.offline_root)
    report['configuration_sha256']=args.configuration_sha256
    _publish_report(args.report,report)
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
