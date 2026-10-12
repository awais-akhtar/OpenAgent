"""Explicit execution entry point; importing never runs comparisons."""
import argparse
import json
from pathlib import Path

from .runner import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'report.json').write_text(json.dumps(dict(
            schema='openagent.wholemodel-validation.v1', status='failed',
            certification=False, errors=[dict(type=type(error).__name__, message=str(error))]),
            indent=2) + '\n', encoding='utf-8')
        return 2
    report = run(manifest, base_directory=args.manifest.resolve().parent,
                 output_directory=args.output)
    print(json.dumps(dict(status=report['status'], report=str((args.output / 'report.json').resolve()))))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
