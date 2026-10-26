"""Exact 108-route Colab planning, resumable capture, and receipt verification."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .device_capture import CaptureError
from .device_observations import load_receipts, verify_route_receipt
from .phase1_capture import build_parser
from .source_scope import source_scope_matrix

FIELDS=('family','model_type','backend','arithmetic','source_format','persisted_format')
TARGET_FAMILIES=('olmoe','inkling','qwen4','glm5-next')
PLAN_SCHEMA='openagent.phase1-colab-plan.v1'


def target_rows():
    rows=tuple(sorted((row for row in source_scope_matrix()['execution_cases']
        if row['family'] in TARGET_FAMILIES and row['combination_status']=='verified-admitted'),
        key=lambda row:row['case_id']))
    counts=Counter(row['family'] for row in rows)
    if counts!=Counter({'olmoe':45,'inkling':21,'qwen4':21,'glm5-next':21}):
        raise CaptureError('canonical 108-route target changed; review the plan before capture')
    return rows


def _command(row):
    return {'olmoe':'olmoe','inkling_text':'inkling','inkling_mm_model':'inkling-mm',
            'qwen4_exp_text':'qwen4','qwen4_exp':'qwen4-media',
            'glm5_next_text':'glm5','glm5_next':'glm5-publisher-media'}.get(row['model_type'])


def make_plan(assets):
    """Expand actual asset locations against canonical rows, never a Cartesian guess."""
    if type(assets) is not dict or assets.get('schema')!='openagent.phase1-colab-assets.v1':
        raise CaptureError('versioned Colab asset configuration required')
    libraries=assets['libraries'];entries=[]
    for row in target_rows():
        family,identity,backend=(row[name] for name in ('family','model_type','backend'))
        checkpoint=assets['checkpoints'][identity][row['source_format']]
        command=_command(row)
        args=[command,'--checkpoint',str(checkpoint),'--backend',backend,'--case-id',row['case_id']]
        media=identity in ('qwen4_exp','glm5_next','inkling_mm_model')
        if family=='olmoe':
            manifest=assets['manifests'].get(row['case_id'],assets['manifests'].get(row['persisted_format']))
            if not manifest:raise CaptureError('exact OLMoE manifest required: '+row['case_id'])
            args += ['--manifest',str(manifest),'--library',str(libraries['segment'] if backend=='public-range' else libraries['olmoe']),
                     '--prompt-ids',assets.get('prompt_ids','1,2,3,4')]
        else:
            args += ['--source-format',row['source_format']]
            if media:
                args += ['--media-library',str(libraries['media']),
                         '--prompt',assets['prompts'][identity],'--image',str(assets['image'])]
                if family!='glm5-next' or backend!='device':
                    args += ['--text-library',str(libraries[family])]
            else:
                args += ['--prompt-ids',assets.get('prompt_ids','1,2,3,4')]
                if backend!='device':args += ['--library',str(libraries[family])]
            if backend=='device':
                args += ['--device',assets.get('device','cuda:0')]
                if family=='glm5-next':args += ['--reference-library',str(libraries[family])]
            elif family=='glm5-next':
                args += ['--reference-device',assets.get('device','cuda:0')]
            if backend=='public-range':args += ['--runtime-library',str(libraries['segment'])]
        args += ['--max-tokens',str(assets.get('max_tokens',4)),
                 '--context-tokens',str(assets.get('context_tokens',512)),
                 '--evidence-root',str(assets['evidence_root']),'--receipts',str(assets['receipts'])]
        entries.append({'case_id':row['case_id'],'route':[row[name] for name in FIELDS],'argv':args})
    plan={'schema':PLAN_SCHEMA,'receipts':str(assets['receipts']),'routes':entries}
    validate_plan(plan)
    return plan


def validate_plan(plan):
    """Reject missing/duplicate/wrong routes and CLI mismatches before allocating models."""
    if type(plan) is not dict or plan.get('schema')!=PLAN_SCHEMA or type(plan.get('routes')) is not list:
        raise CaptureError('versioned Colab route plan required')
    if type(plan.get('receipts')) is not str or not plan['receipts'].strip():
        raise CaptureError('nonempty receipt manifest path required')
    expected={row['case_id']:row for row in target_rows()};seen=set();parser=build_parser()
    for entry in plan['routes']:
        if type(entry) is not dict or type(entry.get('case_id')) is not str:
            raise CaptureError('capture route entry must be an object with a string case_id')
        case=entry['case_id'];row=expected.get(case)
        if row is None or case in seen or entry.get('route')!=[row[name] for name in FIELDS]:
            raise CaptureError('duplicate, foreign or incorrectly bound capture route: '+str(case))
        seen.add(case)
        argv=entry.get('argv')
        if type(argv) is not list or not argv or any(type(arg) is not str for arg in argv):
            raise CaptureError('structured capture arguments required')
        try:args=parser.parse_args(argv)
        except SystemExit as error:raise CaptureError('capture command does not match its CLI: '+case) from error
        if (args.command!=_command(row) or args.case_id!=case or args.backend!=row['backend'] or
                Path(args.receipts)!=Path(plan['receipts']) or
                getattr(args,'source_format',row['source_format'])!=row['source_format'] or
                getattr(args,'model_type',row['model_type'])!=row['model_type']):
            raise CaptureError('capture arguments select a different route: '+case)
    if seen!=set(expected):raise CaptureError(f'plan contains {len(seen)}/108 exact routes')
    return expected


def preflight_plan(plan):
    """Check exact checkpoint identities/encodings, files and media inputs locally."""
    from .source_scope import require_checkpoint_inventory_admission
    rows=validate_plan(plan);parser=build_parser();failures=[]
    for entry in plan['routes']:
        row=rows[entry['case_id']];args=parser.parse_args(entry['argv'])
        try:
            for name in ('library','media_library','text_library','reference_library','runtime_library','manifest'):
                path=getattr(args,name,None)
                if path is not None and not Path(path).is_file():raise CaptureError('missing '+name+': '+str(path))
            if not args.checkpoint.is_dir():raise CaptureError('missing checkpoint: '+str(args.checkpoint))
            config=json.loads((args.checkpoint/'config.json').read_text(encoding='utf-8'))
            if config.get('model_type')!=row['model_type']:raise CaptureError('checkpoint model_type differs from route')
            if row['family']!='olmoe':
                inventory=require_checkpoint_inventory_admission(args.checkpoint,family=row['family'],
                    model_type=row['model_type'],backend=row['backend'],arithmetic=row['arithmetic'])
                types={item.dtype for name,item in inventory.tensors.items()
                       if item.dtype in ('F32','F16','BF16','F8_E4M3') and
                       not name.endswith(('.weight_scale_inv','_scale_inv'))}
                if row['source_format'] not in types or not types<={row['source_format'],'F32'}:
                    raise CaptureError('checkpoint encoding differs from route; separate format artifacts required')
            images=getattr(args,'image',None)
            if not isinstance(images,list):images=[] if images is None else [images]
            for image in images:
                if image is not None and not Path(image).is_file():raise CaptureError('missing image: '+str(image))
        except (OSError,ValueError,CaptureError) as error:
            failures.append({'case_id':entry['case_id'],'error':str(error)})
    if failures:raise CaptureError(json.dumps({'preflight_failures':failures},indent=1))
    return {'routes':len(rows),'preflight':'passed'}


def _write_report(path, report):
    """Replace the summary atomically; a summary is never a route receipt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                dir=path.parent, prefix=path.name+'.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=1)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_plan(plan, *, check_only=False, report_path=None, route_timeout=None):
    if route_timeout is not None and (isinstance(route_timeout, bool) or
            not isinstance(route_timeout, (int, float)) or
            not math.isfinite(route_timeout) or route_timeout <= 0):
        raise CaptureError('route timeout must be finite and positive')
    rows=validate_plan(plan)
    if report_path is not None and Path(report_path).resolve() == Path(plan['receipts']).resolve():
        raise CaptureError('report must not overwrite the receipt manifest')
    outcomes=[]
    def summary(state):
        return {'target':108,'accepted':sum(item['accepted'] for item in outcomes),
                'completed':len(outcomes),'state':state,'routes':list(outcomes)}

    def save(state):
        if report_path is not None:
            _write_report(report_path, summary(state))

    save('running')
    for entry in plan['routes']:
        row=rows[entry['case_id']];accepted=False;error='missing receipt'
        receipt=load_receipts(Path(plan['receipts'])).get(entry['case_id'])
        if receipt is not None:
            try:verify_route_receipt(row,receipt);accepted=True
            except (ValueError,OSError) as failure:error=str(failure)
        if not accepted and not check_only:
            print('Capturing '+entry['case_id'],flush=True)
            try:
                result=subprocess.run([sys.executable,'-m','openagent.adaptive_engine.phase1_capture',*entry['argv']],
                    check=False, timeout=route_timeout)
            except subprocess.TimeoutExpired:
                error='capture exceeded route timeout'
                result=None
            except OSError as failure:
                error='capture could not start: '+str(failure)
                result=None
            except KeyboardInterrupt:
                save('interrupted')
                raise
            if result is not None and result.returncode==0:
                receipt=load_receipts(Path(plan['receipts'])).get(entry['case_id'])
                try:
                    if receipt is None:raise CaptureError('command succeeded without the exact route receipt')
                    verify_route_receipt(row,receipt);accepted=True
                except (ValueError,OSError,CaptureError) as failure:error=str(failure)
            elif result is not None:error='capture exited with code '+str(result.returncode)
        outcomes.append({'case_id':entry['case_id'],'accepted':accepted,
                         **({} if accepted else {'error':error})})
        save('running')
    save('finished')
    return summary('finished')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('plan','preflight','run','check'))
    parser.add_argument('--assets',type=Path)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--route-timeout',type=float,
                        help='optional positive per-route capture timeout in seconds')
    args=parser.parse_args(argv)
    try:
        if args.report is not None and args.report.resolve() == args.plan.resolve():
            raise CaptureError('report must not overwrite the plan')
        if args.command=='plan':
            if args.assets is None:raise CaptureError('--assets is required to create a plan')
            if args.plan.resolve() == args.assets.resolve():
                raise CaptureError('plan must not overwrite the asset configuration')
            if args.report is not None and args.report.resolve() == args.assets.resolve():
                raise CaptureError('report must not overwrite the asset configuration')
            result=make_plan(json.loads(args.assets.read_text(encoding='utf-8')))
            if args.plan.resolve() == Path(result['receipts']).resolve():
                raise CaptureError('plan must not overwrite the receipt manifest')
            if args.report is not None and args.report.resolve() == Path(result['receipts']).resolve():
                raise CaptureError('report must not overwrite the receipt manifest')
            _write_report(args.plan,result)
        else:
            plan=json.loads(args.plan.read_text(encoding='utf-8'))
            validate_plan(plan)
            if args.report is not None and args.report.resolve() in (
                    args.plan.resolve(), Path(plan['receipts']).resolve()):
                raise CaptureError('report must not overwrite the plan or receipt manifest')
            if args.command=='preflight':result=preflight_plan(plan)
            else:
                if args.command=='run':preflight_plan(plan)
                result=run_plan(plan,check_only=args.command=='check',
                    report_path=args.report,route_timeout=args.route_timeout)
        if args.report:_write_report(args.report,result)
        print(json.dumps(result,indent=1))
        return 0 if 'accepted' not in result or result['accepted']==108 else 1
    except (CaptureError,ValueError,OSError,KeyError,TypeError) as error:
        print('Colab capture rejected: '+str(error),file=sys.stderr);return 1


if __name__=='__main__':
    raise SystemExit(main())
