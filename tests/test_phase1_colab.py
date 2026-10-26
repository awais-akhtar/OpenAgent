"""The Colab plan must resolve exactly 108 real CLI contracts before execution."""
from copy import deepcopy
from pathlib import Path
import ast
import json
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from openagent.adaptive_engine.device_capture import CaptureError
from openagent.adaptive_engine.phase1_colab import (
    make_plan, target_rows, validate_plan, run_plan, main, _write_report,
)


def assets():
    checkpoints={}
    for row in target_rows():
        checkpoints.setdefault(row['model_type'],{})[row['source_format']]=(
            '/content/models/'+row['model_type']+'/'+row['source_format'])
    return {'schema':'openagent.phase1-colab-assets.v1','checkpoints':checkpoints,
            'libraries':{name:'/content/build/'+name+'.so'
                         for name in ('inkling','qwen4','glm5-next','olmoe','segment','media')},
            'manifests':{row['persisted_format']:'/content/manifests/'+row['persisted_format']+'.json'
                         for row in target_rows() if row['family']=='olmoe'},
            'prompts':{name:'Describe this image.' for name in ('inkling_mm_model','qwen4_exp','glm5_next')},
            'image':'/content/input.png','receipts':'/content/receipts.json',
            'evidence_root':'/content/evidence'}


class ColabPlanTests(unittest.TestCase):
    def test_every_target_has_one_parseable_exact_command(self):
        plan=make_plan(assets())
        self.assertEqual(len(validate_plan(plan)),108)
        self.assertEqual(len({entry['case_id'] for entry in plan['routes']}),108)
        self.assertEqual(sum(entry['argv'][0]=='inkling-mm' for entry in plan['routes']),12)
        self.assertEqual(sum(entry['argv'][0]=='qwen4-media' for entry in plan['routes']),12)
        self.assertEqual(sum(entry['argv'][0]=='glm5-publisher-media' for entry in plan['routes']),12)

    def test_duplicate_missing_or_foreign_route_cannot_be_counted(self):
        original=make_plan(assets())
        for change in ('duplicate','missing','foreign'):
            plan=deepcopy(original)
            if change=='duplicate':plan['routes'][0]=deepcopy(plan['routes'][1])
            elif change=='missing':plan['routes'].pop()
            else:plan['routes'][0]['case_id']='foreign'
            with self.subTest(change=change),self.assertRaises(CaptureError):validate_plan(plan)

    def test_backend_and_receipt_destination_cannot_drift(self):
        plan=make_plan(assets())
        entry=next(item for item in plan['routes'] if item['argv'][0]=='inkling-mm')
        old=entry['argv'][entry['argv'].index('--backend')+1]
        entry['argv'][entry['argv'].index('--backend')+1]='device' if old!='device' else 'native'
        with self.assertRaises(CaptureError):validate_plan(plan)

    def test_successful_command_without_exact_receipt_never_counts(self):
        plan=make_plan(assets())
        with patch('openagent.adaptive_engine.phase1_colab.load_receipts',return_value={}), \
                patch('openagent.adaptive_engine.phase1_colab.subprocess.run') as run, \
                patch('builtins.print'):
            run.return_value.returncode=0
            report=run_plan(plan)
        self.assertEqual(report['accepted'],0)
        self.assertEqual(len(report['routes']),108)
        self.assertEqual(run.call_count,108)
        self.assertTrue(all('without the exact route receipt' in item['error'] for item in report['routes']))

    def test_notebook_has_python_cells_and_uses_shared_plan_runner(self):
        path=Path(__file__).resolve().parents[1]/'PHASE_1_COLAB_COMPLETE_108.ipynb'
        document=json.loads(path.read_text(encoding='utf-8'))
        code='\n'.join(''.join(cell['source']) for cell in document['cells'] if cell['cell_type']=='code')
        for cell in document['cells']:
            if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
        self.assertIn('phase1_colab',code)
        self.assertNotIn("'inkling_mm',",code)

    def test_launch_failure_and_timeout_are_recorded_without_false_success(self):
        plan=make_plan(assets())
        failures=[OSError('cannot launch'),subprocess.TimeoutExpired('capture',2)]
        with tempfile.TemporaryDirectory() as directory:
            report_path=Path(directory)/'progress.json'
            with patch('openagent.adaptive_engine.phase1_colab.load_receipts',return_value={}), \
                    patch('openagent.adaptive_engine.phase1_colab.subprocess.run',
                          side_effect=(failures*54)) as run, patch('builtins.print'):
                report=run_plan(plan,report_path=report_path,route_timeout=2)
            self.assertEqual(report['accepted'],0)
            self.assertEqual(report['completed'],108)
            self.assertEqual(json.loads(report_path.read_text()),report)
            self.assertIn('could not start',report['routes'][0]['error'])
            self.assertIn('timeout',report['routes'][1]['error'])
            self.assertEqual(run.call_args.kwargs['timeout'],2)

    def test_interruption_keeps_completed_route_summary(self):
        plan=make_plan(assets())
        with tempfile.TemporaryDirectory() as directory:
            report_path=Path(directory)/'progress.json'
            with patch('openagent.adaptive_engine.phase1_colab.load_receipts',return_value={}), \
                    patch('openagent.adaptive_engine.phase1_colab.subprocess.run',
                          side_effect=[OSError('first failure'),KeyboardInterrupt]), \
                    patch('builtins.print'),self.assertRaises(KeyboardInterrupt):
                run_plan(plan,report_path=report_path)
            report=json.loads(report_path.read_text())
            self.assertEqual(report['state'],'interrupted')
            self.assertEqual(report['completed'],1)
            self.assertEqual(report['accepted'],0)
            self.assertEqual(list(Path(directory).glob('*.tmp')),[])

    def test_timeout_and_report_destination_fail_before_capture(self):
        plan=make_plan(assets())
        with patch('openagent.adaptive_engine.phase1_colab.subprocess.run') as run:
            for timeout in (True,0,-1,float('inf'),float('nan'),'2'):
                with self.subTest(timeout=timeout),self.assertRaises(CaptureError):
                    run_plan(plan,route_timeout=timeout)
            with self.assertRaises(CaptureError):
                run_plan(plan,report_path=plan['receipts'])
            run.assert_not_called()

    def test_failed_atomic_report_write_preserves_previous_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'summary.json'
            _write_report(path,{'completed':1})
            with patch('openagent.adaptive_engine.phase1_colab.os.replace',
                       side_effect=OSError('disk failure')),self.assertRaises(OSError):
                _write_report(path,{'completed':2})
            self.assertEqual(json.loads(path.read_text()),{'completed':1})
            self.assertEqual(list(Path(directory).glob('*.tmp')),[])

    def test_cli_refuses_report_overwriting_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'plan.json'
            path.write_text(json.dumps(make_plan(assets())),encoding='utf-8')
            original=path.read_bytes()
            with patch('sys.stderr'):
                result=main(['check','--plan',str(path),'--report',str(path)])
            self.assertEqual(result,1)
            self.assertEqual(path.read_bytes(),original)

    def test_malformed_entries_and_manifest_paths_fail_closed(self):
        original=make_plan(assets())
        for entry in (None,42,[],{'case_id':[]},{'case_id':{}},'route'):
            plan=deepcopy(original)
            plan['routes'][0]=entry
            with self.subTest(entry=entry),self.assertRaises(CaptureError):
                validate_plan(plan)
        for receipts in (None,[],{},42,'','   '):
            plan=deepcopy(original)
            plan['receipts']=receipts
            with self.subTest(receipts=receipts),self.assertRaises(CaptureError):
                run_plan(plan,check_only=True)

    def test_plan_output_cannot_overwrite_inputs_or_receipts(self):
        for collision in ('assets','receipts'):
            with self.subTest(collision=collision),tempfile.TemporaryDirectory() as directory:
                config=Path(directory)/'assets.json'
                receipts=Path(directory)/'receipts.json'
                data=assets()
                data['receipts']=str(receipts)
                config.write_text(json.dumps(data),encoding='utf-8')
                receipts.write_text('existing receipt manifest',encoding='utf-8')
                before=(config.read_bytes(),receipts.read_bytes())
                output=config if collision=='assets' else receipts
                with patch('sys.stderr'):
                    result=main(['plan','--assets',str(config),'--plan',str(output)])
                self.assertEqual(result,1)
                self.assertEqual((config.read_bytes(),receipts.read_bytes()),before)


if __name__=='__main__':unittest.main()
