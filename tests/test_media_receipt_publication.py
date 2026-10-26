"""Isolated publication failure tests, not real model evidence/certification."""
from contextlib import contextmanager, ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from openagent.adaptive_engine.device_capture import _pin
from openagent.adaptive_engine.media_route_capture import publish_media_measurements


class MediaReceiptPublicationTests(unittest.TestCase):
    def exercise(self, directory, *, failure):
        root=Path(directory)
        config=root/'config.json'
        config.write_text('{}',encoding='utf-8')
        measured={
            'case_id':'fixture/media','route':['qwen4','qwen4_exp','native','fp32','F32','float32'],
            'consumer':'fixture','abi_owner':'fixture','implementation':'fixture',
            'prompt_ids':[1],'generated_ids':[2],'reference_generated_ids':[2],
            'finish_reason':'length','candidate_logits':[],'reference_logits':[],
            'tolerance':0,'cache':{},'routing_layers':[],'recovery':{},
            'placement':{'device':'cpu'},'comparison_reference':{'implementation':'fixture'},
            'artifacts':[_pin(config,'config')],'cache_schema':'fixture',
        }
        @contextmanager
        def fail_on_unlock(path):
            yield
            raise OSError('injected post-commit unlock failure')

        with ExitStack() as stack:
            prefix='openagent.adaptive_engine.media_route_capture.'
            stack.enter_context(patch(prefix+'canonical_row',return_value={'case_id':'fixture/media'}))
            stack.enter_context(patch(prefix+'coverage_applicability',return_value={
                name:False for name in ('media_lifecycle','speculative_state','distributed_recovery')}))
            stack.enter_context(patch(prefix+'encode_matrix',return_value={}))
            stack.enter_context(patch(prefix+'verify_route_receipt',
                side_effect=ValueError('injected gate rejection') if failure=='gate' else None))
            if failure=='unlock':
                stack.enter_context(patch('openagent.adaptive_engine.device_capture._receipt_manifest_lock',
                    fail_on_unlock))
            else:
                stack.enter_context(patch(prefix+'_write_receipts',side_effect=OSError('before commit')))
            with self.assertRaises((ValueError,OSError)):
                publish_media_measurements(measured,evidence_root=root/'evidence',
                    receipts_path=root/'receipts.json')
        return root

    def test_post_commit_error_retains_every_committed_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=self.exercise(directory,failure='unlock')
            receipt=json.loads((root/'receipts.json').read_text())['routes'][0]
            self.assertTrue(receipt['observations'])
            for pin in receipt['observations']:
                self.assertTrue(Path(pin['path']).is_file())
                self.assertEqual(_pin(Path(pin['path']),pin['kind'])['sha256'],pin['sha256'])

    def test_uncertain_publication_retains_validated_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=self.exercise(directory,failure='write')
            self.assertFalse((root/'receipts.json').exists())
            self.assertEqual(len(list((root/'evidence').iterdir())),1)

    def test_gate_rejection_removes_unpublished_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root=self.exercise(directory,failure='gate')
            self.assertFalse((root/'receipts.json').exists())
            self.assertEqual(list((root/'evidence').iterdir()),[])


if __name__=='__main__':
    unittest.main()
