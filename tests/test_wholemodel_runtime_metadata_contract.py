import unittest

import hashlib
import json
import unittest

from openagent.adaptive_engine.wholemodel_validation.runner import (
    adapter_runtime_metadata,
    observation_receipt_binding,
    report_receipt_binding,
    validate_report_receipt_binding,
)


class WholeModelRuntimeMetadataContractTests(unittest.TestCase):
    def test_missing_runtime_metadata_is_rejected(self):
        with self.assertRaisesRegex(TypeError, 'runtime_metadata'):
            adapter_runtime_metadata(object())

    def test_empty_runtime_metadata_is_rejected(self):
        class Adapter:
            def runtime_metadata(self):
                return []

        with self.assertRaisesRegex(ValueError, 'nonempty runtime evidence'):
            adapter_runtime_metadata(Adapter())

    def test_nonempty_runtime_metadata_is_preserved(self):
        class Adapter:
            def runtime_metadata(self):
                return [dict(device='cpu', deterministic_algorithms=True)]

        self.assertEqual(adapter_runtime_metadata(Adapter()),
                         [dict(device='cpu', deterministic_algorithms=True)])


class WholeModelReportReceiptContractTests(unittest.TestCase):
    @staticmethod
    def _execution_binding():
        payload = dict(contract={'family': 'test'},
                       oracle_factory='oracle:build',
                       candidate_factory='candidate:build',
                       artifact_pins={'oracle': [], 'candidate': []})
        digest = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(',', ':'),
            allow_nan=False).encode('utf-8')).hexdigest()
        return dict(payload, binding_sha256=digest, receipt_required=True,
                    receipt_present=False, executed=False, certification=False)

    @classmethod
    def _report(cls):
        manifest = 'a' * 64
        execution = cls._execution_binding()
        archive = 'b' * 64
        receipt = observation_receipt_binding(
            artifact_sha256=archive, manifest_sha256=manifest,
            execution_binding_sha256=execution['binding_sha256'])
        return dict(manifest_sha256=manifest, execution_binding=execution,
                    observation_artifact=dict(sha256=archive,
                                              execution_binding=receipt))

    def test_report_binding_validates_nested_observation_receipt_digest(self):
        report = self._report()
        report['report_binding'] = report_receipt_binding(report)
        self.assertIs(validate_report_receipt_binding(report), report['report_binding'])

        report['observation_artifact']['execution_binding']['binding_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'observation receipt digest'):
            report_receipt_binding(report)

    def test_report_binding_validates_nested_execution_digest(self):
        report = self._report()
        report['execution_binding']['binding_sha256'] = '0' * 64
        report['observation_artifact']['execution_binding'] = observation_receipt_binding(
            artifact_sha256=report['observation_artifact']['sha256'],
            manifest_sha256=report['manifest_sha256'],
            execution_binding_sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'execution binding digest'):
            report_receipt_binding(report)

    def test_report_binding_rejects_receipt_claiming_certification(self):
        report = self._report()
        report['observation_artifact']['execution_binding']['certification'] = True
        with self.assertRaisesRegex(ValueError, 'observation receipt'):
            report_receipt_binding(report)


if __name__ == '__main__':
    unittest.main()
