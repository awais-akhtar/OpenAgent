import unittest

from openagent.adaptive_engine.k3_execution.exact_admission import _media_coverage


class K3MediaExactAdmissionTests(unittest.TestCase):
    def test_media_evidence_with_error_inventory_is_rejected(self):
        value = {
            'execution_identity': 'k3-native-v1',
            'arithmetic_identity': 'bf16-v1',
        }
        report = {
            'schema': 'openagent.k3-media-wholemodel.v2',
            'status': 'passed',
            'errors': ['failed resource cleanup'],
            'candidate_runtime': {
                'execution_identity': value['execution_identity'],
                'arithmetic_identity': value['arithmetic_identity'],
            },
        }
        with self.assertRaisesRegex(
                ValueError, 'error-free media comparison required'):
            _media_coverage(report, value)


if __name__ == '__main__':
    unittest.main()
