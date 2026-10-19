"""P02 identity contract for the generic native Segment/Edge factory."""
import ast
from pathlib import Path
import unittest


class SegmentEdgeFactoryIdentityContractTest(unittest.TestCase):
    def test_factory_binds_claimed_family_and_model_to_checkpoint_before_runtime(self):
        source_path=(Path(__file__).parents[1]/'openagent'/'adaptive_engine'/'execution_factories.py')
        text=source_path.read_text(encoding='utf-8')
        module=ast.parse(text,filename=str(source_path))
        factory=next(node for node in module.body if isinstance(node,ast.FunctionDef)
                     and node.name=='create_segment_edge_execution_service')
        body=ast.get_source_segment(text,factory)
        self.assertIn('source_family_for_identity(model_type)!=source_family',body)
        self.assertIn("inventory.config.get('model_type')!=model_type",body)
        self.assertLess(body.index('source_family_for_identity(model_type)'),
                        body.index('SegmentEdgeExecution('))
        self.assertLess(body.index("inventory.config.get('model_type')"),
                        body.index('SegmentEdgeExecution('))


if __name__=='__main__':
    unittest.main()
