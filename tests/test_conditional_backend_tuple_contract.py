"""P02 backend-tuple admission ordering for conditional publisher loading."""
import ast
from pathlib import Path
import unittest


class ConditionalBackendTupleContractTest(unittest.TestCase):
    def test_device_and_segment_tuple_is_rejected_before_checkpoint_planning(self):
        source_path=(Path(__file__).parents[1]/'openagent'/'adaptive_engine'/'publisher_conditional_loader.py')
        text=source_path.read_text(encoding='utf-8')
        module=ast.parse(text,filename=str(source_path))
        loader=next(node for node in module.body if isinstance(node,ast.FunctionDef)
                    and node.name=='load_publisher_conditional_decoder')
        body=ast.get_source_segment(text,loader)
        guard=body.index('publisher device text and Segment runtime backends are mutually exclusive')
        self.assertLess(guard,body.index('inventory=snapshot('))
        self.assertLess(guard,body.index('reader=PublisherTensorReader('))
        self.assertIn('text_device is not None and segment_runtime_library is not None',body)


if __name__=='__main__':
    unittest.main()
