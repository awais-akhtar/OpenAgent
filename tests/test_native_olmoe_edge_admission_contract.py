"""Source contract for the standalone native OLMoE edge loader."""
import ast
from pathlib import Path
import unittest


class NativeOlmoeEdgeAdmissionContractTest(unittest.TestCase):
    def test_edge_admits_exact_checkpoint_tuple_before_reading_dense_tensors(self):
        source_path=(Path(__file__).parents[1]/'openagent'/'adaptive_engine'/'native_olmoe_segments.py')
        text=source_path.read_text(encoding='utf-8')
        module=ast.parse(text,filename=str(source_path))
        edge=next(node for node in module.body if isinstance(node,ast.ClassDef)
                  and node.name=='NativeOlmoeEdge')
        constructor=next(node for node in edge.body if isinstance(node,ast.FunctionDef)
                         and node.name=='__init__')
        segment=ast.get_source_segment(text,constructor)
        self.assertIn('require_checkpoint_inventory_admission(',segment)
        for exact in ("family='olmoe'", "model_type='olmoe'", "backend='native'",
                      "arithmetic='fp32'", "persisted_format='float32'"):
            self.assertIn(exact,segment)
        self.assertLess(segment.index('require_checkpoint_inventory_admission('),
                        segment.index('OlmoeConfig.read(self.root)'))


if __name__=='__main__':
    unittest.main()
