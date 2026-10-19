"""Source contract for direct public ABI publisher composition."""
import ast
from pathlib import Path
import unittest


class PublisherPublicGeometryContractTest(unittest.TestCase):
    def test_public_adapter_rejects_service_geometry_mismatch_before_abi_loading(self):
        source_path=(Path(__file__).parents[1]/'openagent'/'adaptive_engine'/'publisher_public_adapters.py')
        module=ast.parse(source_path.read_text(encoding='utf-8'),filename=str(source_path))
        adapter=next(node for node in module.body if isinstance(node,ast.ClassDef)
                     and node.name=='NativePublisherPublicAdapters')
        constructor=next(node for node in adapter.body if isinstance(node,ast.FunctionDef)
                         and node.name=='__init__')
        source=ast.get_source_segment(source_path.read_text(encoding='utf-8'),constructor)
        self.assertLess(source.index("limits=getattr(backing_service,'limits',None)"),
                        source.index('NativeLibraryIdentity.read(runtime_library)'))
        self.assertIn("getattr(limits,'vocabulary',None)!=vocab",source)
        self.assertIn("getattr(limits,'context',None)!=context",source)


if __name__=='__main__':
    unittest.main()
