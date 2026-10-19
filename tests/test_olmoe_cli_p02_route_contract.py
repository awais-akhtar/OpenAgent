"""P02 fail-closed contract for the OLMoE Python reference CLI route."""
import ast
from pathlib import Path
import unittest


class OlmoeCliP02RouteContractTest(unittest.TestCase):
    def test_reference_execution_commands_are_rejected_before_model_creation(self):
        source_path=(Path(__file__).parents[1]/'openagent'/'adaptive_engine'/'olmoe_cli.py')
        text=source_path.read_text(encoding='utf-8')
        module=ast.parse(text,filename=str(source_path))
        dispatch=next(node for node in module.body if isinstance(node,ast.FunctionDef)
                      and node.name=='_dispatch')
        body=ast.get_source_segment(text,dispatch)
        guard=body.index("if args.command in ('tune','verify','generate','benchmark')")
        rejection=body.index('not an admitted P02 execution backend')
        model_creation=body.index('OlmoeModel(')
        self.assertLess(guard,rejection)
        self.assertLess(rejection,model_creation)
        for command in ('tune','verify','generate','benchmark'):
            self.assertIn("'"+command+"'",body[guard:rejection])


if __name__=='__main__':
    unittest.main()
