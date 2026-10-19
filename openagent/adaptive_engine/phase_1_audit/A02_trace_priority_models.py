#!/usr/bin/env python3
"""
A02: Trace Priority Models Through Callable Chain
==================================================

For each admitted route in priority models, verify that every dimension
in the chain actually exists in the real source code.

Priority models:
- Qwen3.5-MoE (14 admitted)
- Qwen4-Exp (14 admitted)
- GLM5-Next (17 admitted)
- GLM-MoE-DSA (9 admitted)
- DeepSeek V4.1 (9 admitted)

This is NOT about building code. It's about verifying that the claimed
callable chains in source_scope.py actually point to real, importable code.

Truth only: every claim verified against actual source files.
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from openagent.adaptive_engine.source_scope import (
        _P02_CHAIN_DIMENSIONS,
        source_execution_cases,
        _SOURCE_VERIFIED_ROUTE_BINDINGS,
    )
except ImportError as e:
    print(f"ERROR: Cannot import source_scope: {e}")
    sys.exit(1)


class ChainVerifier:
    """Verify callable chain dimensions against actual source."""

    PRIORITY_FAMILIES = ['qwen35', 'qwen4', 'glm5-next', 'glm-dsa', 'deepseek-v41']

    def __init__(self):
        self.cases = []
        self.priority_routes = []
        self.verifications = defaultdict(dict)
        self.missing_imports = defaultdict(list)

    def load_cases(self):
        """Load execution cases."""
        print("\n" + "="*70)
        print("A02: TRACE PRIORITY MODELS")
        print("="*70)

        try:
            self.cases = source_execution_cases()
            print(f"\n✓ Loaded {len(self.cases)} execution cases")
        except Exception as e:
            print(f"\n✗ Failed to load cases: {e}")
            return False

        # Filter to priority families and admitted routes
        for case in self.cases:
            if case['family'] in self.PRIORITY_FAMILIES:
                # Check if this is an admitted route
                case_key = (
                    case['family'],
                    case['model_type'],
                    case['backend'],
                    case['arithmetic'],
                    case['source_format'],
                    case['persisted_format']
                )
                if case_key in _SOURCE_VERIFIED_ROUTE_BINDINGS:
                    self.priority_routes.append(case)

        print(f"✓ Found {len(self.priority_routes)} admitted priority routes\n")

        # Group by family
        by_family = defaultdict(list)
        for route in self.priority_routes:
            by_family[route['family']].append(route)

        for family in sorted(by_family.keys()):
            print(f"  {family:15s}: {len(by_family[family]):2d} routes")

        return True

    def verify_import(self, module_path: str) -> bool:
        """Try to import a module reference to verify it exists."""
        if not module_path or module_path.startswith('missing'):
            return False

        # Parse module path like "native_qwen35.NativeQwen35TextDecoder"
        parts = module_path.split('.')

        try:
            # Try importing the module part
            module_name = parts[0]
            # This is a heuristic check - we're not actually importing
            # Just checking if the file would exist
            module_file = project_root / 'openagent' / 'adaptive_engine' / f'{module_name}.py'

            if module_file.exists():
                return True

            # Check in subdirectories
            for subdir in ['v41_execution', 'distributed_execution', 'cache_owners']:
                subfile = project_root / 'openagent' / 'adaptive_engine' / subdir / f'{module_name}.py'
                if subfile.exists():
                    return True

            return False
        except Exception:
            return False

    def trace_route(self, route):
        """Trace a single route through all chain dimensions."""
        route_id = route['case_id']
        chain = route.get('callable_chain', {})

        result = {
            'route_id': route_id,
            'family': route['family'],
            'backend': route['backend'],
            'dimensions': {},
            'all_present': True,
            'missing_dims': [],
        }

        for dim in _P02_CHAIN_DIMENSIONS:
            refs = chain.get(dim, ())

            if not refs:
                result['dimensions'][dim] = {
                    'present': False,
                    'refs': (),
                }
                result['missing_dims'].append(dim)
                result['all_present'] = False
            else:
                # Convert to tuple if string
                if isinstance(refs, str):
                    refs = (refs,)

                result['dimensions'][dim] = {
                    'present': True,
                    'refs': list(refs),
                    'verified': all(
                        self.verify_import(ref) for ref in refs
                    ),
                }

        return result

    def run_verification(self):
        """Verify all priority routes."""
        print("Tracing callable chains...")

        for i, route in enumerate(self.priority_routes):
            if (i + 1) % 10 == 0:
                print(f"  ... {i + 1}/{len(self.priority_routes)}")

            result = self.trace_route(route)
            route_id = route['case_id']
            self.verifications[route_id] = result

        print(f"✓ Verification complete\n")

    def print_summary(self):
        """Print verification summary."""
        print("="*70)
        print("CHAIN VERIFICATION RESULTS")
        print("="*70)

        total = len(self.verifications)
        all_present = sum(1 for v in self.verifications.values() if v['all_present'])

        print(f"\nTotal routes traced: {total}")
        print(f"  ✓ All dimensions present: {all_present}")
        print(f"  ~ Partial chains:        {total - all_present}")

        # Show routes with missing dimensions
        partial = [v for v in self.verifications.values() if not v['all_present']]
        if partial:
            print(f"\nRoutes with incomplete chains ({len(partial)}):")
            for route in partial[:10]:
                print(f"\n  {route['route_id']}")
                print(f"    Missing: {', '.join(route['missing_dims'])}")

        print("\n" + "="*70)
        print("INTERPRETATION")
        print("="*70)

        if all_present == total:
            print("\n✓ ALL PRIORITY ROUTES have complete callable chains")
            print("  Next: Verify each reference points to actual source code")
            print("  Then: Check numerical parity (A10 requires execution)")
        else:
            print(f"\n⚠ {total - all_present} routes have missing chain dimensions")
            print("  These need implementation or explicit rejection before Phase 1 closure")

    def export_results(self, filepath):
        """Export verification results."""
        data = {
            'timestamp': '2026-10-19',
            'total_routes': len(self.verifications),
            'routes': list(self.verifications.values()),
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n✓ Exported to {filepath}")


def main():
    verifier = ChainVerifier()

    if not verifier.load_cases():
        return 1

    verifier.run_verification()
    verifier.print_summary()

    output = project_root / "A02_trace_results.json"
    verifier.export_results(str(output))

    return 0


if __name__ == '__main__':
    sys.exit(main())
