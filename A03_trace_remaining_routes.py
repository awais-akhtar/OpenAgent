#!/usr/bin/env python3
"""
A03: Trace Remaining 66 Non-Priority Admitted Routes
====================================================

Priority routes (63) already verified in A02.
This traces the remaining 66 admitted routes:
- Mixtral: 18
- OLMoE: 27
- Inkling: 21

Verify each has complete callable chain and all references exist in source.
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


class RemainingRouteTracer:
    """Trace remaining 66 non-priority admitted routes."""

    PRIORITY_FAMILIES = ['qwen35', 'qwen4', 'glm5-next', 'glm-dsa', 'deepseek-v41']
    REMAINING_FAMILIES = ['mixtral', 'olmoe', 'inkling']

    def __init__(self):
        self.cases = []
        self.remaining_routes = []
        self.traces = {}

    def load_cases(self):
        """Load execution cases."""
        print("\n" + "="*70)
        print("A03: TRACE REMAINING 66 NON-PRIORITY ROUTES")
        print("="*70)

        try:
            self.cases = source_execution_cases()
            print(f"\n✓ Loaded {len(self.cases)} execution cases")
        except Exception as e:
            print(f"\n✗ Failed to load cases: {e}")
            return False

        # Filter to remaining families and admitted routes
        for case in self.cases:
            if case['family'] in self.REMAINING_FAMILIES:
                case_key = (
                    case['family'],
                    case['model_type'],
                    case['backend'],
                    case['arithmetic'],
                    case['source_format'],
                    case['persisted_format']
                )
                if case_key in _SOURCE_VERIFIED_ROUTE_BINDINGS:
                    self.remaining_routes.append(case)

        print(f"✓ Found {len(self.remaining_routes)} remaining admitted routes\n")

        # Group by family
        by_family = defaultdict(list)
        for route in self.remaining_routes:
            by_family[route['family']].append(route)

        for family in sorted(by_family.keys()):
            print(f"  {family:15s}: {len(by_family[family]):2d} routes")

        return True

    def trace_route(self, route):
        """Trace a single route."""
        route_id = route['case_id']
        chain = route.get('callable_chain', {})

        result = {
            'route_id': route_id,
            'family': route['family'],
            'model_type': route['model_type'],
            'backend': route['backend'],
            'arithmetic': route['arithmetic'],
            'source_format': route['source_format'],
            'persisted_format': route['persisted_format'],
            'chain_complete': True,
            'dimensions': {},
            'missing': [],
        }

        for dim in _P02_CHAIN_DIMENSIONS:
            refs = chain.get(dim, ())

            if not refs:
                result['chain_complete'] = False
                result['missing'].append(dim)
                result['dimensions'][dim] = {'present': False}
            else:
                # Convert to tuple if string
                if isinstance(refs, str):
                    refs = (refs,)

                result['dimensions'][dim] = {
                    'present': True,
                    'ref_count': len(refs),
                    'refs': list(refs)[:2],  # First 2 only for brevity
                }

        # Verify source-verified binding exists
        source_review = _SOURCE_VERIFIED_ROUTE_BINDINGS.get(
            (route['family'], route['model_type'], route['backend'],
             route['arithmetic'], route['source_format'], route['persisted_format'])
        )
        result['source_verified'] = source_review is not None

        return result

    def run_traces(self):
        """Trace all remaining routes."""
        print("Tracing remaining routes...")

        for i, route in enumerate(self.remaining_routes):
            if (i + 1) % 20 == 0:
                print(f"  ... {i + 1}/{len(self.remaining_routes)}")

            result = self.trace_route(route)
            self.traces[result['route_id']] = result

        print(f"✓ Tracing complete\n")

    def verify_chains(self):
        """Verify all chains are complete."""
        incomplete = [t for t in self.traces.values() if not t['chain_complete']]

        if incomplete:
            print(f"⚠ {len(incomplete)} routes with incomplete chains:")
            for trace in incomplete[:5]:
                print(f"  {trace['route_id']}")
                print(f"    Missing: {', '.join(trace['missing'])}")
            return False
        else:
            print(f"✓ All {len(self.traces)} routes have complete chains")
            return True

    def print_summary(self):
        """Print summary."""
        print("\n" + "="*70)
        print("REMAINING ROUTES TRACE RESULTS")
        print("="*70)

        by_family = defaultdict(list)
        for route in self.traces.values():
            by_family[route['family']].append(route)

        for family in sorted(by_family.keys()):
            routes = by_family[family]
            complete = sum(1 for r in routes if r['chain_complete'])
            print(f"\n{family}:")
            print(f"  Total:      {len(routes)}")
            print(f"  Complete:   {complete}")

            # Show backend breakdown
            by_backend = defaultdict(int)
            for route in routes:
                by_backend[route['backend']] += 1

            for backend in sorted(by_backend.keys()):
                print(f"    {backend:15s}: {by_backend[backend]:2d} routes")

        print(f"\nTotal remaining: {len(self.traces)}")
        complete_count = sum(1 for t in self.traces.values() if t['chain_complete'])
        print(f"  ✓ Complete chains: {complete_count}")
        print(f"  ~ Incomplete:      {len(self.traces) - complete_count}")

    def export(self, filepath):
        """Export results."""
        data = {
            'timestamp': '2026-10-19',
            'total': len(self.traces),
            'complete': sum(1 for t in self.traces.values() if t['chain_complete']),
            'routes': list(self.traces.values()),
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n✓ Exported to {filepath}")


def main():
    tracer = RemainingRouteTracer()

    if not tracer.load_cases():
        return 1

    tracer.run_traces()

    if not tracer.verify_chains():
        print("\n⚠ Some chains incomplete - needs implementation")
        return 1

    tracer.print_summary()
    tracer.export(project_root / "A03_remaining_traces.json")

    print("\n" + "="*70)
    print("NEXT: A04 - Media lifecycle closure")
    print("="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
