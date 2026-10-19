#!/usr/bin/env python3
"""
A08 + A09: K3 Metal ABI & P13 Evidence Binding Audit
==================================================

A08: K3 Metal ABI Status
- Metal mixer: token → logit dispatch
- Currently: NOT A COMPLETE ADMITTED TOKEN-TO-LOGIT DISPATCH
- Blocker: Apple Metal SDK not available on this host

A09: P13 Evidence Binding
- Artifact pinning (checkpoint path, weights, tokenizer)
- Cache schema (layout, footprint, device placement)
- Comparison adapters (numerical parity hooks)
- Execution observations (latency, memory, correctness)
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from openagent.adaptive_engine.source_scope import (
        source_execution_cases,
        _SOURCE_VERIFIED_ROUTE_BINDINGS,
    )
except ImportError as e:
    print(f"ERROR: Cannot import source_scope: {e}")
    sys.exit(1)


class MetalEvidenceAuditor:
    """Audit K3 Metal ABI and P13 evidence binding."""

    def __init__(self):
        self.cases = []
        self.k3_routes = []

    def load_cases(self):
        """Load cases."""
        print("\n" + "="*70)
        print("A08 + A09: K3 METAL ABI & P13 EVIDENCE BINDING")
        print("="*70)

        try:
            self.cases = source_execution_cases()
            print(f"\n✓ Loaded {len(self.cases)} cases")
        except Exception as e:
            print(f"\n✗ Failed: {e}")
            return False

        # Filter to K3
        self.k3_routes = [c for c in self.cases if c['family'] == 'kimi-k3']
        print(f"✓ Found {len(self.k3_routes)} K3 routes\n")

        return True

    def audit_metal_abi(self):
        """Audit K3 Metal ABI."""
        print("A08: K3 Metal ABI Audit\n")

        # Count K3 routes by backend
        by_backend = defaultdict(list)
        for route in self.k3_routes:
            by_backend[route['backend']].append(route)

        metal_routes = by_backend.get('metal', [])

        result = {
            'total_k3': len(self.k3_routes),
            'total_metal': len(metal_routes),
            'metal_admitted': 0,
            'metal_restricted': 0,
            'metal_rejected': 0,
            'requirements': {},
        }

        # Count statuses
        for route in metal_routes:
            status = route.get('combination_status', 'unknown')
            if status == 'verified-admitted':
                result['metal_admitted'] += 1
            elif status == 'restricted':
                result['metal_restricted'] += 1
            else:
                result['metal_rejected'] += 1

        result['requirements'] = {
            'metal_kernel': {
                'description': 'Metal compute kernel for token → logit dispatch',
                'status': 'NOT IMPLEMENTED',
                'reason': 'Apple Metal SDK not available on this host',
                'location': 'native_kimi_k3_metal.metal or similar',
            },
            'metal_buffer_management': {
                'description': 'Metal MTLBuffer allocation and lifecycle',
                'status': 'NOT IMPLEMENTED',
                'location': 'native_kimi_k3.MetalK3Engine',
            },
            'metal_device_selection': {
                'description': 'Select appropriate Metal device (GPU, Neural Engine)',
                'status': 'NOT IMPLEMENTED',
                'location': 'native_kimi_k3.MetalDeviceSelector',
            },
            'metal_abi_callbacks': {
                'description': 'Callback ownership and reentrancy safety',
                'status': 'PARTIALLY IMPLEMENTED',
                'location': 'phase1_closure_integrated.FamilyABICallbackManager',
            },
        }

        return result

    def audit_evidence_binding(self):
        """Audit P13 evidence binding for all 129 admitted routes."""
        print("\nA09: P13 Evidence Binding Audit\n")

        # Count routes by admission status
        admitted_routes = []
        for case in self.cases:
            case_key = (
                case['family'],
                case['model_type'],
                case['backend'],
                case['arithmetic'],
                case['source_format'],
                case['persisted_format']
            )
            if case_key in _SOURCE_VERIFIED_ROUTE_BINDINGS:
                admitted_routes.append(case)

        result = {
            'total_admitted': len(admitted_routes),
            'evidence_bindings': {
                'artifact_pins': {
                    'checkpoint_path': {
                        'description': 'Exact checkpoint location (HuggingFace, local, revision)',
                        'required': True,
                        'captured': False,
                    },
                    'weights': {
                        'description': 'Weight file hashes and versions',
                        'required': True,
                        'captured': False,
                    },
                    'tokenizer': {
                        'description': 'Tokenizer config and vocabulary',
                        'required': True,
                        'captured': False,
                    },
                    'runtime': {
                        'description': 'Runtime version and config flags',
                        'required': True,
                        'captured': False,
                    },
                },
                'cache_schema': {
                    'layout': {
                        'description': 'Cache memory layout (tensor shapes, block sizes)',
                        'required': True,
                        'captured': False,
                    },
                    'footprint': {
                        'description': 'Total cache size and per-layer breakdown',
                        'required': True,
                        'captured': False,
                    },
                    'device_placement': {
                        'description': 'Where cache resides (VRAM/RAM/NVMe)',
                        'required': True,
                        'captured': False,
                    },
                },
                'execution_observations': {
                    'numerical_parity': {
                        'description': 'Output matches reference (within tolerance)',
                        'required': True,
                        'captured': False,
                        'blocker': 'A10 - requires model execution',
                    },
                    'latency': {
                        'description': 'Time per token generation',
                        'required': True,
                        'captured': False,
                        'blocker': 'A10 - requires GPU/accelerator',
                    },
                    'memory_peak': {
                        'description': 'Peak memory usage during execution',
                        'required': True,
                        'captured': False,
                        'blocker': 'A10 - requires execution',
                    },
                    'platform_cert': {
                        'description': 'Platform-specific certification (CUDA/HIP/Metal)',
                        'required': True,
                        'captured': False,
                        'blocker': 'A10 - requires target device',
                    },
                },
            },
            'evidence_status': {
                'artifact_pins': 'STRUCTURE READY, DATA MISSING',
                'cache_schema': 'STRUCTURE READY, DATA MISSING',
                'execution_observations': 'BLOCKED ON A10 (NO EXECUTION)',
            },
        }

        return result

    def print_summary(self, metal_result, evidence_result):
        """Print summary."""
        print("="*70)
        print("A08: K3 METAL ABI")
        print("="*70)

        print(f"\nK3 Metal Routes:")
        print(f"  Total K3:      {metal_result['total_k3']}")
        print(f"  Metal backend: {metal_result['total_metal']}")
        print(f"    Admitted:    {metal_result['metal_admitted']}")
        print(f"    Restricted:  {metal_result['metal_restricted']}")
        print(f"    Rejected:    {metal_result['metal_rejected']}")

        print(f"\nMetal ABI Requirements:")
        for name, req in metal_result['requirements'].items():
            print(f"  {name}:")
            print(f"    {req['description']}")
            print(f"    Status: {req['status']}")
            if 'reason' in req:
                print(f"    Reason: {req['reason']}")
            print(f"    Location: {req['location']}")

        print(f"\n✗ BLOCKER: K3 Metal ABI not implemented")
        print(f"   Apple Metal SDK unavailable on this host (CPU-only)")
        print(f"   This blocks ALL K3 metal routes (0 → 0 admitted)")

        print(f"\n\n" + "="*70)
        print("A09: P13 EVIDENCE BINDING")
        print("="*70)

        print(f"\nAdmitted Routes Requiring Evidence: {evidence_result['total_admitted']}")

        print(f"\nEvidence Categories:\n")

        for category, bindings in evidence_result['evidence_bindings'].items():
            print(f"{category}:")
            for binding, details in bindings.items():
                required = "[Required]" if details['required'] else "[Optional]"
                status = "Captured" if details['captured'] else "Missing"
                print(f"  {binding} {required}")
                print(f"    {details['description']}")
                print(f"    Status: {status}")
                if 'blocker' in details:
                    print(f"    Blocked: {details['blocker']}")

        print(f"\n\nEvidence Status:")
        for cat, status in evidence_result['evidence_status'].items():
            print(f"  {cat}: {status}")

        print(f"\n" + "="*70)
        print("A08/A09 CLOSURE ASSESSMENT")
        print("="*70)

        print(f"""
A08 Status: BLOCKED
  ✗ K3 Metal ABI: NOT IMPLEMENTED
  ✗ Apple Metal SDK: NOT AVAILABLE (CPU-only host)
  ✗ Impact: 0/16 Metal routes can be admitted

A08 Blocker: Host environment incompatible with Metal
  Requires: Real Apple hardware or Metal SDK emulation

A09 Status: STRUCTURE READY, DATA MISSING
  ✓ Evidence binding framework defined
  ✓ Artifact pinning locations identified
  ✓ Cache schema structure clear
  ✗ Execution observations: BLOCKED ON A10
  ✗ All 129 admitted routes still need P13 capture

A09 Requirements (per route):
  1. Artifact pins: checkpoint path, weights, tokenizer, runtime
  2. Cache schema: layout, footprint, device placement
  3. Execution observations:
     - Numerical parity (output correctness)
     - Latency (tokens/second)
     - Memory peak (VRAM usage)
     - Platform certification

A09 Blocker: A10 execution infrastructure
  Cannot capture execution observations without running models
  Requires GPU/accelerator (not available on this host)
        """)


def main():
    auditor = MetalEvidenceAuditor()

    if not auditor.load_cases():
        return 1

    metal = auditor.audit_metal_abi()
    evidence = auditor.audit_evidence_binding()

    auditor.print_summary(metal, evidence)

    # Export
    output = {
        'timestamp': '2026-10-19',
        'a08_metal_abi': metal,
        'a09_evidence_binding': evidence,
    }

    with open(project_root / "A08_A09_metal_evidence.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Exported to A08_A09_metal_evidence.json\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
