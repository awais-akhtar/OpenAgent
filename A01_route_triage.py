#!/usr/bin/env python3
"""
A01: Phase 1 Tuple Triage
=========================

Triage all 1,334+ execution candidates into required routes vs unsupported combinations.
Record the decision (admitted/restricted/rejected) with reason and source references for each.

**TRUTH ONLY**: No fake claims. Every decision is backed by actual source code references.
Every candidate reviewed, every reason documented.
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
        _BACKEND_RESTRICTIONS,
        _BACKEND_MEDIA_RESTRICTIONS,
        _BACKEND_POLICIES,
        _FAMILIES,
        _PERSISTED_FORMAT_POLICIES,
        _V41_BANK_FORMATS,
    )
except ImportError as e:
    print(f"ERROR: Cannot import source_scope: {e}")
    sys.exit(1)


class RouteTriageDecision:
    """Record a single route decision with full reasoning."""

    def __init__(self, route_id: str, family: str, backend: str,
                 source_format: str, persisted_format: str):
        self.route_id = route_id
        self.family = family
        self.backend = backend
        self.source_format = source_format
        self.persisted_format = persisted_format

        self.decision = None  # 'admitted', 'restricted', 'rejected'
        self.reason = None
        self.source_refs = []
        self.blocker = None

    def admit(self, reason="Source-verified callable chain present"):
        """Mark as admitted with reason."""
        self.decision = 'admitted'
        self.reason = reason

    def restrict(self, reason, source_ref=None):
        """Mark as restricted with specific reason."""
        self.decision = 'restricted'
        self.reason = reason
        if source_ref:
            self.source_refs.append(source_ref)

    def reject(self, reason, source_ref=None, blocker=None):
        """Mark as rejected with specific reason."""
        self.decision = 'rejected'
        self.reason = reason
        self.blocker = blocker
        if source_ref:
            self.source_refs.append(source_ref)

    def to_dict(self):
        return {
            'route_id': self.route_id,
            'family': self.family,
            'backend': self.backend,
            'source_format': self.source_format,
            'persisted_format': self.persisted_format,
            'decision': self.decision,
            'reason': self.reason,
            'source_refs': self.source_refs,
            'blocker': self.blocker,
        }


class RouteTriage:
    """Triage all execution cases."""

    def __init__(self):
        self.cases = []
        self.decisions = {}
        self.summary = {
            'total': 0,
            'admitted': 0,
            'restricted': 0,
            'rejected': 0,
            'by_family': defaultdict(lambda: {'admitted': 0, 'restricted': 0, 'rejected': 0}),
            'blockers': defaultdict(int),
        }

    def load_cases(self):
        """Load all execution cases from source_scope."""
        print("\n" + "="*70)
        print("A01: PHASE 1 ROUTE TRIAGE")
        print("="*70)

        try:
            self.cases = source_execution_cases()
            print(f"\n✓ Loaded {len(self.cases)} execution cases from source_scope")
        except Exception as e:
            print(f"\n✗ Failed to load cases: {e}")
            return False

        return True

    def triage_case(self, case):
        """Apply triage logic to a single case."""
        family = case['family']
        backend = case['backend']
        source_fmt = case['source_format']
        persisted_fmt = case['persisted_format']
        route_id = case['case_id']

        decision = RouteTriageDecision(
            route_id, family, backend, source_fmt, persisted_fmt
        )

        # Check 1: Backend restrictions
        backend_restriction = _BACKEND_RESTRICTIONS.get((family, backend))
        if backend_restriction is not None:
            decision.restrict(
                f"Backend restricted: {backend_restriction}",
                f"_BACKEND_RESTRICTIONS[({family}, {backend})]"
            )
            return decision

        # Check 2: Media restrictions
        media_restriction = _BACKEND_MEDIA_RESTRICTIONS.get(backend)
        if media_restriction is not None:
            decision.restrict(
                f"Media restricted: {media_restriction}",
                f"_BACKEND_MEDIA_RESTRICTIONS[{backend}]"
            )
            return decision

        # Check 3: V4.1-specific format restrictions
        if family == 'deepseek-v41':
            if (source_fmt, persisted_fmt) not in _V41_BANK_FORMATS:
                decision.reject(
                    "DeepSeek V4.1 admits only BF16/bf16, FP8/fp8, FP4/fp4 routes",
                    "_V41_BANK_FORMATS",
                    blocker='V41_FORMAT_RESTRICTION'
                )
                return decision

        # Check 4: K3 format restrictions
        if family == 'kimi-k3':
            if persisted_fmt not in ('float32', 'mxfp4-e2m1-g32-ue8m0'):
                decision.reject(
                    "K3 checkpoint formats require exact publisher projection route",
                    "source_scope.py:1138",
                    blocker='K3_FORMAT_RESTRICTION'
                )
                return decision

        # Check 5: Source-verified routes
        source_review = _SOURCE_VERIFIED_ROUTE_BINDINGS.get(
            (family, case['model_type'], backend, case['arithmetic'], source_fmt, persisted_fmt)
        )
        if source_review is not None:
            decision.admit(
                f"Source-verified: {source_review}"
            )
            decision.source_refs.append("_SOURCE_VERIFIED_ROUTE_BINDINGS")
            return decision

        # Check 6: Format admitted but not source-verified
        combination_status = case.get('combination_status')
        if combination_status == 'candidate-unverified':
            # Further analysis needed
            callable_chain = case.get('callable_chain', {})

            # Check if chain is complete
            chain_complete = all(
                callable_chain.get(dim) for dim in case.get('chain_order', [])
                if dim not in ('media',)  # media can be optional
            )

            if chain_complete and case.get('cells', {}).get('format_route', {}).get('classification') == 'candidate-unverified':
                # Has complete chain, format is admitted but not verified
                decision.restrict(
                    "Candidate route with complete chain structure; awaiting P13 evidence capture (numerical parity, placement data, runtime execution)",
                    "source_scope.py:801-804"
                )
                return decision
            else:
                # Incomplete chain
                missing = [
                    dim for dim in case.get('chain_order', [])
                    if not callable_chain.get(dim) and dim != 'media'
                ]
                decision.reject(
                    f"Incomplete callable chain; missing dimensions: {', '.join(missing)}",
                    "source_scope.py:915-918",
                    blocker='INCOMPLETE_CHAIN'
                )
                return decision

        if combination_status == 'restricted':
            decision.restrict(
                "Format or backend marked as restricted in source_scope",
                "source_scope.py:970-974"
            )
            return decision

        if combination_status == 'rejected-unadmitted':
            decision.reject(
                "Format or backend explicitly rejected in source",
                "source_scope.py:970-974",
                blocker='SOURCE_REJECTED'
            )
            return decision

        # Fallback
        decision.restrict(
            f"Unknown status: {combination_status}",
            "A01_route_triage.py:triage_case"
        )
        return decision

    def run_triage(self):
        """Triage all cases."""
        if not self.load_cases():
            return False

        print(f"\nTriaging {len(self.cases)} routes...")

        for i, case in enumerate(self.cases):
            if (i + 1) % 100 == 0:
                print(f"  ... {i + 1}/{len(self.cases)}")

            decision = self.triage_case(case)
            self.decisions[decision.route_id] = decision

            # Update summary
            self.summary['total'] += 1
            self.summary[decision.decision] += 1
            self.summary['by_family'][decision.family][decision.decision] += 1
            if decision.blocker:
                self.summary['blockers'][decision.blocker] += 1

        print(f"✓ Triage complete")
        return True

    def print_summary(self):
        """Print triage summary."""
        print("\n" + "="*70)
        print("TRIAGE RESULTS")
        print("="*70)

        print(f"\nTotal routes: {self.summary['total']}")
        print(f"  ✓ Admitted:   {self.summary['admitted']:4d}")
        print(f"  ~ Restricted: {self.summary['restricted']:4d}")
        print(f"  ✗ Rejected:   {self.summary['rejected']:4d}")

        print(f"\nBy Family:")
        for family in sorted(self.summary['by_family'].keys()):
            counts = self.summary['by_family'][family]
            total = sum(counts.values())
            print(f"  {family:15s}: {counts['admitted']:3d} admitted, {counts['restricted']:3d} restricted, {counts['rejected']:3d} rejected (total {total})")

        if self.summary['blockers']:
            print(f"\nTop Blockers:")
            for blocker, count in sorted(self.summary['blockers'].items(), key=lambda x: -x[1])[:10]:
                print(f"  {blocker:30s}: {count:3d} routes")

    def export_decisions(self, filepath):
        """Export all decisions to JSON."""
        data = {
            'timestamp': str(Path(__file__).stat().st_mtime),
            'summary': dict(self.summary),
            'decisions': {
                rid: dec.to_dict() for rid, dec in self.decisions.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n✓ Exported {len(self.decisions)} decisions to {filepath}")


def main():
    triage = RouteTriage()

    if not triage.run_triage():
        return 1

    triage.print_summary()

    # Export decisions
    output_path = project_root / "A01_triage_decisions.json"
    triage.export_decisions(str(output_path))

    print(f"\n{'='*70}")
    print("NEXT: Verify decisions against actual source code.")
    print("Then: Implement missing callable chains for restricted routes.")
    print("="*70 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
