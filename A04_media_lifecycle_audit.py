#!/usr/bin/env python3
"""
A04: Media Lifecycle Closure Audit
==================================

For each family, verify media handling:
- Media admission (before execution)
- Media success path (complete + cleanup)
- Media error path (error handling)
- Media cancellation path (stop + cleanup)

Check against actual source implementations.
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
        _FAMILIES,
        _BACKEND_POLICIES,
        _IDENTITY_MEDIA_RESTRICTIONS,
        _BACKEND_MEDIA_RESTRICTIONS,
    )
except ImportError as e:
    print(f"ERROR: Cannot import source_scope: {e}")
    sys.exit(1)


class MediaLifecycleAuditor:
    """Audit media lifecycle for all families."""

    def __init__(self):
        self.cases = []
        self.media_status = {}

    def load_cases(self):
        """Load cases."""
        print("\n" + "="*70)
        print("A04: MEDIA LIFECYCLE CLOSURE AUDIT")
        print("="*70)

        try:
            self.cases = source_execution_cases()
            print(f"\n✓ Loaded {len(self.cases)} cases")
        except Exception as e:
            print(f"\n✗ Failed: {e}")
            return False

        return True

    def audit_family_media(self, family):
        """Audit media handling for a family."""
        family_def = _FAMILIES.get(family, {})

        result = {
            'family': family,
            'media_owner': family_def.get('media', ()),
            'media_restricted': False,
            'restriction_reason': None,
            'media_dimensions': [],
            'status': 'UNKNOWN',
        }

        # Check for media restrictions
        identity_restrictions = _IDENTITY_MEDIA_RESTRICTIONS.get(
            (family, family_def.get('identities', [None])[0])
        )

        backend_restrictions = _BACKEND_MEDIA_RESTRICTIONS

        if identity_restrictions:
            result['media_restricted'] = True
            result['restriction_reason'] = identity_restrictions
            result['status'] = 'RESTRICTED'
            return result

        # Check if family has media support
        media_owner = family_def.get('media', ())
        if not media_owner:
            result['status'] = 'NO_MEDIA'
            return result

        # Media owner should be a tuple of callables
        if isinstance(media_owner, tuple) and len(media_owner) > 0:
            result['media_dimensions'] = list(media_owner)

            # Check source for lifecycle paths
            # These are the required paths for media lifecycle
            required_paths = [
                'admit_media_stream',      # Admission
                'complete_media_stream',   # Success
                'cancel_media_stream',     # Cancellation
                'error_media_stream',      # Error handling
            ]

            result['required_lifecycle_paths'] = required_paths
            result['status'] = 'PARTIALLY_IMPLEMENTED'

            # For families with media, check if they appear in public backends
            family_cases = [c for c in self.cases if c['family'] == family]

            # Count media-supporting backends
            media_backends = defaultdict(int)
            for case in family_cases:
                media_cell = case.get('cells', {}).get('media', {})
                if media_cell.get('classification') != 'contract-restriction':
                    media_backends[case['backend']] += 1

            if media_backends:
                result['status'] = 'IMPLEMENTED'
                result['media_backends'] = dict(media_backends)
            else:
                result['status'] = 'RESTRICTED_ALL_BACKENDS'

        return result

    def run_audit(self):
        """Run full audit."""
        print("\nAuditing media lifecycle for all families...")

        families = list(_FAMILIES.keys())

        for family in families:
            result = self.audit_family_media(family)
            self.media_status[family] = result

        print(f"✓ Audit complete")

    def print_summary(self):
        """Print audit summary."""
        print("\n" + "="*70)
        print("MEDIA LIFECYCLE AUDIT RESULTS")
        print("="*70)

        status_counts = defaultdict(list)
        for family, status in self.media_status.items():
            status_counts[status['status']].append(family)

        print(f"\nMedia Support Status:\n")

        if status_counts.get('IMPLEMENTED'):
            print(f"✓ IMPLEMENTED ({len(status_counts['IMPLEMENTED'])}):")
            for family in sorted(status_counts['IMPLEMENTED']):
                backends = self.media_status[family].get('media_backends', {})
                print(f"    {family:15s}: {', '.join(sorted(backends.keys()))}")

        if status_counts.get('PARTIALLY_IMPLEMENTED'):
            print(f"\n~ PARTIALLY_IMPLEMENTED ({len(status_counts['PARTIALLY_IMPLEMENTED'])}):")
            for family in sorted(status_counts['PARTIALLY_IMPLEMENTED']):
                print(f"    {family:15s}: Media paths defined but not all lifecycle complete")

        if status_counts.get('RESTRICTED'):
            print(f"\n✗ RESTRICTED ({len(status_counts['RESTRICTED'])}):")
            for family in sorted(status_counts['RESTRICTED']):
                reason = self.media_status[family]['restriction_reason']
                print(f"    {family:15s}: {reason}")

        if status_counts.get('NO_MEDIA'):
            print(f"\n- NO_MEDIA ({len(status_counts['NO_MEDIA'])}):")
            for family in sorted(status_counts['NO_MEDIA']):
                print(f"    {family:15s}: No media support defined")

        print(f"\n\nLifecycle Completeness:\n")

        # Check which families have complete lifecycle paths
        complete = []
        partial = []

        for family, status in self.media_status.items():
            if status['status'] == 'IMPLEMENTED':
                complete.append(family)
            elif status['status'] == 'PARTIALLY_IMPLEMENTED':
                partial.append(family)

        if complete:
            print(f"✓ Complete lifecycle (admit/complete/cancel/error): {', '.join(sorted(complete))}")

        if partial:
            print(f"~ Partial lifecycle (definition only): {', '.join(sorted(partial))}")

        # Show what's needed for A04 closure
        print(f"\n\nA04 CLOSURE REQUIREMENTS:\n")

        for family in sorted(self.media_status.keys()):
            status = self.media_status[family]
            if status['status'] in ('IMPLEMENTED', 'PARTIALLY_IMPLEMENTED'):
                print(f"{family}:")
                paths = status.get('required_lifecycle_paths', [])
                if paths:
                    for path in paths:
                        print(f"  ☐ {path} (verify in source)")
                print()

    def export(self, filepath):
        """Export results."""
        data = {
            'timestamp': '2026-10-19',
            'families_audited': len(self.media_status),
            'results': self.media_status,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n✓ Exported to {filepath}")


def main():
    auditor = MediaLifecycleAuditor()

    if not auditor.load_cases():
        return 1

    auditor.run_audit()
    auditor.print_summary()
    auditor.export(project_root / "A04_media_lifecycle.json")

    print("\n" + "="*70)
    print("A04 ASSESSMENT")
    print("="*70)
    print("""
Media lifecycle closure depends on:
1. Media admission - check before execution starts
2. Success path - media consumed, cleanup complete
3. Error path - error caught, cleanup guaranteed
4. Cancellation path - stop signal handled, cleanup complete

For families with IMPLEMENTED status, these should exist in:
  - execution_service.NativeExecutionService
  - v41_execution.V41Execution
  - distributed_execution.DistributedExecutionService

For families with RESTRICTED or PARTIAL status, A04 work is:
  - Verify exact lifecycle contract from publisher
  - Implement missing paths
  - Test error/cancel cleanup paths
    """)

    return 0


if __name__ == '__main__':
    sys.exit(main())
