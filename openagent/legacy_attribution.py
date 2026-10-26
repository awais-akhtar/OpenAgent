"""Legacy attribution policy enforcement for ownership-bound records."""

from __future__ import annotations

from typing import Any

from .models import AuditEvent


class LegacyAttributionError(RuntimeError):
    """Raised when legacy attribution policies are violated."""

    pass


class LegacyAttributionPolicy:
    """Enforces ownership-bound access and immutability for legacy records."""

    def __init__(self, store: Any) -> None:
        self.store = store

    def validate_ownership(
        self,
        record_id: str,
        resource_type: str,
        current_owner: str,
        requested_owner: str | None = None,
    ) -> dict[str, Any]:
        """Validate that access/modification respects ownership boundaries.

        Args:
            record_id: ID of the record being accessed
            resource_type: Type of resource (notes, drafts, etc)
            current_owner: Owner from authentication context
            requested_owner: Owner specified in request (for listing)

        Returns:
            Validation result {allowed, reason, owner}

        Raises:
            LegacyAttributionError: If access violates ownership policy
        """
        # Rule 1: Self-only access
        # Users can only access their own records
        if requested_owner and requested_owner != current_owner:
            raise LegacyAttributionError(
                f"Cannot access {resource_type} owned by {requested_owner}. "
                f"Only owner {requested_owner} can access their own records."
            )

        return {
            "allowed": True,
            "reason": "ownership validated",
            "owner": current_owner,
            "resource_type": resource_type,
        }

    def validate_creation(
        self,
        resource_type: str,
        owner: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate that record creation respects ownership binding.

        Args:
            resource_type: Type of resource being created
            owner: Owner to bind record to
            metadata: Optional metadata (created_at, source, etc)

        Returns:
            Creation validation result {allowed, owner}
        """
        if not owner or not owner.strip():
            raise LegacyAttributionError("Owner is required for legacy attribution binding.")

        metadata = metadata or {}
        return {
            "allowed": True,
            "owner": owner.strip(),
            "resource_type": resource_type,
            "immutable_after_creation": True,
            "metadata": metadata,
        }

    def validate_modification(
        self,
        record_id: str,
        resource_type: str,
        record_owner: str,
        current_owner: str,
        modification_type: str = "update",
    ) -> dict[str, Any]:
        """Validate modifications respect ownership and immutability rules.

        Args:
            record_id: ID of record being modified
            resource_type: Type of resource
            record_owner: Current owner of the record
            current_owner: Owner from authentication
            modification_type: Type of modification (update, delete, etc)

        Returns:
            Modification validation result {allowed, reason}

        Raises:
            LegacyAttributionError: If modification violates policy
        """
        # Rule 2: Owner-only modifications
        if record_owner != current_owner:
            raise LegacyAttributionError(
                f"Cannot {modification_type} {resource_type} owned by {record_owner}. "
                f"Only the owner can modify their own records."
            )

        # Rule 3: Deletions create audit trail
        if modification_type == "delete":
            return {
                "allowed": True,
                "reason": "owner authorized",
                "requires_audit_trail": True,
                "preservation_mode": "soft_delete_with_audit",
            }

        return {
            "allowed": True,
            "reason": "owner authorized",
            "modification_type": modification_type,
        }

    def enforce_access_boundary(
        self,
        resource_ids: list[str],
        resource_type: str,
        current_owner: str,
        fetch_owner_fn: Any,
    ) -> dict[str, Any]:
        """Enforce that resources are owned by current user before operations.

        Args:
            resource_ids: IDs to verify ownership of
            resource_type: Type of resources
            current_owner: User requesting access
            fetch_owner_fn: Callable(resource_id) -> owner_string

        Returns:
            {valid_ids, denied_ids, accessible_count}

        Raises:
            LegacyAttributionError: If ANY resource is not owned by user
        """
        valid_ids = []
        denied_ids = []

        for resource_id in resource_ids:
            try:
                record_owner = fetch_owner_fn(resource_id)
                if record_owner == current_owner:
                    valid_ids.append(resource_id)
                else:
                    denied_ids.append(resource_id)
            except (KeyError, ValueError):
                denied_ids.append(resource_id)

        if denied_ids:
            raise LegacyAttributionError(
                f"Access denied for {len(denied_ids)} {resource_type} "
                f"not owned by {current_owner}: {denied_ids[:3]}"
            )

        return {
            "valid_ids": valid_ids,
            "denied_ids": denied_ids,
            "accessible_count": len(valid_ids),
            "resource_type": resource_type,
            "owner": current_owner,
        }

    def audit_cross_owner_denial(
        self,
        action: str,
        resource_type: str,
        attempted_owner: str,
        requested_owner: str,
        context: str = "",
    ) -> dict[str, Any]:
        """Log a cross-owner access denial for audit trail.

        Args:
            action: What was attempted (read, update, delete)
            resource_type: Type of resource
            attempted_owner: User who attempted access
            requested_owner: Owner of the record
            context: Additional context

        Returns:
            Audit event record
        """
        event = AuditEvent(
            actor=attempted_owner,
            action=f"legacy_attribution.{action}_denied",
            resource=f"{resource_type}:cross_owner",
            project="system",
            status="denied",
            metadata={
                "reason": "cross_owner_access_attempt",
                "attempted_owner": attempted_owner,
                "record_owner": requested_owner,
                "resource_type": resource_type,
                "context": context,
            },
        )

        self.store.add_audit_event(event)

        return {
            "action": f"legacy_attribution.{action}_denied",
            "actor": attempted_owner,
            "denied_reason": "cross_owner_access",
        }

    def get_policy_summary(self) -> dict[str, Any]:
        """Return summary of legacy attribution enforcement policies."""
        return {
            "name": "Legacy Attribution Policy",
            "version": "1.0",
            "rules": [
                {
                    "rule": "Self-only access",
                    "description": "Users can only access records they own",
                    "enforcement": "error on cross-owner read attempt",
                },
                {
                    "rule": "Owner-only modifications",
                    "description": "Only record owner can update/delete",
                    "enforcement": "error on cross-owner modification",
                },
                {
                    "rule": "Audit trail on deletion",
                    "description": "Deletions create permanent audit records",
                    "enforcement": "soft delete with audit logging",
                },
                {
                    "rule": "Immutable ownership",
                    "description": "Record ownership cannot be changed after creation",
                    "enforcement": "owner field is set once at creation",
                },
            ],
            "scope": ["notes", "drafts", "signatures", "personal_records"],
            "boundary_type": "ownership_based",
        }
