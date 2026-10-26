"""UI placement decision framework for privileged operations."""

from __future__ import annotations

from typing import Any


class UIPlacementDecisions:
    """Framework for deciding where UI controls should be placed."""

    PLACEMENT_CRITERIA = {
        "frequency": "How often the operation is used (daily, weekly, rare)",
        "privilege": "Whether operation requires special permissions",
        "visibility": "Whether operation should be discoverable vs hidden",
        "context": "In which views/screens operation is relevant",
    }

    OPERATION_PLACEMENTS = {
        "vault_request_access": {
            "primary": "Advanced workspace - Vault panel",
            "secondary": "Security settings",
            "privilege": "requires: vault:read",
            "visibility": "prominent",
        },
        "approve_request": {
            "primary": "Advanced workspace - Vault panel",
            "secondary": "Admin dashboard",
            "privilege": "requires: vault:admin",
            "visibility": "prominent for admins",
        },
        "create_workflow": {
            "primary": "Workflows screen",
            "secondary": "Quick actions menu",
            "privilege": "none",
            "visibility": "discoverable",
        },
        "schedule_task": {
            "primary": "Tasks screen - Advanced options",
            "secondary": "Calendar integration",
            "privilege": "requires: tasks:write",
            "visibility": "discoverable",
        },
        "archive_document": {
            "primary": "Document menu - Archive option",
            "secondary": "Hidden (admin only)",
            "privilege": "requires: documents:admin",
            "visibility": "hidden",
        },
        "view_audit_log": {
            "primary": "Admin panel - Audit section",
            "secondary": "Security dashboard",
            "privilege": "requires: audit:read",
            "visibility": "hidden",
        },
    }

    def get_operation_placement(self, operation: str) -> dict[str, Any]:
        """Get placement decision for an operation."""
        if operation not in self.OPERATION_PLACEMENTS:
            return {
                "operation": operation,
                "placement": "not_configured",
                "message": "No placement decision defined for this operation",
            }

        placement = self.OPERATION_PLACEMENTS[operation]
        return {
            "operation": operation,
            "placement": placement,
            "decision_criteria": self.PLACEMENT_CRITERIA,
        }

    def list_placements(self) -> dict[str, Any]:
        """List all operation placements."""
        return {
            "total_operations": len(self.OPERATION_PLACEMENTS),
            "operations": self.OPERATION_PLACEMENTS,
            "criteria": self.PLACEMENT_CRITERIA,
        }

    def get_privileged_operations(self) -> list[dict[str, Any]]:
        """List all operations requiring privileges."""
        privileged = []
        for op_name, placement in self.OPERATION_PLACEMENTS.items():
            if "requires:" in placement.get("privilege", ""):
                privileged.append({
                    "operation": op_name,
                    "privilege": placement["privilege"],
                    "primary": placement["primary"],
                    "visibility": placement.get("visibility", "hidden"),
                })
        return privileged

    def get_placement_policy(self) -> dict[str, Any]:
        """Get the UI placement policy framework."""
        return {
            "name": "UI Placement Decision Framework",
            "description": "Determines where operations appear in the UI",
            "decision_model": "4-criteria framework",
            "criteria": self.PLACEMENT_CRITERIA,
            "status": "Defined - awaiting product decisions",
            "scope": "Operations with security/visibility implications",
            "examples_configured": len(self.OPERATION_PLACEMENTS),
        }
