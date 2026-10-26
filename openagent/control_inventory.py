"""Control inventory and discovery API."""

from __future__ import annotations

from typing import Any


class ControlInventory:
    """Discovers and catalogs all available UI controls."""

    CONTROL_GROUPS = {
        "voice": {
            "controls": ["record", "transcribe", "play", "pause", "stop"],
            "enabled": True,
        },
        "agents": {
            "controls": ["run", "pause", "resume", "cancel", "retry"],
            "enabled": True,
        },
        "workflows": {
            "controls": ["create", "edit", "execute", "schedule", "delete"],
            "enabled": True,
        },
        "vault": {
            "controls": ["request_access", "approve", "deny", "view_entries"],
            "enabled": True,
        },
        "knowledge": {
            "controls": ["search", "add", "edit", "delete", "export"],
            "enabled": True,
        },
        "personal_os": {
            "controls": ["create_note", "edit_draft", "sign_document", "version_history"],
            "enabled": True,
        },
        "calendar": {
            "controls": ["create_event", "reschedule", "set_reminder", "share"],
            "enabled": True,
        },
        "image_studio": {
            "controls": ["upload", "rotate", "crop", "undo", "redo"],
            "enabled": True,
        },
    }

    def get_all_controls(self) -> dict[str, Any]:
        """Get inventory of all available controls."""
        total = sum(len(group["controls"]) for group in self.CONTROL_GROUPS.values())
        enabled = sum(
            len(group["controls"]) for group in self.CONTROL_GROUPS.values() if group["enabled"]
        )

        return {
            "total_controls": total,
            "enabled_controls": enabled,
            "groups": self.CONTROL_GROUPS,
        }

    def get_controls_by_group(self, group: str) -> dict[str, Any]:
        """Get controls for a specific group."""
        if group not in self.CONTROL_GROUPS:
            raise ValueError(f"Unknown control group: {group}")

        group_data = self.CONTROL_GROUPS[group]
        return {
            "group": group,
            "controls": group_data["controls"],
            "enabled": group_data["enabled"],
            "count": len(group_data["controls"]),
        }

    def search_controls(self, query: str) -> list[dict[str, Any]]:
        """Search for controls by name/keyword."""
        query = query.lower().strip()
        if not query:
            return []

        results = []
        for group_name, group_data in self.CONTROL_GROUPS.items():
            for control in group_data["controls"]:
                if query in control.lower() or query in group_name.lower():
                    results.append({
                        "group": group_name,
                        "control": control,
                        "enabled": group_data["enabled"],
                    })
        return results

    def get_control_stats(self) -> dict[str, Any]:
        """Get statistics about available controls."""
        return {
            "total_groups": len(self.CONTROL_GROUPS),
            "total_controls": sum(len(g["controls"]) for g in self.CONTROL_GROUPS.values()),
            "enabled_groups": sum(1 for g in self.CONTROL_GROUPS.values() if g["enabled"]),
            "coverage": {
                group: {
                    "total": len(data["controls"]),
                    "enabled": len(data["controls"]) if data["enabled"] else 0,
                }
                for group, data in self.CONTROL_GROUPS.items()
            },
        }
