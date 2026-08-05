"""Contacts and personal CRM service."""

from __future__ import annotations

import json
from typing import Any

from .communication import CommunicationHub
from .models import AuditEvent, utc_now
from .storage import OpenAgentStore


class ContactsManager:
    """Standalone contacts/CRM facade over local communication contacts."""

    def __init__(self, store: OpenAgentStore) -> None:
        self.store = store
        self.communication = CommunicationHub(store)

    def list_contacts(self, *, project: str = "default", query: str = "", owner: str = "") -> list[dict[str, Any]]:
        contacts = self.communication.contacts(project=project, query=query, owner=owner)
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for contact in contacts:
            identities = self._identity_values(contact)
            if identities and seen.intersection(identities):
                continue
            seen.update(identities)
            deduped.append(contact)
        return [self._enrich(contact, owner=owner) for contact in deduped]

    def search(self, query: str, *, project: str = "default", owner: str = "") -> list[dict[str, Any]]:
        return self.list_contacts(project=project, query=query, owner=owner)

    def create_contact(
        self,
        *,
        name: str,
        project: str = "default",
        owner: str = "local",
        handles: dict[str, str] | None = None,
        notes: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        contact = self.communication.upsert_contact(
            name,
            project=project,
            owner=owner,
            handles=handles or {},
            notes=notes,
            metadata=metadata or {},
        )
        self.store.add_audit_event(
            AuditEvent(
                actor="contacts",
                action="contact.upsert",
                resource=f"contact:{contact['id']}",
                project=project,
                metadata={"owner": owner, "handles": sorted((handles or {}).keys())},
            )
        )
        return self._enrich(contact, owner=owner)

    def update_contact(
        self,
        contact_id: str,
        *,
        owner: str = "",
        name: str | None = None,
        handles: dict[str, str] | None = None,
        notes: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        existing = self.get_contact(contact_id, owner=owner)
        now = utc_now()
        merged_metadata = dict(existing.get("metadata", {}))
        if metadata is not None:
            merged_metadata.update(metadata)
        with self.store.session() as connection:
            connection.execute(
                """
                UPDATE contacts
                SET name = ?,
                    handles_json = ?,
                    notes = ?,
                    metadata_json = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    name.strip() if isinstance(name, str) and name.strip() else existing["name"],
                    json.dumps(handles if handles is not None else existing.get("handles", {}), sort_keys=True),
                    notes if notes is not None else existing.get("notes", ""),
                    json.dumps(merged_metadata, sort_keys=True),
                    now,
                    contact_id,
                ),
            )
        updated = self.get_contact(contact_id, owner=owner)
        self.store.add_audit_event(
            AuditEvent(
                actor="contacts",
                action="contact.update",
                resource=f"contact:{contact_id}",
                project=str(existing.get("project", "default")),
                metadata={"owner": owner or existing.get("owner", "")},
            )
        )
        return updated

    def delete_contact(self, contact_id: str, *, owner: str = "") -> dict[str, Any]:
        existing = self.get_contact(contact_id, owner=owner)
        identities = self._identity_values(existing)
        with self.store.session() as connection:
            if identities:
                clauses = ["id = ?"]
                params: list[Any] = [contact_id]
                for identity in identities:
                    clauses.append("lower(name) = ?")
                    params.append(identity)
                    clauses.append("lower(handles_json) LIKE ?")
                    params.append(f"%{identity}%")
                connection.execute(
                    f"""
                    DELETE FROM contacts
                    WHERE project = ?
                      AND (? = '' OR owner = ?)
                      AND ({' OR '.join(clauses)})
                    """,
                    (existing["project"], owner, owner, *params),
                )
            else:
                connection.execute("DELETE FROM contacts WHERE id = ?", (contact_id,))
        self.store.add_audit_event(
            AuditEvent(
                actor="contacts",
                action="contact.delete",
                resource=f"contact:{contact_id}",
                project=str(existing.get("project", "default")),
                metadata={"owner": owner or existing.get("owner", "")},
            )
        )
        return {"deleted": True, "contact_id": contact_id}

    def get_contact(self, contact_id: str, *, owner: str = "") -> dict[str, Any]:
        contact = self.communication.get_contact(contact_id)
        if contact is None or (owner and contact.get("owner") != owner):
            raise KeyError(f"Unknown contact '{contact_id}'.")
        return self._enrich(contact, owner=owner)

    def import_contacts(
        self,
        records: list[dict[str, Any]],
        *,
        project: str = "default",
        owner: str = "local",
    ) -> dict[str, Any]:
        imported = []
        for record in records:
            if not isinstance(record, dict):
                continue
            name = str(record.get("name", "")).strip()
            if not name:
                continue
            imported.append(
                self.create_contact(
                    name=name,
                    project=str(record.get("project", project)).strip() or project,
                    owner=str(record.get("owner", owner)).strip() or owner,
                    handles=record.get("handles") if isinstance(record.get("handles"), dict) else {},
                    notes=str(record.get("notes", "")),
                    metadata=record.get("metadata") if isinstance(record.get("metadata"), dict) else {},
                )
            )
        self.store.add_audit_event(
            AuditEvent(
                actor="contacts",
                action="contact.import",
                resource=f"contacts:{project}",
                project=project,
                metadata={"count": len(imported), "owner": owner},
            )
        )
        return {"imported": imported, "count": len(imported)}

    def export_contacts(self, *, project: str = "default", owner: str = "") -> dict[str, Any]:
        contacts = self.list_contacts(project=project, owner=owner)
        return {
            "project": project,
            "owner": owner or "all",
            "count": len(contacts),
            "contacts": contacts,
            "privacy_boundary": "Contacts export is generated from the local OpenAgent SQLite database.",
        }

    def _enrich(self, contact: dict[str, Any], *, owner: str = "") -> dict[str, Any]:
        identities = self._identity_values(contact)
        clauses = ["project = ?"]
        params: list[Any] = [contact.get("project", "default")]
        if owner:
            clauses.append("owner = ?")
            params.append(owner)
        if identities:
            identity_clauses = []
            for identity in identities:
                identity_clauses.append("(lower(sender) = ? OR lower(recipients_json) LIKE ?)")
                params.extend([identity, f"%{identity}%"])
            clauses.append(f"({' OR '.join(identity_clauses)})")
        with self.store.session() as connection:
            count = connection.execute(
                f"SELECT count(*) FROM inbox_messages WHERE {' AND '.join(clauses)}",
                tuple(params),
            ).fetchone()[0]
        return {
            **contact,
            "message_count": count,
            "privacy_boundary": "Contact data stays in local SQLite unless an explicit connector sync is approved.",
        }

    @staticmethod
    def _identity_values(contact: dict[str, Any]) -> set[str]:
        handles = contact.get("handles") if isinstance(contact.get("handles"), dict) else {}
        identities = {str(contact.get("name", "")).strip().lower()}
        identities.update(str(value).strip().lower() for value in handles.values() if str(value).strip())
        return {identity for identity in identities if identity}
