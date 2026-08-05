"""Document and editor-draft lifecycle service."""

from __future__ import annotations

from typing import Any

from ...documents import DocumentManager


class DocumentLifecycleService:
    """Route-facing service for documents, versions, restores, and drafts."""

    def __init__(self, documents: DocumentManager) -> None:
        self.documents = documents

    def create_document(
        self,
        *,
        title: str,
        content: str = "",
        project: str = "default",
        owner: str = "local",
        doc_format: str = "markdown",
        tags: list[str] | None = None,
        folder: str = "",
    ) -> dict[str, Any]:
        return self.documents.create_document(
            title,
            content=content,
            project=project,
            owner=owner,
            doc_format=doc_format,
            tags=tags,
            folder=folder,
        )

    def get_document(self, doc_id: str, *, owner: str = "") -> dict[str, Any]:
        return self.documents.get_document(doc_id, owner=owner)

    def library(
        self,
        *,
        project: str = "default",
        query: str = "",
        tag: str = "",
        status: str = "active",
        owner: str = "",
    ) -> dict[str, Any]:
        return self.documents.list_library(
            project=project,
            query=query,
            tag=tag,
            status=status,
            owner=owner,
        )

    def update_document(
        self,
        doc_id: str,
        *,
        title: str | None = None,
        content: str | None = None,
        if_version: int | None = None,
        owner: str = "",
    ) -> dict[str, Any]:
        return self.documents.update_document(
            doc_id,
            title=title,
            content=content,
            if_version=if_version,
            owner=owner,
        )

    def archive_document(self, doc_id: str, *, archived: bool = True, owner: str = "") -> dict[str, Any]:
        return self.documents.archive_document(doc_id, archived=archived, owner=owner)

    def patch_document(
        self,
        doc_id: str,
        *,
        tags: list[str] | None = None,
        folder: str | None = None,
        title: str | None = None,
        owner: str = "",
    ) -> dict[str, Any]:
        return self.documents.patch_document(
            doc_id,
            tags=tags,
            folder=folder,
            title=title,
            owner=owner,
        )

    def versions(self, doc_id: str, *, owner: str = "") -> list[dict[str, Any]]:
        return self.documents.list_versions(doc_id, owner=owner)

    def version(self, doc_id: str, version: int, *, owner: str = "") -> dict[str, Any]:
        return self.documents.get_version(doc_id, version, owner=owner)

    def restore_version(self, doc_id: str, version: int, *, owner: str = "") -> dict[str, Any]:
        return self.documents.restore_version(doc_id, version, owner=owner)

    def delete_document(self, doc_id: str, *, owner: str = "") -> dict[str, Any]:
        return self.documents.delete_document(doc_id, owner=owner)

    def create_draft(
        self,
        *,
        doc_id: str,
        content: str = "",
        project: str = "default",
        owner: str = "local",
        base_version: int = 0,
    ) -> dict[str, Any]:
        return self.documents.create_draft(
            doc_id=doc_id,
            content=content,
            project=project,
            owner=owner,
            base_version=base_version,
        )

    def get_draft(self, draft_id: str, *, owner: str = "") -> dict[str, Any]:
        return self.documents.get_draft(draft_id, owner=owner)

    def update_draft(
        self,
        draft_id: str,
        *,
        content: str,
        base_version: int | None = None,
        owner: str = "",
    ) -> dict[str, Any]:
        return self.documents.update_draft(
            draft_id,
            content=content,
            base_version=base_version,
            owner=owner,
        )
