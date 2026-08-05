"""Communication provider onboarding and sync service."""

from __future__ import annotations

from typing import Any

from ...communication import CommunicationHub


class CommunicationProviderService:
    """Service facade for OAuth accounts, mailbox policy, and sync workers."""

    def __init__(self, communication: CommunicationHub) -> None:
        self.communication = communication

    def oauth_providers(self) -> list[dict[str, Any]]:
        return self.communication.oauth_providers()

    def configure_oauth_account(
        self,
        *,
        provider: str,
        platform: str = "email",
        project: str = "default",
        owner: str = "local",
        label: str = "",
        client_id: str = "",
        client_secret: str = "",
        auth_endpoint: str = "",
        token_endpoint: str = "",
        redirect_uri: str = "",
        scopes: list[str] | None = None,
        mailbox_policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        account = self.communication.configure_oauth_account(
            provider=provider,
            platform=platform,
            project=project,
            owner=owner,
            label=label,
            client_id=client_id,
            client_secret=client_secret,
            auth_endpoint=auth_endpoint,
            token_endpoint=token_endpoint,
            redirect_uri=redirect_uri,
            scopes=scopes,
            mailbox_policy=mailbox_policy,
        )
        account.setdefault("account_id", account.get("id", ""))
        return account

    def oauth_accounts(
        self,
        *,
        project: str = "default",
        owner: str = "",
        platform: str = "",
    ) -> list[dict[str, Any]]:
        return self.communication.oauth_accounts(project=project, owner=owner, platform=platform)

    def authorization_url(
        self,
        account_id: str,
        *,
        owner: str = "",
        state: str = "",
        code_challenge: str = "",
    ) -> dict[str, Any]:
        return self.communication.oauth_authorization_url(
            account_id,
            owner=owner,
            state=state,
            code_challenge=code_challenge,
        )

    def refresh_oauth_token(
        self,
        account_id: str,
        *,
        owner: str = "",
        allow_external: bool = False,
        authorization_code: str = "",
        code_verifier: str = "",
    ) -> dict[str, Any]:
        return self.communication.refresh_oauth_token(
            account_id,
            owner=owner,
            allow_external=allow_external,
            authorization_code=authorization_code,
            code_verifier=code_verifier,
        )

    def mailbox_policy(self, *, provider: str = "generic", account_id: str = "", owner: str = "") -> dict[str, Any]:
        return self.communication.mailbox_policy(provider=provider, account_id=account_id, owner=owner)

    def sync_resilient(
        self,
        *,
        platform: str,
        project: str = "default",
        owner: str = "local",
        payload: dict[str, Any] | None = None,
        allow_external: bool = False,
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        return self.communication.sync_connector_resilient(
            platform,
            project=project,
            owner=owner,
            payload=payload or {},
            allow_external=allow_external,
            max_attempts=max_attempts,
        )

    def dead_letters(self, *, project: str = "default", owner: str = "") -> list[dict[str, Any]]:
        return self.communication.list_dead_letters(project=project, owner=owner)
