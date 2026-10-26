"""Native desktop entry point for OpenAgent.

The desktop shell uses the local Python runtime and SQLite store directly. It
does not require a browser, webview, cloud service, telemetry service, or
external desktop framework.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .agents import AgentRegistry
from .assistant import AssistantManager
from .capabilities import vision_coverage
from .chat import ChatService
from .communication import CommunicationHub
from .config import desktop_home
from .context import ContextEngine
from .developer import DeveloperPlatform
from .knowledge import KnowledgeEngine
from .local_ai import LocalAIInfrastructure
from .media import MediaManager
from .model_gateway import ModelGateway
from .models import AuditEvent
from .production_runtime import ProductionRuntime
from .security_privacy import SecurityPrivacyManager
from .semantic_memory import SemanticMemoryManager
from .storage import OpenAgentStore
from .voice import VoiceRuntime, synthesize_tone_wav
from .workbench import (
    configured_workbench_port,
    launch_workbench,
    maybe_reexec_project_venv,
    show_desktop_launch_error,
    workbench_status,
)
from .workflows import WorkflowEngine
from .workspace import WorkspaceManager


DESKTOP_PRIVACY_BOUNDARY = (
    "OpenAgent Desktop reads and writes only the configured local data directory. "
    "It serves the modern workbench from localhost, does not start telemetry, and does not call external providers unless "
    "a configured local action explicitly allows that boundary crossing."
)

WINDOWS_INSTALLER_FILES = (
    "installers/windows/install.ps1",
    "installers/windows/uninstall.ps1",
    "installers/windows/OpenAgent.cmd",
    "installers/windows/README.md",
)

DESKTOP_THEME = {
    "mode": "dark",
    "background": "#0d100f",
    "surface": "#151a18",
    "surface_raised": "#1b211f",
    "surface_soft": "#202824",
    "border": "#2e3833",
    "text": "#f7faf7",
    "muted": "#a8b3ad",
    "accent": "#19c37d",
    "accent_soft": "#103b2a",
    "info": "#67e8f9",
    "warning": "#f4b942",
    "danger": "#ff5c5c",
}

DESKTOP_NAVIGATION = (
    "Home",
    "Chat",
    "Agents",
    "Memory",
    "Workspace",
    "Knowledge",
    "Assistant",
    "Models",
    "Voice + Media",
    "Automation",
    "Security + Dev",
    "Audit",
)

DESKTOP_MODULES = (
    {
        "id": "workspace",
        "name": "Workspace",
        "features": ("folders", "documents", "notes", "wiki", "dashboards", "canvas", "whiteboards", "mind maps", "templates"),
    },
    {
        "id": "knowledge",
        "name": "Knowledge and Research",
        "features": ("document ingestion", "hybrid search", "semantic memory", "RAG synthesis", "graph", "timeline"),
    },
    {
        "id": "assistant",
        "name": "Assistant and Communications",
        "features": ("tasks", "goals", "daily brief", "contacts", "inbox", "calendar", "CRM"),
    },
    {
        "id": "models",
        "name": "Models and Local AI",
        "features": ("model registry", "hardware detection", "provider readiness", "Colab detection", "benchmarks", "GPU planning", "fine-tune records"),
    },
    {
        "id": "voice_media",
        "name": "Voice and Media",
        "features": ("voice sessions", "STT", "TTS WAV", "voiceprint clone", "media analysis"),
    },
    {
        "id": "automation",
        "name": "Automation and Production",
        "features": ("workflows", "HTML form simulation", "optional browser tests", "desktop file actions", "queues", "packages", "hard audit"),
    },
    {
        "id": "security_developer",
        "name": "Security and Developer",
        "features": (
            "privacy report",
            "secrets",
            "sandbox",
            "API keys",
            "SDK",
            "plugins",
            "MCP",
            "unified tool runtime",
            "compute capability planning",
            "live MCP invocation",
        ),
    },
)


def desktop_status(
    data_dir: Path | str | None = None,
    *,
    project: str = "default",
    ui_root: Path | str | None = None,
) -> dict[str, Any]:
    """Return a non-GUI status object for tests, installers, and audits."""

    data_path = Path(data_dir).expanduser() if data_dir else desktop_home()
    store = OpenAgentStore(data_path)
    health = store.health()
    coverage = vision_coverage()
    return {
        "status": "ready",
        "mode": "native-desktop",
        "default_surface": "modern-web-workbench",
        "project": project,
        "data_dir": str(data_path),
        "browser_required": False,
        "browser_tab_required": False,
        "windows_installable": True,
        "entrypoints": {
            "package_script": "openagent-desktop",
            "modern_workbench_script": "openagent-workbench",
            "python_module": "python -m openagent.desktop",
            "windows_launcher": "installers/windows/OpenAgent.cmd",
        },
        "health": health,
        "capability_verdict": coverage["summary"]["verdict"],
        "current_release": coverage["summary"]["current_release"],
        "implemented_items": coverage["summary"]["counts"].get("implemented", 0),
        "desktop_modules": desktop_feature_manifest(),
        "runtime_tools": desktop_runtime_status(),
        "desktop_layout": desktop_layout_manifest(),
        "modern_workbench": workbench_status(data_path, project=project, ui_root=ui_root),
        "privacy_boundary": DESKTOP_PRIVACY_BOUNDARY,
    }


def desktop_layout_manifest() -> dict[str, Any]:
    """Describe the native desktop shell without opening a GUI."""

    return {
        "shell": "native-desktop-workbench",
        "default_surface": "modern-web-workbench",
        "browser_required": False,
        "browser_tab_required": False,
        "theme": {
            "mode": DESKTOP_THEME["mode"],
            "background": DESKTOP_THEME["background"],
            "surface": DESKTOP_THEME["surface"],
            "accent": DESKTOP_THEME["accent"],
        },
        "first_screen": {
            "navigation": "left-sidebar",
            "workspace": "runtime-dashboard",
            "panels": ["status-strip", "module-grid", "quick-actions", "command-palette", "audit-preview"],
        },
        "navigation": list(DESKTOP_NAVIGATION),
        "keyboard_shortcuts": {
            "command_palette": "Ctrl+K",
            "refresh_status": "Ctrl+R",
            "refresh_audit": "Ctrl+L",
            "navigation": "Ctrl+1 through Ctrl+0",
        },
        "modern_workbench": {
            "script": "openagent-workbench",
            "surface": "same UI as local web workbench",
            "legacy_tk_flag": "--legacy-tk",
        },
        "module_cards": desktop_feature_manifest(),
        "privacy_boundary": DESKTOP_PRIVACY_BOUNDARY,
    }


def desktop_feature_manifest() -> list[dict[str, Any]]:
    return [
        {"id": module["id"], "name": module["name"], "features": list(module["features"])}
        for module in DESKTOP_MODULES
    ]


def desktop_runtime_status() -> dict[str, Any]:
    """Report optional runtime readiness without starting a browser or model server."""

    from importlib.util import find_spec

    from .runtimes.colab import detect_colab, recommend_model

    gateway = ModelGateway()
    office_modules = {name: find_spec(name) is not None for name in ("docx", "pptx", "openpyxl")}
    browser_ready = find_spec("playwright") is not None
    return {
        "browser": {
            "installed": browser_ready,
            "install_hint": "pip install openagent[browser] && playwright install chromium" if not browser_ready else "ready for user-triggered local or approved external browser tests",
        },
        "office_exports": {
            "modules": office_modules,
            "ready": all(office_modules.values()),
            "install_hint": "pip install openagent[office]" if not all(office_modules.values()) else "DOCX, PPTX, and XLSX export dependencies are installed",
        },
        "colab": {
            "detected": detect_colab(),
            "recommended_model_without_probe": recommend_model({"gpu": False}),
            "launch_import": "from openagent.colab import launch",
        },
        "providers": {
            "registered": sorted(gateway.providers),
            "external_requires_explicit_approval": sorted(provider_id for provider_id, provider in gateway.providers.items() if not provider.local),
        },
    }


def run_desktop_feature_smoke(data_dir: Path | str | None = None, *, project: str = "desktop-smoke") -> dict[str, Any]:
    """Exercise desktop-visible features without opening a GUI."""

    data_path = Path(data_dir).expanduser() if data_dir else desktop_home()
    store = OpenAgentStore(data_path)
    workspace = WorkspaceManager(store)
    knowledge = KnowledgeEngine(store)
    semantic = SemanticMemoryManager(store)
    assistant = AssistantManager(store)
    communication = CommunicationHub(store)
    gateway = ModelGateway()
    local_ai = LocalAIInfrastructure(store, gateway)
    voice = VoiceRuntime(store)
    media = MediaManager(store)
    workflows = WorkflowEngine(store, gateway)
    production = ProductionRuntime(store, gateway)
    security = SecurityPrivacyManager(store)
    developer = DeveloperPlatform(store)

    checks = {
        "workspace": workspace.create_item(kind="document", title="Desktop smoke document", content="Native desktop workspace.", project=project),
        "knowledge": knowledge.ingest_text(title="Desktop smoke knowledge", content="OpenAgent desktop exposes knowledge search and synthesis.", project=project),
        "semantic": semantic.add_memory(title="Desktop smoke memory", content="Expose complete desktop modules.", project=project, memory_type="procedural"),
        "assistant": assistant.create_task(title="Review desktop workbench", project=project),
        "communication": communication.import_message(platform="email", sender="owner@example.com", subject="Desktop smoke", body="Check desktop modules.", project=project),
        "calendar": communication.add_calendar_event(title="Desktop smoke review", starts_at="2026-06-26T09:00:00", ends_at="2026-06-26T09:30:00", project=project),
        "local_ai": local_ai.detect_hardware(),
        "voice": voice.synthesize_text(session_id=voice.create_session(project=project)["id"], text="Desktop smoke audio."),
        "media": media.ingest_bytes(b"%PDF-1.4\nBT (Desktop smoke PDF) Tj ET\n%%EOF", filename="desktop-smoke.pdf", project=project),
        "workflow": workflows.run(
            workflows.create(
                name="Desktop smoke workflow",
                project=project,
                steps=[{"action": "add_event", "title": "Desktop smoke event", "content": "workflow"}],
            )["id"]
        ),
        "production": production.package_app(app_type="desktop", platform="windows", project=project),
        "security": security.create_privacy_report(),
        "developer": developer.generate_sdk(language="python"),
    }
    store.add_audit_event(
        AuditEvent(
            actor="desktop",
            action="desktop.feature_smoke",
            resource="desktop:runtime-dashboard",
            project=project,
            metadata={"modules": [module["id"] for module in DESKTOP_MODULES]},
        )
    )
    return {
        "ok": True,
        "project": project,
        "modules": desktop_feature_manifest(),
        "checks": checks,
        "privacy_boundary": "Desktop feature smoke writes only local records in the configured OpenAgent data directory.",
    }


def windows_installation_manifest(repo_root: Path | str | None = None) -> dict[str, Any]:
    """Describe the Windows install assets shipped with the source tree."""

    root = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parent.parent
    files = {}
    for relative in WINDOWS_INSTALLER_FILES:
        path = root / relative
        files[relative] = {
            "exists": path.exists(),
            "path": str(path),
        }
    return {
        "platform": "windows",
        "installer_root": str(root / "installers" / "windows"),
        "files": files,
        "install_command": r".\installers\windows\install.ps1",
        "launch_command": "openagent-desktop",
        "browser_required": False,
        "privacy_boundary": DESKTOP_PRIVACY_BOUNDARY,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OpenAgent as a native local desktop app.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory for local OpenAgent data. Defaults to OPENAGENT_HOME or ~/.openagent.",
    )
    parser.add_argument("--project", default="default", help="Project scope for desktop actions.")
    parser.add_argument("--host", default="127.0.0.1", help="Host for the modern workbench surface.")
    parser.add_argument(
        "--port",
        default=configured_workbench_port(),
        type=int,
        help="Stable localhost port. Installed desktop launches always default to 8765.",
    )
    parser.add_argument("--no-open", action="store_true", help="Start the modern workbench server without opening an app window.")
    parser.add_argument("--ui-root", default=None, help="Override the built static UI folder for local acceptance testing.")
    parser.add_argument("--legacy-tk", action="store_true", help="Open the legacy Tk desktop shell instead of the modern workbench.")
    parser.add_argument(
        "--status-json",
        action="store_true",
        help="Print desktop install/runtime status as JSON without opening the GUI.",
    )
    parser.add_argument(
        "--hard-audit-json",
        action="store_true",
        help="Run the executable hard audit as JSON without opening the GUI.",
    )
    parser.add_argument(
        "--feature-manifest-json",
        action="store_true",
        help="Print the native desktop module manifest as JSON without opening the GUI.",
    )
    parser.add_argument(
        "--layout-json",
        action="store_true",
        help="Print the native desktop layout manifest as JSON without opening the GUI.",
    )
    parser.add_argument(
        "--feature-smoke-json",
        action="store_true",
        help="Run desktop-visible feature smoke checks as JSON without opening the GUI.",
    )
    return parser


def launch_desktop(data_dir: Path | str | None = None, *, project: str = "default") -> int:
    data_path = Path(data_dir).expanduser() if data_dir else desktop_home()
    store = OpenAgentStore(data_path)
    store.add_audit_event(
        AuditEvent(
            actor="desktop",
            action="desktop.launch",
            resource="openagent-desktop",
            project=project,
            metadata={"data_dir": str(data_path), "browser_required": False},
        )
    )

    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except Exception as exc:  # pragma: no cover - platform dependent
        raise RuntimeError("OpenAgent Desktop requires Python tkinter support on this system.") from exc

    root = tk.Tk()
    root.title("OpenAgent Desktop")
    root.geometry("1320x840")
    root.minsize(1080, 700)
    app = DesktopShell(root=root, ttk=ttk, messagebox=messagebox, store=store, project=project, data_dir=data_path)
    app.render()
    root.mainloop()
    return 0


class DesktopShell:
    def __init__(self, *, root: Any, ttk: Any, messagebox: Any, store: OpenAgentStore, project: str, data_dir: Path):
        self.root = root
        self.ttk = ttk
        self.messagebox = messagebox
        self.store = store
        self.project = project
        self.data_dir = data_dir
        self.context = ContextEngine(store)
        self.gateway = ModelGateway()
        self.chat = ChatService(store, self.gateway)
        self.agents = AgentRegistry(store, self.gateway)
        self.workspace = WorkspaceManager(store)
        self.knowledge = KnowledgeEngine(store)
        self.semantic = SemanticMemoryManager(store)
        self.assistant = AssistantManager(store)
        self.communication = CommunicationHub(store)
        self.local_ai = LocalAIInfrastructure(store, self.gateway)
        self.voice = VoiceRuntime(store)
        self.media = MediaManager(store)
        self.workflows = WorkflowEngine(store, self.gateway)
        self.production = ProductionRuntime(store, self.gateway)
        self.security = SecurityPrivacyManager(store)
        self.developer = DeveloperPlatform(store)
        self.theme = DESKTOP_THEME
        self.notebook: Any = None
        self.nav_buttons: list[tuple[Any, Any]] = []
        self.status_value_widgets: dict[str, Any] = {}
        self.command_palette_window: Any = None
        self.home_text: Any = None
        self.chat_log: Any = None
        self.chat_input: Any = None
        self.agent_goal: Any = None
        self.agent_picker: Any = None
        self.agent_output: Any = None
        self.note_title: Any = None
        self.note_content: Any = None
        self.search_input: Any = None
        self.memory_output: Any = None
        self.workspace_kind: Any = None
        self.workspace_title: Any = None
        self.workspace_content: Any = None
        self.workspace_output: Any = None
        self.knowledge_title: Any = None
        self.knowledge_content: Any = None
        self.knowledge_query: Any = None
        self.knowledge_output: Any = None
        self.assistant_input: Any = None
        self.assistant_output: Any = None
        self.models_output: Any = None
        self.voice_input: Any = None
        self.voice_output: Any = None
        self.automation_output: Any = None
        self.browser_url: Any = None
        self.browser_external_approval: Any = None
        self.security_output: Any = None
        self.audit_output: Any = None

    def render(self) -> None:
        self._configure_style()
        self.root.configure(bg=self.theme["background"])

        shell = self.ttk.Frame(self.root, padding=0, style="App.TFrame")
        shell.pack(fill="both", expand=True)

        header = self.ttk.Frame(shell, padding=(18, 14), style="Header.TFrame")
        header.pack(fill="x")
        title_area = self.ttk.Frame(header, style="Header.TFrame")
        title_area.pack(side="left", fill="x", expand=True)
        self.ttk.Label(title_area, text="OpenAgent", style="Brand.TLabel").pack(anchor="w")
        self.ttk.Label(
            title_area,
            text=f"Native local AI OS workbench | Project: {self.project} | Data: {self.data_dir}",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(2, 0))
        header_actions = self.ttk.Frame(header, style="Header.TFrame")
        header_actions.pack(side="right")
        self.ttk.Button(header_actions, text="Command Palette", command=self.open_command_palette).pack(side="left", padx=(0, 8))
        self.ttk.Button(header_actions, text="Refresh", command=self.refresh_home, style="Accent.TButton").pack(side="left", padx=(0, 8))
        self.ttk.Button(header_actions, text="Hard Audit", command=self.run_hard_audit).pack(side="left")

        body = self.ttk.Frame(shell, padding=(12, 12, 14, 14), style="App.TFrame")
        body.pack(fill="both", expand=True)

        sidebar = self.ttk.Frame(body, width=230, padding=(12, 12), style="Sidebar.TFrame")
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        main = self.ttk.Frame(body, padding=(12, 0, 0, 0), style="App.TFrame")
        main.pack(side="left", fill="both", expand=True)

        status_strip = self.ttk.Frame(main, style="App.TFrame")
        status_strip.pack(fill="x", pady=(0, 10))
        self._build_status_strip(status_strip)

        notebook = self.ttk.Notebook(main, style="Workbench.TNotebook")
        notebook.pack(fill="both", expand=True)
        self.notebook = notebook

        self._build_home_tab(notebook)
        self._build_chat_tab(notebook)
        self._build_agents_tab(notebook)
        self._build_memory_tab(notebook)
        self._build_workspace_tab(notebook)
        self._build_knowledge_tab(notebook)
        self._build_assistant_tab(notebook)
        self._build_models_tab(notebook)
        self._build_voice_media_tab(notebook)
        self._build_automation_tab(notebook)
        self._build_security_developer_tab(notebook)
        self._build_audit_tab(notebook)
        self._build_sidebar(sidebar, notebook)
        self._bind_shortcuts(notebook)
        self.refresh_home()
        self.refresh_audit()

    def _configure_style(self) -> None:
        style = self.ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("App.TFrame", background=self.theme["background"])
        style.configure("Header.TFrame", background=self.theme["surface"])
        style.configure("Sidebar.TFrame", background=self.theme["surface"])
        style.configure("Panel.TFrame", background=self.theme["surface_raised"])
        style.configure("TFrame", background=self.theme["background"])
        style.configure("TLabel", background=self.theme["background"], foreground=self.theme["text"], font=("Segoe UI", 10))
        style.configure("Brand.TLabel", background=self.theme["surface"], foreground=self.theme["text"], font=("Segoe UI", 22, "bold"))
        style.configure("Title.TLabel", background=self.theme["background"], foreground=self.theme["text"], font=("Segoe UI", 15, "bold"))
        style.configure("Muted.TLabel", background=self.theme["surface"], foreground=self.theme["muted"], font=("Segoe UI", 9))
        style.configure("BodyMuted.TLabel", background=self.theme["background"], foreground=self.theme["muted"], font=("Segoe UI", 9))
        style.configure("PanelTitle.TLabel", background=self.theme["surface_raised"], foreground=self.theme["text"], font=("Segoe UI", 13, "bold"))
        style.configure("PanelMuted.TLabel", background=self.theme["surface_raised"], foreground=self.theme["muted"], font=("Segoe UI", 9))
        style.configure("TButton", padding=(10, 6), font=("Segoe UI", 9))
        style.configure("Accent.TButton", padding=(12, 7), font=("Segoe UI", 9, "bold"))
        style.map(
            "Accent.TButton",
            background=[("active", self.theme["accent"]), ("!disabled", self.theme["accent_soft"])],
            foreground=[("!disabled", self.theme["text"])],
        )
        style.configure(
            "TEntry",
            fieldbackground=self.theme["surface_raised"],
            foreground=self.theme["text"],
            insertcolor=self.theme["text"],
            bordercolor=self.theme["border"],
            lightcolor=self.theme["border"],
            darkcolor=self.theme["border"],
            padding=7,
        )
        style.configure(
            "TCombobox",
            fieldbackground=self.theme["surface_raised"],
            background=self.theme["surface_raised"],
            foreground=self.theme["text"],
            arrowcolor=self.theme["text"],
            padding=7,
        )
        style.configure(
            "Workbench.TNotebook",
            background=self.theme["background"],
            borderwidth=0,
            tabmargins=(0, 0, 0, 0),
        )
        style.configure(
            "Workbench.TNotebook.Tab",
            background=self.theme["surface"],
            foreground=self.theme["muted"],
            padding=(12, 7),
            font=("Segoe UI", 9),
        )
        style.map(
            "Workbench.TNotebook.Tab",
            background=[("selected", self.theme["surface_raised"])],
            foreground=[("selected", self.theme["text"])],
        )

    def _build_status_strip(self, parent: Any) -> None:
        cards = (
            ("health", "Health", "starting", self.theme["accent"]),
            ("records", "Local Records", "-", self.theme["info"]),
            ("agents", "Agent Runs", "-", self.theme["warning"]),
            ("audit", "Audit Trail", "-", self.theme["accent"]),
        )
        for key, title, value, color in cards:
            self._status_card(parent, key=key, title=title, value=value, color=color)

    def _status_card(self, parent: Any, *, key: str, title: str, value: str, color: str) -> None:
        import tkinter as tk

        card = tk.Frame(
            parent,
            bg=self.theme["surface_raised"],
            highlightbackground=self.theme["border"],
            highlightthickness=1,
            bd=0,
        )
        card.pack(side="left", fill="x", expand=True, padx=(0, 10))
        tk.Label(card, text=title, bg=self.theme["surface_raised"], fg=self.theme["muted"], font=("Segoe UI", 9)).pack(
            anchor="w", padx=12, pady=(10, 0)
        )
        value_label = tk.Label(card, text=value, bg=self.theme["surface_raised"], fg=color, font=("Segoe UI", 18, "bold"))
        value_label.pack(anchor="w", padx=12, pady=(1, 10))
        self.status_value_widgets[key] = value_label

    def _build_sidebar(self, parent: Any, notebook: Any) -> None:
        import tkinter as tk

        tk.Label(parent, text="WORKBENCH", bg=self.theme["surface"], fg=self.theme["muted"], font=("Segoe UI", 8, "bold")).pack(
            anchor="w", pady=(0, 8)
        )
        nav = tk.Frame(parent, bg=self.theme["surface"])
        nav.pack(fill="both", expand=True)
        for tab_id in notebook.tabs():
            text = notebook.tab(tab_id, "text")
            button = tk.Button(
                nav,
                text=text,
                anchor="w",
                relief="flat",
                bd=0,
                padx=12,
                pady=8,
                bg=self.theme["surface"],
                activebackground=self.theme["surface_soft"],
                activeforeground=self.theme["text"],
                fg=self.theme["muted"],
                font=("Segoe UI", 10),
                command=lambda item=tab_id: self._select_nav(notebook, item),
            )
            button.pack(fill="x", pady=1)
            self.nav_buttons.append((tab_id, button))
        footer = tk.Frame(parent, bg=self.theme["surface"])
        footer.pack(fill="x", pady=(10, 0))
        tk.Label(footer, text="Privacy boundary", bg=self.theme["surface"], fg=self.theme["muted"], font=("Segoe UI", 8, "bold")).pack(anchor="w")
        tk.Label(
            footer,
            text="Local files, local database, no browser required.",
            bg=self.theme["surface"],
            fg=self.theme["text"],
            justify="left",
            wraplength=190,
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(4, 0))
        notebook.bind("<<NotebookTabChanged>>", lambda _event: self._sync_nav_buttons(notebook))
        self._sync_nav_buttons(notebook)

    def _select_nav(self, notebook: Any, tab_id: Any) -> None:
        notebook.select(tab_id)
        self._sync_nav_buttons(notebook)

    def _sync_nav_buttons(self, notebook: Any) -> None:
        selected = notebook.select()
        for tab_id, button in self.nav_buttons:
            active = str(tab_id) == str(selected)
            button.configure(
                bg=self.theme["accent_soft"] if active else self.theme["surface"],
                fg=self.theme["text"] if active else self.theme["muted"],
                font=("Segoe UI", 10, "bold" if active else "normal"),
            )

    def _bind_shortcuts(self, notebook: Any) -> None:
        self.root.bind_all("<Control-k>", lambda _event: self._shortcut(self.open_command_palette))
        self.root.bind_all("<Control-K>", lambda _event: self._shortcut(self.open_command_palette))
        self.root.bind_all("<Control-r>", lambda _event: self._shortcut(self.refresh_home))
        self.root.bind_all("<Control-R>", lambda _event: self._shortcut(self.refresh_home))
        self.root.bind_all("<Control-l>", lambda _event: self._shortcut(self.refresh_audit))
        self.root.bind_all("<Control-L>", lambda _event: self._shortcut(self.refresh_audit))
        for index in range(10):
            key = "0" if index == 9 else str(index + 1)
            self.root.bind_all(
                f"<Control-Key-{key}>",
                lambda _event, tab_index=index: self._shortcut(lambda: self._select_tab_index(notebook, tab_index)),
            )

    def _shortcut(self, action: Any) -> str:
        action()
        return "break"

    def _select_tab_index(self, notebook: Any, index: int) -> None:
        tabs = notebook.tabs()
        if 0 <= index < len(tabs):
            self._select_nav(notebook, tabs[index])

    def _go_to_tab(self, title: str) -> None:
        if self.notebook is None:
            return
        for tab_id in self.notebook.tabs():
            if self.notebook.tab(tab_id, "text") == title:
                self._select_nav(self.notebook, tab_id)
                return

    def _command_actions(self) -> list[tuple[str, Any]]:
        actions: list[tuple[str, Any]] = [
            ("Refresh status", self.refresh_home),
            ("Run hard audit", self.run_hard_audit),
            ("Run feature smoke", self.run_feature_smoke),
            ("Refresh audit trail", self.refresh_audit),
        ]
        actions.extend((f"Go to {title}", lambda item=title: self._go_to_tab(item)) for title in DESKTOP_NAVIGATION)
        return actions

    def open_command_palette(self) -> None:
        import tkinter as tk

        if self.command_palette_window is not None and self.command_palette_window.winfo_exists():
            self.command_palette_window.lift()
            self.command_palette_window.focus_force()
            return

        palette = tk.Toplevel(self.root)
        self.command_palette_window = palette
        palette.title("OpenAgent Command Palette")
        palette.configure(bg=self.theme["surface"])
        palette.geometry("560x430")
        palette.transient(self.root)
        palette.resizable(False, False)

        tk.Label(palette, text="Command Palette", bg=self.theme["surface"], fg=self.theme["text"], font=("Segoe UI", 16, "bold")).pack(
            anchor="w", padx=16, pady=(16, 4)
        )
        tk.Label(
            palette,
            text="Type to find a screen or action.",
            bg=self.theme["surface"],
            fg=self.theme["muted"],
            font=("Segoe UI", 9),
        ).pack(anchor="w", padx=16, pady=(0, 12))
        query = tk.Entry(
            palette,
            bg=self.theme["surface_raised"],
            fg=self.theme["text"],
            insertbackground=self.theme["text"],
            relief="flat",
            font=("Segoe UI", 11),
        )
        query.pack(fill="x", padx=16, ipady=8)
        listbox = tk.Listbox(
            palette,
            bg="#0f1311",
            fg=self.theme["text"],
            selectbackground=self.theme["accent_soft"],
            selectforeground=self.theme["text"],
            activestyle="none",
            highlightbackground=self.theme["border"],
            highlightcolor=self.theme["accent"],
            relief="flat",
            font=("Segoe UI", 10),
        )
        listbox.pack(fill="both", expand=True, padx=16, pady=16)
        actions = self._command_actions()
        visible: list[tuple[str, Any]] = []

        def refresh_list(_event: Any | None = None) -> None:
            search = query.get().strip().lower()
            listbox.delete(0, "end")
            visible.clear()
            for label, action in actions:
                if not search or search in label.lower():
                    visible.append((label, action))
                    listbox.insert("end", label)
            if visible:
                listbox.selection_set(0)

        def invoke_selected(_event: Any | None = None) -> str:
            selection = listbox.curselection()
            if selection:
                _label, action = visible[selection[0]]
                palette.destroy()
                action()
            return "break"

        query.bind("<KeyRelease>", refresh_list)
        query.bind("<Return>", invoke_selected)
        query.bind("<Escape>", lambda _event: palette.destroy())
        listbox.bind("<Return>", invoke_selected)
        listbox.bind("<Double-Button-1>", invoke_selected)
        listbox.bind("<Escape>", lambda _event: palette.destroy())
        refresh_list()
        query.focus_set()

    def _tab_header(self, parent: Any, title: str, subtitle: str) -> None:
        self.ttk.Label(parent, text=title, style="Title.TLabel").pack(anchor="w")
        self.ttk.Label(parent, text=subtitle, style="BodyMuted.TLabel").pack(anchor="w", pady=(2, 10))

    def _panel(self, parent: Any) -> Any:
        frame = self.ttk.Frame(parent, padding=14, style="Panel.TFrame")
        return frame

    def _module_grid(self, parent: Any) -> Any:
        import tkinter as tk

        grid = tk.Frame(parent, bg=self.theme["background"])
        for index, module in enumerate(DESKTOP_MODULES):
            card = tk.Frame(
                grid,
                bg=self.theme["surface_raised"],
                highlightbackground=self.theme["border"],
                highlightthickness=1,
                bd=0,
            )
            row = index // 2
            column = index % 2
            card.grid(row=row, column=column, sticky="ew", padx=(0 if column == 0 else 8, 8 if column == 0 else 0), pady=(0, 8))
            tk.Label(card, text=module["name"], bg=self.theme["surface_raised"], fg=self.theme["text"], font=("Segoe UI", 11, "bold")).pack(
                anchor="w", padx=12, pady=(10, 2)
            )
            tk.Label(
                card,
                text=" | ".join(module["features"][:5]),
                bg=self.theme["surface_raised"],
                fg=self.theme["muted"],
                wraplength=430,
                justify="left",
                font=("Segoe UI", 9),
            ).pack(anchor="w", padx=12, pady=(0, 10))
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        return grid

    def _build_home_tab(self, notebook: Any) -> None:
        tab = self.ttk.Frame(notebook, padding=10)
        notebook.add(tab, text="Home")
        self._tab_header(tab, "Runtime Dashboard", "Native workspace, agents, memory, automations, security, and audit in one local app.")
        hero = self._panel(tab)
        hero.pack(fill="x", pady=(0, 10))
        self.ttk.Label(hero, text="OpenAgent is running as a local desktop operating layer.", style="PanelTitle.TLabel").pack(anchor="w")
        self.ttk.Label(
            hero,
            text="Use the sidebar to move between chats, agents, knowledge, workflows, model controls, and privacy tools.",
            style="PanelMuted.TLabel",
        ).pack(anchor="w", pady=(2, 10))
        actions = self.ttk.Frame(hero, style="Panel.TFrame")
        actions.pack(fill="x")
        self.ttk.Button(actions, text="Refresh Status", command=self.refresh_home, style="Accent.TButton").pack(side="left")
        self.ttk.Button(actions, text="Run Hard Audit", command=self.run_hard_audit).pack(side="left", padx=8)
        self.ttk.Button(actions, text="Feature Smoke", command=self.run_feature_smoke).pack(side="left")

        grid = self._module_grid(tab)
        grid.pack(fill="x", pady=(0, 10))
        self.home_text = self._text(tab, height=14)

    def _build_chat_tab(self, notebook: Any) -> None:
        tab = self.ttk.Frame(notebook, padding=10)
        notebook.add(tab, text="Chat")
        self._tab_header(tab, "Conversation", "Local chat with project memory and audit logging.")
        self.chat_log = self._text(tab, height=22)
        row = self.ttk.Frame(tab)
        row.pack(fill="x", pady=(8, 0))
        self.chat_input = self.ttk.Entry(row)
        self.chat_input.pack(side="left", fill="x", expand=True)
        self.chat_input.bind("<Return>", lambda _event: self.send_chat())
        self.ttk.Button(row, text="Send Local Chat", command=self.send_chat).pack(side="left", padx=(8, 0))

    def _build_agents_tab(self, notebook: Any) -> None:
        tab = self.ttk.Frame(notebook, padding=10)
        notebook.add(tab, text="Agents")
        self._tab_header(tab, "Multi-Agent Runs", "Choose a built-in specialist and run local delegated work.")
        ids = [spec["id"] for spec in self.agents.specs()]
        self.agent_picker = self.ttk.Combobox(tab, values=ids, state="readonly")
        self.agent_picker.set(ids[0] if ids else "executive")
        self.agent_picker.pack(fill="x")
        self.agent_goal = self.ttk.Entry(tab)
        self.agent_goal.pack(fill="x", pady=8)
        self.ttk.Button(tab, text="Run Agent", command=self.run_agent).pack(anchor="w")
        self.agent_output = self._text(tab, height=22)

    def _build_memory_tab(self, notebook: Any) -> None:
        tab = self.ttk.Frame(notebook, padding=10)
        notebook.add(tab, text="Memory")
        self._tab_header(tab, "Context and Memory", "Capture notes, search local context, and keep a project trail.")
        self.note_title = self.ttk.Entry(tab)
        self.note_title.insert(0, "Desktop note")
        self.note_title.pack(fill="x")
        self.note_content = self._text(tab, height=6)
        self.ttk.Button(tab, text="Save Local Note", command=self.save_note).pack(anchor="w", pady=(8, 10))
        row = self.ttk.Frame(tab)
        row.pack(fill="x")
        self.search_input = self.ttk.Entry(row)
        self.search_input.pack(side="left", fill="x", expand=True)
        self.search_input.bind("<Return>", lambda _event: self.search_memory())
        self.ttk.Button(row, text="Search Memory", command=self.search_memory).pack(side="left", padx=(8, 0))
        self.memory_output = self._text(tab, height=16)

    def _build_workspace_tab(self, notebook: Any) -> None:
        tab = self.ttk.Frame(notebook, padding=10)
        notebook.add(tab, text="Workspace")
        self._tab_header(tab, "Workspace System", "Create folders, docs, dashboards, canvases, whiteboards, and templates.")
        kinds = ["folder", "document", "note", "wiki", "dashboard", "research_canvas", "whiteboard", "mind_map", "template"]
        self.workspace_kind = self.ttk.Combobox(tab, values=kinds, state="readonly")
        self.workspace_kind.set("document")
        self.workspace_kind.pack(fill="x")
        self.workspace_title = self.ttk.Entry(tab)
        self.workspace_title.insert(0, "OpenAgent workspace item")
        self.workspace_title.pack(fill="x", pady=8)
        self.workspace_content = self._text(tab, height=6)
        row = self.ttk.Frame(tab)
        row.pack(fill="x", pady=(8, 8))
        self.ttk.Button(row, text="Create Item", command=self.create_workspace_item).pack(side="left")
        self.ttk.Button(row, text="Dashboard", command=self.show_workspace_dashboard).pack(side="left", padx=8)
        self.ttk.Button(row, text="Canvas", command=self.show_workspace_canvas).pack(side="left")
        self.ttk.Button(row, text="List Items", command=self.list_workspace_items).pack(side="left", padx=8)
        self.workspace_output = self._text(tab, height=14)

    def _build_knowledge_tab(self, notebook: Any) -> None:
        tab = self.ttk.Frame(notebook, padding=10)
        notebook.add(tab, text="Knowledge")
        self._tab_header(tab, "Knowledge and RAG", "Ingest, search, synthesize, graph, and timeline local knowledge.")
        self.knowledge_title = self.ttk.Entry(tab)
        self.knowledge_title.insert(0, "Research note")
        self.knowledge_title.pack(fill="x")
        self.knowledge_content = self._text(tab, height=6)
        self.knowledge_query = self.ttk.Entry(tab)
        self.knowledge_query.insert(0, "OpenAgent")
        self.knowledge_query.pack(fill="x", pady=8)
        row = self.ttk.Frame(tab)
        row.pack(fill="x", pady=(0, 8))
        self.ttk.Button(row, text="Ingest Text", command=self.ingest_knowledge_text).pack(side="left")
        self.ttk.Button(row, text="Hybrid Search", command=self.search_knowledge).pack(side="left", padx=8)
        self.ttk.Button(row, text="Synthesize", command=self.synthesize_knowledge).pack(side="left")
        self.ttk.Button(row, text="Graph", command=self.show_knowledge_graph).pack(side="left", padx=8)
        self.ttk.Button(row, text="Timeline", command=self.show_knowledge_timeline).pack(side="left")
        self.ttk.Button(row, text="Semantic Memory", command=self.add_semantic_memory).pack(side="left", padx=8)
        self.knowledge_output = self._text(tab, height=14)

    def _build_assistant_tab(self, notebook: Any) -> None:
        tab = self.ttk.Frame(notebook, padding=10)
        notebook.add(tab, text="Assistant")
        self._tab_header(tab, "Assistant Hub", "Tasks, daily brief, meetings, inbox, calendar, and CRM records.")
        self.assistant_input = self.ttk.Entry(tab)
        self.assistant_input.insert(0, "Follow up on OpenAgent project")
        self.assistant_input.pack(fill="x")
        row = self.ttk.Frame(tab)
        row.pack(fill="x", pady=8)
        self.ttk.Button(row, text="Create Task", command=self.create_assistant_task).pack(side="left")
        self.ttk.Button(row, text="Daily Brief", command=self.show_daily_brief).pack(side="left", padx=8)
        self.ttk.Button(row, text="Meeting Summary", command=self.create_meeting_summary).pack(side="left")
        self.ttk.Button(row, text="Import Message", command=self.import_desktop_message).pack(side="left", padx=8)
        self.ttk.Button(row, text="Add Calendar", command=self.add_desktop_calendar_event).pack(side="left")
        self.ttk.Button(row, text="CRM", command=self.create_crm_record).pack(side="left", padx=8)
        self.assistant_output = self._text(tab)

    def _build_models_tab(self, notebook: Any) -> None:
        tab = self.ttk.Frame(notebook, padding=10)
        notebook.add(tab, text="Models")
        self._tab_header(tab, "Models and Local AI", "Hardware detection, local model registry, benchmarks, and GPU planning.")
        row = self.ttk.Frame(tab)
        row.pack(fill="x", pady=(0, 8))
        self.ttk.Button(row, text="Hardware", command=self.show_hardware).pack(side="left")
        self.ttk.Button(row, text="Model Registry", command=self.show_models).pack(side="left", padx=8)
        self.ttk.Button(row, text="Register Local Model", command=self.register_local_model).pack(side="left")
        self.ttk.Button(row, text="Benchmark", command=self.benchmark_local_model).pack(side="left", padx=8)
        self.ttk.Button(row, text="GPU Plan", command=self.plan_gpu).pack(side="left")
        self.ttk.Button(row, text="Runtime Readiness", command=self.show_runtime_readiness).pack(side="left", padx=8)
        self.models_output = self._text(tab)

    def _build_voice_media_tab(self, notebook: Any) -> None:
        tab = self.ttk.Frame(notebook, padding=10)
        notebook.add(tab, text="Voice + Media")
        self._tab_header(tab, "Voice and Media", "Voice sessions, speech samples, local TTS artifacts, and media status.")
        self.voice_input = self.ttk.Entry(tab)
        self.voice_input.insert(0, "OpenAgent local voice is ready")
        self.voice_input.pack(fill="x")
        row = self.ttk.Frame(tab)
        row.pack(fill="x", pady=8)
        self.ttk.Button(row, text="Create Voice Session", command=self.create_voice_session).pack(side="left")
        self.ttk.Button(row, text="TTS WAV", command=self.synthesize_voice).pack(side="left", padx=8)
        self.ttk.Button(row, text="STT Sample", command=self.transcribe_voice_sample).pack(side="left")
        self.ttk.Button(row, text="Clone Sample", command=self.clone_voice_sample).pack(side="left", padx=8)
        self.ttk.Button(row, text="Media Status", command=self.show_media_status).pack(side="left")
        self.voice_output = self._text(tab)

    def _build_automation_tab(self, notebook: Any) -> None:
        import tkinter as tk

        tab = self.ttk.Frame(notebook, padding=10)
        notebook.add(tab, text="Automation")
        self._tab_header(tab, "Automation Runtime", "Run workflows and local actions. The form demo is offline HTML simulation; browser tests use optional Playwright.")
        row = self.ttk.Frame(tab)
        row.pack(fill="x", pady=(0, 8))
        self.ttk.Button(row, text="Run Workflow", command=self.run_desktop_workflow).pack(side="left")
        self.ttk.Button(row, text="HTML Form Simulation", command=self.run_browser_form).pack(side="left", padx=8)
        self.ttk.Button(row, text="Desktop File Action", command=self.run_desktop_file_action).pack(side="left")
        self.ttk.Button(row, text="Queue Job", command=self.create_queue_job).pack(side="left", padx=8)
        self.ttk.Button(row, text="Package App", command=self.package_desktop_app).pack(side="left")
        self.ttk.Button(row, text="Hard Audit", command=self.run_hard_audit).pack(side="left", padx=8)
        browser_row = self.ttk.Frame(tab)
        browser_row.pack(fill="x", pady=(4, 0))
        self.browser_url = self.ttk.Entry(browser_row)
        self.browser_url.insert(0, "http://127.0.0.1:8765/")
        self.browser_url.pack(side="left", fill="x", expand=True)
        self.browser_external_approval = tk.BooleanVar(value=False)
        self.ttk.Checkbutton(browser_row, text="Approve external URL", variable=self.browser_external_approval).pack(side="left", padx=8)
        self.ttk.Button(browser_row, text="Run Browser Test", command=self.run_browser_test).pack(side="left")
        self.automation_output = self._text(tab)

    def _build_security_developer_tab(self, notebook: Any) -> None:
        tab = self.ttk.Frame(notebook, padding=10)
        notebook.add(tab, text="Security + Dev")
        self._tab_header(tab, "Security and Developer", "Privacy reports, sandbox checks, API keys, SDK artifacts, and capability ledgers.")
        row = self.ttk.Frame(tab)
        row.pack(fill="x", pady=(0, 8))
        self.ttk.Button(row, text="Privacy Report", command=self.show_privacy_report).pack(side="left")
        self.ttk.Button(row, text="Sandbox Check", command=self.run_sandbox_check).pack(side="left", padx=8)
        self.ttk.Button(row, text="API Key", command=self.create_api_key).pack(side="left")
        self.ttk.Button(row, text="SDK Artifact", command=self.generate_sdk).pack(side="left", padx=8)
        self.ttk.Button(row, text="Capability Ledger", command=self.show_capability_ledger).pack(side="left")
        self.security_output = self._text(tab)

    def _build_audit_tab(self, notebook: Any) -> None:
        tab = self.ttk.Frame(notebook, padding=10)
        notebook.add(tab, text="Audit")
        self._tab_header(tab, "Audit Trail", "Recent local actions written to the OpenAgent audit log.")
        self.ttk.Button(tab, text="Refresh Audit", command=self.refresh_audit).pack(anchor="w")
        self.audit_output = self._text(tab, height=28)

    def _text(self, parent: Any, *, height: int = 28) -> Any:
        import tkinter as tk

        widget = tk.Text(
            parent,
            height=height,
            wrap="word",
            font=("Consolas", 10),
            bg="#0f1311",
            fg=self.theme["text"],
            insertbackground=self.theme["text"],
            selectbackground=self.theme["accent_soft"],
            selectforeground=self.theme["text"],
            relief="flat",
            bd=0,
            padx=12,
            pady=10,
            highlightbackground=self.theme["border"],
            highlightcolor=self.theme["accent"],
            highlightthickness=1,
        )
        widget.pack(fill="both", expand=True, pady=(8, 0))
        return widget

    def refresh_home(self) -> None:
        self._set_text(self.home_text, json.dumps(desktop_status(self.data_dir, project=self.project), indent=2))
        self._refresh_status_strip()

    def run_feature_smoke(self) -> None:
        self._run_to(self.home_text, lambda: run_desktop_feature_smoke(self.data_dir, project=self.project))

    def run_hard_audit(self) -> None:
        try:
            from .hard_audit import run_hard_audit

            result = run_hard_audit(self.store, project=self.project)
            self._set_text(self.home_text, json.dumps(result, indent=2, sort_keys=True))
            self.refresh_audit()
        except Exception as exc:
            self.messagebox.showerror("Hard Audit Failed", str(exc))

    def create_workspace_item(self) -> None:
        self._run_to(
            self.workspace_output,
            lambda: self.workspace.create_item(
                kind=self.workspace_kind.get().strip() or "document",
                title=self.workspace_title.get().strip() or "OpenAgent workspace item",
                content=self.workspace_content.get("1.0", "end").strip(),
                project=self.project,
                metadata={"client": "openagent-desktop"},
            ),
        )

    def show_workspace_dashboard(self) -> None:
        self._run_to(self.workspace_output, lambda: self.workspace.dashboard(project=self.project))

    def show_workspace_canvas(self) -> None:
        self._run_to(self.workspace_output, lambda: self.workspace.canvas_model(project=self.project))

    def list_workspace_items(self) -> None:
        self._run_to(self.workspace_output, lambda: self.workspace.list_items(project=self.project, limit=50))

    def ingest_knowledge_text(self) -> None:
        self._run_to(
            self.knowledge_output,
            lambda: self.knowledge.ingest_text(
                title=self.knowledge_title.get().strip() or "Research note",
                content=self.knowledge_content.get("1.0", "end").strip() or "OpenAgent desktop knowledge note.",
                project=self.project,
            ),
        )

    def search_knowledge(self) -> None:
        self._run_to(self.knowledge_output, lambda: self.knowledge.hybrid_search(self.knowledge_query.get(), project=self.project))

    def synthesize_knowledge(self) -> None:
        self._run_to(self.knowledge_output, lambda: self.knowledge.synthesize(self.knowledge_query.get(), project=self.project))

    def show_knowledge_graph(self) -> None:
        self._run_to(self.knowledge_output, lambda: self.knowledge.graph(project=self.project))

    def show_knowledge_timeline(self) -> None:
        self._run_to(self.knowledge_output, lambda: self.knowledge.timeline(project=self.project))

    def add_semantic_memory(self) -> None:
        self._run_to(
            self.knowledge_output,
            lambda: self.semantic.add_memory(
                title=self.knowledge_title.get().strip() or "Desktop semantic memory",
                content=self.knowledge_content.get("1.0", "end").strip() or "Remember the full desktop workbench.",
                project=self.project,
                memory_type="semantic",
                source="desktop",
            ),
        )

    def create_assistant_task(self) -> None:
        self._run_to(self.assistant_output, lambda: self.assistant.create_task(title=self.assistant_input.get(), project=self.project))

    def show_daily_brief(self) -> None:
        self._run_to(self.assistant_output, lambda: self.assistant.daily_brief(project=self.project))

    def create_meeting_summary(self) -> None:
        self._run_to(
            self.assistant_output,
            lambda: self.assistant.summarize_meeting(
                transcript=self.assistant_input.get() or "Discuss OpenAgent roadmap and next actions.",
                project=self.project,
                title="Desktop meeting summary",
            ),
        )

    def import_desktop_message(self) -> None:
        self._run_to(
            self.assistant_output,
            lambda: self.communication.import_message(
                platform="email",
                sender="owner@example.com",
                subject="OpenAgent desktop",
                body=self.assistant_input.get() or "OpenAgent desktop message.",
                project=self.project,
            ),
        )

    def add_desktop_calendar_event(self) -> None:
        self._run_to(
            self.assistant_output,
            lambda: self.communication.add_calendar_event(
                title=self.assistant_input.get() or "OpenAgent review",
                starts_at="2026-06-26T10:00:00",
                ends_at="2026-06-26T10:30:00",
                project=self.project,
            ),
        )

    def create_crm_record(self) -> None:
        self._run_to(
            self.assistant_output,
            lambda: self.assistant.upsert_crm_record(record_type="relationship", name=self.assistant_input.get() or "OpenAgent owner", project=self.project),
        )

    def show_hardware(self) -> None:
        self._run_to(self.models_output, self.local_ai.detect_hardware)

    def show_models(self) -> None:
        self._run_to(self.models_output, self.local_ai.models)

    def show_runtime_readiness(self) -> None:
        self._run_to(self.models_output, desktop_runtime_status)

    def register_local_model(self) -> None:
        self._run_to(self.models_output, lambda: self.local_ai.register_model(name="openagent-desktop-local", provider="local", metadata={"client": "desktop"}))

    def benchmark_local_model(self) -> None:
        def action() -> dict[str, Any]:
            models = self.local_ai.models()
            model = models[0] if models else self.local_ai.register_model(name="openagent-desktop-local", provider="local")
            return self.local_ai.benchmark(model["id"], prompt="OpenAgent desktop benchmark")

        self._run_to(self.models_output, action)

    def plan_gpu(self) -> None:
        self._run_to(self.models_output, lambda: self.production.plan_gpu_management(model_id="openagent-desktop-local", project=self.project))

    def create_voice_session(self) -> None:
        self._run_to(self.voice_output, lambda: self.voice.create_session(project=self.project, mode="push_to_talk"))

    def synthesize_voice(self) -> None:
        def action() -> dict[str, Any]:
            session = self.voice.create_session(project=self.project, mode="push_to_talk")
            return self.voice.synthesize_text(session_id=session["id"], text=self.voice_input.get() or "OpenAgent voice ready.")

        self._run_to(self.voice_output, action)

    def transcribe_voice_sample(self) -> None:
        def action() -> dict[str, Any]:
            session = self.voice.create_session(project=self.project, mode="push_to_talk")
            return self.voice.transcribe_audio(
                session["id"],
                audio_data=synthesize_tone_wav("desktop speech sample"),
                filename="desktop-speech.wav",
                transcript_hint=self.voice_input.get() or "Desktop speech sample transcribed.",
            )

        self._run_to(self.voice_output, action)

    def clone_voice_sample(self) -> None:
        self._run_to(
            self.voice_output,
            lambda: self.voice.clone_voice_from_audio(
                name="Desktop Local Voice",
                audio_data=synthesize_tone_wav("desktop voice clone"),
                project=self.project,
                consent="Desktop user submitted this local voice sample.",
            ),
        )

    def show_media_status(self) -> None:
        self._run_to(self.voice_output, lambda: self.media.status(project=self.project))

    def run_desktop_workflow(self) -> None:
        def action() -> dict[str, Any]:
            workflow = self.workflows.create(
                name="Desktop quick workflow",
                project=self.project,
                steps=[{"action": "add_event", "title": "Desktop workflow", "content": "Workflow created from desktop."}],
            )
            return self.workflows.run(workflow["id"])

        self._run_to(self.automation_output, action)

    def run_browser_form(self) -> None:
        self._run_to(
            self.automation_output,
            lambda: self.production.run_browser_automation(
                actions=[
                    {"action": "open", "url": "http://127.0.0.1"},
                    {"action": "fill", "selector": "email", "value": "owner@example.com"},
                    {"action": "extract_title"},
                ],
                project=self.project,
                html='<title>OpenAgent Desktop</title><form><input name="email" value=""></form>',
            ),
        )

    def run_browser_test(self) -> None:
        def action() -> dict[str, Any]:
            from .tools.browser import BrowserRuntime

            url = self.browser_url.get().strip()
            approved = bool(self.browser_external_approval.get())
            result = BrowserRuntime(self.data_dir).run(url, approved=approved)
            self.store.add_audit_event(
                AuditEvent(
                    actor="desktop",
                    action="desktop.browser_test",
                    resource="browser:playwright",
                    project=self.project,
                    metadata={"url": url, "approved_external": approved, "artifact": result.get("artifact", "")},
                )
            )
            return result

        self._run_to(self.automation_output, action)

    def run_desktop_file_action(self) -> None:
        self._run_to(
            self.automation_output,
            lambda: self.production.run_desktop_automation(
                actions=[
                    {"action": "write_file", "path": "desktop/status.txt", "content": "OpenAgent desktop action"},
                    {"action": "read_file", "path": "desktop/status.txt"},
                ],
                project=self.project,
            ),
        )

    def create_queue_job(self) -> None:
        self._run_to(self.automation_output, lambda: self.production.enqueue_job(kind="desktop.check", payload={"source": "desktop"}, project=self.project))

    def package_desktop_app(self) -> None:
        self._run_to(self.automation_output, lambda: self.production.package_app(app_type="desktop", platform="windows", project=self.project))

    def show_privacy_report(self) -> None:
        self._run_to(self.security_output, self.security.create_privacy_report)

    def run_sandbox_check(self) -> None:
        def action() -> dict[str, Any]:
            profile = self.security.create_sandbox_profile(name="Desktop Sandbox", permissions=["context.read"], network_allowed=False)
            return self.security.evaluate_sandbox_request(profile_id=profile["id"], permissions=["context.read"], network=False)

        self._run_to(self.security_output, action)

    def create_api_key(self) -> None:
        self._run_to(self.security_output, lambda: self.developer.create_api_key(label="Desktop API key", scopes=["context.read"]))

    def generate_sdk(self) -> None:
        self._run_to(self.security_output, lambda: self.developer.generate_sdk(language="python"))

    def show_capability_ledger(self) -> None:
        self._run_to(self.security_output, vision_coverage)

    def send_chat(self) -> None:
        message = self.chat_input.get().strip()
        if not message:
            return
        try:
            result = self.chat.send(message, project=self.project, provider="local")
            transcript = "\n\n".join(f"{item['role']}: {item['content']}" for item in result["messages"])
            self._append_text(self.chat_log, transcript + "\n")
            self.chat_input.delete(0, "end")
            self.refresh_audit()
        except Exception as exc:
            self.messagebox.showerror("Chat Failed", str(exc))

    def run_agent(self) -> None:
        agent_id = self.agent_picker.get().strip() or "executive"
        goal = self.agent_goal.get().strip() or "Review OpenAgent desktop readiness"
        try:
            run = self.agents.run(agent_id, goal, project=self.project)
            self._set_text(self.agent_output, json.dumps(run, indent=2))
            self.refresh_audit()
        except Exception as exc:
            self.messagebox.showerror("Agent Failed", str(exc))

    def save_note(self) -> None:
        title = self.note_title.get().strip() or "Desktop note"
        content = self.note_content.get("1.0", "end").strip()
        if not content:
            self.messagebox.showwarning("No Content", "Write a note before saving it.")
            return
        event = self.context.ingest(
            title=title,
            content=content,
            type="note",
            source="desktop",
            project=self.project,
            metadata={"client": "openagent-desktop"},
        )
        self.store.add_audit_event(
            AuditEvent(
                actor="desktop",
                action="desktop.note.create",
                resource=f"event:{event['id']}",
                project=self.project,
                metadata={"title": title},
            )
        )
        self._set_text(self.memory_output, json.dumps(event, indent=2))
        self.note_content.delete("1.0", "end")
        self.refresh_audit()

    def search_memory(self) -> None:
        query = self.search_input.get().strip()
        results = self.context.search(query, project=self.project, limit=20)
        self._set_text(self.memory_output, json.dumps(results, indent=2))

    def refresh_audit(self) -> None:
        events = self.store.list_audit_events(project=self.project, limit=50)
        self._set_text(self.audit_output, json.dumps(events, indent=2))
        self._refresh_status_strip()

    def _run_to(self, widget: Any, action: Any) -> None:
        try:
            result = action()
            self._set_text(widget, json.dumps(result, indent=2, sort_keys=True))
            self.refresh_audit()
        except Exception as exc:
            self.messagebox.showerror("OpenAgent Desktop Action Failed", str(exc))

    def _set_text(self, widget: Any, value: str) -> None:
        if widget is None:
            return
        widget.delete("1.0", "end")
        widget.insert("1.0", value)

    def _append_text(self, widget: Any, value: str) -> None:
        if widget is None:
            return
        widget.insert("end", value)
        widget.see("end")

    def _refresh_status_strip(self) -> None:
        if not self.status_value_widgets:
            return
        health = self.store.health()
        total_records = sum(
            int(health.get(key, 0))
            for key in ("event_count", "conversation_count", "workflow_count", "agent_run_count")
        )
        values = {
            "health": str(health.get("status", "ok")).upper(),
            "records": str(total_records),
            "agents": str(health.get("agent_run_count", 0)),
            "audit": str(health.get("audit_event_count", 0)),
        }
        for key, value in values.items():
            widget = self.status_value_widgets.get(key)
            if widget is not None:
                widget.configure(text=value)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else desktop_home()
    if args.status_json:
        print(json.dumps(desktop_status(data_dir, project=args.project, ui_root=args.ui_root), indent=2, sort_keys=True))
        return 0
    if args.hard_audit_json:
        from .hard_audit import run_hard_audit

        store = OpenAgentStore(data_dir)
        print(json.dumps(run_hard_audit(store, project=args.project), indent=2, sort_keys=True))
        return 0
    if args.feature_manifest_json:
        print(json.dumps(desktop_feature_manifest(), indent=2, sort_keys=True))
        return 0
    if args.layout_json:
        print(json.dumps(desktop_layout_manifest(), indent=2, sort_keys=True))
        return 0
    if args.feature_smoke_json:
        print(json.dumps(run_desktop_feature_smoke(data_dir, project=args.project), indent=2, sort_keys=True))
        return 0
    if argv is None and not args.legacy_tk:
        maybe_reexec_project_venv()
    try:
        if args.legacy_tk:
            return launch_desktop(data_dir, project=args.project)
        return launch_workbench(
            data_dir,
            project=args.project,
            host=args.host,
            port=args.port,
            open_window=not args.no_open,
            ui_root=args.ui_root,
        )
    except (RuntimeError, OSError) as exc:
        show_desktop_launch_error(f"{type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
