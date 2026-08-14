"""Framework and toolchain preflight for CodeQ generation runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import platform
import shutil
from typing import Any

from .profiles import classify_request


@dataclass(frozen=True)
class FrameworkSpec:
    id: str
    label: str
    languages: tuple[str, ...]
    manifests: tuple[str, ...]
    required_tools: tuple[tuple[str, ...], ...] = ()
    host_platforms: tuple[str, ...] = ()
    external_requirements: tuple[str, ...] = ()


FRAMEWORKS: tuple[FrameworkSpec, ...] = (
    FrameworkSpec("model-selected", "Model-selected stack", (), ()),
    FrameworkSpec("browser", "Browser application", ("html", "css", "javascript"), ("index.html",)),
    FrameworkSpec("react", "React", ("javascript", "typescript"), ("package.json",), (("node",), ("npm", "npm.cmd"))),
    FrameworkSpec("nextjs", "Next.js", ("typescript", "javascript"), ("package.json",), (("node",), ("npm", "npm.cmd"))),
    FrameworkSpec("vue", "Vue", ("javascript", "typescript"), ("package.json",), (("node",), ("npm", "npm.cmd"))),
    FrameworkSpec("svelte", "Svelte", ("typescript", "javascript"), ("package.json",), (("node",), ("npm", "npm.cmd"))),
    FrameworkSpec("angular", "Angular", ("typescript",), ("package.json", "angular.json"), (("node",), ("npm", "npm.cmd"))),
    FrameworkSpec("node", "Node.js", ("javascript", "typescript"), ("package.json",), (("node",), ("npm", "npm.cmd"))),
    FrameworkSpec("python", "Python", ("python",), ("pyproject.toml or requirements.txt",), (("python", "python.exe"),)),
    FrameworkSpec("django", "Django", ("python",), ("manage.py",), (("python", "python.exe"),)),
    FrameworkSpec("fastapi", "FastAPI", ("python",), ("pyproject.toml or requirements.txt",), (("python", "python.exe"),)),
    FrameworkSpec("flask", "Flask", ("python",), ("pyproject.toml or requirements.txt",), (("python", "python.exe"),)),
    FrameworkSpec("go", "Go", ("go",), ("go.mod",), (("go",),)),
    FrameworkSpec("rust", "Rust", ("rust",), ("Cargo.toml",), (("cargo",), ("rustc",))),
    FrameworkSpec("cmake", "C or C++ with CMake", ("c", "cpp"), ("CMakeLists.txt",), (("cmake",),)),
    FrameworkSpec("zig", "Zig", ("zig",), ("build.zig",), (("zig",),)),
    FrameworkSpec("spring", "Spring", ("java", "kotlin"), ("pom.xml or build.gradle",), (("java",),)),
    FrameworkSpec("dotnet", ".NET", ("csharp",), (".csproj",), (("dotnet",),)),
    FrameworkSpec("rails", "Ruby on Rails", ("ruby",), ("Gemfile",), (("ruby",), ("bundle",))),
    FrameworkSpec("laravel", "Laravel", ("php",), ("composer.json", "artisan"), (("php",), ("composer",))),
    FrameworkSpec("phoenix", "Phoenix", ("elixir",), ("mix.exs",), (("elixir",), ("mix",))),
    FrameworkSpec("flutter", "Flutter", ("dart",), ("pubspec.yaml",), (("flutter",), ("dart",))),
    FrameworkSpec("react-native", "React Native", ("typescript", "javascript"), ("package.json",), (("node",), ("npm", "npm.cmd")), external_requirements=("mobile emulator or physical device for device validation",)),
    FrameworkSpec("android", "Native Android", ("kotlin", "java"), ("settings.gradle or settings.gradle.kts",), (("java",),), external_requirements=("Android SDK and emulator or physical device",)),
    FrameworkSpec("ios", "Native iOS", ("swift",), (".xcodeproj or Package.swift",), (("xcodebuild",),), ("darwin",), ("Apple signing identity and provisioning profile for signed distribution",)),
    FrameworkSpec("electron", "Electron", ("javascript", "typescript"), ("package.json",), (("node",), ("npm", "npm.cmd"))),
    FrameworkSpec("tauri", "Tauri", ("rust", "javascript", "typescript"), ("package.json", "src-tauri/Cargo.toml"), (("cargo",), ("node",), ("npm", "npm.cmd"))),
    FrameworkSpec("solidity", "Solidity", ("solidity",), ("foundry.toml or hardhat.config",), (("forge", "npx"),), external_requirements=("RPC endpoint, funded signing account, and chain selection for deployment",)),
    FrameworkSpec("godot", "Godot", ("gdscript",), ("project.godot",), (("godot", "godot4"),)),
    FrameworkSpec("unity", "Unity", ("csharp",), ("ProjectSettings/ProjectVersion.txt",), (("unity", "Unity"),), external_requirements=("Unity editor installation and platform build modules",)),
    FrameworkSpec("unreal", "Unreal Engine", ("cpp",), (".uproject",), (("UnrealEditor", "UnrealEditor.exe"),), external_requirements=("Unreal Engine installation and target-platform SDK",)),
)

FRAMEWORK_BY_ID = {item.id: item for item in FRAMEWORKS}


def framework_catalog() -> list[dict[str, Any]]:
    return [asdict(item) for item in FRAMEWORKS]


def toolchain_preflight(goal: str, project_map: dict[str, Any] | None = None) -> dict[str, Any]:
    project_map = project_map or {"file_count": 0, "framework": "", "languages": []}
    profile = classify_request(goal, project_map)
    framework_id = str(profile.get("framework", "model-selected"))
    spec = FRAMEWORK_BY_ID.get(framework_id, FRAMEWORK_BY_ID["model-selected"])
    return preflight_for_spec(spec, goal, project_map, profile=profile)


def preflight_for_spec(
    spec: FrameworkSpec,
    goal: str,
    project_map: dict[str, Any] | None = None,
    *,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    project_map = project_map or {"file_count": 0, "framework": "", "languages": []}
    profile = profile or classify_request(goal, project_map)
    profile = {**profile, "framework": spec.id}
    host = platform.system().lower()
    host_supported = not spec.host_platforms or host in spec.host_platforms
    tools = []
    for alternatives in spec.required_tools:
        resolved = next((shutil.which(name) for name in alternatives if shutil.which(name)), None)
        tools.append({"alternatives": list(alternatives), "available": bool(resolved), "path": resolved or ""})
    missing_tools = [item["alternatives"] for item in tools if not item["available"]]
    external = list(spec.external_requirements)
    requested_boundaries = _external_boundaries(goal)
    return {
        "profile": profile,
        "framework": asdict(spec),
        "host": {"system": platform.system(), "release": platform.release(), "supported": host_supported},
        "toolchains": tools,
        "missing_toolchains": missing_tools,
        "source_generation": "available",
        "local_validation": "ready" if host_supported and not missing_tools else "requires_setup",
        "external_boundaries": requested_boundaries,
        "external_requirements": external,
        "production_readiness": "not_evaluated",
        "production_gates": [
            "requested build and test commands pass",
            "runtime smoke checks pass on the target platform",
            "security checks and acceptance criteria pass",
            "required credentials are supplied through the secret vault",
            "external release actions receive explicit approval",
        ],
        "environment": {"path_entries": len(os.environ.get("PATH", "").split(os.pathsep))},
    }


def _external_boundaries(goal: str) -> list[dict[str, str]]:
    text = str(goal).casefold()
    boundaries = (
        (("deploy", "production", "cloud", "aws", "azure", "gcp"), "deployment", "approval_required"),
        (("app store", "play store", "testflight"), "app_store_submission", "approval_required"),
        (("payment", "stripe", "purchase"), "payments", "approval_required"),
        (("dns", "domain"), "dns", "approval_required"),
        (("kubernetes", "cluster", "k8s"), "cluster", "external_environment_required"),
        (("firmware", "microcontroller", "embedded"), "hardware", "physical_target_required"),
        (("enterprise", "sso", "saml", "scim"), "enterprise_integration", "provider_configuration_required"),
    )
    return [
        {"boundary": name, "status": status}
        for needles, name, status in boundaries
        if any(needle in text for needle in needles)
    ]
