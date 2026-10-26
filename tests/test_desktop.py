from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib
import unittest
from unittest.mock import patch

from openagent.desktop import (
    desktop_feature_manifest,
    desktop_runtime_status,
    desktop_layout_manifest,
    desktop_status,
    main,
    run_desktop_feature_smoke,
    windows_installation_manifest,
)
from openagent.hard_check import main as hard_check_main
from openagent.workbench import (
    DEFAULT_WORKBENCH_PORT,
    DesktopAssetError,
    WorkbenchInstanceLock,
    _existing_instance_is_healthy,
    build_parser as build_workbench_parser,
    configure_managed_accelerator,
    configure_managed_openvino_model,
    installation_source_root,
    launch_workbench,
    project_venv_python,
    ui_build_token,
    workbench_status,
)


class DesktopTests(unittest.TestCase):
    def test_workbench_uses_one_stable_default_port(self) -> None:
        args = build_workbench_parser().parse_args([])
        self.assertEqual(args.port, DEFAULT_WORKBENCH_PORT)
        self.assertEqual(args.port, 8765)
        self.assertRegex(ui_build_token(), r"^[a-f0-9]{16}$")

    def test_workbench_instance_lock_prevents_duplicate_server(self) -> None:
        with TemporaryDirectory() as tmp:
            first = WorkbenchInstanceLock(Path(tmp))
            second = WorkbenchInstanceLock(Path(tmp))
            self.assertTrue(first.acquire())
            try:
                self.assertFalse(second.acquire())
            finally:
                first.release()
            self.assertTrue(second.acquire())
            second.release()

    def test_desktop_opens_existing_headless_runtime(self) -> None:
        with TemporaryDirectory() as tmp:
            runtime = Path(tmp) / "runtime"
            runtime.mkdir()
            with \
             patch("openagent.workbench.workbench_runtime_dir", return_value=runtime), \
             patch("openagent.workbench.WorkbenchInstanceLock.acquire", return_value=False), \
             patch("openagent.workbench._running_instance_url", return_value="http://127.0.0.1:8765/"), \
             patch("openagent.workbench._existing_instance_is_healthy", return_value=True), \
             patch("openagent.workbench.focus_openagent_window", return_value=False), \
             patch("openagent.workbench.open_app_window") as open_window:
                self.assertEqual(launch_workbench(tmp, open_window=True), 0)
                open_window.assert_called_once()
                self.assertEqual(open_window.call_args.args[0], "http://127.0.0.1:8765/")

    def test_stale_saved_instance_url_never_opens_a_blank_window(self) -> None:
        with TemporaryDirectory() as tmp:
            runtime = Path(tmp) / "runtime"
            runtime.mkdir()
            with \
             patch("openagent.workbench.workbench_runtime_dir", return_value=runtime), \
             patch("openagent.workbench.WorkbenchInstanceLock.acquire", return_value=False), \
             patch("openagent.workbench._running_instance_url", return_value="http://127.0.0.1:8878/"), \
             patch("openagent.workbench._existing_instance_is_healthy", return_value=False), \
             patch("openagent.workbench.open_app_window") as open_window:
                with self.assertRaisesRegex(DesktopAssetError, "saved workbench address is not healthy"):
                    launch_workbench(tmp, open_window=True)
                open_window.assert_not_called()

    def test_inaccessible_workbench_lock_is_handled_as_contention(self) -> None:
        with TemporaryDirectory() as tmp, patch.object(Path, "open", side_effect=PermissionError("locked")):
            self.assertFalse(WorkbenchInstanceLock(Path(tmp) / "runtime").acquire())

    def test_existing_instance_health_check_rejects_non_loopback_url(self) -> None:
        self.assertFalse(_existing_instance_is_healthy("http://example.com/"))

    def test_desktop_reports_stale_runtime_instead_of_opening_blank_page(self) -> None:
        with TemporaryDirectory() as tmp:
            runtime = Path(tmp) / "runtime"
            runtime.mkdir()
            with \
             patch("openagent.workbench.workbench_runtime_dir", return_value=runtime), \
             patch("openagent.workbench.WorkbenchInstanceLock.acquire", return_value=False), \
             patch("openagent.workbench._running_instance_url", return_value="http://127.0.0.1:8878/"), \
             patch("openagent.workbench._existing_instance_is_healthy", return_value=False), \
             patch("openagent.workbench.open_app_window") as open_window, \
             patch("openagent.desktop.show_desktop_launch_error") as show_error:
                self.assertEqual(main(["--no-open", "--data-dir", tmp]), 2)
                show_error.assert_called_once()
                self.assertIn("saved workbench address is not healthy", show_error.call_args.args[0])
                open_window.assert_not_called()

    def test_managed_openvino_configuration_requires_complete_local_model(self) -> None:
        with TemporaryDirectory() as tmp:
            model = Path(tmp) / "models" / "openvino" / "qwen2.5-1.5b-instruct-int4-ov"
            model.mkdir(parents=True)
            self.assertNotEqual(configure_managed_openvino_model(Path(tmp)), str(model.resolve()))
            (model / "openvino_model.xml").write_text("model", encoding="ascii")
            (model / "openvino_model.bin").write_bytes(b"weights")
            import os
            from unittest.mock import patch

            with patch.dict(os.environ, {"OPENAGENT_OPENVINO_MODEL_DIR": str(model)}, clear=False):
                selected = configure_managed_openvino_model(Path(tmp))
                self.assertEqual(selected, str(model.resolve()))

    def test_installed_runtime_discovers_manifest_source_accelerator_and_model(self) -> None:
        import os
        from unittest.mock import patch

        with TemporaryDirectory() as tmp:
            local_app_data = Path(tmp) / "local"
            source = Path(tmp) / "source"
            (source / "openagent").mkdir(parents=True)
            accelerator = source / ".accelerator-venv" / "Scripts" / "python.exe"
            accelerator.parent.mkdir(parents=True)
            accelerator.write_bytes(b"local-python")
            model = source / "models" / "openvino" / "qwen2.5-1.5b-instruct-int4-ov"
            model.mkdir(parents=True)
            (model / "openvino_model.xml").write_text("model", encoding="ascii")
            (model / "openvino_model.bin").write_bytes(b"weights")
            manifest = local_app_data / "OpenAgent" / "install-manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"source": str(source)}), encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "LOCALAPPDATA": str(local_app_data),
                    "OPENAGENT_SOURCE_ROOT": "",
                    "OPENAGENT_ACCELERATOR_PYTHON": "",
                    "OPENAGENT_OPENVINO_MODEL_DIR": "",
                },
                clear=False,
            ):
                self.assertEqual(installation_source_root(), source.resolve())
                self.assertEqual(configure_managed_accelerator(), str(accelerator.resolve()))
                self.assertEqual(configure_managed_openvino_model(Path(tmp) / "data"), str(model.resolve()))

    def test_installed_runtime_does_not_reexec_into_manifest_source_venv(self) -> None:
        import os
        from unittest.mock import patch

        with TemporaryDirectory() as tmp:
            local_app_data = Path(tmp) / "local"
            source = Path(tmp) / "source"
            (source / "openagent").mkdir(parents=True)
            (source / "pyproject.toml").write_text("[project]\nname='openagent'\n", encoding="utf-8")
            source_interpreter = source / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            source_interpreter.parent.mkdir(parents=True)
            source_interpreter.write_bytes(b"source-python")
            installed = Path(tmp) / "install" / ".venv" / ("Lib/site-packages" if os.name == "nt" else "lib/python3/site-packages")
            installed_package = installed / "openagent"
            installed_package.mkdir(parents=True)
            installed_entry = installed_package / "workbench.py"
            installed_entry.write_text("# installed wheel entry\n", encoding="utf-8")
            manifest = local_app_data / "OpenAgent" / "install-manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"source": str(source)}), encoding="utf-8")
            with patch("openagent.workbench.__file__", str(installed_entry)), patch.dict(
                os.environ, {"LOCALAPPDATA": str(local_app_data), "OPENAGENT_SOURCE_ROOT": ""}, clear=False
            ):
                self.assertEqual(installation_source_root(), source.resolve())
                self.assertIsNone(project_venv_python())

    def test_source_checkout_uses_its_virtualenv_for_development(self) -> None:
        import os
        from unittest.mock import patch

        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            package = source / "openagent"
            package.mkdir(parents=True)
            (source / "pyproject.toml").write_text("[project]\nname='openagent'\n", encoding="utf-8")
            interpreter = source / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            interpreter.parent.mkdir(parents=True)
            interpreter.write_bytes(b"source-python")
            with patch("openagent.workbench.__file__", str(package / "workbench.py")), patch.dict(
                os.environ, {"OPENAGENT_SOURCE_ROOT": ""}, clear=False
            ):
                self.assertEqual(project_venv_python(), interpreter.resolve())

    def test_installed_editable_package_keeps_its_managed_install_venv(self) -> None:
        import os
        from unittest.mock import patch

        with TemporaryDirectory() as tmp:
            local_app_data = Path(tmp) / "local"
            source = Path(tmp) / "source"
            package = source / "openagent"
            package.mkdir(parents=True)
            (source / "pyproject.toml").write_text("[project]\nname='openagent'\n", encoding="utf-8")
            source_interpreter = source / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            source_interpreter.parent.mkdir(parents=True)
            source_interpreter.write_bytes(b"source-python")
            install_venv = local_app_data / "OpenAgent" / ".venv"
            manifest = local_app_data / "OpenAgent" / "install-manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"source": str(source)}), encoding="utf-8")
            with patch("openagent.workbench.__file__", str(package / "workbench.py")), patch(
                "openagent.workbench.sys.prefix", str(install_venv)
            ), patch.dict(
                os.environ,
                {"LOCALAPPDATA": str(local_app_data), "OPENAGENT_SOURCE_ROOT": ""},
                clear=False,
            ):
                self.assertIsNone(project_venv_python())

    def test_desktop_status_is_native_and_local(self) -> None:
        with TemporaryDirectory() as tmp:
            status = desktop_status(tmp, project="core")
            self.assertEqual(status["mode"], "native-desktop")
            self.assertFalse(status["browser_required"])
            self.assertEqual(status["default_surface"], "modern-web-workbench")
            self.assertFalse(status["browser_tab_required"])
            self.assertTrue(status["windows_installable"])
            self.assertTrue(status["modern_workbench"]["single_instance"])
            self.assertEqual(status["modern_workbench"]["default_port"], 8765)
            self.assertEqual(status["entrypoints"]["package_script"], "openagent-desktop")
            self.assertEqual(status["entrypoints"]["modern_workbench_script"], "openagent-workbench")
            self.assertEqual(status["capability_verdict"], "implemented_local_ai_os")
            self.assertIn("local data directory", status["privacy_boundary"])
            module_ids = {module["id"] for module in status["desktop_modules"]}
            self.assertIn("workspace", module_ids)
            self.assertIn("automation", module_ids)
            self.assertIn("security_developer", module_ids)
            self.assertEqual(status["desktop_layout"]["shell"], "native-desktop-workbench")
            self.assertEqual(status["modern_workbench"]["mode"], "modern-desktop-workbench")
            self.assertIn("browser", status["runtime_tools"])
            self.assertIn("providers", status["runtime_tools"])

    def test_desktop_feature_manifest_exposes_mission_modules(self) -> None:
        modules = desktop_feature_manifest()
        ids = {module["id"] for module in modules}
        self.assertSetEqual(
            ids,
            {
                "workspace",
                "knowledge",
                "assistant",
                "models",
                "voice_media",
                "automation",
                "security_developer",
            },
        )
        features = " ".join(feature for module in modules for feature in module["features"])
        self.assertIn("RAG synthesis", features)
        self.assertIn("HTML form simulation", features)
        self.assertIn("optional browser tests", features)
        self.assertIn("SDK", features)

    def test_desktop_runtime_status_reports_optional_tools_without_starting_them(self) -> None:
        status = desktop_runtime_status()
        self.assertIn("installed", status["browser"])
        self.assertIn("office_exports", status)
        self.assertIn("colab", status)
        self.assertIn("openai-compatible", status["providers"]["registered"])

    def test_desktop_layout_manifest_exposes_native_workbench(self) -> None:
        layout = desktop_layout_manifest()
        self.assertEqual(layout["shell"], "native-desktop-workbench")
        self.assertEqual(layout["default_surface"], "modern-web-workbench")
        self.assertEqual(layout["first_screen"]["navigation"], "left-sidebar")
        self.assertIn("module-grid", layout["first_screen"]["panels"])
        self.assertFalse(layout["browser_required"])
        self.assertIn("OpenAgent Desktop reads and writes only", layout["privacy_boundary"])
        self.assertGreaterEqual(len(layout["navigation"]), 10)
        self.assertEqual(layout["keyboard_shortcuts"]["command_palette"], "Ctrl+K")
        self.assertEqual(layout["keyboard_shortcuts"]["navigation"], "Ctrl+1 through Ctrl+0")
        self.assertEqual(layout["modern_workbench"]["script"], "openagent-workbench")

    def test_workbench_status_exposes_modern_app_window_assets(self) -> None:
        with TemporaryDirectory() as tmp:
            status = workbench_status(tmp, project="core")
            self.assertEqual(status["mode"], "modern-desktop-workbench")
            self.assertTrue(status["localhost_only"])
            self.assertFalse(status["browser_tab_required"])
            self.assertEqual(status["entrypoints"]["package_script"], "openagent-workbench")
            self.assertTrue(Path(status["assets"]["index"]).exists())
            self.assertTrue(Path(status["assets"]["css"]).exists())
            self.assertTrue(Path(status["assets"]["js"]).exists())
            self.assertRegex(status["assets"]["build"], r"^[a-f0-9]{16}$")

    def test_desktop_feature_smoke_exercises_visible_modules(self) -> None:
        with TemporaryDirectory() as tmp:
            result = run_desktop_feature_smoke(tmp, project="core")
            self.assertTrue(result["ok"], result)
            self.assertIn("workspace", result["checks"])
            self.assertIn("knowledge", result["checks"])
            self.assertIn("developer", result["checks"])

    def test_status_json_cli_does_not_open_gui(self) -> None:
        with TemporaryDirectory() as tmp:
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["--status-json", "--data-dir", tmp, "--project", "core"])
            payload = json.loads(output.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(payload["project"], "core")
            self.assertFalse(payload["browser_required"])
            self.assertEqual(payload["default_surface"], "modern-web-workbench")

    def test_feature_manifest_json_cli_does_not_open_gui(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["--feature-manifest-json"])
        payload = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertGreaterEqual(len(payload), 7)

    def test_layout_json_cli_does_not_open_gui(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["--layout-json"])
        payload = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload["shell"], "native-desktop-workbench")
        self.assertFalse(payload["browser_required"])
        self.assertEqual(payload["default_surface"], "modern-web-workbench")

    def test_feature_smoke_json_cli_does_not_open_gui(self) -> None:
        with TemporaryDirectory() as tmp:
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["--feature-smoke-json", "--data-dir", tmp, "--project", "core"])
            payload = json.loads(output.getvalue())
            self.assertEqual(code, 0)
            self.assertTrue(payload["ok"], payload)

    def test_windows_installer_assets_exist(self) -> None:
        manifest = windows_installation_manifest()
        missing = [relative for relative, item in manifest["files"].items() if not item["exists"]]
        self.assertEqual(missing, [])
        self.assertFalse(manifest["browser_required"])
        self.assertEqual(manifest["launch_command"], "openagent-desktop")

    def test_hard_check_cli_returns_zero_when_audit_passes(self) -> None:
        with TemporaryDirectory() as tmp:
            output = io.StringIO()
            with redirect_stdout(output):
                code = hard_check_main(["--data-dir", tmp, "--project", "core"])
            payload = json.loads(output.getvalue())
            self.assertEqual(code, 0)
            self.assertTrue(payload["ok"], payload)

    def test_windows_installer_has_runtime_preflight_checks(self) -> None:
        installer = Path(__file__).resolve().parents[1] / "installers" / "windows" / "install.ps1"
        content = installer.read_text(encoding="ascii")
        self.assertIn("$managedRuntimeRoot = [System.IO.Path]::GetFullPath", content)
        self.assertIn("$activeManagedProcessIds", content)
        self.assertIn("Close its desktop window and local server before upgrading", content)
        self.assertIn("Python 3.11 or newer is required", content)
        self.assertIn("$pythonExe -m venv", content)
        self.assertIn('$frontendBuildRoot = [System.IO.Path]::GetFullPath((Join-Path $env:TEMP', content)
        self.assertIn("$frontendNpmCache = Join-Path $frontendBuildRoot \"npm-cache\"", content)
        self.assertIn("$env:npm_config_cache = $frontendNpmCache", content)
        self.assertIn('foreach ($relativeDirectory in @("src", "public", "scripts", "reference"))', content)
        self.assertIn("Push-Location $frontendBuild", content)
        self.assertIn('Remove-Item -LiteralPath $frontendBuildRoot -Recurse -Force', content)
        self.assertNotIn("Push-Location $frontendSource", content)
        self.assertIn("[switch]$UpgradePip", content)
        self.assertIn("[switch]$SkipVoiceExtras", content)
        self.assertIn("[switch]$CreateShortcutsForCustomInstall", content)
        self.assertIn("if ($UpgradePip)", content)
        self.assertIn("$SourcePath[media,rag]", content)
        self.assertIn("$SourcePath[full]", content)
        self.assertIn("install-manifest.json", content)
        self.assertIn('Scripts\\pythonw.exe', content)
        self.assertIn('$shortcut.WindowStyle = 7', content)
        self.assertIn('-m openagent.desktop --data-dir', content)
        self.assertIn('$normalizedInstallDir = [System.IO.Path]::GetFullPath($InstallDir)', content)
        self.assertIn('$defaultInstallSuffix = "\\AppData\\Local\\OpenAgent"', content)
        self.assertIn('$userProfileRoot = $normalizedInstallDir.Substring', content)
        self.assertIn('$programsDir = Join-Path $userProfileRoot "AppData\\Roaming\\Microsoft\\Windows\\Start Menu\\Programs"', content)
        self.assertIn("$shouldCreateShortcuts = $isDefaultInstall -or $CreateShortcutsForCustomInstall", content)
        self.assertIn("Custom install path detected; existing Start Menu and Desktop shortcuts were left unchanged", content)
        pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="ascii")
        self.assertIn("openagent-workbench", pyproject)

    def test_distribution_discovers_nested_openagent_packages(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = tomllib.loads((root / "pyproject.toml").read_text(encoding="ascii"))
        includes = config["tool"]["setuptools"]["packages"]["find"]["include"]
        self.assertEqual(includes, ["openagent*"])
        self.assertTrue((root / "openagent" / "routes" / "handlers" / "__init__.py").is_file())


if __name__ == "__main__":
    unittest.main()
