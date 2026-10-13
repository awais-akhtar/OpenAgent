from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class WebUiTests(unittest.TestCase):
    def test_react_spa_shell_and_production_assets_are_present(self) -> None:
        index = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        app = (ROOT / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")
        package = (ROOT / "package.json").read_text(encoding="utf-8")
        service_worker = (ROOT / "web" / "sw.js").read_text(encoding="utf-8")

        self.assertTrue((ROOT / "web" / "app.js").is_file())
        self.assertTrue((ROOT / "web" / "styles.css").is_file())
        self.assertIn('id="app"', index)
        self.assertRegex(index, r'<script type="module"[^>]+src="/app\.js"')
        self.assertIn('href="/styles.css"', index)
        self.assertIn('"react":', package)
        self.assertIn('"vite":', package)
        self.assertIn('"@arco-design/web-react":', package)
        self.assertIn('"unocss":', package)
        main = (ROOT / "frontend" / "src" / "main.tsx").read_text(encoding="utf-8")
        self.assertIn("createRoot", main)
        self.assertIn("ConfigProvider", main)
        self.assertIn('aria-label="Search OpenAgent"', app)
        self.assertIn("aria-label={`Use ${themeMode", app)
        self.assertIn("specialPageIds", app)
        self.assertRegex(service_worker, r"CACHE_VERSION\s*=\s*['\"]openagent-vite-[a-f0-9]{16,}")

    def test_every_original_view_has_a_react_screen_mapping(self) -> None:
        navigation = (ROOT / "frontend" / "src" / "lib" / "navigation.ts").read_text(encoding="utf-8")
        app = (ROOT / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")
        registry = (ROOT / "frontend" / "src" / "pages" / "SpecialPages.tsx").read_text(encoding="utf-8")
        resources = (ROOT / "frontend" / "src" / "pages" / "ResourcePages.tsx").read_text(encoding="utf-8")
        registry_block = navigation.split("export const groups", 1)[1].split("export const views", 1)[0]
        view_ids = re.findall(r"\{ id: '([a-z]+)', label:", registry_block)
        expected = {
            "dashboard", "chat", "codebase", "voice", "research", "phd", "agents", "knowledge", "workspace", "automation", "assistant", "communications", "models", "security", "developer", "settings", "engine", "science", "calendar", "twin", "vault", "scraper", "finetune",
        }
        self.assertEqual(len(view_ids), 23)
        self.assertEqual(set(view_ids), expected)
        self.assertRegex(app, r"view\s*===\s*'chat'\s*\?\s*<ChatPage")
        self.assertRegex(app, r"view\s*===\s*'dashboard'\s*\?\s*<DashboardPage")
        self.assertRegex(app, r"specialPageIds\.has\(view\)\s*\?\s*<SpecialPages")
        for view_id in view_ids:
            if view_id in {"chat", "dashboard"}:
                continue
            with self.subTest(view=view_id):
                is_special_page = re.search(rf"\b{re.escape(view_id)}\s*:", registry) is not None
                is_resource_page = re.search(rf"\b{re.escape(view_id)}\s*:\s*\{{\s*title:", resources) is not None
                self.assertTrue(is_special_page or is_resource_page, f"No page mapping found for {view_id}")


if __name__ == "__main__":
    unittest.main()
