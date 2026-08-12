from __future__ import annotations

import gzip
import hashlib
import json
import unittest

from openagent.science.materials.catalog import (
    PublicMaterialsDataset,
    analyse_public_materials_dataset,
    import_public_materials_dataset,
    list_public_materials_datasets,
)


def _fixture(rows: int = 500) -> bytes:
    columns = ["density", "temperature", "bulk_modulus"]
    data = []
    for index in range(rows):
        density = 2.0 + index * 0.01
        temperature = 250.0 + index % 20
        modulus = 30.0 * density - 0.04 * temperature + (index % 3) * 0.02
        data.append([density, temperature, modulus])
    return gzip.compress(json.dumps({"columns": columns, "index": list(range(rows)), "data": data}).encode("utf-8"), mtime=0)


def _entry(content: bytes, *, rows: int = 500, digest: str | None = None) -> PublicMaterialsDataset:
    return PublicMaterialsDataset(
        id="test-public-materials",
        title="Test public materials",
        description="Network-free test fixture.",
        rows=rows,
        target="bulk_modulus",
        download_url="https://datasets.example.test/materials.json.gz",
        source_url="https://datasets.example.test/record",
        sha256=digest or hashlib.sha256(content).hexdigest(),
        license_id="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        citation="Test fixture citation.",
        file_name="materials.json.gz",
    )


class PublicMaterialsCatalogTests(unittest.TestCase):
    def test_catalog_contains_only_metadata_and_explicit_provenance(self) -> None:
        catalog = list_public_materials_datasets()
        self.assertGreaterEqual(len(catalog), 3)
        for entry in catalog:
            self.assertGreaterEqual(entry["rows"], 500)
            self.assertLessEqual(entry["rows"], 5000)
            self.assertEqual(len(entry["sha256"]), 64)
            self.assertTrue(entry["download_url"].startswith("https://"))
            self.assertTrue(entry["source_url"].startswith("https://"))
            self.assertTrue(entry["license_id"])
            self.assertFalse(entry["bundled_records"])

    def test_import_requires_network_and_license_approval_without_fetching(self) -> None:
        content = _fixture()
        calls: list[str] = []

        def fetch(url: str, limit: int) -> bytes:
            calls.append(url)
            return content

        with self.assertRaises(PermissionError):
            import_public_materials_dataset("test-public-materials", fetcher=fetch, catalog=[_entry(content)])
        with self.assertRaises(PermissionError):
            import_public_materials_dataset("test-public-materials", allow_external=True, fetcher=fetch, catalog=[_entry(content)])
        self.assertEqual(calls, [])

    def test_import_verifies_checksum_and_expected_row_count(self) -> None:
        content = _fixture()
        fetch = lambda _url, _limit: content
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            import_public_materials_dataset(
                "test-public-materials",
                allow_external=True,
                license_accepted=True,
                fetcher=fetch,
                catalog=[_entry(content, digest="0" * 64)],
            )
        with self.assertRaisesRegex(ValueError, "row count mismatch"):
            import_public_materials_dataset(
                "test-public-materials",
                allow_external=True,
                license_accepted=True,
                fetcher=fetch,
                catalog=[_entry(content, rows=501)],
            )

    def test_public_analysis_is_reproducible_and_uses_an_80_20_split(self) -> None:
        content = _fixture()
        kwargs = {
            "allow_external": True,
            "license_accepted": True,
            "fetcher": lambda _url, _limit: content,
            "catalog": [_entry(content)],
            "seed": 47,
        }
        first = analyse_public_materials_dataset("test-public-materials", **kwargs)
        second = analyse_public_materials_dataset("test-public-materials", **kwargs)
        self.assertEqual(first["training_rows"], 400)
        self.assertEqual(first["test_rows"], 100)
        self.assertEqual(first["predictions"], second["predictions"])
        self.assertEqual(first["test_metrics"], second["test_metrics"])
        self.assertEqual(first["provenance"]["verified_sha256"], hashlib.sha256(content).hexdigest())
        self.assertEqual(first["split"], {"method": "seeded-random", "training_fraction": 0.8, "test_fraction": 0.2, "seed": 47})


if __name__ == "__main__":
    unittest.main()
