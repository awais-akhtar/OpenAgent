import tempfile
import unittest
from pathlib import Path

from openagent.adaptive_engine.public_abi_audit import audit_public_abi


ROOT = Path(__file__).resolve().parents[1]


class PublicABIAuditTests(unittest.TestCase):
    def test_repository_public_header_definitions_and_ctypes_names_reconcile(self):
        audit = audit_public_abi(ROOT).require_ok()
        self.assertGreater(len(audit.exported_symbols), 20)
        self.assertGreater(len(audit.ctypes_symbols), 20)
        self.assertIn("oa_glm5_next_range_version", audit.ctypes_symbols)
        self.assertIn("oa_v41_execution_op", audit.ctypes_symbols)

    def test_audit_rejects_an_export_without_implementation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "adaptive_engine"
            native = root / "native"
            native.mkdir(parents=True)
            (native / "contract.h").write_text(
                "#define OA_API\nOA_API int oa_missing(void);\n", encoding="utf-8")
            (native / "impl.c").write_text("int unrelated(void) { return 0; }\n", encoding="utf-8")
            (root / "adapter.py").write_text("value = 1\n", encoding="utf-8")
            audit = audit_public_abi(root)
        self.assertEqual(audit.missing_export_definitions, ("oa_missing",))
        self.assertFalse(audit.ok)

    def test_audit_rejects_stale_ctypes_symbol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "adaptive_engine"
            native = root / "native"
            native.mkdir(parents=True)
            (native / "contract.h").write_text("#define OA_API\n", encoding="utf-8")
            (native / "impl.c").write_text("int present(void) { return 0; }\n", encoding="utf-8")
            (root / "adapter.py").write_text("library.oa_stale.argtypes = []\n", encoding="utf-8")
            audit = audit_public_abi(root)
        self.assertEqual(audit.missing_ctypes_sources, ("oa_stale",))
        self.assertFalse(audit.ok)

    def test_audit_rejects_stale_symbol_through_nested_native_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "adaptive_engine"
            native = root / "native"
            native.mkdir(parents=True)
            (native / "contract.h").write_text("#define OA_API\n", encoding="utf-8")
            (native / "impl.c").write_text("int present(void) { return 0; }\n", encoding="utf-8")
            (root / "adapter.py").write_text(
                "self.kernel.oa_stale.argtypes = []\n", encoding="utf-8")
            audit = audit_public_abi(root)
        self.assertEqual(audit.missing_ctypes_sources, ("oa_stale",))
        self.assertFalse(audit.ok)

    def test_audit_rejects_export_definition_parameter_count_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "adaptive_engine"
            native = root / "native"
            native.mkdir(parents=True)
            (native / "contract.h").write_text(
                "#define OA_API\n"
                "OA_API int OA_CALL oa_mismatch(int value, const float *input);\n"
                "OA_API int OA_CALL oa_callback(int (*callback)(void *, int), size_t count);\n",
                encoding="utf-8")
            (native / "impl.c").write_text(
                "int oa_mismatch(int value) { return value; }\n"
                "int oa_callback(int (*callback)(void *, int), size_t count) { return count; }\n",
                encoding="utf-8")
            (root / "adapter.py").write_text("value = 1\n", encoding="utf-8")
            audit = audit_public_abi(root)
        self.assertEqual(audit.mismatched_export_parameters, ("oa_mismatch",))
        self.assertFalse(audit.ok)

    def test_audit_rejects_same_arity_pointer_type_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "adaptive_engine"
            native = root / "native"
            native.mkdir(parents=True)
            (native / "contract.h").write_text(
                "#define OA_API\n"
                "OA_API int OA_CALL oa_pointer(const float *input, size_t count);\n",
                encoding="utf-8")
            (native / "impl.c").write_text(
                "int oa_pointer(const int *input, size_t count) { return (int)count; }\n",
                encoding="utf-8")
            (root / "adapter.py").write_text("value = 1\n", encoding="utf-8")
            audit = audit_public_abi(root)
        self.assertEqual(audit.mismatched_export_types, ("oa_pointer",))
        self.assertFalse(audit.ok)

    def test_audit_rejects_export_calling_convention_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "adaptive_engine"
            native = root / "native"
            native.mkdir(parents=True)
            (native / "contract.h").write_text(
                "#define OA_API\n#define OA_CALL __cdecl\n"
                "OA_API int OA_CALL oa_calling_convention(void);\n",
                encoding="utf-8",
            )
            (native / "impl.c").write_text(
                "int oa_calling_convention(void) { return 0; }\n",
                encoding="utf-8",
            )
            (root / "adapter.py").write_text("value = 1\n", encoding="utf-8")
            audit = audit_public_abi(root)
        self.assertEqual(audit.mismatched_calling_conventions,
                         ("oa_calling_convention",))
        self.assertFalse(audit.ok)

    def test_audit_accepts_matching_export_calling_convention(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "adaptive_engine"
            native = root / "native"
            native.mkdir(parents=True)
            (native / "contract.h").write_text(
                "#define OA_API\n#define OA_CALL __cdecl\n"
                "OA_API int OA_CALL oa_calling_convention(void);\n",
                encoding="utf-8",
            )
            (native / "impl.c").write_text(
                "int OA_CALL oa_calling_convention(void) { return 0; }\n",
                encoding="utf-8",
            )
            (root / "adapter.py").write_text("value = 1\n", encoding="utf-8")
            audit = audit_public_abi(root)
        self.assertEqual(audit.mismatched_calling_conventions, ())
        self.assertTrue(audit.ok)


if __name__ == "__main__":
    unittest.main()
