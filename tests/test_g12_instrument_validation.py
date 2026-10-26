"""Tests for G12: Scientific AI physical instrument validation and safety."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from openagent.science.instruments import (
    CameraInstrument,
    InstrumentCapability,
    InstrumentRegistry,
    InstrumentStatus,
    InstrumentValidation,
    SpectrographInstrument,
    validate_instrument_code,
)


class InstrumentValidationTests(unittest.TestCase):
    """Test instrument validation and status reporting."""

    def test_camera_validation_available(self) -> None:
        """Test camera validation when available."""
        camera = CameraInstrument()
        validation = camera.validate()

        self.assertTrue(validation.valid)
        self.assertEqual(validation.status, InstrumentStatus.AVAILABLE)
        self.assertEqual(validation.identity, "camera_default")
        self.assertEqual(validation.model, "MockCamera")
        self.assertEqual(len(validation.capabilities), 2)

    def test_camera_capabilities(self) -> None:
        """Test camera capabilities declaration."""
        camera = CameraInstrument()
        validation = camera.validate()

        capture_cap = next(cap for cap in validation.capabilities if cap.name == "capture")
        self.assertFalse(capture_cap.requires_approval)
        self.assertFalse(capture_cap.safety_critical)

        record_cap = next(cap for cap in validation.capabilities if cap.name == "record")
        self.assertTrue(record_cap.requires_approval)
        self.assertEqual(record_cap.required_scope, "science:hardware")

    def test_spectrograph_validation(self) -> None:
        """Test spectrograph validation."""
        spec = SpectrographInstrument()
        validation = spec.validate()

        self.assertTrue(validation.valid)
        self.assertEqual(validation.status, InstrumentStatus.AVAILABLE)
        self.assertEqual(validation.serial_number, "SPEC-001")
        self.assertEqual(validation.identity, "spectrograph_default")

    def test_spectrograph_safety_critical_capabilities(self) -> None:
        """Test that spectrograph marks safety-critical operations."""
        spec = SpectrographInstrument()
        validation = spec.validate()

        measure_cap = next(cap for cap in validation.capabilities if cap.name == "measure")
        self.assertTrue(measure_cap.safety_critical)

        calibrate_cap = next(cap for cap in validation.capabilities if cap.name == "calibrate")
        self.assertTrue(calibrate_cap.safety_critical)
        self.assertTrue(calibrate_cap.requires_approval)


class InstrumentExecutionTests(unittest.TestCase):
    """Test bounded instrument execution."""

    def test_camera_capture_without_approval(self) -> None:
        """Test camera capture (non-approval method)."""
        camera = CameraInstrument()
        result = camera.execute_bounded("capture", {})

        self.assertIn("image_id", result)
        self.assertTrue(result["image_id"].startswith("capture-"))

    def test_camera_record_requires_approval(self) -> None:
        """Test camera record requires approval token."""
        camera = CameraInstrument()

        # Should fail without approval token
        with self.assertRaises(ValueError) as context:
            camera.execute_bounded("record", {})
        self.assertIn("approval", str(context.exception).lower())

        # Should succeed with approval token
        result = camera.execute_bounded("record", {}, approval_token="approved-token-123")
        self.assertIn("recording_id", result)

    def test_spectrograph_measure_with_valid_parameters(self) -> None:
        """Test spectrograph measurement with valid parameters."""
        spec = SpectrographInstrument()
        result = spec.execute_bounded(
            "measure",
            {"wavelength_min": 400, "wavelength_max": 700},
        )

        self.assertIn("spectrum_id", result)

    def test_spectrograph_measure_wavelength_bounds(self) -> None:
        """Test spectrograph enforces wavelength bounds."""
        spec = SpectrographInstrument()

        # Below minimum
        with self.assertRaises(ValueError) as context:
            spec.execute_bounded("measure", {"wavelength_min": 100, "wavelength_max": 700})
        self.assertIn("range", str(context.exception).lower())

        # Above maximum
        with self.assertRaises(ValueError) as context:
            spec.execute_bounded("measure", {"wavelength_min": 400, "wavelength_max": 3000})
        self.assertIn("range", str(context.exception).lower())

    def test_spectrograph_measure_inverted_bounds(self) -> None:
        """Test spectrograph rejects inverted wavelength bounds."""
        spec = SpectrographInstrument()

        with self.assertRaises(ValueError) as context:
            spec.execute_bounded("measure", {"wavelength_min": 700, "wavelength_max": 400})
        self.assertIn("less than", str(context.exception).lower())

    def test_spectrograph_calibrate_requires_approval(self) -> None:
        """Test spectrograph calibration requires approval."""
        spec = SpectrographInstrument()

        with self.assertRaises(ValueError) as context:
            spec.execute_bounded("calibrate", {})
        self.assertIn("approval", str(context.exception).lower())

        result = spec.execute_bounded("calibrate", {}, approval_token="admin-token-456")
        self.assertIn("calibration_id", result)

    def test_instrument_unknown_method(self) -> None:
        """Test error on unknown method."""
        camera = CameraInstrument()

        with self.assertRaises(ValueError) as context:
            camera.execute_bounded("unknown_method", {})
        self.assertIn("unknown", str(context.exception).lower())


class InstrumentRegistryTests(unittest.TestCase):
    """Test instrument registry."""

    def setUp(self) -> None:
        self.registry = InstrumentRegistry()

    def test_register_and_retrieve_instrument(self) -> None:
        """Test registering and retrieving instruments."""
        camera = CameraInstrument()
        self.registry.register(camera)

        retrieved = self.registry.get_instrument("camera_default")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.instrument_id, "camera_default")

    def test_list_available_instruments(self) -> None:
        """Test listing available instruments."""
        camera = CameraInstrument()
        spec = SpectrographInstrument("lab-1")
        self.registry.register(camera)
        self.registry.register(spec)

        available = self.registry.list_available()
        self.assertEqual(len(available), 2)
        identities = {inst.identity for inst in available}
        self.assertIn("camera_default", identities)
        self.assertIn("spectrograph_lab-1", identities)

    def test_execute_method_via_registry(self) -> None:
        """Test executing method through registry."""
        camera = CameraInstrument()
        self.registry.register(camera)

        result = self.registry.execute_method("camera_default", "capture", {})
        self.assertIn("image_id", result)

    def test_execute_unknown_instrument(self) -> None:
        """Test error on unknown instrument."""
        with self.assertRaises(ValueError) as context:
            self.registry.execute_method("unknown_id", "capture", {})
        self.assertIn("unknown", str(context.exception).lower())

    def test_execute_unknown_method(self) -> None:
        """Test error on unknown method for instrument."""
        camera = CameraInstrument()
        self.registry.register(camera)

        with self.assertRaises(ValueError) as context:
            self.registry.execute_method("camera_default", "invalid_method", {})
        self.assertIn("not support", str(context.exception).lower())

    def test_approval_validation(self) -> None:
        """Test approval token validation."""
        camera = CameraInstrument()

        # Register with approval validator
        def validate_approval(token: str) -> bool:
            return token.startswith("valid-")

        self.registry.register(camera, approval_validator=validate_approval)

        # Valid token
        result = self.registry.execute_method(
            "camera_default", "record", {}, approval_token="valid-token"
        )
        self.assertIn("recording_id", result)

        # Invalid token
        with self.assertRaises(ValueError) as context:
            self.registry.execute_method(
                "camera_default", "record", {}, approval_token="invalid-token"
            )
        self.assertIn("approval", str(context.exception).lower())


class InstrumentCodeValidationTests(unittest.TestCase):
    """Test code validation for instrument safety."""

    def test_valid_instrument_code(self) -> None:
        """Test validation of safe instrument code."""
        code = """
        result = registry.get_instrument("camera_default").execute_bounded("capture", {})
        """
        validation = validate_instrument_code(code)

        self.assertTrue(validation["valid"])
        self.assertEqual(validation["violations"], [])
        self.assertEqual(validation["referenced_instruments"], ["camera_default"])

    def test_dangerous_import_os(self) -> None:
        """Test rejection of os module import."""
        code = """
        import os
        os.system("rm -rf /")
        """
        validation = validate_instrument_code(code)

        self.assertFalse(validation["valid"])
        self.assertTrue(any("os module" in v for v in validation["violations"]))

    def test_dangerous_subprocess(self) -> None:
        """Test rejection of subprocess module."""
        code = """
        import subprocess
        subprocess.run(["dangerous", "command"])
        """
        validation = validate_instrument_code(code)

        self.assertFalse(validation["valid"])
        self.assertTrue(any("subprocess" in v for v in validation["violations"]))

    def test_dangerous_file_io(self) -> None:
        """Test rejection of file I/O."""
        code = """
        with open("/etc/passwd", "r") as f:
            data = f.read()
        """
        validation = validate_instrument_code(code)

        self.assertFalse(validation["valid"])
        self.assertTrue(any("file" in v.lower() for v in validation["violations"]))

    def test_dangerous_dynamic_import(self) -> None:
        """Test rejection of dynamic imports."""
        code = """
        mod = __import__("os")
        mod.system("bad")
        """
        validation = validate_instrument_code(code)

        self.assertFalse(validation["valid"])
        self.assertTrue(any("dynamic" in v.lower() for v in validation["violations"]))

    def test_allowed_instruments_restriction(self) -> None:
        """Test restriction to allowed instruments."""
        code = """
        registry.get_instrument("camera_default").execute_bounded("capture", {})
        registry.get_instrument("spectrograph_lab1").execute_bounded("measure", {})
        """
        validation = validate_instrument_code(code, allowed_instruments=["camera_default"])

        self.assertFalse(validation["valid"])
        violations_text = " ".join(validation["violations"])
        self.assertIn("spectrograph_lab1", violations_text)

    def test_multiple_referenced_instruments(self) -> None:
        """Test detection of multiple instrument references."""
        code = """
        cam = registry.get_instrument("camera_lab1")
        spec = registry.get_instrument("spectrograph_lab2")
        """
        validation = validate_instrument_code(code)

        self.assertTrue(validation["valid"])
        self.assertEqual(
            set(validation["referenced_instruments"]),
            {"camera_lab1", "spectrograph_lab2"},
        )


class InstrumentCapabilityTests(unittest.TestCase):
    """Test instrument capability declaration."""

    def test_capability_creation(self) -> None:
        """Test creating capability."""
        cap = InstrumentCapability(
            name="measure",
            description="Measure spectrum",
            requires_approval=True,
            required_scope="science:admin",
            safety_critical=True,
        )

        self.assertEqual(cap.name, "measure")
        self.assertTrue(cap.requires_approval)
        self.assertEqual(cap.required_scope, "science:admin")
        self.assertTrue(cap.safety_critical)

    def test_capability_defaults(self) -> None:
        """Test capability defaults."""
        cap = InstrumentCapability(
            name="read",
            description="Read sensor",
        )

        self.assertFalse(cap.requires_approval)
        self.assertIsNone(cap.required_scope)
        self.assertFalse(cap.safety_critical)


class InstrumentIntegrationTests(unittest.TestCase):
    """Integration tests for instrument system."""

    def test_bounded_execution_integration(self) -> None:
        """Test full instrument execution workflow."""
        # Create registry
        registry = InstrumentRegistry()

        # Register camera with approval validator
        camera = CameraInstrument("lab-a")
        def check_approval(token: str) -> bool:
            return token == "secret-admin-key"
        registry.register(camera, approval_validator=check_approval)

        # Validate instrument
        validation = registry.list_available()
        self.assertEqual(len(validation), 1)
        self.assertEqual(validation[0].identity, "camera_lab-a")

        # Execute capture (no approval needed)
        capture = registry.execute_method("camera_lab-a", "capture", {})
        self.assertIn("image_id", capture)

        # Execute record (approval needed)
        record = registry.execute_method(
            "camera_lab-a", "record", {}, approval_token="secret-admin-key"
        )
        self.assertIn("recording_id", record)

        # Reject with bad approval
        with self.assertRaises(ValueError):
            registry.execute_method(
                "camera_lab-a", "record", {}, approval_token="wrong-key"
            )

    def test_code_validation_before_execution(self) -> None:
        """Test code validation before execution."""
        # Validate code first
        code = """
        result = registry.get_instrument("camera_default").execute_bounded("capture", {})
        """
        code_validation = validate_instrument_code(code)
        self.assertTrue(code_validation["valid"])

        # Then execute
        registry = InstrumentRegistry()
        camera = CameraInstrument()
        registry.register(camera)

        result = registry.execute_method("camera_default", "capture", {})
        self.assertIn("image_id", result)


if __name__ == "__main__":
    unittest.main()
