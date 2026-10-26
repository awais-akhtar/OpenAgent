"""Instrument validation and safety framework for physical device control.

Supports bounded, verified execution of code targeting physical instruments
with explicit safety checks, provenance validation, and resource limits.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Sequence


class InstrumentStatus(str, Enum):
    """Status of physical instrument availability."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    NOT_SUPPORTED = "not_supported"


@dataclass(frozen=True)
class InstrumentCapability:
    """Describes a capability of a physical instrument."""

    name: str
    description: str
    requires_approval: bool = False
    required_scope: str | None = None
    safety_critical: bool = False


@dataclass(frozen=True)
class InstrumentValidation:
    """Result of instrument validation."""

    valid: bool
    status: InstrumentStatus
    errors: list[str]
    warnings: list[str]
    capabilities: list[InstrumentCapability]
    identity: str  # Unique instrument identifier
    model: str
    serial_number: str | None = None


class Instrument(ABC):
    """Base class for physical instruments."""

    @property
    @abstractmethod
    def instrument_id(self) -> str:
        """Unique identifier for this instrument."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier for this instrument."""

    @property
    @abstractmethod
    def serial_number(self) -> str | None:
        """Serial number or None if not available."""

    @abstractmethod
    def validate(self) -> InstrumentValidation:
        """Validate instrument availability and capabilities."""

    @abstractmethod
    def execute_bounded(
        self,
        method: str,
        parameters: Mapping[str, Any],
        *,
        timeout_seconds: float = 30.0,
        approval_token: str | None = None,
    ) -> dict[str, Any]:
        """Execute bounded operation on instrument.

        Args:
            method: Method name on instrument
            parameters: Method parameters
            timeout_seconds: Maximum execution time
            approval_token: Approval token for restricted operations

        Returns:
            Operation result

        Raises:
            ValueError: Invalid method, parameters, or missing approval
            TimeoutError: Operation exceeded timeout
            RuntimeError: Instrument error or unavailable
        """


class CameraInstrument(Instrument):
    """Example camera instrument with bounded control."""

    def __init__(self, device_id: str = "default") -> None:
        self.device_id = device_id
        self._available = False
        self._model = "MockCamera"
        self._serial = None

    @property
    def instrument_id(self) -> str:
        return f"camera_{self.device_id}"

    @property
    def model(self) -> str:
        return self._model

    @property
    def serial_number(self) -> str | None:
        return self._serial

    def validate(self) -> InstrumentValidation:
        """Validate camera availability."""
        errors: list[str] = []
        warnings: list[str] = []

        # Check device accessibility (would be real check in production)
        self._available = True  # Mock: assume available

        capabilities = [
            InstrumentCapability(
                name="capture",
                description="Capture single image",
                requires_approval=False,
                safety_critical=False,
            ),
            InstrumentCapability(
                name="record",
                description="Record video stream",
                requires_approval=True,
                required_scope="science:hardware",
                safety_critical=False,
            ),
        ]

        return InstrumentValidation(
            valid=self._available and not errors,
            status=InstrumentStatus.AVAILABLE if self._available else InstrumentStatus.UNAVAILABLE,
            errors=errors,
            warnings=warnings,
            capabilities=capabilities,
            identity=self.instrument_id,
            model=self.model,
            serial_number=self.serial_number,
        )

    def execute_bounded(
        self,
        method: str,
        parameters: Mapping[str, Any],
        *,
        timeout_seconds: float = 30.0,
        approval_token: str | None = None,
    ) -> dict[str, Any]:
        """Execute bounded camera operation."""
        validation = self.validate()
        if not validation.valid:
            raise RuntimeError(f"Instrument validation failed: {'; '.join(validation.errors)}")

        if method == "capture":
            return {"image_id": f"capture-{hash(str(parameters))}", "timestamp": None}
        elif method == "record":
            if not approval_token:
                raise ValueError("Recording requires explicit approval_token")
            return {"recording_id": f"record-{hash(approval_token)}", "duration_seconds": 0}
        else:
            raise ValueError(f"Unknown method: {method}")


class SpectrographInstrument(Instrument):
    """Example spectrograph instrument with measurement bounds."""

    def __init__(self, device_id: str = "default") -> None:
        self.device_id = device_id
        self._available = False
        self._model = "MockSpectrograph"
        self._serial = "SPEC-001"

    @property
    def instrument_id(self) -> str:
        return f"spectrograph_{self.device_id}"

    @property
    def model(self) -> str:
        return self._model

    @property
    def serial_number(self) -> str | None:
        return self._serial

    def validate(self) -> InstrumentValidation:
        """Validate spectrograph availability."""
        errors: list[str] = []
        warnings: list[str] = []

        self._available = True  # Mock: assume available

        capabilities = [
            InstrumentCapability(
                name="measure",
                description="Measure spectrum",
                requires_approval=False,
                safety_critical=True,
            ),
            InstrumentCapability(
                name="calibrate",
                description="Calibrate instrument",
                requires_approval=True,
                required_scope="science:admin",
                safety_critical=True,
            ),
        ]

        return InstrumentValidation(
            valid=self._available and not errors,
            status=InstrumentStatus.AVAILABLE if self._available else InstrumentStatus.UNAVAILABLE,
            errors=errors,
            warnings=warnings,
            capabilities=capabilities,
            identity=self.instrument_id,
            model=self.model,
            serial_number=self.serial_number,
        )

    def execute_bounded(
        self,
        method: str,
        parameters: Mapping[str, Any],
        *,
        timeout_seconds: float = 30.0,
        approval_token: str | None = None,
    ) -> dict[str, Any]:
        """Execute bounded spectrograph operation."""
        validation = self.validate()
        if not validation.valid:
            raise RuntimeError(f"Instrument validation failed: {'; '.join(validation.errors)}")

        # Validate measurement parameters
        wavelength_min = float(parameters.get("wavelength_min", 400))
        wavelength_max = float(parameters.get("wavelength_max", 700))

        if wavelength_min < 200 or wavelength_max > 2000:
            raise ValueError("Wavelength out of supported range: 200-2000 nm")
        if wavelength_min >= wavelength_max:
            raise ValueError("wavelength_min must be less than wavelength_max")

        if method == "measure":
            return {"spectrum_id": f"spectrum-{hash(str(parameters))}", "wavelengths": None}
        elif method == "calibrate":
            if not approval_token:
                raise ValueError("Calibration requires explicit approval_token")
            return {"calibration_id": f"cal-{hash(approval_token)}", "success": True}
        else:
            raise ValueError(f"Unknown method: {method}")


class InstrumentRegistry:
    """Registry of available instruments with validation and safety checks."""

    def __init__(self) -> None:
        self._instruments: dict[str, Instrument] = {}
        self._approval_validators: dict[str, Callable[[str], bool]] = {}

    def register(self, instrument: Instrument, approval_validator: Callable[[str], bool] | None = None) -> None:
        """Register an instrument."""
        self._instruments[instrument.instrument_id] = instrument
        if approval_validator:
            self._approval_validators[instrument.instrument_id] = approval_validator

    def get_instrument(self, instrument_id: str) -> Instrument | None:
        """Get registered instrument."""
        return self._instruments.get(instrument_id)

    def list_available(self) -> list[InstrumentValidation]:
        """List all available instruments."""
        results = []
        for instrument in self._instruments.values():
            validation = instrument.validate()
            if validation.valid:
                results.append(validation)
        return results

    def execute_method(
        self,
        instrument_id: str,
        method: str,
        parameters: Mapping[str, Any],
        *,
        approval_token: str | None = None,
    ) -> dict[str, Any]:
        """Execute method on registered instrument."""
        instrument = self.get_instrument(instrument_id)
        if not instrument:
            raise ValueError(f"Unknown instrument: {instrument_id}")

        validation = instrument.validate()
        if not validation.valid:
            raise RuntimeError(f"Instrument not available: {'; '.join(validation.errors)}")

        # Check method in capabilities
        capability = next(
            (cap for cap in validation.capabilities if cap.name == method),
            None,
        )
        if not capability:
            raise ValueError(f"Instrument does not support method: {method}")

        # Check approval for restricted operations
        if capability.requires_approval:
            if not approval_token:
                raise ValueError(f"Method '{method}' requires explicit approval")
            validator = self._approval_validators.get(instrument_id)
            if validator and not validator(approval_token):
                raise ValueError("Invalid or expired approval token")

        return instrument.execute_bounded(
            method,
            parameters,
            approval_token=approval_token,
        )


def validate_instrument_code(
    code: str,
    allowed_instruments: Sequence[str] | None = None,
) -> dict[str, object]:
    """Validate that code references only allowed instruments.

    Args:
        code: Python source code to validate
        allowed_instruments: List of allowed instrument IDs (None = all allowed)

    Returns:
        Validation result with any violations found
    """
    violations: list[str] = []

    # Check for dangerous patterns
    dangerous_patterns = [
        (r"import\s+os\b", "os module import not allowed"),
        (r"import\s+subprocess\b", "subprocess module import not allowed"),
        (r"__import__\s*\(\s*['\"]os", "Dynamic os import not allowed"),
        (r"open\s*\(", "File I/O not allowed"),
    ]

    for pattern, message in dangerous_patterns:
        if re.search(pattern, code):
            violations.append(message)

    # Check instrument references
    instrument_pattern = r"registry\.get_instrument\s*\(\s*['\"]([^'\"]+)['\"]"
    referenced_instruments = set(re.findall(instrument_pattern, code))

    if allowed_instruments is not None:
        allowed_set = set(allowed_instruments)
        unauthorized = referenced_instruments - allowed_set
        if unauthorized:
            violations.append(f"Unauthorized instrument references: {', '.join(sorted(unauthorized))}")

    return {
        "valid": not violations,
        "violations": violations,
        "referenced_instruments": sorted(referenced_instruments),
    }
