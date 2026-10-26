# Scientific Instrument Control Framework

## Overview

OpenAgent's Science module provides a bounded, validated framework for controlling physical instruments. This framework keeps proposal generation, code execution, and physical control strictly separate, with explicit safety checks at each layer.

## Architecture

### Three-Layer Separation

```
Layer 1: Proposal Generation (Bounded Python)
  └─> Code validation
  └─> Citation checking
  └─> Safety analysis

Layer 2: Scientific Code Execution (Isolated Subprocess)
  └─> Resource limits (memory, time)
  └─> Policy enforcement
  └─> Execution profiling

Layer 3: Physical Control (Instrument Registry)
  └─> Capability validation
  └─> Approval verification
  └─> Operation bounds enforcement
```

### Key Principles

1. **Proposal ≠ Execution**: Generated code is never automatically executed
2. **Execution ≠ Physical**: Execution happens in isolation; physical control is explicit
3. **Explicit Approval**: Safety-critical operations require explicit approval tokens
4. **Bounded Operations**: All physical device operations are method/parameter bounded
5. **Provenance**: All operations logged with execution history and metadata

## Instrument Model

### Base Instrument Interface

Every physical instrument implements the `Instrument` interface:

```python
from openagent.science.instruments import Instrument, InstrumentValidation

class MyInstrument(Instrument):
    @property
    def instrument_id(self) -> str:
        """Unique identifier."""
        return "my_device_lab1"
    
    @property
    def model(self) -> str:
        """Model identifier."""
        return "ModelXYZ"
    
    @property
    def serial_number(self) -> str | None:
        """Serial number if available."""
        return "SN-12345"
    
    def validate(self) -> InstrumentValidation:
        """Validate instrument is available and report capabilities."""
        # Check physical device status
        # Return capabilities, errors, warnings
        pass
    
    def execute_bounded(
        self,
        method: str,
        parameters: dict[str, Any],
        *,
        timeout_seconds: float = 30.0,
        approval_token: str | None = None,
    ) -> dict[str, Any]:
        """Execute bounded operation on instrument."""
        # Validate parameters
        # Check approval if required
        # Execute operation
        # Return result
        pass
```

### Capability Declaration

Each instrument declares its capabilities:

```python
from openagent.science.instruments import InstrumentCapability

capabilities = [
    InstrumentCapability(
        name="measure",
        description="Perform measurement",
        requires_approval=False,
        safety_critical=False,
    ),
    InstrumentCapability(
        name="calibrate",
        description="Calibrate instrument",
        requires_approval=True,  # Requires approval token
        required_scope="science:admin",  # Required user scope
        safety_critical=True,  # Safety-critical operation
    ),
]
```

### Validation Results

Instruments report detailed validation status:

```python
validation = instrument.validate()
# Returns InstrumentValidation with:
# - valid: bool (ready for operation)
# - status: InstrumentStatus (AVAILABLE, UNAVAILABLE, ERROR, NOT_SUPPORTED)
# - errors: list[str] (critical issues)
# - warnings: list[str] (non-critical alerts)
# - capabilities: list[InstrumentCapability] (available operations)
# - identity: str (unique ID)
# - model: str (model identifier)
# - serial_number: str | None (serial if available)
```

## Using the Instrument Registry

### Registration

```python
from openagent.science.instruments import (
    InstrumentRegistry,
    CameraInstrument,
    SpectrographInstrument,
)

registry = InstrumentRegistry()

# Register camera (non-approval operations)
camera = CameraInstrument()
registry.register(camera)

# Register spectrograph with approval validator
spec = SpectrographInstrument()
def validate_approval(token: str) -> bool:
    return token.startswith("approved-")
registry.register(spec, approval_validator=validate_approval)
```

### Listing Available Instruments

```python
# Get all available instruments
available = registry.list_available()
for inst in available:
    print(f"{inst.identity}: {inst.model}")
    for cap in inst.capabilities:
        approval_required = " (requires approval)" if cap.requires_approval else ""
        safety = " [SAFETY-CRITICAL]" if cap.safety_critical else ""
        print(f"  - {cap.name}: {cap.description}{approval_required}{safety}")
```

### Executing Operations

```python
# Simple operation (no approval)
result = registry.execute_method(
    "camera_default",
    "capture",
    {"resolution": "1920x1080"},
)

# Safety-critical operation (requires approval)
result = registry.execute_method(
    "spectrograph_lab1",
    "calibrate",
    {},
    approval_token="admin-secret-key",
)
```

## Code Validation

Before execution, validate that code uses only allowed instruments:

```python
from openagent.science.instruments import validate_instrument_code

code = """
result = registry.get_instrument("camera_default").execute_bounded("capture", {})
"""

validation = validate_instrument_code(code, allowed_instruments=["camera_default"])
if not validation["valid"]:
    for violation in validation["violations"]:
        print(f"ERROR: {violation}")
else:
    print(f"Instruments used: {validation['referenced_instruments']}")
```

### Code Safety Checks

Validation rejects dangerous patterns:
- `import os` - System module access
- `import subprocess` - Process spawning
- `__import__("os")` - Dynamic imports
- `open()` - File I/O

## Adding New Instruments

### Example: Temperature Controller

```python
from openagent.science.instruments import (
    Instrument,
    InstrumentCapability,
    InstrumentValidation,
    InstrumentStatus,
)

class TemperatureController(Instrument):
    def __init__(self, device_id: str = "default"):
        self.device_id = device_id
        self._min_temp = -50  # °C
        self._max_temp = 300  # °C
    
    @property
    def instrument_id(self) -> str:
        return f"tempcontroller_{self.device_id}"
    
    @property
    def model(self) -> str:
        return "TC-9000"
    
    @property
    def serial_number(self) -> str | None:
        return "TC-12345"
    
    def validate(self) -> InstrumentValidation:
        """Validate temperature controller availability."""
        errors = []
        warnings = []
        
        # Check device connection
        device_available = True  # Would check actual device
        
        capabilities = [
            InstrumentCapability(
                name="set_temperature",
                description="Set target temperature",
                requires_approval=False,
                safety_critical=True,  # Temperature control is safety-critical
            ),
            InstrumentCapability(
                name="set_ramp",
                description="Set temperature ramp rate",
                requires_approval=True,  # Requires approval
                required_scope="science:hardware",
                safety_critical=True,
            ),
        ]
        
        return InstrumentValidation(
            valid=device_available and not errors,
            status=InstrumentStatus.AVAILABLE if device_available else InstrumentStatus.UNAVAILABLE,
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
        parameters: dict[str, Any],
        *,
        timeout_seconds: float = 30.0,
        approval_token: str | None = None,
    ) -> dict[str, Any]:
        """Execute temperature control operation."""
        if method == "set_temperature":
            target_temp = float(parameters.get("temperature", 20))
            if target_temp < self._min_temp or target_temp > self._max_temp:
                raise ValueError(
                    f"Temperature must be between {self._min_temp} and {self._max_temp}°C"
                )
            return {"status": "set", "target": target_temp}
        
        elif method == "set_ramp":
            if not approval_token:
                raise ValueError("Ramp control requires approval")
            rate = float(parameters.get("rate_per_minute", 1))
            if rate < 0 or rate > 50:
                raise ValueError("Ramp rate must be 0-50°C/minute")
            return {"status": "ramp_configured", "rate": rate}
        
        else:
            raise ValueError(f"Unknown method: {method}")
```

## Safety-Critical Operations

### Marking Operations

Operations that directly control physical equipment should be marked `safety_critical=True`:

```python
InstrumentCapability(
    name="set_high_voltage",
    description="Apply high voltage",
    requires_approval=True,
    required_scope="science:admin",
    safety_critical=True,  # <-- Clearly marked
)
```

### Approval Workflow

```python
# 1. Code validation
code_validation = validate_instrument_code(code)
assert code_validation["valid"]

# 2. Instrument validation
validation = registry.list_available()
spec = next(v for v in validation if v.identity == "spectrograph_lab1")

# 3. Check if operation requires approval
calibrate = next(c for c in spec.capabilities if c.name == "calibrate")
assert calibrate.requires_approval  # True for safety-critical

# 4. User must explicitly approve
approval_token = get_admin_approval("Calibrate spectrograph?")

# 5. Execute with approval
result = registry.execute_method(
    "spectrograph_lab1",
    "calibrate",
    {},
    approval_token=approval_token,
)
```

## Bounded Execution

### Parameter Validation

Validate all parameters before sending to physical device:

```python
def execute_bounded(
    self,
    method: str,
    parameters: dict[str, Any],
    *,
    timeout_seconds: float = 30.0,
    approval_token: str | None = None,
) -> dict[str, Any]:
    """Bounded execution with parameter validation."""
    
    if method == "measure":
        # Validate wavelength range
        wavelength_min = float(parameters.get("wavelength_min", 400))
        wavelength_max = float(parameters.get("wavelength_max", 700))
        
        if wavelength_min < 200 or wavelength_max > 2000:
            raise ValueError("Wavelength out of supported range: 200-2000 nm")
        if wavelength_min >= wavelength_max:
            raise ValueError("wavelength_min must be less than wavelength_max")
        
        # Validate timeout
        timeout = float(parameters.get("timeout_seconds", 30))
        if timeout < 1 or timeout > 300:
            raise ValueError("Timeout must be 1-300 seconds")
        
        # Then execute
        return self._perform_measurement(wavelength_min, wavelength_max, timeout)
```

### Timeout Handling

```python
# Registry enforces timeout
result = registry.execute_method(
    "spectrograph_lab1",
    "measure",
    {"wavelength_min": 400, "wavelength_max": 700},
    # timeout_seconds defaults to 30, can be overridden
)
```

## Testing

```bash
# Run instrument validation tests
.venv\Scripts\python.exe -m unittest tests.test_g12_instrument_validation -v

# Run specific test
.venv\Scripts\python.exe -m unittest tests.test_g12_instrument_validation.InstrumentValidationTests -v

# Run with verbose output
.venv\Scripts\python.exe -m unittest tests.test_g12_instrument_validation -vv
```

## Integration with Science Workflows

### Full Workflow

```python
# 1. Proposal generation with bounds
proposal = science_agent.generate_proposal(
    query="Measure spectral absorption of sample",
    max_iterations=3,
    allow_physical_execution=False,  # Initially no hardware
)

# 2. Code validation
code_validation = validate_instrument_code(proposal.code)
assert code_validation["valid"]

# 3. Human review and approval
review = human_review(proposal, code_validation)

# 4. Register available instruments
registry = InstrumentRegistry()
registry.register(SpectrographInstrument("lab-1"))

# 5. Validate instruments available
available = registry.list_available()
assert len(available) > 0

# 6. Execute with approval
result = science_agent.execute_proposal(
    proposal,
    registry=registry,
    approval_token="review-approved-token",
)
```

## Future Enhancements

1. **Distributed Instrument Control**: Control instruments over network
2. **Instrument Logging**: Comprehensive operation and failure logs
3. **Calibration Management**: Automatic calibration scheduling
4. **Data Provenance**: Complete measurement history with metadata
5. **Multi-User Access**: Shared instrument queue and booking
6. **Failure Recovery**: Automated recovery from transient errors
7. **Performance Monitoring**: Track instrument performance metrics

## References

- [Science Module Architecture](../openagent/science/)
- [Bounded Code Execution](../openagent/science/execution.py)
- [Safety Checks](../openagent/science/safety.py)
- [Test Suite](../tests/test_g12_instrument_validation.py)
