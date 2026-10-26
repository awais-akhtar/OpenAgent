# Provider Configuration Guide

This guide explains how to configure and test external providers (LLM services, voice providers, calendar APIs, etc.) in OpenAgent.

## Overview

OpenAgent supports multiple external providers:
- **LLM Providers**: OpenAI, Anthropic, Google
- **Voice Providers**: Google Speech-to-Text, other TTS/STT services
- **Calendar Providers**: Google Calendar, Microsoft Exchange
- **Communication**: Email, messaging services

## Test Modes

Each provider supports three test modes:

### 1. Mock Mode (Default)
Uses synthetic responses for testing without external calls.
- No credentials required
- Fast test execution
- Suitable for CI/CD pipelines
- Used by default in test suite

### 2. Sandbox Mode
Uses test/sandbox credentials with real provider infrastructure.
- Requires valid test account
- Limited quotas
- Good for integration testing
- Set `*_TEST_MODE=sandbox` environment variable

### 3. Real Mode
Uses production credentials against live provider APIs.
- Requires valid API keys
- Production quotas apply
- Use with caution - can incur costs
- Set `*_TEST_MODE=real` environment variable

## Configuration

### OpenAI Provider

```bash
# Mock mode (default)
export OPENAI_TEST_MODE=mock

# Sandbox mode (requires valid test key)
export OPENAI_TEST_MODE=sandbox
export OPENAI_API_KEY=sk-...

# Real mode (requires production key)
export OPENAI_TEST_MODE=real
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.openai.com/v1  # Optional
```

### Anthropic Provider

```bash
# Mock mode (default)
export ANTHROPIC_TEST_MODE=mock

# Sandbox/Real mode
export ANTHROPIC_TEST_MODE=real
export ANTHROPIC_API_KEY=sk-ant-...
export ANTHROPIC_BASE_URL=https://api.anthropic.com/v1  # Optional
```

### Google Provider

```bash
# Mock mode (default)
export GOOGLE_TEST_MODE=mock

# Real mode
export GOOGLE_TEST_MODE=real
export GOOGLE_API_KEY=your-api-key

# OAuth refresh tokens
export GOOGLE_REFRESH_TOKEN=your-refresh-token
```

### Voice Providers

```bash
# Provider selection
export VOICE_PROVIDER=google  # or other provider
export VOICE_TEST_MODE=mock

# Google Speech-to-Text
export GOOGLE_VOICE_KEY=your-key
export GOOGLE_VOICE_BASE_URL=https://speech.googleapis.com

# Real hardware speech quality testing
export VOICE_HARDWARE_TEST=true
export VOICE_TEST_DEVICE=default  # audio input device
```

### Calendar Providers

```bash
# Google Calendar
export CALENDAR_PROVIDER=google
export CALENDAR_TEST_MODE=mock

# OAuth configuration
export CALENDAR_CLIENT_ID=your-client-id
export CALENDAR_CLIENT_SECRET=your-secret
export CALENDAR_REDIRECT_URI=http://localhost:8000/callback
```

## Running Provider Tests

### Mock Mode (Default)
```bash
# Run all provider tests with mock responses
.venv\Scripts\python.exe -m unittest tests.test_c3_external_providers -v
```

### With Specific Providers
```bash
# Test OpenAI provider
.venv\Scripts\python.exe -m unittest tests.test_c3_external_providers.C3ProviderFixtureTests.test_openai_fixture_mock_mode -v

# Test all fixture tests
.venv\Scripts\python.exe -m unittest tests.test_c3_external_providers.C3ProviderFixtureTests -v
```

### Sandbox/Real Mode Testing
```bash
# Test with sandbox credentials
export OPENAI_TEST_MODE=sandbox
export OPENAI_API_KEY=sk-...
.venv\Scripts\python.exe -m unittest tests.test_c3_external_providers -v
```

## Provider Features

### Authentication
- API key validation
- OAuth token refresh
- Token expiration handling
- Multi-provider chaining

### Error Handling
- Timeout recovery
- Rate limit detection and backoff
- Auth failure detection
- Network error handling
- Graceful degradation with fallbacks

### Security
- Secret redaction in logs
- API key masking
- Sensitive data filtering
- No credential persistence in test results

### Testing Scenarios

#### Auth Refresh
```python
from tests.fixtures.provider_fixtures import GoogleTestFixture

fixture = GoogleTestFixture()
result = fixture.handle_auth_refresh("refresh_token")
# Returns: {"access_token": "...", "expires_in": 3600, "refresh_token": "..."}
```

#### Error Handling
```python
# Test timeout scenario
error = fixture.handle_error("timeout")

# Test rate limiting
error = fixture.handle_error("rate_limit")

# Test authentication failure
error = fixture.handle_error("auth")
```

#### Secret Redaction
```python
sensitive = {"api_key": "sk-12345678", "refresh_token": "token123"}
redacted = fixture.redact_secrets(sensitive)
# Returns: {"api_key": "sk-5678", "refresh_token": "***"}
```

## Adding New Providers

Create a new fixture class:

```python
from tests.fixtures.provider_fixtures import ProviderFixture, ProviderTestConfig

class MyProviderTestFixture(ProviderFixture):
    def get_config(self) -> ProviderTestConfig:
        return ProviderTestConfig(
            provider_name="myprovider",
            test_mode=os.environ.get("MYPROVIDER_TEST_MODE", "mock"),
            api_key=os.environ.get("MYPROVIDER_API_KEY"),
            base_url=os.environ.get("MYPROVIDER_BASE_URL"),
            should_validate_creds=True,
            error_behavior=None,
        )
    
    def validate_credentials(self) -> bool:
        # Implement validation logic
        pass
    
    def create_completion(self, model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        # Implement response generation
        pass
    
    def handle_auth_refresh(self, refresh_token: str) -> dict[str, Any]:
        # Implement token refresh
        pass
    
    def handle_error(self, error_type: str) -> dict[str, Any]:
        # Implement error scenarios
        pass
    
    def redact_secrets(self, data: dict[str, Any]) -> dict[str, Any]:
        # Implement secret redaction
        pass
```

## CI/CD Integration

### GitHub Actions Example
```yaml
- name: Run provider tests
  env:
    OPENAI_TEST_MODE: mock
    ANTHROPIC_TEST_MODE: mock
    GOOGLE_TEST_MODE: mock
    VOICE_TEST_MODE: mock
  run: |
    .venv\Scripts\python.exe -m unittest tests.test_c3_external_providers -v
```

### Important Notes
- Always use `mock` mode in CI/CD pipelines
- Never commit real API keys
- Sandbox/real mode testing requires explicit configuration
- Test cost implications for expensive providers
- Set up credentials in secure secrets management

## Troubleshooting

### Import Error: fixtures.provider_fixtures
```bash
# Ensure tests directory has __init__.py
touch tests/__init__.py
touch tests/fixtures/__init__.py
```

### Provider Validation Fails
- Check API key format for the provider
- Verify environment variable names (case-sensitive)
- Ensure test mode is set correctly
- Check network connectivity for sandbox/real modes

### Rate Limit Errors
- Implement exponential backoff
- Use fixture error simulation: `error_behavior="rate_limit"`
- Test with reduced request frequency

### Auth Token Expired
- Use `handle_auth_refresh()` to refresh OAuth tokens
- Implement token rotation in provider handlers
- Test refresh scenarios with fixtures

## Best Practices

1. **Use Mock Mode for Development**
   - Fast feedback loops
   - No external dependencies
   - No cost implications

2. **Test Auth Refresh Scenarios**
   - Always test token refresh flows
   - Verify refresh token rotation
   - Handle expired token recovery

3. **Secret Management**
   - Never log API keys
   - Use `redact_secrets()` before logging
   - Clear sensitive data from test results

4. **Error Simulation**
   - Test all error scenarios
   - Verify fallback handling
   - Ensure graceful degradation

5. **Provider Certification**
   - Test with real provider if possible
   - Verify response structures match provider docs
   - Update fixtures when provider changes

## References

- Provider APIs:
  - [OpenAI API](https://platform.openai.com/docs)
  - [Anthropic API](https://docs.anthropic.com)
  - [Google AI API](https://ai.google.dev)
- Test fixtures: `tests/fixtures/provider_fixtures.py`
- Test suite: `tests/test_c3_external_providers.py`
