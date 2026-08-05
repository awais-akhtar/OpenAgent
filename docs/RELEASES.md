# Release Policy

OpenAgent publishes a release only from a verified repository snapshot.

## Required Checks

1. Run the complete Python test suite.
2. Run Python compilation and JavaScript syntax checks.
3. Run the executable hard audit.
4. Confirm the feature status and privacy-boundary documentation match the code.
5. Confirm local data, model weights, credentials, runtime files, and private working material are excluded.
6. Build the source and wheel artifacts from a clean tree.
7. Inspect the generated artifacts before publishing.

## Versioning

OpenAgent uses semantic versioning. A release tag is created only after the current snapshot passes the required checks. Release notes describe the shipped behavior, migration requirements, known limitations, and security-relevant changes.

## Package Policy

Packages are built from committed source. Credentials are supplied by the release environment and are never stored in the repository. Published artifacts must include the MIT license, package metadata, public documentation, and the runtime assets declared in `MANIFEST.in` and `pyproject.toml`.

## Verification Commands

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.\.venv\Scripts\python.exe -m compileall -q openagent
node --check web\app.js
.\.venv\Scripts\python.exe -m openagent.hard_audit
python -m build
```
