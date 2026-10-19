from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import hashlib

root = Path.cwd()
output = root / 'OpenAgent_Batch2A_Implementation_2026-10-19.zip'
skip_dirs = {'node_modules', 'dist', 'test-results', 'playwright-report', '.git', '.venv', '__pycache__', '.qa', '.cache', '.vite', 'coverage', 'evidence', 'screenshots', 'audit'}
skip_suffixes = {'.zip', '.png', '.jpg', '.jpeg', '.webp', '.mp4', '.sqlite', '.sqlite3', '.db', '.pyc', '.pyo', '.pem', '.key', '.p12', '.pfx'}
skip_names = {'credentials.json', 'secrets.json'}
include_roots = [
    root / 'frontend-second',
    root / 'openagent',
    root / 'pyproject.toml',
    root / 'requirements.txt',
    root / 'requirements-dev.txt',
    root / 'tests' / 'test_agent_run_access.py',
    root / 'tests' / 'test_auth_policy.py',
    root / 'tests' / 'test_auth_enforcement.py',
    root / 'tests' / 'test_project_identity_contract.py',
    root / 'tests' / 'test_codebase_frameworks.py',
    root / 'tests' / 'test_codebase_agent.py',
    root / 'tests' / 'support' / 'real_integration_server.py',
]

def excluded(path: Path) -> bool:
    rel = path.relative_to(root)
    if any(part in skip_dirs or part.startswith('.tmp-') for part in rel.parts):
        return True
    name = path.name.lower()
    if name in skip_names or name.startswith('.env') or name.endswith(('.secret', '.token')):
        return True
    if path.suffix.lower() in skip_suffixes:
        return True
    if any(token in name for token in ('independent_audit', 'independent-audit', 'handoff_instruction', 'correction_instruction')):
        return True
    return False

files = set()
for item in include_roots:
    if not item.exists():
        continue
    if item.is_file():
        if not excluded(item): files.add(item)
        continue
    for path in item.rglob('*'):
        if path.is_file() and not excluded(path):
            files.add(path)

with ZipFile(output, 'w', ZIP_DEFLATED, compresslevel=7) as archive:
    for path in sorted(files):
        archive.write(path, path.relative_to(root).as_posix())

with ZipFile(output, 'r') as archive:
    bad = archive.testzip()
    names = archive.namelist()
    required = [
        'frontend-second/package.json',
        'frontend-second/src/App.tsx',
        'frontend-second/reference/frontend-final-approved.html',
        'frontend-second/FRONTEND_BACKEND_MASTER_GAPS.md',
        'frontend-second/BATCH2A_IMPLEMENTATION_RESULTS.md',
        'openagent/auth.py',
        'openagent/routes/agents_routes.py',
        'tests/support/real_integration_server.py',
    ]
    missing = [name for name in required if name not in names]
    forbidden = [name for name in names if any(part in skip_dirs for part in Path(name).parts) or name.lower().endswith(tuple(skip_suffixes)) or Path(name).name.lower().startswith('.env')]
    canonical = archive.read('frontend-second/reference/frontend-final-approved.html')
    canonical_hash = hashlib.sha256(canonical).hexdigest().upper()
    assert bad is None, f'corrupt entry: {bad}'
    assert not missing, f'missing required entries: {missing}'
    assert not forbidden, f'forbidden entries: {forbidden[:10]}'
    assert canonical_hash == 'BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1', canonical_hash

print(f'Created {output}')
print(f'Files: {len(files)}; size: {output.stat().st_size:,} bytes')
print(f'ZIP integrity: OK; required source entries: {len(required)}; forbidden entries: 0')
print(f'Canonical HTML SHA-256: {canonical_hash}')
