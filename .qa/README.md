# Local QA workspace

`.qa/` is for generated local QA runs and small, intentionally reviewed
evidence fixtures. It is not a second source tree.

Do not commit package staging/verification copies, dependency directories,
runtime state, or complete frontend/backend snapshots here. Preserve bulky QA
evidence as release artifacts and track only a manifest containing source
revision, artifact hash, environment, and reproduction command.
