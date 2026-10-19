# Served web resources

This directory contains stable resources served by the Python application and
legacy web source retained by the current packaging contract. Generated hashed
bundles, demo output, and build-only CSS are ignored by Git.

Do not treat `web/` as a third canonical frontend. New UI product development
belongs in [`frontend-second/`](../frontend-second/); build tooling may publish
its output here without making generated files source-of-truth.
