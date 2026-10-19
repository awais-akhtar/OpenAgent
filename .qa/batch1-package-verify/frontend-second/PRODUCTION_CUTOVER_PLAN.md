# Build and deployment boundary

No production cutover is authorized or performed as part of visual correction. The original Python-served `web/` frontend and the legacy `frontend/` source remain untouched by the frontend build.

Normal `frontend-second` builds use `dist/` and run `scripts/safe-build.mjs`, which checks that `../web/` is unchanged and that `public/demo/` fixtures were not copied into the output. The Vite development server serves the fixtures through a dev-only middleware so local demo workflows continue to work.

The root development entry already points at `frontend-second/`. Deploying a completed build into the Python-served UI requires a later explicit decision and is not implied by `npm run build` or `npm run build:web`.

Visual acceptance remains open until the exact approved HTML and matched fixture states are verified. Current results and raw screenshots are linked from [PIXEL_FIDELITY_GAPS.md](PIXEL_FIDELITY_GAPS.md). Historical image scores are excluded from current acceptance; only reports carrying the pinned canonical SHA and matching fixture/browser manifests are valid.
