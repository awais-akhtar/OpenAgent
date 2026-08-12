# Engineering Optimisation

Engineering Optimisation turns requirements into variables, objectives,
constraints, candidate evaluations, and a verified result.

## Architecture

```mermaid
flowchart LR
  Requirement[Engineering requirement] --> Variables[Variables and bounds]
  Variables --> Physics[Physics evaluator]
  Physics --> Search[Random, Bayesian, differential evolution, or minimize]
  Search --> Constraints[Stress, deflection, thermal and pressure checks]
  Constraints -->|pass| Ranking[Feasible ranking]
  Constraints -->|fail| Search
  Ranking --> Report[Profiled engineering report]
```

## How it works

The bundled beam example uses a five-stage plan and minimises mass while
checking stress and deflection. The bridge workflow performs an equivalent-
girder screening optimisation and labels the checks that still require a
code-certified structural model.
The heat-exchanger workflow writes and executes a reproducible Python search,
records runtime and memory use, checks effectiveness and pressure-drop limits,
and reports the selected candidate. Bridge requests run the screening workflow
and use project scientific memory for density, Young's modulus, span, load,
stress limit, and deflection limit when those records are available. Missing
values use visible screening defaults; the report does not present the result
as a certified bridge design.

`EngineeringOptimizer` supports deterministic random search, native Bayesian
optimisation, and optional SciPy `differential_evolution` and `minimize`
methods. The Bayesian method fits a small RBF Gaussian process to normalized
design coordinates and chooses subsequent evaluations by expected improvement.
It begins with a seeded Latin-hypercube design, then balances candidates near
the current best design with candidates spanning the full search space.

The Bayesian backend uses the Python standard library, so installing SciPy or
NumPy is not required. Results identify the backend as
`native-rbf-gaussian-process` and include the acquisition policy, initial and
adaptive evaluation counts, improvement over the initial design, candidate-
pool size, and numerical fallback count.
A covariance failure falls back only for the affected acquisition step and is
reported through `fallback_used` and `fallback_evaluations`.

The implementation is intended for bounded, low-dimensional engineering
searches. It is deterministic for a fixed seed and budget, but it is not a
replacement for a calibrated probabilistic optimisation package on noisy or
high-dimensional industrial problems. Benchmark responses report the actual
method, objective, feasibility, and evaluation count; a dependency fallback is
never renamed as the requested optimizer.
