# Scientific Verification

Verification is independent of model wording. A fluent answer does not pass
unless its supplied evidence and numerical checks pass.

## Architecture

```mermaid
flowchart LR
  Result[Agent or simulation result] --> Numeric[Finite values]
  Numeric --> Equation[Residual and formula checks]
  Equation --> Units[Unit vocabulary]
  Units --> Constraints[Physical constraints]
  Constraints --> Citations[Citation presence]
  Citations --> Consistency[Evidence consistency]
  Consistency --> Decision[Pass or needs review]
```

## How it works

The verifier checks finite numerical values, caller-supplied residuals,
restricted symbolic equations through SymPy, recognised engineering units,
and explicit constraints. A restricted standard-library algebra evaluator is
available for minimal installations. The safety layer adds citation
presence, evidence-term overlap, and numeric consistency checks. Every finding
has a status, message, severity, and evidence record. A failed check remains
visible in the UI and audit history.
