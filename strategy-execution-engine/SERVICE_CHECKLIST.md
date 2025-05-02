# Service Checklist

## Service: Strategy Execution Engine

- [ ] **Phase 1: Structure & Deduplication:**
  - [ ] Reviewed duplication with other services.
  - [ ] Identified code to move to `common-lib`.
  - [ ] Moved shared code to `common-lib`.
  - [ ] Deleted local duplicated code.
- [X] **Phase 2: Dependency Management:**
  - [X] `pyproject.toml` created/updated (using Poetry).
  - [X] Service-specific dependencies added.
  - [X] `common-lib` added as path dependency.
  - [X] `poetry.lock` generated/updated.
- [ ] **Phase 3: Documentation:**
  - [X] `README.md` created/updated from template.
  - [ ] API documented in `API_DOCS.md` (if applicable).
  - [ ] Docstrings added to core functions/classes.
- [X] **Phase 4: Error Handling:**
  - [X] Custom exceptions from `common-lib` implemented.
  - [X] Error handling decorators implemented for consistent error handling.
- [X] **Phase 5: Security:**
  - [X] Hardcoded secrets removed (using env vars).
  - [X] `.env.example` created/updated.
  - [X] `README.md` updated with required env vars.
  - [X] API Key/Auth dependency applied (if applicable).
  - [X] CORS settings reviewed/secured (if applicable).
- [ ] **Phase 6: Testing Basics:**
  - [ ] `tests/` directory structure created.
  - [ ] `pytest` and `pytest-cov` added as dev dependencies.
  - [ ] Basic unit tests for core logic added.
- [ ] **Phase 7: Implementation Completion:**
  - [ ] `TODO`/`FIXME`/Placeholders identified.
  - [ ] Core missing logic implemented.
  - [ ] Imports validated.
- [ ] **Phase 8: Integration Improvement:**
  - [ ] Direct inter-service imports reviewed/refactored (if applicable).
  - [ ] Resilience patterns (Retry/Circuit Breaker) added (if applicable).
