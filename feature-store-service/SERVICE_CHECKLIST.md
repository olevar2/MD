# Service Checklist

## Service: Feature Store Service

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
  - [X] FastAPI error handlers registered (if applicable).
- [X] **Phase 5: Security:**
  - [X] Hardcoded secrets removed (checked, none found).
  - [X] `.env.example` created/updated.
  - [X] `README.md` updated with required env vars.
  - [X] API Key/Auth dependency applied (verified, uses API_KEY env var).
  - [X] CORS settings reviewed/secured (if applicable).
- [ ] **Phase 6: Testing Basics:**
  - [X] `tests/` directory structure created.
  - [X] `pytest` and `pytest-cov` added as dev dependencies.
  - [X] Basic unit tests for core logic added.
- [X] **Phase 7: Implementation Completion:**
  - [X] `TODO`/`FIXME`/Placeholders identified.
  - [X] Core missing logic implemented:
    - [X] Implemented Commodity Channel Index (CCI) with optimized calculation
    - [X] Implemented Williams %R with proper error handling
  - [X] Imports validated.
