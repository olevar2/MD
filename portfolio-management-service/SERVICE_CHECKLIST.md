# Service Checklist

## S- [ ] **Phase 6: Testing Basics:**
  - [X] `tests/` directory structure created.
  - [X] `pytest` and `pytest-cov` added as dev dependencies.
  - [ ] Basic unit tests for core logic added.ce: Portfolio Management Service

- [X] **Phase 1: Structure & Deduplication:**
  - [X] Reviewed duplication with other services.
  - [X] Identified code to move to `common-lib`.
  - [X] Moved shared code to `common-lib`.
  - [X] Deleted local duplicated code.
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
  - [X] Hardcoded secrets removed (using env vars).
  - [X] `.env.example` created/updated.
  - [X] `README.md` updated with required env vars.
  - [X] API Key/Auth dependency applied (if applicable).
  - [X] CORS settings reviewed/secured (if applicable).
- [ ] **Phase 6: Testing Basics:**
  - [X] `tests/` directory structure created.
  - [X] `pytest` and `pytest-cov` added as dev dependencies.
  - [ ] Basic unit tests for core logic added.
