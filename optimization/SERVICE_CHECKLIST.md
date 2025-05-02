# Service Checklist

## Service: Optimization

- [ ] **Phase 1: Structure & Deduplication:**
  - [ ] Reviewed duplication with other services.
  - [ ] Identified code to move to `common-lib`.
  - [ ] Moved shared code to `common-lib`.
  - [ ] Deleted local duplicated code.
- [X] **Phase 2: Dependency Management:**
  - [X] `pyproject.toml` created/updated (using Poetry).
  - [X] Service-specific dependencies added.
  - [ ] `common-lib` added as path dependency.
  - [ ] `poetry.lock` generated/updated.
- [ ] **Phase 3: Documentation:**
  - [ ] `README.md` created/updated from template.
  - [ ] API documented in `API_DOCS.md` (if applicable).
  - [ ] Docstrings added to core functions/classes.
- [X] **Phase 4: Error Handling:**
  - [X] Custom exceptions from `common-lib` implemented.
  - [X] Error handling decorators implemented for consistent error handling.
- [X] **Phase 5: Security:**
  - [X] Hardcoded secrets removed (checked, none found).
  - [X] `.env.example` created/updated (N/A - no env vars required).
  - [X] `README.md` updated with required env vars (N/A - no env vars required).
  - [X] API Key/Auth dependency applied (N/A - library module).
  - [X] CORS settings reviewed/secured (N/A - library module).
- [ ] **Phase 6: Testing Basics:**
  - [X] `tests/` directory structure created.
  - [X] `pytest` and `pytest-cov` added as dev dependencies.
  - [X] Basic unit tests for core logic added.
