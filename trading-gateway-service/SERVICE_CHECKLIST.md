# Service Checklist

## Service: Trading Gateway Service

- [ ] **Phase 1: Structure & Deduplication:**
  - [ ] Reviewed duplication with other services.
  - [ ] Identified code to move to `common-lib` and `common-js-lib`.
  - [ ] Moved shared code to shared libraries.
  - [ ] Deleted local duplicated code.
- [X] **Phase 2: Dependency Management:**
  - [X] `pyproject.toml` created/updated (using Poetry) for Python components.
  - [X] `package.json` maintained for Node.js components.
  - [X] Service-specific dependencies added.
  - [X] Shared libraries added as dependencies.
  - [X] Lock files generated/updated.
- [ ] **Phase 3: Documentation:**
  - [X] `README.md` created/updated from template.
  - [ ] API documented in `API_DOCS.md` (if applicable).
  - [ ] Docstrings/JSDoc added to core functions/classes.
- [X] **Phase 4: Error Handling:**
  - [X] Custom exceptions from shared libraries implemented.
  - [X] Error handlers registered (if applicable).
- [X] **Phase 5: Security:**
  - [X] Hardcoded secrets removed (using env vars).
  - [X] `.env.example` created/updated.
  - [X] `README.md` updated with required env vars.
  - [X] API Key/Auth dependency applied (if applicable).
  - [X] CORS settings reviewed/secured (if applicable).
- [ ] **Phase 6: Testing Basics:**
  - [X] `tests/` directory structure created.
  - [X] Testing libraries added as dev dependencies.
  - [X] Basic unit tests for core logic added.
- [X] **Phase 7: Implementation Completion:**
  - [X] `TODO`/`FIXME`/Placeholders identified.
  - [X] Core missing logic implemented.
  - [X] Imports validated.
- [ ] **Phase 8: Integration Improvement:**
  - [ ] Direct inter-service imports reviewed/refactored (if applicable).
  - [ ] Resilience patterns (Retry/Circuit Breaker) added (if applicable).
