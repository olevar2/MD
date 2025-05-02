# Schema Management Strategy

## Current Schema Duplication Issue

There are currently two sources of truth for data models in the platform:

1. **Local models** - Defined in `data_pipeline_service/models/schemas.py`
2. **Common library models** - Imported from `common_lib.schemas`

This duplication creates several issues:
- Import errors when running tests
- Potential for model drift between implementations
- Confusion about which model to use in new code

## Recommended Approach

### Short-term Solution

To immediately address test failures, we're using local schemas in the TimeseriesAggregator component.

### Long-term Solution (Recommended)

Migrate to using only common library schemas:

1. Remove duplicate model definitions from `data_pipeline_service/models/schemas.py`
2. Add common-lib as a proper dependency in pyproject.toml
3. Update all imports to use common_lib.schemas
4. Update tests to use common_lib.schemas

## Implementation Plan

1. **Dependency Management**:
   ```toml
   [tool.poetry.dependencies]
   common-lib = {path = "../common-lib"}
   ```

2. **Import Updates**:
   ```python
   # Change from
   from ..models.schemas import OHLCVData
   # to
   from common_lib.schemas import OHLCVData
   ```

3. **Model Deprecation**:
   Add deprecation notices to local models until all components are migrated.

4. **Testing**:
   Update tests to ensure common-lib is properly loaded in the test environment.

## Validation

After implementation, run the full test suite to verify there are no more import errors or schema conflicts.
