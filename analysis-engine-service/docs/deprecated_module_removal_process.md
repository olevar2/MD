# Deprecated Module Removal Process

This document outlines the process for removing deprecated modules from the Analysis Engine Service codebase after the deprecation period ends on December 31, 2023.

## Timeline

1. **Monitoring Phase (Now - December 31, 2023)**
   - Track usage of deprecated modules
   - Send regular reminders to developers
   - Provide migration assistance

2. **Final Assessment (January 1-7, 2024)**
   - Generate final usage report
   - Identify any remaining usages
   - Create tickets for any remaining migration work

3. **Removal Phase (January 8-15, 2024)**
   - Create backups of deprecated modules
   - Remove deprecated modules
   - Run comprehensive tests
   - Deploy changes

4. **Verification Phase (January 16-31, 2024)**
   - Monitor for any issues
   - Address any problems that arise
   - Update documentation

## Modules to be Removed

The following modules are scheduled for removal:

1. **`analysis_engine.core.config`**
   - **Replacement:** `analysis_engine.config`
   - **Migration Guide:** [Configuration Migration Guide](./configuration_migration_guide.md)
   - **Migration Tool:** `tools/migrate_config_imports.py`

2. **`config.config`**
   - **Replacement:** `analysis_engine.config`
   - **Migration Guide:** [Configuration Migration Guide](./configuration_migration_guide.md)
   - **Migration Tool:** `tools/migrate_config_imports.py`

3. **`analysis_engine.api.router`**
   - **Replacement:** `analysis_engine.api.routes`
   - **Migration Guide:** [API Router Migration Guide](./api_router_migration_guide.md)
   - **Migration Tool:** `tools/migrate_router_imports.py`

## Monitoring Tools

The following tools are available to monitor usage of deprecated modules:

1. **Deprecation Dashboard**
   ```bash
   python tools/deprecation_dashboard.py
   ```
   Generates an HTML dashboard showing usage of deprecated modules, including charts and detailed information about usage locations.

2. **Scheduled Reminders**
   ```bash
   python tools/scheduled_deprecation_reminder.py --email --slack
   ```
   Sends reminders about deprecated modules via email and Slack. This can be scheduled to run weekly using a cron job or scheduled task.

3. **Usage Report**
   ```bash
   python -c "from analysis_engine.core.deprecation_monitor import get_usage_report, save_data; print(get_usage_report()); save_data()"
   ```
   Generates a report of deprecated module usage and saves it to a file.

## Removal Process

### 1. Final Assessment

Before removing deprecated modules, perform a final assessment to ensure all usages have been migrated:

```bash
python tools/prepare_module_removal.py --check-only
```

If any usages are found, they must be migrated before proceeding with removal.

### 2. Create Backups

Create backups of the deprecated modules before removing them:

```bash
python tools/prepare_module_removal.py --backup
```

This will create a timestamped backup of all deprecated modules in the `backups` directory.

### 3. Generate Removal Plan

Generate a detailed removal plan with impact assessment:

```bash
python tools/prepare_module_removal.py --plan
```

This will create a `module_removal_plan.md` document in the `docs` directory with detailed information about the removal process.

### 4. Remove Modules

Remove the deprecated modules:

1. Remove `analysis_engine/core/config.py`
2. Remove `config/config.py`
3. Remove `analysis_engine/api/router.py`

### 5. Run Tests

Run all tests to ensure the system still functions correctly:

```bash
poetry run pytest
```

### 6. Update Documentation

Update documentation to remove references to deprecated modules:

1. Update `README.md` to remove migration notices
2. Archive migration guides
3. Update API documentation

## Rollback Plan

If issues are encountered after removal:

1. Restore modules from backups
2. Run tests to verify functionality
3. Create new migration tickets for any issues discovered

## Communication Plan

### Before Removal

1. Send final reminder email one week before removal
2. Post announcement in Slack channels
3. Update JIRA tickets with removal timeline

### During Removal

1. Send notification when removal begins
2. Provide status updates during the process
3. Notify when removal is complete

### After Removal

1. Send confirmation email when removal is complete
2. Provide summary of changes
3. Share contact information for reporting issues

## Contact Information

If you encounter any issues during or after the removal process, please contact:

- **Analysis Engine Team:** analysis-engine@example.com
- **Support Slack Channel:** #analysis-engine-support
- **JIRA Project:** AES
