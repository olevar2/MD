<!-- filepath: d:\MD\forex_trading_platform\data-pipeline-service\README.md -->
# Service: data-pipeline-service

## Purpose

*(Placeholder: Describe the main goal and responsibilities of this service. What problem does it solve?)*

This service is responsible for ingesting, processing, and managing the flow of financial data required by other services in the platform.

## Structure

*(Placeholder: Briefly describe the main components or modules within this service and their roles.)*

-   **`data_pipeline_service/`**: Main application code.
    -   `__init__.py`
    -   `main.py`: Entry point (if applicable, e.g., for API).
    -   `pipeline/`: Core data processing logic.
    -   `sources/`: Data ingestion modules (e.g., Kafka consumers, API clients).
    -   `sinks/`: Data output modules (e.g., database writers, message queue producers).
    -   `config.py`: Service-specific configuration.
-   **`tests/`**: Unit and integration tests.
-   **`pyproject.toml`**: Project metadata and dependencies (Poetry).
-   **`README.md`**: This file.
-   **`SERVICE_CHECKLIST.md`**: Refactoring checklist for this service.

## Dependencies

-   `common-lib`: Shared utilities (database, config, exceptions).
-   *(List other key external libraries, e.g., Kafka client, database drivers)*

## Setup & Running

*(Instructions on how to set up the development environment and run the service)*

1.  **Install Dependencies:**
    ```powershell
    # Ensure Poetry is installed
    cd data-pipeline-service
    poetry install
    ```
2.  **Environment Variables:**
    See `.env.example` for all required environment variables. Key variables include:
    
    ```
    # Database Settings
    DB_USER=postgres
    DB_PASSWORD=your_db_password_here
    DB_HOST=localhost
    DB_PORT=5432
    DB_NAME=data_pipeline
    
    # Data Provider API Keys
    OANDA_API_KEY=your_oanda_api_key_here
    OANDA_ACCOUNT_ID=your_oanda_account_id_here
    
    # Object Storage (if enabled)
    USE_OBJECT_STORAGE=False
    OBJECT_STORAGE_KEY=your_object_storage_key_here
    OBJECT_STORAGE_SECRET=your_object_storage_secret_here
    ```
    
    Copy `.env.example` to `.env` and update with actual values for development.
3.  **Running the Service:**
    *(Command to start the service, e.g., `poetry run python -m data_pipeline_service.main` or specific pipeline execution commands)*

## API Documentation (if applicable)

*(Link to or brief description of the service's API, if it exposes one)*

## Error Handling

*(Briefly describe how errors are handled, referencing `common-lib.exceptions` if used)*

## Security

The service uses environment variables to store all sensitive information such as:
- Database credentials
- API keys for data providers (Oanda, etc.)
- Object storage credentials

All secrets are handled using Pydantic's SecretStr type to prevent accidental exposure in logs. Authentication mechanisms rely on `common-lib.security` implementations.
