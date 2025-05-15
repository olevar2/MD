# Chat Service

This service handles chat functionalities within the forex trading platform, including message processing, history retrieval, and session management.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [API Endpoints](#api-endpoints)
- [Setup and Installation](#setup-and-installation)
- [Configuration](#configuration)
- [Running the Service](#running-the-service)
- [Running Tests](#running-tests)
- [Deployment](#deployment)
- [Contributing](#contributing)

## Features

- Real-time chat message processing.
- Persistent storage of chat history.
- User session management.
- API key-based authentication.
- Correlation ID for request tracing.
- Event-driven architecture for asynchronous tasks.

## Architecture

The service is built using FastAPI and follows a standard layered architecture:

- **API Layer (`app/api`)**: Handles HTTP requests, validation, and routing.
- **Service Layer (`app/services`)**: Contains business logic and orchestrates operations.
- **Repository Layer (`app/repositories`)**: Manages data access and interaction with the database.
- **Models (`app/models`)**: Defines database schemas (SQLAlchemy models).
- **Schemas (`app/schemas`)**: Defines data validation models (Pydantic models) for requests and responses.
- **Configuration (`app/config`)**: Manages application settings and environment variables.
- **Database (`app/database.py`)**: Handles database connections and session management.
- **Events (`app/events`)**: Implements the event bus for asynchronous communication (e.g., Kafka).
- **Middleware (`app/middleware`)**: Includes custom middleware for logging, correlation IDs, and authentication.
- **Dependencies (`app/dependencies.py`)**: Manages dependency injection for services, repositories, etc.
- **Exceptions (`app/exceptions.py`)**: Defines custom exceptions and handlers.

## API Endpoints

Base URL: `/api/v1`

- **POST `/chat/message`**: Send a new chat message.
  - Headers:
    - `X-API-Key`: Your API key.
    - `X-User-ID`: The ID of the user sending the message.
    - `X-Correlation-ID` (optional): For request tracing.
  - Body (JSON):
    ```json
    {
      "message": "Hello, how is the market today?",
      "context": { "symbol": "EURUSD" }
    }
    ```
- **GET `/chat/history`**: Retrieve chat history for a user.
  - Headers:
    - `X-API-Key`: Your API key.
    - `X-User-ID`: The ID of the user whose history is being requested.
  - Query Parameters:
    - `limit` (optional, default: 50): Number of messages to retrieve.
    - `before` (optional, ISO 8601 timestamp): Retrieve messages created before this timestamp.
- **GET `/health`**: Health check endpoint.

## Setup and Installation

1.  **Clone the repository (if applicable).**
2.  **Navigate to the `chat-service` directory:**
    ```bash
    cd path/to/forex_trading_platform/chat-service
    ```
3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Set up environment variables:**
    Copy the `.env.example` file to `.env` and update the values accordingly:
    ```bash
    cp .env.example .env
    ```
    Ensure you configure `DATABASE_URL`, `SECRET_KEY`, `KAFKA_BOOTSTRAP_SERVERS` (if using Kafka), and other necessary variables.

## Configuration

Configuration is managed through environment variables. Refer to the `.env.example` file for a list of available variables and their descriptions.

Key variables to configure:

- `SECRET_KEY`: A strong secret key for JWT and API key validation.
- `DATABASE_URL`: Connection string for the database.
- `REDIS_URL`: Connection string for Redis (if used for caching).
- `EVENT_BUS_TYPE`: Set to `kafka` or `in-memory`.
- `KAFKA_BOOTSTRAP_SERVERS`: Comma-separated list of Kafka brokers if `EVENT_BUS_TYPE` is `kafka`.

## Running the Service

To run the service locally for development:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The service will be available at `http://localhost:8000`.

## Running Tests

(Test setup and execution commands will be added here once tests are implemented.)

```bash
# Example (assuming pytest is configured)
# pytest
```

## Deployment

A `Dockerfile` is provided for containerizing the application.

1.  **Build the Docker image:**
    ```bash
    docker build -t chat-service .
    ```
2.  **Run the Docker container:**
    ```bash
    docker run -d -p 8000:8000 --env-file .env chat-service
    ```
    Ensure your `.env` file is correctly configured for the deployment environment.

## Contributing

(Contribution guidelines will be added here.)