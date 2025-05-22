# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install Poetry
RUN pip install poetry==1.7.1 # Pinning version for consistency

# Copy only files necessary for dependency installation to leverage Docker cache
COPY pyproject.toml poetry.lock ./

# Install project dependencies
# --no-root: Do not install the project itself as editable, it will be copied later
# --no-dev: Do not install development dependencies
RUN poetry install --no-root --no-dev

# Copy the rest of the application code
# This includes the main application, protos, and generated_protos
COPY . .

# The settings file (analysis_engine/config/settings.py) defines:
# PORT: 8000 (FastAPI)
# GRPC_PORT: 50051 (gRPC)
EXPOSE 8000
EXPOSE 50051

# Command to run the application
# The application entry point is analysis-engine-service/core/main.py
# Ensure this path is correct relative to the WORKDIR /app
CMD ["python", "core/main.py"]
