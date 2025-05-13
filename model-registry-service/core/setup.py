"""
Installation and package information for the Model Registry Service.
"""
from setuptools import setup, find_packages

setup(
    name="model-registry-service",
    version="1.0.0",
    description="Service for managing ML model versioning and lifecycle",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.65.0",
        "uvicorn>=0.13.0",
        "python-multipart>=0.0.5",  # For file uploads
        "pydantic>=1.8.0",
        "joblib>=1.0.0",  # For model serialization
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-asyncio>=0.15.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "isort>=5.9.0",
            "mypy>=0.900",
        ]
    },
    python_requires=">=3.8",
)
