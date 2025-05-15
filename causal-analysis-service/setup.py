"""
Setup script for the causal analysis service.
"""
from setuptools import setup, find_packages

setup(
    name="causal_analysis_service",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "numpy",
        "pandas",
        "pytest",
        "pytest-asyncio",
    ],
)