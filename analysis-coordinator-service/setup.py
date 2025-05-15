from setuptools import setup, find_packages

setup(
    name="analysis-coordinator-service",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.95.0",
        "uvicorn==0.21.1",
        "pydantic==1.10.7",
        "aiohttp==3.8.4",
        "python-dotenv==1.0.0",
    ],
    author="Forex Trading Platform Team",
    author_email="team@forextradingplatform.com",
    description="Service for coordinating analysis tasks across multiple analysis services",
    keywords="forex, trading, analysis, coordination",
    python_requires=">=3.8",
)