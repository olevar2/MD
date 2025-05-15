# d:\MD\forex_trading_platform\backtesting-service\main.py
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.utils.logger import get_logger
from app.utils.correlation_id import CorrelationIdMiddleware
from app.config.settings import settings # Assuming settings.py will be created

logger = get_logger(__name__)

app = FastAPI(
    title="Backtesting Service",
    description="Service for running and managing trading strategy backtests.",
    version="0.1.0",
    openapi_url=f"{settings.API_PREFIX}/openapi.json",
    docs_url=f"{settings.API_PREFIX}/docs",
    redoc_url=f"{settings.API_PREFIX}/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS if settings.ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(CorrelationIdMiddleware)

# Routers
app.include_router(api_router, prefix=settings.API_PREFIX)

@app.on_event("startup")
async def startup_event():
    logger.info("Backtesting Service starting up...")
    # Initialize any resources here if needed
    # e.g., database connections, external service clients

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Backtesting Service shutting down...")
    # Clean up resources here if needed

@app.get(f"{settings.API_PREFIX}/health", tags=["Health"])
async def health_check():
    logger.info("Health check endpoint called.")
    return {"status": "healthy", "service": "Backtesting Service"}

if __name__ == "__main__":
    logger.info(f"Starting Uvicorn server on host {settings.APP_HOST} and port {settings.APP_PORT}")
    # Ensure settings are loaded before uvicorn.run
    # For example, by importing them or ensuring they are set as environment variables
    # that settings.py can read.
    uvicorn.run(app, host=settings.APP_HOST, port=settings.APP_PORT)