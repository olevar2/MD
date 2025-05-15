import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from analysis_coordinator_service.api.v1.coordinator import router as coordinator_router
from analysis_coordinator_service.config.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    settings = get_settings()
    
    app = FastAPI(
        title="Analysis Coordinator Service",
        description="Service for coordinating analysis tasks across multiple analysis services",
        version="1.0.0",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(coordinator_router, prefix=settings.api_prefix)
    
    @app.get("/health")
    async def health_check():
        """
        Health check endpoint.
        """
        return {"status": "ok"}
        
    @app.get("/ready")
    async def readiness_check():
        """
        Readiness check endpoint.
        """
        return {"status": "ready"}
        
    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)