from fastapi import Request, HTTPException, Depends
from analysis_engine.causal.services.causal_inference_service import CausalInferenceService

def get_causal_inference_service(request: Request) -> CausalInferenceService:
    """Dependency function to get the CausalInferenceService instance."""
    try:
        # Resolve using the container stored in app state
        service = request.app.state.service_container.resolve(CausalInferenceService)
        if not service:
            raise HTTPException(status_code=503, detail="CausalInferenceService not available")
        return service
    except Exception as e:
        # Log the error for debugging
        # logger.error(f"Error resolving CausalInferenceService: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not resolve CausalInferenceService")
