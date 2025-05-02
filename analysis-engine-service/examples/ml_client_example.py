"""
ML Pipeline Client Example Usage

This example demonstrates how to use the MLPipelineClient for common ML operations.
"""

import asyncio
import logging
from typing import Dict, Any

from analysis_engine.clients.ml_pipeline_client import MLPipelineClient


# Set up basic logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def demo_ml_pipeline_client():
    """Demonstrate basic MLPipelineClient usage patterns."""
    
    # Initialize client
    client = MLPipelineClient()
    logger.info("MLPipelineClient initialized")

    # Example model ID for demonstration
    model_id = "forex_prediction_model_v1"
    
    try:
        # 1. List available models
        logger.info("Listing available models...")
        models = await client.list_models()
        logger.info(f"Found {len(models)} models")
        
        # 2. Get details about our specific model
        logger.info(f"Getting details for model: {model_id}")
        try:
            model_details = await client.get_model_details(model_id)
            logger.info(f"Model details: {model_details}")
        except Exception as e:
            logger.error(f"Model {model_id} not found: {e}")
            
        # 3. Request a model retraining with custom parameters
        retraining_params = {
            "dataset_version": "latest",
            "hyperparameters": {
                "epochs": 50,
                "learning_rate": 0.001,
                "batch_size": 64
            },
            "features": ["price_momentum", "volatility", "market_sentiment"]
        }
        
        logger.info(f"Starting retraining for model: {model_id}")
        job_id = await client.start_retraining_job(model_id, params=retraining_params)
        logger.info(f"Retraining job submitted with job ID: {job_id}")
        
        # 4. Poll job status
        max_polls = 5  # Limit polling for demo purposes
        polls = 0
        completed = False
        
        logger.info(f"Polling job status for job ID: {job_id}")
        while polls < max_polls and not completed:
            status_data = await client.get_job_status(job_id)
            status = status_data.get("status", "").lower()
            
            logger.info(f"Job status: {status}")
            
            if status in ["completed", "failed", "cancelled"]:
                completed = True
                logger.info(f"Job reached final status: {status}")
                if status == "completed":
                    # 5. Deploy the newly trained model
                    logger.info(f"Deploying model: {model_id}")
                    deployment = await client.deploy_model(
                        model_id, 
                        environment="staging",
                        config={"replicas": 2, "resources": "medium"}
                    )
                    logger.info(f"Deployment initiated: {deployment}")
                    
                    # 6. Make a prediction with the deployed model
                    features = {
                        "price_momentum": 0.75,
                        "volatility": 0.42,
                        "market_sentiment": 0.62,
                        "timestamp": "2025-04-29T12:00:00Z"
                    }
                    
                    prediction = await client.get_prediction(model_id, features)
                    logger.info(f"Prediction results: {prediction}")
            
            # Wait before polling again
            if not completed:
                polls += 1
                logger.info(f"Waiting 2 seconds before next poll ({polls}/{max_polls})...")
                await asyncio.sleep(2)
        
        if not completed:
            # 7. Demonstrate cancellation if job is still running
            logger.info(f"Demo complete but job still running. Cancelling job: {job_id}")
            cancel_result = await client.cancel_job(job_id)
            logger.info(f"Job cancellation result: {cancel_result}")
    
    except Exception as e:
        logger.error(f"Error during ML pipeline client demo: {e}")
        raise


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demo_ml_pipeline_client())
