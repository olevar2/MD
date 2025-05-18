"""
Main module for starting the gRPC server.
"""

import asyncio
import logging
import os
import signal
from typing import Dict, Any, Optional

from causal_analysis.services.causal_graph_service import CausalGraphService
from causal_analysis.services.counterfactual_service import CounterfactualService
from causal_analysis.services.effect_estimation_service import EffectEstimationService
from causal_analysis.services.structure_discovery_service import StructureDiscoveryService
from causal_analysis.services.intervention_service import InterventionService

from causal_analysis.grpc_server.server import GrpcServer

logger = logging.getLogger(__name__)


class GrpcServerRunner:
    """
    Runner for the gRPC server.
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 50051,
        max_workers: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the gRPC server runner.
        
        Args:
            host: The host to bind the server to
            port: The port to bind the server to
            max_workers: The maximum number of workers for the server
            config: Optional configuration dictionary
        """
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.config = config or {}
        self.server = None
        self.logger = logger
    
    async def start(self):
        """
        Start the gRPC server.
        """
        # Create services
        causal_graph_service = CausalGraphService()
        counterfactual_service = CounterfactualService()
        effect_estimation_service = EffectEstimationService()
        structure_discovery_service = StructureDiscoveryService()
        intervention_service = InterventionService()
        
        # Create and start the gRPC server
        grpc_server = GrpcServer(
            host=self.host,
            port=self.port,
            max_workers=self.max_workers,
            config=self.config
        )
        
        self.server = await grpc_server.start(
            causal_graph_service=causal_graph_service,
            counterfactual_service=counterfactual_service,
            effect_estimation_service=effect_estimation_service,
            structure_discovery_service=structure_discovery_service,
            intervention_service=intervention_service
        )
        
        # Set up signal handlers
        self._setup_signal_handlers()
        
        # Wait for the server to terminate
        await self.server.wait_for_termination()
    
    async def stop(self, grace: float = 5.0):
        """
        Stop the gRPC server.
        
        Args:
            grace: Grace period in seconds for stopping the server
        """
        if self.server:
            self.logger.info(f"Stopping gRPC server with {grace}s grace period")
            await self.server.stop(grace)
            self.logger.info("gRPC server stopped")
    
    def _setup_signal_handlers(self):
        """
        Set up signal handlers for graceful shutdown.
        """
        loop = asyncio.get_event_loop()
        
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig,
                lambda: asyncio.create_task(self._shutdown())
            )
    
    async def _shutdown(self):
        """
        Shutdown the server gracefully.
        """
        self.logger.info("Received shutdown signal")
        await self.stop()


def main():
    """
    Main function to start the gRPC server.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get configuration from environment variables
    host = os.environ.get('GRPC_HOST', '0.0.0.0')
    port = int(os.environ.get('GRPC_PORT', '50051'))
    max_workers = int(os.environ.get('GRPC_MAX_WORKERS', '10'))
    
    # Create and start the server runner
    runner = GrpcServerRunner(
        host=host,
        port=port,
        max_workers=max_workers
    )
    
    # Run the server
    asyncio.run(runner.start())


if __name__ == '__main__':
    main()