"""
gRPC server implementation for the Causal Analysis Service.
"""

import asyncio
import logging
import grpc
from concurrent import futures
from typing import Optional, Dict, Any

from common_lib.grpc.causal_analysis import causal_analysis_service_pb2 as pb2
from common_lib.grpc.causal_analysis import causal_analysis_service_pb2_grpc as pb2_grpc
from common_lib.grpc.common import common_types_pb2, error_types_pb2

from causal_analysis.services.causal_graph_service import CausalGraphService
from causal_analysis.services.counterfactual_service import CounterfactualService
from causal_analysis.services.effect_estimation_service import EffectEstimationService
from causal_analysis.services.structure_discovery_service import StructureDiscoveryService
from causal_analysis.services.intervention_service import InterventionService

from causal_analysis.grpc_server.interceptors import AuthInterceptor, LoggingInterceptor, ErrorHandlingInterceptor

logger = logging.getLogger(__name__)


class CausalAnalysisServicer(pb2_grpc.CausalAnalysisServiceServicer):
    """
    gRPC servicer for the Causal Analysis Service.
    Implements the methods defined in the causal_analysis_service.proto file.
    """
    
    def __init__(
        self,
        causal_graph_service: CausalGraphService,
        counterfactual_service: CounterfactualService,
        effect_estimation_service: EffectEstimationService,
        structure_discovery_service: StructureDiscoveryService,
        intervention_service: InterventionService
    ):
        """
        Initialize the CausalAnalysisServicer.
        
        Args:
            causal_graph_service: Service for causal graph operations
            counterfactual_service: Service for counterfactual operations
            effect_estimation_service: Service for effect estimation operations
            structure_discovery_service: Service for structure discovery operations
            intervention_service: Service for intervention operations
        """
        self.causal_graph_service = causal_graph_service
        self.counterfactual_service = counterfactual_service
        self.effect_estimation_service = effect_estimation_service
        self.structure_discovery_service = structure_discovery_service
        self.intervention_service = intervention_service
        self.logger = logger
    
    async def GetCausalGraph(
        self,
        request: pb2.GetCausalGraphRequest,
        context: grpc.aio.ServicerContext
    ) -> pb2.GetCausalGraphResponse:
        """
        Get a causal graph for the specified parameters.
        
        Args:
            request: The request containing parameters for the causal graph
            context: The gRPC service context
            
        Returns:
            A response containing the causal graph or an error
        """
        try:
            self.logger.info(f"GetCausalGraph request received for symbol: {request.symbol.name}")
            
            # Convert request parameters to domain model
            symbol = request.symbol.name
            timeframe = request.timeframe.name
            start_date = request.start_date.seconds
            end_date = request.end_date.seconds
            graph_id = request.graph_id if request.graph_id else None
            parameters = dict(request.parameters)
            
            # Call the service
            if graph_id:
                causal_graph = await self.causal_graph_service.get_causal_graph_by_id(graph_id)
            else:
                causal_graph = await self.causal_graph_service.generate_causal_graph(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    parameters=parameters
                )
            
            # Convert domain model to response
            response = pb2.GetCausalGraphResponse()
            
            # Create CausalGraph message
            graph = pb2.CausalGraph(
                id=causal_graph.id,
                name=causal_graph.name
            )
            
            # Add nodes
            for node in causal_graph.nodes:
                graph_node = pb2.Node(
                    id=node.id,
                    name=node.name,
                    type=node.type
                )
                
                # Add metadata
                if node.metadata:
                    metadata = common_types_pb2.Metadata()
                    for key, value in node.metadata.items():
                        entry = metadata.entries.add()
                        entry.key = key
                        entry.value = str(value)
                    graph_node.metadata.CopyFrom(metadata)
                
                graph.nodes.append(graph_node)
            
            # Add edges
            for edge in causal_graph.edges:
                graph_edge = pb2.Edge(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    weight=edge.weight,
                    type=edge.type
                )
                
                # Add metadata
                if edge.metadata:
                    metadata = common_types_pb2.Metadata()
                    for key, value in edge.metadata.items():
                        entry = metadata.entries.add()
                        entry.key = key
                        entry.value = str(value)
                    graph_edge.metadata.CopyFrom(metadata)
                
                graph.edges.append(graph_edge)
            
            # Add metadata
            if causal_graph.metadata:
                metadata = common_types_pb2.Metadata()
                for key, value in causal_graph.metadata.items():
                    entry = metadata.entries.add()
                    entry.key = key
                    entry.value = str(value)
                graph.metadata.CopyFrom(metadata)
            
            response.graph.CopyFrom(graph)
            
            self.logger.info(f"GetCausalGraph response sent for graph: {graph.id}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error in GetCausalGraph: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error generating causal graph: {str(e)}")
            
            # Create error response
            response = pb2.GetCausalGraphResponse()
            error = error_types_pb2.ErrorResponse(
                code=error_types_pb2.ErrorCode.INTERNAL_ERROR,
                message=f"Error generating causal graph: {str(e)}",
                request_id=context.invocation_metadata().get("x-request-id", "unknown")
            )
            response.error.CopyFrom(error)
            
            return response
    
    async def GenerateCounterfactuals(
        self,
        request: pb2.GenerateCounterfactualsRequest,
        context: grpc.aio.ServicerContext
    ) -> pb2.GenerateCounterfactualsResponse:
        """
        Generate counterfactual scenarios based on a causal graph.
        
        Args:
            request: The request containing parameters for counterfactual generation
            context: The gRPC service context
            
        Returns:
            A response containing the counterfactual scenarios or an error
        """
        try:
            self.logger.info(f"GenerateCounterfactuals request received for graph: {request.graph_id}")
            
            # Convert request parameters to domain model
            graph_id = request.graph_id
            intervention_node_id = request.intervention_node_id
            intervention_value = request.intervention_value
            target_node_id = request.target_node_id
            num_scenarios = request.num_scenarios
            parameters = dict(request.parameters)
            
            # Call the service
            counterfactuals = await self.counterfactual_service.generate_counterfactuals(
                graph_id=graph_id,
                intervention_node_id=intervention_node_id,
                intervention_value=intervention_value,
                target_node_id=target_node_id,
                num_scenarios=num_scenarios,
                parameters=parameters
            )
            
            # Convert domain model to response
            response = pb2.GenerateCounterfactualsResponse()
            
            # Add scenarios
            for scenario in counterfactuals.scenarios:
                cf_scenario = pb2.CounterfactualScenario(
                    id=scenario.id,
                    intervention_node_id=scenario.intervention_node_id,
                    intervention_value=scenario.intervention_value,
                    target_node_id=scenario.target_node_id,
                    counterfactual_outcome=scenario.counterfactual_outcome,
                    confidence=scenario.confidence
                )
                
                # Add metadata
                if scenario.metadata:
                    metadata = common_types_pb2.Metadata()
                    for key, value in scenario.metadata.items():
                        entry = metadata.entries.add()
                        entry.key = key
                        entry.value = str(value)
                    cf_scenario.metadata.CopyFrom(metadata)
                
                response.scenarios.append(cf_scenario)
            
            self.logger.info(f"GenerateCounterfactuals response sent with {len(response.scenarios)} scenarios")
            return response
            
        except Exception as e:
            self.logger.error(f"Error in GenerateCounterfactuals: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error generating counterfactuals: {str(e)}")
            
            # Create error response
            response = pb2.GenerateCounterfactualsResponse()
            error = error_types_pb2.ErrorResponse(
                code=error_types_pb2.ErrorCode.INTERNAL_ERROR,
                message=f"Error generating counterfactuals: {str(e)}",
                request_id=context.invocation_metadata().get("x-request-id", "unknown")
            )
            response.error.CopyFrom(error)
            
            return response
    
    async def EstimateEffects(
        self,
        request: pb2.EstimateEffectsRequest,
        context: grpc.aio.ServicerContext
    ) -> pb2.EstimateEffectsResponse:
        """
        Estimate the effects of interventions on a causal graph.
        
        Args:
            request: The request containing parameters for effect estimation
            context: The gRPC service context
            
        Returns:
            A response containing the estimated effects or an error
        """
        try:
            self.logger.info(f"EstimateEffects request received for graph: {request.graph_id}")
            
            # Convert request parameters to domain model
            graph_id = request.graph_id
            intervention_node_ids = list(request.intervention_node_ids)
            target_node_ids = list(request.target_node_ids)
            parameters = dict(request.parameters)
            
            # Call the service
            effects = await self.effect_estimation_service.estimate_effects(
                graph_id=graph_id,
                intervention_node_ids=intervention_node_ids,
                target_node_ids=target_node_ids,
                parameters=parameters
            )
            
            # Convert domain model to response
            response = pb2.EstimateEffectsResponse()
            
            # Add effects
            for effect in effects:
                effect_msg = pb2.Effect(
                    id=effect.id,
                    intervention_node_id=effect.intervention_node_id,
                    target_node_id=effect.target_node_id,
                    effect_size=effect.effect_size,
                    confidence_lower=effect.confidence_lower,
                    confidence_upper=effect.confidence_upper,
                    p_value=effect.p_value
                )
                
                # Add metadata
                if effect.metadata:
                    metadata = common_types_pb2.Metadata()
                    for key, value in effect.metadata.items():
                        entry = metadata.entries.add()
                        entry.key = key
                        entry.value = str(value)
                    effect_msg.metadata.CopyFrom(metadata)
                
                response.effects.append(effect_msg)
            
            self.logger.info(f"EstimateEffects response sent with {len(response.effects)} effects")
            return response
            
        except Exception as e:
            self.logger.error(f"Error in EstimateEffects: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error estimating effects: {str(e)}")
            
            # Create error response
            response = pb2.EstimateEffectsResponse()
            error = error_types_pb2.ErrorResponse(
                code=error_types_pb2.ErrorCode.INTERNAL_ERROR,
                message=f"Error estimating effects: {str(e)}",
                request_id=context.invocation_metadata().get("x-request-id", "unknown")
            )
            response.error.CopyFrom(error)
            
            return response
    
    async def DiscoverStructure(
        self,
        request: pb2.DiscoverStructureRequest,
        context: grpc.aio.ServicerContext
    ) -> pb2.DiscoverStructureResponse:
        """
        Discover the causal structure from data.
        
        Args:
            request: The request containing parameters for structure discovery
            context: The gRPC service context
            
        Returns:
            A response containing the discovered causal structure or an error
        """
        try:
            self.logger.info(f"DiscoverStructure request received for symbol: {request.symbol.name}")
            
            # Convert request parameters to domain model
            symbol = request.symbol.name
            timeframe = request.timeframe.name
            start_date = request.start_date.seconds
            end_date = request.end_date.seconds
            variables = list(request.variables)
            algorithm = request.algorithm
            parameters = dict(request.parameters)
            
            # Call the service
            causal_graph = await self.structure_discovery_service.discover_structure(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                variables=variables,
                algorithm=algorithm,
                parameters=parameters
            )
            
            # Convert domain model to response
            response = pb2.DiscoverStructureResponse()
            
            # Create CausalGraph message
            graph = pb2.CausalGraph(
                id=causal_graph.id,
                name=causal_graph.name
            )
            
            # Add nodes
            for node in causal_graph.nodes:
                graph_node = pb2.Node(
                    id=node.id,
                    name=node.name,
                    type=node.type
                )
                
                # Add metadata
                if node.metadata:
                    metadata = common_types_pb2.Metadata()
                    for key, value in node.metadata.items():
                        entry = metadata.entries.add()
                        entry.key = key
                        entry.value = str(value)
                    graph_node.metadata.CopyFrom(metadata)
                
                graph.nodes.append(graph_node)
            
            # Add edges
            for edge in causal_graph.edges:
                graph_edge = pb2.Edge(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    weight=edge.weight,
                    type=edge.type
                )
                
                # Add metadata
                if edge.metadata:
                    metadata = common_types_pb2.Metadata()
                    for key, value in edge.metadata.items():
                        entry = metadata.entries.add()
                        entry.key = key
                        entry.value = str(value)
                    graph_edge.metadata.CopyFrom(metadata)
                
                graph.edges.append(graph_edge)
            
            # Add metadata
            if causal_graph.metadata:
                metadata = common_types_pb2.Metadata()
                for key, value in causal_graph.metadata.items():
                    entry = metadata.entries.add()
                    entry.key = key
                    entry.value = str(value)
                graph.metadata.CopyFrom(metadata)
            
            response.graph.CopyFrom(graph)
            
            self.logger.info(f"DiscoverStructure response sent for graph: {graph.id}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error in DiscoverStructure: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error discovering structure: {str(e)}")
            
            # Create error response
            response = pb2.DiscoverStructureResponse()
            error = error_types_pb2.ErrorResponse(
                code=error_types_pb2.ErrorCode.INTERNAL_ERROR,
                message=f"Error discovering structure: {str(e)}",
                request_id=context.invocation_metadata().get("x-request-id", "unknown")
            )
            response.error.CopyFrom(error)
            
            return response
    
    async def GetInterventionEffect(
        self,
        request: pb2.GetInterventionEffectRequest,
        context: grpc.aio.ServicerContext
    ) -> pb2.GetInterventionEffectResponse:
        """
        Calculate the effect of an intervention on a target variable.
        
        Args:
            request: The request containing parameters for intervention effect calculation
            context: The gRPC service context
            
        Returns:
            A response containing the intervention effect or an error
        """
        try:
            self.logger.info(f"GetInterventionEffect request received for graph: {request.graph_id}")
            
            # Convert request parameters to domain model
            graph_id = request.graph_id
            intervention_node_id = request.intervention_node_id
            intervention_value = request.intervention_value
            target_node_id = request.target_node_id
            parameters = dict(request.parameters)
            
            # Call the service
            effect = await self.intervention_service.get_intervention_effect(
                graph_id=graph_id,
                intervention_node_id=intervention_node_id,
                intervention_value=intervention_value,
                target_node_id=target_node_id,
                parameters=parameters
            )
            
            # Convert domain model to response
            response = pb2.GetInterventionEffectResponse()
            
            # Create Effect message
            effect_msg = pb2.Effect(
                id=effect.id,
                intervention_node_id=effect.intervention_node_id,
                target_node_id=effect.target_node_id,
                effect_size=effect.effect_size,
                confidence_lower=effect.confidence_lower,
                confidence_upper=effect.confidence_upper,
                p_value=effect.p_value
            )
            
            # Add metadata
            if effect.metadata:
                metadata = common_types_pb2.Metadata()
                for key, value in effect.metadata.items():
                    entry = metadata.entries.add()
                    entry.key = key
                    entry.value = str(value)
                effect_msg.metadata.CopyFrom(metadata)
            
            response.effect.CopyFrom(effect_msg)
            
            self.logger.info(f"GetInterventionEffect response sent for effect: {effect.id}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error in GetInterventionEffect: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error calculating intervention effect: {str(e)}")
            
            # Create error response
            response = pb2.GetInterventionEffectResponse()
            error = error_types_pb2.ErrorResponse(
                code=error_types_pb2.ErrorCode.INTERNAL_ERROR,
                message=f"Error calculating intervention effect: {str(e)}",
                request_id=context.invocation_metadata().get("x-request-id", "unknown")
            )
            response.error.CopyFrom(error)
            
            return response


class GrpcServer:
    """
    gRPC server for the Causal Analysis Service.
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 50051,
        max_workers: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the gRPC server.
        
        Args:
            host: The host to bind the server to
            port: The port to bind the server to
            max_workers: The maximum number of workers for the server
            config: Optional configuration dictionary
        """
        self.host = host
        self.port = port
        self.max_workers = max_workers or (asyncio.get_event_loop().get_default_executor()._max_workers)
        self.config = config or {}
        self.server = None
        self.logger = logger
    
    async def start(
        self,
        causal_graph_service: CausalGraphService,
        counterfactual_service: CounterfactualService,
        effect_estimation_service: EffectEstimationService,
        structure_discovery_service: StructureDiscoveryService,
        intervention_service: InterventionService
    ):
        """
        Start the gRPC server.
        
        Args:
            causal_graph_service: Service for causal graph operations
            counterfactual_service: Service for counterfactual operations
            effect_estimation_service: Service for effect estimation operations
            structure_discovery_service: Service for structure discovery operations
            intervention_service: Service for intervention operations
            
        Returns:
            The running gRPC server
        """
        # Create interceptors
        auth_interceptor = AuthInterceptor()
        logging_interceptor = LoggingInterceptor()
        error_handling_interceptor = ErrorHandlingInterceptor()
        
        # Create server with interceptors
        self.server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=self.max_workers),
            options=[
                ('grpc.max_send_message_length', 50 * 1024 * 1024),
                ('grpc.max_receive_message_length', 50 * 1024 * 1024),
            ],
            interceptors=[auth_interceptor, logging_interceptor, error_handling_interceptor]
        )
        
        # Create servicer
        servicer = CausalAnalysisServicer(
            causal_graph_service=causal_graph_service,
            counterfactual_service=counterfactual_service,
            effect_estimation_service=effect_estimation_service,
            structure_discovery_service=structure_discovery_service,
            intervention_service=intervention_service
        )
        
        # Add servicer to server
        pb2_grpc.add_CausalAnalysisServiceServicer_to_server(servicer, self.server)
        
        # Add secure port if TLS is enabled
        if self.config.get("tls_enabled", False):
            server_credentials = grpc.ssl_server_credentials(
                [(self.config["private_key"], self.config["certificate"])]
            )
            address = f"{self.host}:{self.port}"
            self.server.add_secure_port(address, server_credentials)
            self.logger.info(f"gRPC server starting on {address} with TLS")
        else:
            address = f"{self.host}:{self.port}"
            self.server.add_insecure_port(address)
            self.logger.info(f"gRPC server starting on {address} without TLS")
        
        # Start the server
        await self.server.start()
        
        self.logger.info(f"gRPC server started on {address}")
        
        return self.server
    
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