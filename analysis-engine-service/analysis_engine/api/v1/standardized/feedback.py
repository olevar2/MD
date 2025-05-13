"""
Standardized Feedback API for Analysis Engine Service.

This module provides standardized API endpoints for accessing feedback-related functionality,
providing endpoints to retrieve feedback statistics, trigger model retraining,
and manage feedback rules.

All endpoints follow the platform's standardized API design patterns.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, Query, HTTPException, Body, Request
from pydantic import BaseModel, Field
from analysis_engine.adaptive_layer.feedback_integration_service import FeedbackIntegrationService
from analysis_engine.services.service_factory import get_feedback_service
from analysis_engine.core.exceptions_bridge import ForexTradingPlatformError, AnalysisError, FeedbackError, get_correlation_id_from_request
from analysis_engine.monitoring.structured_logging import get_structured_logger


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class RuleUpdate(BaseModel):
    """Model for a feedback rule update"""
    rule_id: str = Field(..., description='ID of the rule to update')
    enabled: Optional[bool] = Field(None, description=
        'Whether the rule is enabled')
    priority: Optional[int] = Field(None, description=
        'Priority of the rule (lower numbers = higher priority)')
    conditions: Optional[Dict[str, Any]] = Field(None, description=
        'Conditions for rule matching')
    actions: Optional[Dict[str, Any]] = Field(None, description=
        'Actions to take when rule matches')


class RuleUpdateRequest(BaseModel):
    """Request model for updating feedback rules"""
    updates: List[RuleUpdate] = Field(..., description=
        'List of rule updates to apply')


    class Config:
    """
    Config class.
    
    Attributes:
        Add attributes here
    """

        schema_extra = {'example': {'updates': [{'rule_id':
            'high_confidence_signals', 'enabled': True, 'priority': 1,
            'conditions': {'confidence_threshold': 0.85, 'signal_types': [
            'buy', 'sell']}, 'actions': {'weight': 2.0, 'auto_approve': 
            True}}]}}


class StatisticsResponse(BaseModel):
    """Response model for feedback statistics"""
    total_feedback_count: int
    feedback_by_source: Dict[str, int]
    feedback_by_category: Dict[str, int]
    feedback_by_outcome: Dict[str, int]
    recent_trend: Dict[str, Any]
    parameter_performance: Optional[Dict[str, Any]] = None
    filters_applied: Dict[str, Any]


class RetrainingResponse(BaseModel):
    """Response model for model retraining"""
    model_id: str
    status: str
    job_id: Optional[str] = None
    estimated_completion_time: Optional[datetime] = None
    message: str


class RuleUpdateResponse(BaseModel):
    """Response model for rule updates"""
    updated_rules: List[str]
    status: str
    message: str


class ParameterPerformanceResponse(BaseModel):
    """Response model for parameter performance"""
    strategy_id: str
    min_samples: int
    parameters: Dict[str, Any]


router = APIRouter(prefix='/v1/analysis/feedback', tags=['Feedback'])
logger = get_structured_logger(__name__)


@router.get('/statistics', response_model=StatisticsResponse, summary=
    'Get feedback statistics', description=
    'Get feedback statistics with optional filtering.')
@async_with_exception_handling
async def get_feedback_statistics(request_obj: Request, strategy_id:
    Optional[str]=Query(None, description='Filter by strategy ID'),
    model_id: Optional[str]=Query(None, description='Filter by model ID'),
    instrument: Optional[str]=Query(None, description=
    'Filter by instrument'), start_time: Optional[datetime]=Query(None,
    description='Start time for filtering (ISO format)'), end_time:
    Optional[datetime]=Query(None, description=
    'End time for filtering (ISO format)'), feedback_service:
    FeedbackIntegrationService=Depends(get_feedback_service)):
    """
    Get feedback statistics with optional filtering.

    This endpoint returns statistics about collected feedback, including counts by
    source, category, and outcome. Results can be filtered by strategy, model,
    instrument, and time range.
    """
    correlation_id = get_correlation_id_from_request(request_obj)
    try:
        stats = await feedback_service.get_feedback_statistics(strategy_id=
            strategy_id, model_id=model_id, instrument=instrument,
            start_time=start_time, end_time=end_time)
        logger.info('Retrieved feedback statistics', extra={
            'correlation_id': correlation_id, 'strategy_id': strategy_id,
            'model_id': model_id, 'instrument': instrument, 'total_count':
            stats.get('total_feedback_count', 0)})
        stats['filters_applied'] = {'strategy_id': strategy_id, 'model_id':
            model_id, 'instrument': instrument, 'start_time': start_time.
            isoformat() if start_time else None, 'end_time': end_time.
            isoformat() if end_time else None}
        return stats
    except Exception as e:
        logger.error(f'Error retrieving feedback statistics: {str(e)}',
            extra={'correlation_id': correlation_id}, exc_info=True)
        raise FeedbackError(message=
            f'Error retrieving feedback statistics: {str(e)}',
            correlation_id=correlation_id)


@router.post('/models/{model_id}/retrain', response_model=
    RetrainingResponse, summary='Trigger model retraining', description=
    'Trigger retraining of a specific model based on collected feedback.')
@async_with_exception_handling
async def trigger_model_retraining(model_id: str, request_obj: Request,
    feedback_service: FeedbackIntegrationService=Depends(get_feedback_service)
    ):
    """
    Trigger retraining of a specific model based on collected feedback.

    This endpoint manually initiates the retraining process for a model,
    incorporating feedback data collected since the last training.
    """
    correlation_id = get_correlation_id_from_request(request_obj)
    try:
        result = await feedback_service.trigger_model_retraining(model_id)
        logger.info(f'Triggered retraining for model {model_id}', extra={
            'correlation_id': correlation_id, 'model_id': model_id,
            'job_id': result.get('job_id')})
        response = RetrainingResponse(model_id=model_id, status=result.get(
            'status', 'initiated'), job_id=result.get('job_id'),
            estimated_completion_time=result.get(
            'estimated_completion_time'), message=result.get('message',
            'Model retraining initiated successfully'))
        return response
    except Exception as e:
        logger.error(
            f'Error triggering model retraining for {model_id}: {str(e)}',
            extra={'correlation_id': correlation_id}, exc_info=True)
        raise FeedbackError(message=
            f'Error triggering model retraining: {str(e)}', correlation_id=
            correlation_id)


@router.put('/rules', response_model=RuleUpdateResponse, summary=
    'Update feedback rules', description=
    'Update feedback categorization rules.')
@async_with_exception_handling
async def update_feedback_rules(request: RuleUpdateRequest, request_obj:
    Request, feedback_service: FeedbackIntegrationService=Depends(
    get_feedback_service)):
    """
    Update feedback categorization rules.

    This endpoint allows modification of the rules used to categorize and route feedback,
    enabling adaptation of the feedback system behavior without code changes.
    """
    correlation_id = get_correlation_id_from_request(request_obj)
    try:
        rule_updates = {}
        for update in request.updates:
            rule_data = update.dict(exclude_unset=True)
            rule_id = rule_data.pop('rule_id')
            rule_updates[rule_id] = rule_data
        results = await feedback_service.update_feedback_rules(rule_updates)
        logger.info(f"Updated feedback rules: {', '.join(rule_updates.keys())}"
            , extra={'correlation_id': correlation_id, 'rule_count': len(
            rule_updates), 'rule_ids': list(rule_updates.keys())})
        response = RuleUpdateResponse(updated_rules=list(rule_updates.keys(
            )), status='success', message=
            f'Successfully updated {len(rule_updates)} rules')
        return response
    except Exception as e:
        logger.error(f'Error updating feedback rules: {str(e)}', extra={
            'correlation_id': correlation_id}, exc_info=True)
        raise FeedbackError(message=
            f'Error updating feedback rules: {str(e)}', correlation_id=
            correlation_id)


@router.get('/strategies/{strategy_id}/parameters', response_model=
    ParameterPerformanceResponse, summary='Get parameter performance',
    description='Get performance statistics for strategy parameters.')
@async_with_exception_handling
async def get_parameter_performance(strategy_id: str, request_obj: Request,
    min_samples: int=Query(10, description=
    'Minimum number of samples required for parameter statistics'),
    feedback_service: FeedbackIntegrationService=Depends(get_feedback_service)
    ):
    """
    Get performance statistics for strategy parameters.

    This endpoint returns performance metrics for different parameter values used in a strategy,
    enabling analysis of which parameter settings yield the best results.
    """
    correlation_id = get_correlation_id_from_request(request_obj)
    try:
        stats = await feedback_service.get_feedback_statistics(strategy_id=
            strategy_id)
        if 'parameter_performance' not in stats:
            logger.info(
                f'No parameter performance data available for strategy {strategy_id}'
                , extra={'correlation_id': correlation_id, 'strategy_id':
                strategy_id, 'min_samples': min_samples})
            return ParameterPerformanceResponse(strategy_id=strategy_id,
                min_samples=min_samples, parameters={})
        filtered_parameters = {}
        for param_key, metrics in stats['parameter_performance'].items():
            total_samples = metrics['success_count'] + metrics['failure_count']
            if total_samples >= min_samples:
                filtered_parameters[param_key] = metrics
        logger.info(
            f'Retrieved parameter performance for strategy {strategy_id}',
            extra={'correlation_id': correlation_id, 'strategy_id':
            strategy_id, 'min_samples': min_samples, 'parameter_count': len
            (filtered_parameters)})
        return ParameterPerformanceResponse(strategy_id=strategy_id,
            min_samples=min_samples, parameters=filtered_parameters)
    except Exception as e:
        logger.error(
            f'Error retrieving parameter performance for strategy {strategy_id}: {str(e)}'
            , extra={'correlation_id': correlation_id}, exc_info=True)
        raise FeedbackError(message=
            f'Error retrieving parameter performance: {str(e)}',
            correlation_id=correlation_id)


@router.post('/submit', summary='Submit feedback', description=
    'Submit feedback for a signal, model, or strategy.')
@async_with_exception_handling
async def submit_feedback(feedback_data: Dict[str, Any]=Body(...,
    description='Feedback data to submit'), request_obj: Request=None,
    feedback_service: FeedbackIntegrationService=Depends(get_feedback_service)
    ):
    """
    Submit feedback for a signal, model, or strategy.

    This endpoint allows users to submit feedback about signals, models, or strategies,
    which can be used to improve the system over time.
    """
    correlation_id = get_correlation_id_from_request(request_obj)
    try:
        required_fields = ['source', 'target_id', 'feedback_type']
        for field in required_fields:
            if field not in feedback_data:
                raise HTTPException(status_code=400, detail=
                    f'Missing required field: {field}')
        if 'timestamp' not in feedback_data:
            feedback_data['timestamp'] = datetime.utcnow().isoformat()
        feedback_data['correlation_id'] = correlation_id
        result = await feedback_service.submit_feedback(feedback_data)
        logger.info(f"Submitted feedback for {feedback_data['target_id']}",
            extra={'correlation_id': correlation_id, 'source':
            feedback_data['source'], 'target_id': feedback_data['target_id'
            ], 'feedback_type': feedback_data['feedback_type']})
        return {'status': 'success', 'message':
            'Feedback submitted successfully', 'feedback_id': result.get(
            'feedback_id')}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error submitting feedback: {str(e)}', extra={
            'correlation_id': correlation_id}, exc_info=True)
        raise FeedbackError(message=f'Error submitting feedback: {str(e)}',
            correlation_id=correlation_id)


legacy_router = APIRouter(prefix='/api/v1/feedback', tags=['Feedback (Legacy)']
    )


@legacy_router.get('/statistics')
async def legacy_get_feedback_statistics(strategy_id: Optional[str]=Query(
    None, description='Filter by strategy ID'), model_id: Optional[str]=
    Query(None, description='Filter by model ID'), instrument: Optional[str
    ]=Query(None, description='Filter by instrument'), start_time: Optional
    [datetime]=Query(None, description=
    'Start time for filtering (ISO format)'), end_time: Optional[datetime]=
    Query(None, description='End time for filtering (ISO format)'),
    request_obj: Request=None, feedback_service: FeedbackIntegrationService
    =Depends(get_feedback_service)):
    """
    Legacy endpoint for getting feedback statistics.
    Consider migrating to /api/v1/analysis/feedback/statistics
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/feedback/statistics'
        )
    return await get_feedback_statistics(request_obj, strategy_id, model_id,
        instrument, start_time, end_time, feedback_service)


@legacy_router.post('/model/retrain/{model_id}')
async def legacy_trigger_model_retraining(model_id: str, request_obj:
    Request=None, feedback_service: FeedbackIntegrationService=Depends(
    get_feedback_service)):
    """
    Legacy endpoint for triggering model retraining.
    Consider migrating to /api/v1/analysis/feedback/models/{model_id}/retrain
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/feedback/models/{model_id}/retrain'
        )
    result = await trigger_model_retraining(model_id, request_obj,
        feedback_service)
    return result.dict()


@legacy_router.put('/rules')
async def legacy_update_feedback_rules(rule_updates: Dict[str, Any]=Body(
    ..., description='Rule updates to apply'), request_obj: Request=None,
    feedback_service: FeedbackIntegrationService=Depends(get_feedback_service)
    ):
    """
    Legacy endpoint for updating feedback rules.
    Consider migrating to /api/v1/analysis/feedback/rules
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/feedback/rules'
        )
    updates = []
    for rule_id, rule_data in rule_updates.items():
        update = {'rule_id': rule_id, **rule_data}
        updates.append(RuleUpdate(**update))
    request = RuleUpdateRequest(updates=updates)
    result = await update_feedback_rules(request, request_obj, feedback_service
        )
    return result.dict()


@legacy_router.get('/parameters/{strategy_id}')
async def legacy_get_parameter_performance(strategy_id: str, min_samples:
    int=Query(10, description=
    'Minimum number of samples required for parameter statistics'),
    request_obj: Request=None, feedback_service: FeedbackIntegrationService
    =Depends(get_feedback_service)):
    """
    Legacy endpoint for getting parameter performance.
    Consider migrating to /api/v1/analysis/feedback/strategies/{strategy_id}/parameters
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/feedback/strategies/{strategy_id}/parameters'
        )
    result = await get_parameter_performance(strategy_id, request_obj,
        min_samples, feedback_service)
    return result.dict()


@legacy_router.post('/submit')
async def legacy_submit_feedback(feedback_data: Dict[str, Any]=Body(...,
    description='Feedback data to submit'), request_obj: Request=None,
    feedback_service: FeedbackIntegrationService=Depends(get_feedback_service)
    ):
    """
    Legacy endpoint for submitting feedback.
    Consider migrating to /api/v1/analysis/feedback/submit
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/feedback/submit'
        )
    return await submit_feedback(feedback_data, request_obj, feedback_service)


def setup_feedback_routes(app: FastAPI) ->None:
    """
    Set up feedback routes.

    Args:
        app: FastAPI application
    """
    app.include_router(router, prefix='/api')
    app.include_router(legacy_router)
