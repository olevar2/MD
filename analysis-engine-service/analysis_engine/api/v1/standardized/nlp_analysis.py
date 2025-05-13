"""
Standardized NLP Analysis API for Analysis Engine Service.

This module provides standardized API endpoints for performing NLP analysis on news
and economic reports, and retrieving insights for trading decisions.

All endpoints follow the platform's standardized API design patterns.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Body, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from core_foundations.models.auth import User
from analysis_engine.analysis.nlp.nlp_integration import NLPIntegration
from analysis_engine.db.connection import get_db_session
from analysis_engine.api.auth import get_current_user
from analysis_engine.core.exceptions_bridge import ForexTradingPlatformError, AnalysisError, NLPAnalysisError, InsufficientDataError, get_correlation_id_from_request
from analysis_engine.monitoring.structured_logging import get_structured_logger


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class NewsItem(BaseModel):
    """Model for a news item"""
    id: str = Field(..., description='Unique identifier for the news item')
    title: str = Field(..., description='Title of the news article')
    content: str = Field(..., description='Full content of the news article')
    source: str = Field(..., description='Source of the news article')
    timestamp: str = Field(..., description=
        'ISO datetime string of when the news was published')


class NewsAnalysisRequest(BaseModel):
    """Request model for news analysis"""
    news_items: List[NewsItem] = Field(..., description=
        'List of news items to analyze')


    class Config:
    """
    Config class.
    
    Attributes:
        Add attributes here
    """

        schema_extra = {'example': {'news_items': [{'id': 'news_12345',
            'title':
            'Federal Reserve Raises Interest Rates by 25 Basis Points',
            'content':
            'The Federal Reserve announced today that it will raise interest rates by 25 basis points...'
            , 'source': 'Financial Times', 'timestamp':
            '2025-04-01T14:30:00Z'}]}}


class EconomicReport(BaseModel):
    """Model for an economic report"""
    title: str = Field(..., description='Title of the economic report')
    content: str = Field(..., description='Full content of the economic report'
        )
    type: Optional[str] = Field(None, description=
        "Type of economic report (e.g., 'NFP', 'GDP')")
    source: str = Field(..., description='Source of the economic report')
    timestamp: str = Field(..., description=
        'ISO datetime string of when the report was published')


class EconomicReportAnalysisRequest(BaseModel):
    """Request model for economic report analysis"""
    report: EconomicReport = Field(..., description=
        'Economic report to analyze')


    class Config:
    """
    Config class.
    
    Attributes:
        Add attributes here
    """

        schema_extra = {'example': {'report': {'title':
            'Non-Farm Payroll Report - April 2025', 'content':
            'The U.S. economy added 200,000 jobs in April 2025, exceeding expectations...'
            , 'type': 'NFP', 'source': 'Bureau of Labor Statistics',
            'timestamp': '2025-05-01T12:00:00Z'}}}


class CombinedInsightsRequest(BaseModel):
    """Request model for combined insights"""
    news_data: Optional[Dict[str, Any]] = Field(None, description=
        'News data to analyze')
    economic_reports: Optional[List[EconomicReport]] = Field(None,
        description='Economic reports to analyze')
    currency_pairs: Optional[List[str]] = Field(None, description=
        'Currency pairs to filter insights for')


    class Config:
    """
    Config class.
    
    Attributes:
        Add attributes here
    """

        schema_extra = {'example': {'news_data': {'news_items': [{'id':
            'news_12345', 'title':
            'Federal Reserve Raises Interest Rates by 25 Basis Points',
            'content':
            'The Federal Reserve announced today that it will raise interest rates by 25 basis points...'
            , 'source': 'Financial Times', 'timestamp':
            '2025-04-01T14:30:00Z'}]}, 'economic_reports': [{'title':
            'Non-Farm Payroll Report - April 2025', 'content':
            'The U.S. economy added 200,000 jobs in April 2025, exceeding expectations...'
            , 'type': 'NFP', 'source': 'Bureau of Labor Statistics',
            'timestamp': '2025-05-01T12:00:00Z'}], 'currency_pairs': [
            'EUR/USD', 'GBP/USD']}}


class AnalysisResponse(BaseModel):
    """Response model for NLP analysis"""
    status: str = Field(..., description='Status of the analysis')
    results: Dict[str, Any] = Field(..., description='Analysis results')
    timestamp: str = Field(..., description=
        'ISO datetime string of when the analysis was performed')


def get_nlp_integration():
    """Get NLP integration dependency"""
    return NLPIntegration()


router = APIRouter(prefix='/v1/analysis/nlp', tags=['NLP Analysis'])
logger = get_structured_logger(__name__)


@router.post('/news/analyze', response_model=AnalysisResponse, summary=
    'Analyze news', description=
    'Analyze financial news content and assess potential market impact.')
@async_with_exception_handling
async def analyze_news(request: NewsAnalysisRequest, request_obj: Request,
    current_user: User=Depends(get_current_user), nlp_integration:
    NLPIntegration=Depends(get_nlp_integration)):
    """
    Analyze financial news content and assess potential market impact.
    """
    correlation_id = get_correlation_id_from_request(request_obj)
    try:
        news_data = {'news_items': [news_item.dict() for news_item in
            request.news_items]}
        result = await nlp_integration.process_news_data(news_data)
        logger.info(f'Analyzed {len(request.news_items)} news items', extra
            ={'correlation_id': correlation_id, 'news_count': len(request.
            news_items), 'sources': list(set(item.source for item in
            request.news_items))})
        response = AnalysisResponse(status='success', results=result.
            result_data, timestamp=datetime.now().isoformat())
        return response
    except InsufficientDataError:
        raise
    except ForexTradingPlatformError:
        raise
    except Exception as e:
        logger.error(f'Error analyzing news: {str(e)}', extra={
            'correlation_id': correlation_id}, exc_info=True)
        raise NLPAnalysisError(message='Failed to analyze news',
            correlation_id=correlation_id)


@router.post('/economic-reports/analyze', response_model=AnalysisResponse,
    summary='Analyze economic report', description=
    'Analyze economic report content and assess potential market impact.')
@async_with_exception_handling
async def analyze_economic_report(request: EconomicReportAnalysisRequest,
    request_obj: Request, current_user: User=Depends(get_current_user),
    nlp_integration: NLPIntegration=Depends(get_nlp_integration)):
    """
    Analyze economic report content and assess potential market impact.
    """
    correlation_id = get_correlation_id_from_request(request_obj)
    try:
        report_data = {'report': request.report.dict()}
        result = await nlp_integration.process_economic_report(report_data)
        logger.info(f'Analyzed economic report: {request.report.title}',
            extra={'correlation_id': correlation_id, 'report_type': request
            .report.type, 'source': request.report.source})
        response = AnalysisResponse(status='success', results=result.
            result_data, timestamp=datetime.now().isoformat())
        return response
    except InsufficientDataError:
        raise
    except ForexTradingPlatformError:
        raise
    except Exception as e:
        logger.error(f'Error analyzing economic report: {str(e)}', extra={
            'correlation_id': correlation_id}, exc_info=True)
        raise NLPAnalysisError(message='Failed to analyze economic report',
            correlation_id=correlation_id)


@router.post('/insights/combined', response_model=AnalysisResponse, summary
    ='Get combined insights', description=
    'Generate combined insights from news and economic reports.')
@async_with_exception_handling
async def get_combined_insights(request: CombinedInsightsRequest,
    request_obj: Request, current_user: User=Depends(get_current_user),
    nlp_integration: NLPIntegration=Depends(get_nlp_integration)):
    """
    Generate combined insights from news and economic reports.
    """
    correlation_id = get_correlation_id_from_request(request_obj)
    try:
        if not request.news_data and not request.economic_reports:
            raise InsufficientDataError(message=
                "Request must contain at least 'news_data' or 'economic_reports'"
                , correlation_id=correlation_id)
        economic_reports = None
        if request.economic_reports:
            economic_reports = [report.dict() for report in request.
                economic_reports]
        results = await nlp_integration.generate_nlp_insights(request.
            news_data, economic_reports)
        if request.currency_pairs and 'aggregate_insights' in results:
            pair_insights = results['aggregate_insights'].get(
                'currency_pair_insights', {})
            filtered_insights = {pair: insights for pair, insights in
                pair_insights.items() if pair in request.currency_pairs}
            results['aggregate_insights']['currency_pair_insights'
                ] = filtered_insights
        logger.info('Generated combined insights', extra={'correlation_id':
            correlation_id, 'has_news_data': bool(request.news_data),
            'economic_reports_count': len(request.economic_reports) if
            request.economic_reports else 0, 'currency_pairs': request.
            currency_pairs})
        response = AnalysisResponse(status='success', results=results,
            timestamp=datetime.now().isoformat())
        return response
    except InsufficientDataError:
        raise
    except ForexTradingPlatformError:
        raise
    except Exception as e:
        logger.error(f'Error generating combined insights: {str(e)}', extra
            ={'correlation_id': correlation_id}, exc_info=True)
        raise NLPAnalysisError(message=
            'Failed to generate combined insights', correlation_id=
            correlation_id)


@router.get('/market-sentiment', response_model=AnalysisResponse, summary=
    'Get market sentiment', description=
    'Get current market sentiment based on recent news and economic reports.')
@async_with_exception_handling
async def get_market_sentiment(currency_pair: Optional[str]=Query(None,
    description='Specific currency pair to analyze'), request_obj: Request=
    None, current_user: User=Depends(get_current_user), db: Session=Depends
    (get_db_session), nlp_integration: NLPIntegration=Depends(
    get_nlp_integration)):
    """
    Get current market sentiment based on recent news and economic reports.
    """
    correlation_id = get_correlation_id_from_request(request_obj)
    try:
        logger.info(
            f"Getting market sentiment for {currency_pair if currency_pair else 'all currency pairs'}"
            , extra={'correlation_id': correlation_id, 'currency_pair':
            currency_pair})
        response = AnalysisResponse(status='success', results={'message':
            'Sentiment analysis requires recent news and economic data',
            'note':
            'This endpoint should be called with recent data using the combined-insights endpoint'
            }, timestamp=datetime.now().isoformat())
        return response
    except ForexTradingPlatformError:
        raise
    except Exception as e:
        logger.error(f'Error getting market sentiment: {str(e)}', extra={
            'correlation_id': correlation_id}, exc_info=True)
        raise NLPAnalysisError(message='Failed to get market sentiment',
            correlation_id=correlation_id)


legacy_router = APIRouter(prefix='/api/v1/nlp', tags=['NLP Analysis (Legacy)'])


@legacy_router.post('/analyze-news', response_model=AnalysisResponse)
async def legacy_analyze_news(news_data: Dict[str, Any], request_obj:
    Request=None, current_user: User=Depends(get_current_user),
    nlp_integration: NLPIntegration=Depends(get_nlp_integration)):
    """
    Legacy endpoint for analyzing news.
    Consider migrating to /api/v1/analysis/nlp/news/analyze
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/nlp/news/analyze'
        )
    if 'news_items' not in news_data or not isinstance(news_data[
        'news_items'], list):
        raise HTTPException(status_code=400, detail=
            "Request must contain 'news_items' list")
    request = NewsAnalysisRequest(news_items=[NewsItem(**item) for item in
        news_data['news_items']])
    return await analyze_news(request, request_obj, current_user,
        nlp_integration)


@legacy_router.post('/analyze-economic-report', response_model=AnalysisResponse
    )
async def legacy_analyze_economic_report(report_data: Dict[str, Any],
    request_obj: Request=None, current_user: User=Depends(get_current_user),
    nlp_integration: NLPIntegration=Depends(get_nlp_integration)):
    """
    Legacy endpoint for analyzing economic reports.
    Consider migrating to /api/v1/analysis/nlp/economic-reports/analyze
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/nlp/economic-reports/analyze'
        )
    if 'report' not in report_data or not isinstance(report_data['report'],
        dict):
        raise HTTPException(status_code=400, detail=
            "Request must contain 'report' object")
    request = EconomicReportAnalysisRequest(report=EconomicReport(**
        report_data['report']))
    return await analyze_economic_report(request, request_obj, current_user,
        nlp_integration)


@legacy_router.post('/combined-insights', response_model=AnalysisResponse)
async def legacy_get_combined_insights(data: Dict[str, Any], request_obj:
    Request=None, current_user: User=Depends(get_current_user),
    nlp_integration: NLPIntegration=Depends(get_nlp_integration)):
    """
    Legacy endpoint for getting combined insights.
    Consider migrating to /api/v1/analysis/nlp/insights/combined
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/nlp/insights/combined'
        )
    news_data = data.get('news_data')
    economic_reports = data.get('economic_reports')
    currency_pairs = data.get('currency_pairs')
    economic_report_models = None
    if economic_reports and isinstance(economic_reports, list):
        economic_report_models = [EconomicReport(**report) for report in
            economic_reports]
    request = CombinedInsightsRequest(news_data=news_data, economic_reports
        =economic_report_models, currency_pairs=currency_pairs)
    return await get_combined_insights(request, request_obj, current_user,
        nlp_integration)


@legacy_router.get('/market-sentiment', response_model=AnalysisResponse)
async def legacy_get_market_sentiment(currency_pair: Optional[str]=Query(
    None, description='Specific currency pair to analyze'), request_obj:
    Request=None, current_user: User=Depends(get_current_user), db: Session
    =Depends(get_db_session), nlp_integration: NLPIntegration=Depends(
    get_nlp_integration)):
    """
    Legacy endpoint for getting market sentiment.
    Consider migrating to /api/v1/analysis/nlp/market-sentiment
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/nlp/market-sentiment'
        )
    return await get_market_sentiment(currency_pair, request_obj,
        current_user, db, nlp_integration)


def setup_nlp_analysis_routes(app: FastAPI) ->None:
    """
    Set up NLP analysis routes.

    Args:
        app: FastAPI application
    """
    app.include_router(router, prefix='/api')
    app.include_router(legacy_router)
