"""
NLP Analysis API

This module provides API endpoints for performing NLP analysis on news
and economic reports, and retrieving insights for trading decisions.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Body
from sqlalchemy.orm import Session
import logging
from core_foundations.models.auth import User
from analysis_engine.analysis.nlp.nlp_integration import NLPIntegration
from analysis_engine.db.connection import get_db_session
from analysis_engine.api.auth import get_current_user
router = APIRouter(prefix='/nlp', tags=['nlp'])
logger = logging.getLogger(__name__)
nlp_integration = NLPIntegration()


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@router.post('/analyze-news', response_model=Dict[str, Any])
@async_with_exception_handling
async def analyze_news(news_data: Dict[str, Any], current_user: User=
    Depends(get_current_user)):
    """
    Analyze financial news content and assess potential market impact
    
    Request body format:
    {
        "news_items": [
            {
                "id": "unique_id",
                "title": "News title",
                "content": "Full news content",
                "source": "Source name",
                "timestamp": "ISO datetime string"
            },
            ...
        ]
    }
    """
    try:
        if 'news_items' not in news_data or not isinstance(news_data[
            'news_items'], list):
            raise HTTPException(status_code=400, detail=
                "Request must contain 'news_items' list")
        result = await nlp_integration.process_news_data(news_data)
        return {'status': 'success', 'results': result.result_data,
            'timestamp': datetime.now().isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error analyzing news: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Failed to analyze news: {str(e)}')


@router.post('/analyze-economic-report', response_model=Dict[str, Any])
@async_with_exception_handling
async def analyze_economic_report(report_data: Dict[str, Any], current_user:
    User=Depends(get_current_user)):
    """
    Analyze economic report content and assess potential market impact
    
    Request body format:
    {
        "report": {
            "title": "Report title",
            "content": "Full report content",
            "type": "NFP", # Optional, can be auto-detected
            "source": "Source name",
            "timestamp": "ISO datetime string"
        }
    }
    """
    try:
        if 'report' not in report_data or not isinstance(report_data[
            'report'], dict):
            raise HTTPException(status_code=400, detail=
                "Request must contain 'report' object")
        result = await nlp_integration.process_economic_report(report_data)
        return {'status': 'success', 'results': result.result_data,
            'timestamp': datetime.now().isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error analyzing economic report: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Failed to analyze economic report: {str(e)}')


@router.post('/combined-insights', response_model=Dict[str, Any])
@async_with_exception_handling
async def get_combined_insights(data: Dict[str, Any], current_user: User=
    Depends(get_current_user)):
    """
    Generate combined insights from news and economic reports
    
    Request body format:
    {
        "news_data": {
            "news_items": [...]
        },
        "economic_reports": [
            {
                "title": "Report title",
                "content": "Full report content",
                "type": "NFP",
                "source": "Source name",
                "timestamp": "ISO datetime string"
            },
            ...
        ],
        "currency_pairs": ["EUR/USD", "GBP/USD", ...] # Optional filter
    }
    """
    try:
        news_data = data.get('news_data')
        economic_reports = data.get('economic_reports')
        currency_pairs = data.get('currency_pairs')
        if not news_data and not economic_reports:
            raise HTTPException(status_code=400, detail=
                "Request must contain at least 'news_data' or 'economic_reports'"
                )
        results = await nlp_integration.generate_nlp_insights(news_data,
            economic_reports)
        if currency_pairs and isinstance(currency_pairs, list
            ) and 'aggregate_insights' in results:
            pair_insights = results['aggregate_insights'].get(
                'currency_pair_insights', {})
            filtered_insights = {pair: insights for pair, insights in
                pair_insights.items() if pair in currency_pairs}
            results['aggregate_insights']['currency_pair_insights'
                ] = filtered_insights
        return {'status': 'success', 'results': results, 'timestamp':
            datetime.now().isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error generating combined insights: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Failed to generate combined insights: {str(e)}')


@router.get('/market-sentiment', response_model=Dict[str, Any])
@async_with_exception_handling
async def get_market_sentiment(currency_pair: Optional[str]=Query(None,
    description='Specific currency pair to analyze'), current_user: User=
    Depends(get_current_user), db: Session=Depends(get_db_session)):
    """
    Get current market sentiment based on recent news and economic reports
    """
    try:
        return {'status': 'success', 'message':
            'Sentiment analysis requires recent news and economic data',
            'timestamp': datetime.now().isoformat(), 'note':
            'This endpoint should be called with recent data using the combined-insights endpoint'
            }
    except Exception as e:
        logger.error(f'Error getting market sentiment: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Failed to get market sentiment: {str(e)}')
