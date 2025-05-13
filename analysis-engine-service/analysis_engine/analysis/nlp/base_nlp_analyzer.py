"""
Base NLP Analyzer Module

This module provides the foundation for all NLP analysis components
with standardized interfaces for text processing and analysis.
"""
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging
import re
import spacy
from abc import abstractmethod
from analysis_engine.analysis.base_analyzer import BaseAnalyzer
from analysis_engine.models.analysis_result import AnalysisResult
logger = logging.getLogger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class BaseNLPAnalyzer(BaseAnalyzer):
    """
    Abstract base class for all NLP analysis components.
    
    This class extends the BaseAnalyzer with NLP-specific functionality
    for text preprocessing, entity recognition, and sentiment analysis.
    """

    def __init__(self, name: str, parameters: Dict[str, Any]=None):
        """
        Initialize the base NLP analyzer
        
        Args:
            name: Name identifier for the analyzer
            parameters: Configuration parameters for the analyzer
        """
        super().__init__(name, parameters)
        self.nlp = None
        self._initialize_nlp_pipeline()

    @with_exception_handling
    def _initialize_nlp_pipeline(self):
        """Initialize the NLP pipeline based on parameters"""
        try:
            model_name = self.parameters.get('spacy_model', 'en_core_web_sm')
            self.nlp = spacy.load(model_name)
            logger.info(f'Loaded NLP model: {model_name}')
        except Exception as e:
            logger.error(f'Failed to load NLP model: {str(e)}')
            self.nlp = None

    @with_exception_handling
    def _ensure_nlp_pipeline(self):
        """Ensure NLP pipeline is loaded, attempt to download if missing"""
        if self.nlp is None:
            try:
                model_name = self.parameters.get('spacy_model',
                    'en_core_web_sm')
                import spacy.cli
                logger.info(f'Downloading NLP model: {model_name}')
                spacy.cli.download(model_name)
                self.nlp = spacy.load(model_name)
                logger.info(
                    f'Successfully downloaded and loaded NLP model: {model_name}'
                    )
            except Exception as e:
                logger.error(f'Failed to download/load NLP model: {str(e)}')
                raise RuntimeError(
                    f'NLP pipeline initialization failed: {str(e)}')

    def preprocess_text(self, text: str) ->str:
        """
        Preprocess text for NLP analysis
        
        Args:
            text: Raw text to process
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ''
        text = text.lower()
        text = re.sub('\\s+', ' ', text)
        text = re.sub('[^\\w\\s]', '', text)
        return text.strip()

    def extract_entities(self, text: str) ->List[Dict[str, Any]]:
        """
        Extract named entities from text
        
        Args:
            text: Text to analyze
            
        Returns:
            List of extracted entities with type and text
        """
        self._ensure_nlp_pipeline()
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({'text': ent.text, 'label': ent.label_, 'start':
                ent.start_char, 'end': ent.end_char})
        return entities

    @with_analysis_resilience('analyze_sentiment')
    def analyze_sentiment(self, text: str) ->Dict[str, Any]:
        """
        Analyze sentiment of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        self._ensure_nlp_pipeline()
        positive_words = set(['increase', 'growth', 'positive', 'bullish',
            'upward', 'gain', 'profit', 'strengthen', 'improved',
            'recovery', 'strong', 'robust', 'exceed'])
        negative_words = set(['decrease', 'decline', 'negative', 'bearish',
            'downward', 'loss', 'deficit', 'weaken', 'deteriorate',
            'recession', 'weak', 'below', 'fail'])
        tokens = [token.text.lower() for token in self.nlp(text)]
        positive_count = sum(1 for token in tokens if token in positive_words)
        negative_count = sum(1 for token in tokens if token in negative_words)
        total_count = len(tokens)
        if total_count > 0:
            positive_score = positive_count / total_count
            negative_score = negative_count / total_count
            compound_score = (positive_count - negative_count
                ) / total_count if total_count > 0 else 0
        else:
            positive_score = negative_score = compound_score = 0
        return {'positive': positive_score, 'negative': negative_score,
            'neutral': 1 - (positive_score + negative_score), 'compound':
            compound_score}

    @abstractmethod
    def analyze(self, data: Any) ->AnalysisResult:
        """
        Perform analysis on provided data
        
        Args:
            data: Data to analyze
            
        Returns:
            AnalysisResult containing analysis results
        """
        pass
