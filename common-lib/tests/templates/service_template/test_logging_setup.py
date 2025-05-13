"""
Unit tests for the service template logging setup module.
"""

import os
import logging
import pytest
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from common_lib.templates.service_template.logging_setup import setup_logging
from common_lib.config import LoggingConfig


@patch("common_lib.templates.service_template.logging_setup.get_logging_config")
def test_setup_logging_with_default_config(mock_get_logging_config):
    """Test setup_logging with default config."""
    # Mock the get_logging_config function
    mock_logging_config = LoggingConfig(
        level="INFO",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        file=None
    )
    mock_get_logging_config.return_value = mock_logging_config

    # Call the function
    logger = setup_logging()

    # Verify the result
    assert isinstance(logger, logging.Logger)
    assert logger.name == "service-template"
    assert logger.level == logging.INFO

    # Verify the mock was called
    mock_get_logging_config.assert_called_once()

    # Verify the logger has a console handler
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)


@patch("common_lib.templates.service_template.logging_setup.get_logging_config")
def test_setup_logging_with_custom_config(mock_get_logging_config):
    """Test setup_logging with custom config."""
    # Create a custom logging config
    custom_logging_config = LoggingConfig(
        level="DEBUG",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        file=None
    )

    # Call the function with custom config
    logger = setup_logging(
        service_name="test-service",
        logging_config=custom_logging_config
    )

    # Verify the result
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test-service"
    assert logger.level == logging.DEBUG

    # Verify the mock was not called
    mock_get_logging_config.assert_not_called()

    # Verify the logger has a console handler
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)


@patch("common_lib.templates.service_template.logging_setup.get_logging_config")
@patch("common_lib.templates.service_template.logging_setup.os.makedirs")
@patch("common_lib.templates.service_template.logging_setup.logging.FileHandler")
def test_setup_logging_with_file(mock_file_handler, mock_makedirs, mock_get_logging_config):
    """Test setup_logging with file."""
    # Mock the get_logging_config function
    mock_logging_config = LoggingConfig(
        level="INFO",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        file="logs/test.log"
    )
    mock_get_logging_config.return_value = mock_logging_config

    # Mock the file handler
    mock_handler = MagicMock()
    mock_file_handler.return_value = mock_handler

    # Call the function
    logger = setup_logging()

    # Verify the result
    assert isinstance(logger, logging.Logger)
    assert logger.name == "service-template"
    assert logger.level == logging.INFO

    # Verify the mocks were called
    mock_get_logging_config.assert_called_once()
    mock_makedirs.assert_called_once()
    mock_file_handler.assert_called_once_with("logs/test.log")

    # Verify the logger has a console handler
    assert len(logger.handlers) >= 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
