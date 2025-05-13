"""
Forex LSTM Model Training Script.

This script trains a baseline LSTM model for forex price prediction
using advanced technical analysis features from the feature store.
"""
import os
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core_foundations.utils.logger import get_logger
from ml_workbench_service.clients.feature_store_client import FeatureStoreClient
from ml_workbench_service.model_registry.model_registry import ModelRegistry
from ml_workbench_service.models.forex_lstm_model import ForexLSTMModel
from ml_workbench_service.services.experiment_tracker import ExperimentTracker
logger = get_logger('forex_lstm_training')


from ml_workbench_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Forex LSTM Model')
    parser.add_argument('--symbol', type=str, default='EUR_USD', help=
        'Forex symbol to train model for')
    parser.add_argument('--timeframe', type=str, default='1h', help=
        'Timeframe for training data (e.g., 1h, 4h, 1d)')
    parser.add_argument('--train_days', type=int, default=365, help=
        'Number of days of data to use for training')
    parser.add_argument('--val_days', type=int, default=60, help=
        'Number of days of data to use for validation')
    parser.add_argument('--test_days', type=int, default=30, help=
        'Number of days of data to use for testing')
    parser.add_argument('--sequence_length', type=int, default=60, help=
        'Number of time steps to use for each prediction')
    parser.add_argument('--forecast_horizon', type=int, default=5, help=
        'Number of time steps to predict ahead')
    parser.add_argument('--lstm_layers', type=str, default='128,64', help=
        'Comma-separated list of LSTM units per layer')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help=
        'Dropout rate for regularization')
    parser.add_argument('--learning_rate', type=float, default=0.001, help=
        'Learning rate for optimizer')
    parser.add_argument('--batch_size', type=int, default=32, help=
        'Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help=
        'Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=10, help=
        'Patience for early stopping')
    parser.add_argument('--model_name', type=str, default=
        'forex_lstm_model', help='Name for the model in the registry')
    parser.add_argument('--output_dir', type=str, default='./models', help=
        'Directory to save model artifacts')
    parser.add_argument('--register_model', action='store_true', help=
        'Whether to register the model in the model registry')
    parser.add_argument('--track_experiment', action='store_true', help=
        'Whether to track the experiment with MLflow')
    return parser.parse_args()


@with_exception_handling
def fetch_data(client, symbol, timeframe, start_date, end_date):
    """Fetch data from feature store with advanced technical indicators."""
    logger.info(
        f'Fetching data for {symbol} ({timeframe}) from {start_date} to {end_date}'
        )
    try:
        ohlcv_data = client.get_historical_data(symbol=symbol, timeframe=
            timeframe, start_date=start_date, end_date=end_date)
        standard_indicators = client.get_standard_indicators(symbol=symbol,
            timeframe=timeframe, start_date=start_date, end_date=end_date,
            indicators=[{'name': 'sma', 'params': {'period': 20}}, {'name':
            'ema', 'params': {'period': 20}}, {'name': 'macd', 'params': {
            'fast_period': 12, 'slow_period': 26, 'signal_period': 9}}, {
            'name': 'adx', 'params': {'period': 14}}, {'name': 'atr',
            'params': {'period': 14}}, {'name': 'bollinger_bands', 'params':
            {'period': 20, 'std_dev': 2}}, {'name': 'rsi', 'params': {
            'period': 14}}, {'name': 'stochastic', 'params': {'k_period': 
            14, 'd_period': 3}}, {'name': 'cci', 'params': {'period': 20}}])
        advanced_features = client.get_advanced_features(symbol=symbol,
            timeframe=timeframe, start_date=start_date, end_date=end_date,
            features=[{'name': 'elliott_wave_pattern', 'params': {}}, {
            'name': 'elliott_wave_position', 'params': {}}, {'name':
            'fibonacci_retracement_levels', 'params': {}}, {'name':
            'fibonacci_extension_levels', 'params': {}}, {'name':
            'pivot_points', 'params': {'method': 'traditional'}}, {'name':
            'fractal_dimension', 'params': {'window': 30}}, {'name':
            'hurst_exponent', 'params': {'window': 50}}, {'name':
            'mtf_trend_alignment', 'params': {}}, {'name':
            'support_resistance_confluence', 'params': {}}, {'name':
            'indicator_confluence', 'params': {}}, {'name':
            'currency_correlation_features', 'params': {}}])
        data = ohlcv_data.merge(standard_indicators, on='datetime', how='left')
        data = data.merge(advanced_features, on='datetime', how='left')
        data = data.ffill().bfill()
        logger.info(
            f'Successfully fetched data with {len(data)} rows and {len(data.columns)} features'
            )
        return data
    except Exception as e:
        logger.error(f'Error fetching data: {str(e)}')
        raise


def train_model(args):
    """Train and evaluate the LSTM model."""
    end_date = datetime.now()
    test_start = end_date - timedelta(days=args.test_days)
    val_start = test_start - timedelta(days=args.val_days)
    train_start = val_start - timedelta(days=args.train_days)
    feature_client = FeatureStoreClient()
    train_data = fetch_data(feature_client, args.symbol, args.timeframe,
        train_start, val_start)
    val_data = fetch_data(feature_client, args.symbol, args.timeframe,
        val_start, test_start)
    test_data = fetch_data(feature_client, args.symbol, args.timeframe,
        test_start, end_date)
    lstm_units = [int(units) for units in args.lstm_layers.split(',')]
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    model_dir = os.path.join(args.output_dir, f'{args.model_name}_{timestamp}')
    os.makedirs(model_dir, exist_ok=True)
    experiment = None
    if args.track_experiment:
        experiment_tracker = ExperimentTracker()
        experiment = experiment_tracker.create_experiment(experiment_name=
            f'{args.model_name}_{args.symbol}_{args.timeframe}', tags={
            'model_type': 'lstm', 'symbol': args.symbol, 'timeframe': args.
            timeframe})
    if experiment:
        params = vars(args)
        params['train_size'] = len(train_data)
        params['val_size'] = len(val_data)
        params['test_size'] = len(test_data)
        params['feature_count'] = len(train_data.columns)
        experiment_tracker.log_parameters(params)
    logger.info('Initializing LSTM model')
    model = ForexLSTMModel(sequence_length=args.sequence_length,
        forecast_horizon=args.forecast_horizon, lstm_units=lstm_units,
        dropout_rate=args.dropout_rate, learning_rate=args.learning_rate,
        batch_size=args.batch_size, epochs=args.epochs, patience=args.
        patience, target_column='close')
    logger.info('Starting model training')
    model_save_path = os.path.join(model_dir, 'model.h5')
    training_results = model.train(train_data=train_data, validation_data=
        val_data, model_save_path=model_save_path)
    if experiment:
        experiment_tracker.log_metrics(training_results)
        if model.history:
            for metric, values in model.history.history.items():
                for epoch, value in enumerate(values):
                    experiment_tracker.log_metric(metric, value, step=epoch)
    logger.info('Evaluating model on test data')
    test_metrics = model.evaluate(test_data)
    if experiment:
        test_metric_dict = {f'test_{k}': v for k, v in test_metrics.items()}
        experiment_tracker.log_metrics(test_metric_dict)
    model.save(model_dir)
    if args.register_model:
        logger.info('Registering model in the model registry')
        registry = ModelRegistry()
        model_metadata = model.get_metadata()
        model_metadata.metadata.update({'symbol': args.symbol, 'timeframe':
            args.timeframe, 'training_date': datetime.now().isoformat(),
            'test_metrics': test_metrics, 'advanced_features_used': True,
            'training_samples': len(train_data), 'validation_samples': len(
            val_data), 'test_samples': len(test_data)})
        registry.register_model(name=args.model_name, version=timestamp,
            path=model_dir, metadata=model_metadata)
        registry.set_stage(model_name=args.model_name, version_id=timestamp,
            stage='staging')
        logger.info(
            f'Model registered with version ID: {timestamp}, stage: staging')
    logger.info('Training completed with the following results:')
    logger.info(f"Final training loss: {training_results['final_loss']:.6f}")
    logger.info(
        f"Final validation loss: {training_results['final_val_loss']:.6f}")
    logger.info(f"Test RMSE: {test_metrics['rmse_original_scale']:.6f}")
    logger.info(
        f"Direction accuracy: {test_metrics['direction_accuracy']:.2f}%")
    return {'model_dir': model_dir, 'training_results': training_results,
        'test_metrics': test_metrics}


if __name__ == '__main__':
    args = parse_arguments()
    train_model(args)
