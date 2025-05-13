"""
Load testing script for the Analysis Engine API.

This script performs load testing on the Analysis Engine API to evaluate
performance under various load conditions.

Usage:
    python load_test.py [--url=http://localhost:8000] [--duration=60] [--users=10] [--spawn-rate=1]
"""
import argparse
import json
import time
import random
import statistics
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class LoadTester:
    """Load tester for the Analysis Engine API."""

    def __init__(self, base_url: str='http://localhost:8000', api_key:
        Optional[str]=None, duration: int=60, users: int=10, spawn_rate: int=1
        ):
        """
        Initialize the load tester.
        
        Args:
            base_url: Base URL of the API
            api_key: API key for authentication
            duration: Test duration in seconds
            users: Number of concurrent users
            spawn_rate: User spawn rate per second
        """
        self.base_url = base_url
        self.api_key = api_key
        self.duration = duration
        self.users = users
        self.spawn_rate = spawn_rate
        self.headers = {'Content-Type': 'application/json'}
        if api_key:
            self.headers['X-API-Key'] = api_key
        self.results = {'confluence': [], 'divergence': [],
            'currency_strength': [], 'related_pairs': []}
        self.errors = {'confluence': [], 'divergence': [],
            'currency_strength': [], 'related_pairs': []}
        self.symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
            'EURGBP', 'EURJPY', 'GBPJPY']
        self.signal_types = ['trend', 'reversal', 'breakout']
        self.signal_directions = ['bullish', 'bearish']
        self.timeframes = ['M15', 'M30', 'H1', 'H4', 'D1']
        logger.info(
            f'Initialized load tester with {users} users, {duration}s duration, {spawn_rate} spawn rate'
            )

    def run(self):
        """Run the load test."""
        logger.info(f'Starting load test against {self.base_url}')
        start_time = time.time()
        end_time = start_time + self.duration
        with ThreadPoolExecutor(max_workers=self.users) as executor:
            futures = []
            for i in range(min(self.users, int(self.spawn_rate))):
                futures.append(executor.submit(self._user_task))
            spawned_users = min(self.users, int(self.spawn_rate))
            next_spawn_time = start_time + 1.0 / self.spawn_rate
            with tqdm(total=self.duration, desc='Load Test Progress') as pbar:
                while time.time() < end_time:
                    current_time = time.time()
                    if (current_time >= next_spawn_time and spawned_users <
                        self.users):
                        futures.append(executor.submit(self._user_task))
                        spawned_users += 1
                        next_spawn_time = current_time + 1.0 / self.spawn_rate
                    elapsed = min(int(current_time - start_time), self.duration
                        )
                    pbar.n = elapsed
                    pbar.refresh()
                    time.sleep(0.1)
        self._calculate_results()

    @with_exception_handling
    def _user_task(self):
        """Simulate a user making API requests."""
        while time.time() < time.time() + self.duration:
            endpoint = random.choice(['confluence', 'divergence',
                'currency_strength', 'related_pairs'])
            try:
                if endpoint == 'confluence':
                    self._test_confluence()
                elif endpoint == 'divergence':
                    self._test_divergence()
                elif endpoint == 'currency_strength':
                    self._test_currency_strength()
                elif endpoint == 'related_pairs':
                    self._test_related_pairs()
            except Exception as e:
                logger.error(f'Error in {endpoint} test: {e}')
                self.errors[endpoint].append(str(e))
            time.sleep(random.uniform(0.5, 2.0))

    def _test_confluence(self):
        """Test the confluence endpoint."""
        symbol = random.choice(self.symbols)
        signal_type = random.choice(self.signal_types)
        signal_direction = random.choice(self.signal_directions)
        timeframe = random.choice(self.timeframes)
        data = {'symbol': symbol, 'signal_type': signal_type,
            'signal_direction': signal_direction, 'timeframe': timeframe,
            'use_currency_strength': random.choice([True, False]),
            'min_confirmation_strength': random.uniform(0.2, 0.5)}
        start_time = time.time()
        response = requests.post(f'{self.base_url}/confluence', headers=
            self.headers, json=data)
        response_time = time.time() - start_time
        self.results['confluence'].append({'response_time': response_time,
            'status_code': response.status_code, 'timestamp': time.time()})

    def _test_divergence(self):
        """Test the divergence endpoint."""
        symbol = random.choice(self.symbols)
        timeframe = random.choice(self.timeframes)
        data = {'symbol': symbol, 'timeframe': timeframe}
        start_time = time.time()
        response = requests.post(f'{self.base_url}/divergence', headers=
            self.headers, json=data)
        response_time = time.time() - start_time
        self.results['divergence'].append({'response_time': response_time,
            'status_code': response.status_code, 'timestamp': time.time()})

    def _test_currency_strength(self):
        """Test the currency strength endpoint."""
        timeframe = random.choice(self.timeframes)
        method = random.choice(['momentum', 'trend', 'combined'])
        start_time = time.time()
        response = requests.get(
            f'{self.base_url}/currency-strength?timeframe={timeframe}&method={method}'
            , headers=self.headers)
        response_time = time.time() - start_time
        self.results['currency_strength'].append({'response_time':
            response_time, 'status_code': response.status_code, 'timestamp':
            time.time()})

    def _test_related_pairs(self):
        """Test the related pairs endpoint."""
        symbol = random.choice(self.symbols)
        min_correlation = random.uniform(0.5, 0.8)
        timeframe = random.choice(self.timeframes)
        start_time = time.time()
        response = requests.get(
            f'{self.base_url}/related-pairs/{symbol}?min_correlation={min_correlation}&timeframe={timeframe}'
            , headers=self.headers)
        response_time = time.time() - start_time
        self.results['related_pairs'].append({'response_time':
            response_time, 'status_code': response.status_code, 'timestamp':
            time.time()})

    def _calculate_results(self):
        """Calculate and print test results."""
        logger.info('Calculating test results...')
        os.makedirs('performance_results', exist_ok=True)
        stats = {}
        for endpoint, results in self.results.items():
            if not results:
                stats[endpoint] = {'count': 0, 'success_rate': 0, 'min': 0,
                    'max': 0, 'mean': 0, 'median': 0, 'p95': 0, 'p99': 0}
                continue
            response_times = [r['response_time'] for r in results]
            status_codes = [r['status_code'] for r in results]
            success_count = sum(1 for code in status_codes if 200 <= code < 300
                )
            stats[endpoint] = {'count': len(results), 'success_rate': 
                success_count / len(results) if results else 0, 'min': min(
                response_times) if response_times else 0, 'max': max(
                response_times) if response_times else 0, 'mean': 
                statistics.mean(response_times) if response_times else 0,
                'median': statistics.median(response_times) if
                response_times else 0, 'p95': np.percentile(response_times,
                95) if response_times else 0, 'p99': np.percentile(
                response_times, 99) if response_times else 0}
            plt.figure(figsize=(10, 6))
            plt.hist(response_times, bins=20, alpha=0.7)
            plt.axvline(stats[endpoint]['mean'], color='r', linestyle=
                'dashed', linewidth=1, label=
                f"Mean: {stats[endpoint]['mean']:.3f}s")
            plt.axvline(stats[endpoint]['p95'], color='g', linestyle=
                'dashed', linewidth=1, label=
                f"95th: {stats[endpoint]['p95']:.3f}s")
            plt.axvline(stats[endpoint]['p99'], color='b', linestyle=
                'dashed', linewidth=1, label=
                f"99th: {stats[endpoint]['p99']:.3f}s")
            plt.title(f'Response Time Distribution - {endpoint.capitalize()}')
            plt.xlabel('Response Time (seconds)')
            plt.ylabel('Count')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(
                f'performance_results/{endpoint}_response_time_distribution.png'
                )
            timestamps = [r['timestamp'] for r in results]
            relative_timestamps = [(t - timestamps[0]) for t in timestamps]
            plt.figure(figsize=(10, 6))
            plt.scatter(relative_timestamps, response_times, alpha=0.7, s=10)
            plt.title(f'Response Time Over Time - {endpoint.capitalize()}')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Response Time (seconds)')
            plt.grid(True, alpha=0.3)
            plt.savefig(
                f'performance_results/{endpoint}_response_time_over_time.png')
        logger.info('\nLoad Test Results Summary:')
        logger.info(f'Duration: {self.duration} seconds')
        logger.info(f'Users: {self.users}')
        logger.info(f'Spawn Rate: {self.spawn_rate} users/second')
        for endpoint, stat in stats.items():
            logger.info(f'\n{endpoint.capitalize()}:')
            logger.info(f"  Requests: {stat['count']}")
            logger.info(f"  Success Rate: {stat['success_rate']:.2%}")
            logger.info(f"  Min: {stat['min']:.3f}s")
            logger.info(f"  Max: {stat['max']:.3f}s")
            logger.info(f"  Mean: {stat['mean']:.3f}s")
            logger.info(f"  Median: {stat['median']:.3f}s")
            logger.info(f"  95th Percentile: {stat['p95']:.3f}s")
            logger.info(f"  99th Percentile: {stat['p99']:.3f}s")
            logger.info(f'  Errors: {len(self.errors[endpoint])}')
        with open('performance_results/load_test_results.json', 'w') as f:
            json.dump({'config': {'base_url': self.base_url, 'duration':
                self.duration, 'users': self.users, 'spawn_rate': self.
                spawn_rate}, 'stats': stats, 'errors': self.errors}, f,
                indent=2)
        logger.info(
            '\nResults saved to performance_results/load_test_results.json')
        logger.info('Plots saved to performance_results/ directory')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
        'Load test the Analysis Engine API')
    parser.add_argument('--url', type=str, default='http://localhost:8000',
        help='Base URL of the API')
    parser.add_argument('--api-key', type=str, help=
        'API key for authentication')
    parser.add_argument('--duration', type=int, default=60, help=
        'Test duration in seconds')
    parser.add_argument('--users', type=int, default=10, help=
        'Number of concurrent users')
    parser.add_argument('--spawn-rate', type=float, default=1, help=
        'User spawn rate per second')
    args = parser.parse_args()
    tester = LoadTester(base_url=args.url, api_key=args.api_key, duration=
        args.duration, users=args.users, spawn_rate=args.spawn_rate)
    tester.run()
