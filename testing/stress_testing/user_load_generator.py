#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
User Load Generator for Stress Testing

This module implements a simulated user load generator for stress testing the
forex trading platform. It creates virtual users that perform realistic user actions
such as logging in, requesting market data, placing orders, and analyzing portfolio
performance with configurable patterns and frequencies.

The generator supports both synchronous and asynchronous interfaces and can
simulate various user behavior patterns and connection scenarios.
"""

import asyncio
import json
import logging
import random
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union, Callable, Any, AsyncGenerator, Generator

import aiohttp
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from .environment_config import StressLevel

# Configure logging
logger = logging.getLogger(__name__)


class UserActivityType(Enum):
    """Types of user activities that can be simulated."""
    LOGIN = "login"
    VIEW_MARKET_DATA = "view_market_data"
    PLACE_ORDER = "place_order"
    CANCEL_ORDER = "cancel_order"
    MODIFY_ORDER = "modify_order"
    VIEW_PORTFOLIO = "view_portfolio"
    RUN_ANALYSIS = "run_analysis"
    DOWNLOAD_REPORT = "download_report"
    CONFIGURE_ALERTS = "configure_alerts"
    VIEW_CHARTS = "view_charts"
    EXECUTE_STRATEGY = "execute_strategy"
    LOGOUT = "logout"


class UserType(Enum):
    """Types of users with different behavior patterns."""
    RETAIL = "retail"  # Individual retail traders
    INSTITUTIONAL = "institutional"  # Institutional clients
    ALGORITHMIC = "algorithmic"  # Algorithm-based trading
    HFT = "high_frequency"  # High-frequency traders
    API_CONSUMER = "api_consumer"  # External systems using APIs


@dataclass
class UserProfile:
    """Represents a simulated user profile with behavior parameters."""
    user_id: str
    user_type: UserType
    activities: Dict[UserActivityType, float] = field(default_factory=dict)  # Activity -> frequency
    session_duration_mean: float = 300.0  # Mean session time in seconds
    session_duration_std: float = 60.0  # Standard deviation of session time
    think_time_mean: float = 5.0  # Mean time between activities in seconds
    think_time_std: float = 2.0  # Standard deviation of think time
    error_injection_rate: float = 0.05  # Rate of deliberate error injection
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "user_id": self.user_id,
            "user_type": self.user_type.value,
            "activities": {activity.value: freq for activity, freq in self.activities.items()},
            "session_duration_mean": self.session_duration_mean,
            "session_duration_std": self.session_duration_std,
            "think_time_mean": self.think_time_mean,
            "think_time_std": self.think_time_std,
            "error_injection_rate": self.error_injection_rate
        }


@dataclass
class LoadProfile:
    """Defines the load characteristics to be generated."""
    concurrent_users: int
    ramp_up_time: float  # Time in seconds to reach full load
    steady_state_time: float  # Time in seconds to maintain full load
    ramp_down_time: float  # Time in seconds to decrease to zero load
    user_distribution: Dict[UserType, float] = field(default_factory=dict)  # UserType -> percentage
    connection_pattern: str = "constant"  # constant, cyclic, random, spike
    geographic_distribution: Dict[str, float] = field(default_factory=dict)  # Region -> percentage
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "concurrent_users": self.concurrent_users,
            "ramp_up_time": self.ramp_up_time,
            "steady_state_time": self.steady_state_time,
            "ramp_down_time": self.ramp_down_time,
            "user_distribution": {user_type.value: pct for user_type, pct in self.user_distribution.items()},
            "connection_pattern": self.connection_pattern,
            "geographic_distribution": self.geographic_distribution
        }


class UserLoadGenerator(ABC):
    """Base abstract class for user load generators."""
    
    def __init__(self, 
                 load_profile: LoadProfile,
                 endpoints: Dict[str, str],
                 credentials: Optional[Dict[str, str]] = None,
                 stress_level: StressLevel = StressLevel.MODERATE):
        """
        Initialize the user load generator.
        
        Args:
            load_profile: Definition of the load to be generated
            endpoints: Dictionary of service endpoints to target
            credentials: Optional credentials for authentication
            stress_level: Intensity level of the stress test
        """
        self.load_profile = load_profile
        self.endpoints = endpoints
        self.credentials = credentials or {}
        self.stress_level = stress_level
        self.users: List[UserProfile] = []
        self._active_sessions: Set[str] = set()
        self._stop_event = asyncio.Event()
        
        # Generate user profiles based on load profile
        self._generate_user_profiles()
    
    def _generate_user_profiles(self) -> None:
        """Generate user profiles based on the load profile configuration."""
        self.users = []
        for user_type, percentage in self.load_profile.user_distribution.items():
            num_users = int(self.load_profile.concurrent_users * percentage)
            
            for i in range(num_users):
                user_id = f"{user_type.value}_{uuid.uuid4()}"
                
                # Define activity distribution based on user type
                activities = self._generate_activity_distribution(user_type)
                
                # Create user profile with type-specific parameters
                profile = UserProfile(
                    user_id=user_id,
                    user_type=user_type,
                    activities=activities,
                    # Customize other parameters based on user type
                    **self._get_user_type_parameters(user_type)
                )
                self.users.append(profile)
        
        # If we didn't generate enough users due to rounding, add more of the most common type
        if len(self.users) < self.load_profile.concurrent_users:
            most_common_type = max(self.load_profile.user_distribution.items(), key=lambda x: x[1])[0]
            remaining = self.load_profile.concurrent_users - len(self.users)
            
            for i in range(remaining):
                user_id = f"{most_common_type.value}_{uuid.uuid4()}"
                activities = self._generate_activity_distribution(most_common_type)
                profile = UserProfile(
                    user_id=user_id,
                    user_type=most_common_type,
                    activities=activities,
                    **self._get_user_type_parameters(most_common_type)
                )
                self.users.append(profile)
    
    def _generate_activity_distribution(self, user_type: UserType) -> Dict[UserActivityType, float]:
        """Generate activity distribution based on user type."""
        activities = {}
        
        if user_type == UserType.RETAIL:
            activities = {
                UserActivityType.LOGIN: 0.05,
                UserActivityType.VIEW_MARKET_DATA: 0.3,
                UserActivityType.PLACE_ORDER: 0.15,
                UserActivityType.CANCEL_ORDER: 0.05,
                UserActivityType.VIEW_PORTFOLIO: 0.2,
                UserActivityType.VIEW_CHARTS: 0.2,
                UserActivityType.LOGOUT: 0.05
            }
        elif user_type == UserType.INSTITUTIONAL:
            activities = {
                UserActivityType.LOGIN: 0.02,
                UserActivityType.VIEW_MARKET_DATA: 0.2,
                UserActivityType.PLACE_ORDER: 0.3,
                UserActivityType.MODIFY_ORDER: 0.1,
                UserActivityType.RUN_ANALYSIS: 0.15,
                UserActivityType.DOWNLOAD_REPORT: 0.15,
                UserActivityType.EXECUTE_STRATEGY: 0.05,
                UserActivityType.LOGOUT: 0.03
            }
        elif user_type == UserType.ALGORITHMIC:
            activities = {
                UserActivityType.LOGIN: 0.01,
                UserActivityType.VIEW_MARKET_DATA: 0.4,
                UserActivityType.PLACE_ORDER: 0.3,
                UserActivityType.MODIFY_ORDER: 0.1,
                UserActivityType.CANCEL_ORDER: 0.1,
                UserActivityType.EXECUTE_STRATEGY: 0.08,
                UserActivityType.LOGOUT: 0.01
            }
        elif user_type == UserType.HFT:
            activities = {
                UserActivityType.LOGIN: 0.01,
                UserActivityType.VIEW_MARKET_DATA: 0.45,
                UserActivityType.PLACE_ORDER: 0.45,
                UserActivityType.CANCEL_ORDER: 0.08,
                UserActivityType.LOGOUT: 0.01
            }
        elif user_type == UserType.API_CONSUMER:
            activities = {
                UserActivityType.LOGIN: 0.02,
                UserActivityType.VIEW_MARKET_DATA: 0.3,
                UserActivityType.PLACE_ORDER: 0.2,
                UserActivityType.VIEW_PORTFOLIO: 0.2,
                UserActivityType.DOWNLOAD_REPORT: 0.1,
                UserActivityType.RUN_ANALYSIS: 0.15,
                UserActivityType.LOGOUT: 0.03
            }
            
        return activities
    
    def _get_user_type_parameters(self, user_type: UserType) -> Dict[str, Any]:
        """Get user profile parameters specific to user type."""
        params = {}
        
        if user_type == UserType.RETAIL:
            params = {
                "session_duration_mean": 900,  # 15 minutes
                "session_duration_std": 300,
                "think_time_mean": 10.0,
                "think_time_std": 5.0,
                "error_injection_rate": 0.08
            }
        elif user_type == UserType.INSTITUTIONAL:
            params = {
                "session_duration_mean": 1800,  # 30 minutes
                "session_duration_std": 600,
                "think_time_mean": 5.0,
                "think_time_std": 2.0,
                "error_injection_rate": 0.03
            }
        elif user_type == UserType.ALGORITHMIC:
            params = {
                "session_duration_mean": 7200,  # 2 hours
                "session_duration_std": 1800,
                "think_time_mean": 0.5,
                "think_time_std": 0.2,
                "error_injection_rate": 0.02
            }
        elif user_type == UserType.HFT:
            params = {
                "session_duration_mean": 14400,  # 4 hours
                "session_duration_std": 3600,
                "think_time_mean": 0.01,
                "think_time_std": 0.005,
                "error_injection_rate": 0.01
            }
        elif user_type == UserType.API_CONSUMER:
            params = {
                "session_duration_mean": 3600,  # 1 hour
                "session_duration_std": 900,
                "think_time_mean": 1.0,
                "think_time_std": 0.5,
                "error_injection_rate": 0.05
            }
            
        return params
    
    @abstractmethod
    async def start(self) -> None:
        """Start the load generation."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the load generation."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the current load test."""
        pass


class AsyncUserLoadGenerator(UserLoadGenerator):
    """Asynchronous implementation of user load generator."""
    
    def __init__(self, 
                 load_profile: LoadProfile,
                 endpoints: Dict[str, str],
                 credentials: Optional[Dict[str, str]] = None,
                 stress_level: StressLevel = StressLevel.MODERATE):
        """Initialize the asynchronous user load generator."""
        super().__init__(load_profile, endpoints, credentials, stress_level)
        self._session: Optional[aiohttp.ClientSession] = None
        self._active_tasks: Set[asyncio.Task] = set()
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": [],
            "active_sessions": 0,
            "completed_sessions": 0,
            "start_time": None,
            "end_time": None,
            "activities": {activity.value: 0 for activity in UserActivityType}
        }
    
    async def start(self) -> None:
        """Start the load generation."""
        if self._session is not None:
            return  # Already started
        
        self._session = aiohttp.ClientSession()
        self._stats["start_time"] = datetime.now()
        self._stop_event.clear()
        
        # Start the load according to the specified pattern
        if self.load_profile.connection_pattern == "constant":
            await self._start_constant_load()
        elif self.load_profile.connection_pattern == "cyclic":
            await self._start_cyclic_load()
        elif self.load_profile.connection_pattern == "random":
            await self._start_random_load()
        elif self.load_profile.connection_pattern == "spike":
            await self._start_spike_load()
        else:
            # Default to constant load
            await self._start_constant_load()
    
    async def _start_constant_load(self) -> None:
        """Start a constant user load."""
        # Ramp up phase
        num_users = len(self.users)
        if self.load_profile.ramp_up_time > 0:
            delay_per_user = self.load_profile.ramp_up_time / num_users
            for i, user in enumerate(self.users):
                if self._stop_event.is_set():
                    break
                task = asyncio.create_task(self._simulate_user_session(user))
                self._active_tasks.add(task)
                task.add_done_callback(self._active_tasks.discard)
                await asyncio.sleep(delay_per_user)
        else:
            # Start all users immediately
            for user in self.users:
                if self._stop_event.is_set():
                    break
                task = asyncio.create_task(self._simulate_user_session(user))
                self._active_tasks.add(task)
                task.add_done_callback(self._active_tasks.discard)
        
        # Steady state - just wait
        if not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), 
                                      timeout=self.load_profile.steady_state_time)
            except asyncio.TimeoutError:
                pass  # Expected timeout, steady state duration completed
    
    async def _start_cyclic_load(self) -> None:
        """Start a cyclical user load that varies over time."""
        cycles = 3  # Number of complete sine waves during the steady state
        cycle_duration = self.load_profile.steady_state_time / cycles
        cycle_amplitude = len(self.users) * 0.5  # 50% variation
        base_users = len(self.users) * 0.5  # 50% base load
        
        # Ramp up to base load
        ramp_up_delay_per_user = self.load_profile.ramp_up_time / base_users
        for i in range(int(base_users)):
            if self._stop_event.is_set():
                break
            task = asyncio.create_task(self._simulate_user_session(self.users[i]))
            self._active_tasks.add(task)
            task.add_done_callback(self._active_tasks.discard)
            await asyncio.sleep(ramp_up_delay_per_user)
        
        # Cyclic load during steady state
        start_time = time.time()
        end_time = start_time + self.load_profile.steady_state_time
        
        while time.time() < end_time and not self._stop_event.is_set():
            elapsed = time.time() - start_time
            cycle_position = (elapsed % cycle_duration) / cycle_duration
            
            # Calculate target users with sine wave
            target_users = int(base_users + cycle_amplitude * np.sin(2 * np.pi * cycle_position))
            target_users = max(1, min(target_users, len(self.users)))
            
            # Adjust current users to target
            current_users = len(self._active_sessions)
            
            if current_users < target_users:
                # Add users
                users_to_add = target_users - current_users
                for i in range(users_to_add):
                    if self._stop_event.is_set() or len(self._active_sessions) >= len(self.users):
                        break
                    
                    # Find an inactive user
                    for user in self.users:
                        if user.user_id not in self._active_sessions:
                            task = asyncio.create_task(self._simulate_user_session(user))
                            self._active_tasks.add(task)
                            task.add_done_callback(self._active_tasks.discard)
                            break
            
            await asyncio.sleep(cycle_duration / 20)  # Check 20 times per cycle
    
    async def _start_random_load(self) -> None:
        """Start a random user load pattern."""
        # Ramp up to initial random number of users (between 10%-90% of max)
        initial_users = random.randint(
            int(0.1 * len(self.users)),
            int(0.9 * len(self.users))
        )
        
        ramp_up_delay_per_user = self.load_profile.ramp_up_time / initial_users
        for i in range(initial_users):
            if self._stop_event.is_set():
                break
            task = asyncio.create_task(self._simulate_user_session(self.users[i]))
            self._active_tasks.add(task)
            task.add_done_callback(self._active_tasks.discard)
            await asyncio.sleep(ramp_up_delay_per_user)
        
        # Random fluctuations during steady state
        num_changes = int(self.load_profile.steady_state_time / 30)  # Change every ~30 seconds
        
        for _ in range(num_changes):
            if self._stop_event.is_set():
                break
                
            # Random target between 10% and 100% of users
            target_users = random.randint(
                int(0.1 * len(self.users)),
                len(self.users)
            )
            
            current_users = len(self._active_sessions)
            
            if current_users < target_users:
                # Add users
                users_to_add = target_users - current_users
                for i in range(users_to_add):
                    if self._stop_event.is_set() or len(self._active_sessions) >= len(self.users):
                        break
                    
                    # Find an inactive user
                    for user in self.users:
                        if user.user_id not in self._active_sessions:
                            task = asyncio.create_task(self._simulate_user_session(user))
                            self._active_tasks.add(task)
                            task.add_done_callback(self._active_tasks.discard)
                            break
            
            await asyncio.sleep(30 + random.uniform(-5, 5))  # ~30 seconds between changes
    
    async def _start_spike_load(self) -> None:
        """Start a spike load pattern with sudden surges in users."""
        # Divide steady state into normal periods and spike periods
        steady_duration = self.load_profile.steady_state_time
        base_load_percentage = 0.3  # 30% of users during normal periods
        spike_percentage = 0.9  # 90% of users during spikes
        
        # Start with base load
        base_users = int(len(self.users) * base_load_percentage)
        ramp_up_delay_per_user = self.load_profile.ramp_up_time / base_users
        
        for i in range(base_users):
            if self._stop_event.is_set():
                break
            task = asyncio.create_task(self._simulate_user_session(self.users[i]))
            self._active_tasks.add(task)
            task.add_done_callback(self._active_tasks.discard)
            await asyncio.sleep(ramp_up_delay_per_user)
        
        # Schedule 2-3 spikes during the steady state
        num_spikes = random.randint(2, 3)
        steady_state_start = time.time()
        
        for spike_num in range(num_spikes):
            if self._stop_event.is_set():
                break
                
            # Wait until spike start time
            spike_delay = steady_duration * (spike_num + 1) / (num_spikes + 1)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=spike_delay)
            except asyncio.TimeoutError:
                pass  # Expected timeout
                
            if self._stop_event.is_set():
                break
                
            # Calculate additional users needed for spike
            spike_users = int(len(self.users) * spike_percentage)
            current_users = len(self._active_sessions)
            users_to_add = spike_users - current_users
            
            logger.info(f"Generating load spike: Adding {users_to_add} users")
            
            # Add users rapidly (within 5-15 seconds)
            spike_ramp_up = random.uniform(5, 15)
            delay_per_spike_user = spike_ramp_up / users_to_add if users_to_add > 0 else 0
            
            for i in range(users_to_add):
                if self._stop_event.is_set() or len(self._active_sessions) >= len(self.users):
                    break
                
                # Find an inactive user
                for user in self.users:
                    if user.user_id not in self._active_sessions:
                        task = asyncio.create_task(self._simulate_user_session(
                            user, 
                            session_duration_override=random.uniform(30, 90)  # Short spike sessions
                        ))
                        self._active_tasks.add(task)
                        task.add_done_callback(self._active_tasks.discard)
                        break
                
                await asyncio.sleep(delay_per_spike_user)
            
            # Wait for spike duration (30-90 seconds)
            spike_duration = random.uniform(30, 90)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=spike_duration)
            except asyncio.TimeoutError:
                pass  # Expected timeout
    
    async def _simulate_user_session(self, 
                                   user: UserProfile, 
                                   session_duration_override: Optional[float] = None) -> None:
        """Simulate a single user session with the given profile."""
        user_id = user.user_id
        
        if user_id in self._active_sessions:
            return  # User already has an active session
        
        self._active_sessions.add(user_id)
        self._stats["active_sessions"] += 1
        
        try:
            # Determine session duration (with some randomness)
            if session_duration_override is not None:
                session_duration = session_duration_override
            else:
                session_duration = max(
                    10.0,  # Minimum session time
                    np.random.normal(user.session_duration_mean, user.session_duration_std)
                )
            
            session_start_time = time.time()
            session_end_time = session_start_time + session_duration
            
            # Simulate login activity
            await self._perform_activity(user, UserActivityType.LOGIN)
            
            # Perform activities until session ends
            while time.time() < session_end_time and not self._stop_event.is_set():
                # Select a random activity based on frequencies
                activity = self._select_random_activity(user)
                
                # Skip login/logout activities during the main session
                if activity in [UserActivityType.LOGIN, UserActivityType.LOGOUT]:
                    continue
                
                # Perform the activity
                await self._perform_activity(user, activity)
                
                # Wait for "think time" before next activity
                think_time = max(
                    0.1,  # Minimum think time
                    np.random.normal(user.think_time_mean, user.think_time_std)
                )
                await asyncio.sleep(think_time)
            
            # Simulate logout activity at the end of the session
            await self._perform_activity(user, UserActivityType.LOGOUT)
            
        except Exception as e:
            logger.error(f"Error in user session {user_id}: {str(e)}")
        finally:
            self._active_sessions.remove(user_id)
            self._stats["active_sessions"] -= 1
            self._stats["completed_sessions"] += 1
    
    def _select_random_activity(self, user: UserProfile) -> UserActivityType:
        """Select a random activity based on the frequency distribution."""
        activities = list(user.activities.keys())
        weights = list(user.activities.values())
        
        # Normalize weights if they don't sum to 1
        total_weight = sum(weights)
        if total_weight != 1.0:
            weights = [w / total_weight for w in weights]
            
        return random.choices(activities, weights=weights, k=1)[0]
    
    async def _perform_activity(self, user: UserProfile, activity: UserActivityType) -> None:
        """Perform a specific user activity by making appropriate API calls."""
        if not self._session:
            return
            
        start_time = time.time()
        error_injected = random.random() < user.error_injection_rate
        
        try:
            self._stats["total_requests"] += 1
            self._stats["activities"][activity.value] += 1
            
            if activity == UserActivityType.LOGIN:
                endpoint = self.endpoints.get("auth", "http://localhost:8080/api/auth")
                payload = {"username": f"user_{user.user_id}", "password": "password123"}
                
                if error_injected:
                    payload["password"] = "wrong_password"  # Deliberate error
                
                async with self._session.post(endpoint, json=payload) as response:
                    _ = await response.json()
                    
            elif activity == UserActivityType.VIEW_MARKET_DATA:
                endpoint = self.endpoints.get("market_data", "http://localhost:8080/api/market-data")
                params = {"symbols": "EUR/USD,GBP/USD,USD/JPY"}
                
                if error_injected:
                    params["symbols"] = "INVALID/PAIR"  # Deliberate error
                    
                async with self._session.get(endpoint, params=params) as response:
                    _ = await response.json()
                    
            elif activity == UserActivityType.PLACE_ORDER:
                endpoint = self.endpoints.get("orders", "http://localhost:8080/api/orders")
                order_data = {
                    "symbol": "EUR/USD",
                    "side": "BUY",
                    "type": "MARKET",
                    "quantity": round(random.uniform(0.1, 10.0), 2)
                }
                
                if error_injected:
                    order_data["quantity"] = -1.0  # Deliberate error
                    
                async with self._session.post(endpoint, json=order_data) as response:
                    _ = await response.json()
                    
            elif activity == UserActivityType.CANCEL_ORDER:
                endpoint = self.endpoints.get("orders", "http://localhost:8080/api/orders")
                order_id = f"order_{uuid.uuid4()}"
                
                if error_injected:
                    order_id = "non_existent_order"  # Deliberate error
                    
                async with self._session.delete(f"{endpoint}/{order_id}") as response:
                    _ = await response.text()
                    
            elif activity == UserActivityType.MODIFY_ORDER:
                endpoint = self.endpoints.get("orders", "http://localhost:8080/api/orders")
                order_id = f"order_{uuid.uuid4()}"
                order_data = {
                    "quantity": round(random.uniform(0.1, 10.0), 2),
                    "price": round(random.uniform(1.0, 1.5), 4)
                }
                
                if error_injected:
                    order_data["price"] = -1.0  # Deliberate error
                    
                async with self._session.put(f"{endpoint}/{order_id}", json=order_data) as response:
                    _ = await response.json()
                    
            elif activity == UserActivityType.VIEW_PORTFOLIO:
                endpoint = self.endpoints.get("portfolio", "http://localhost:8080/api/portfolio")
                
                async with self._session.get(endpoint) as response:
                    _ = await response.json()
                    
            elif activity == UserActivityType.RUN_ANALYSIS:
                endpoint = self.endpoints.get("analysis", "http://localhost:8080/api/analysis")
                analysis_data = {
                    "type": "technical",
                    "symbol": "EUR/USD",
                    "timeframe": "1h",
                    "indicators": ["SMA", "EMA", "RSI"]
                }
                
                if error_injected:
                    analysis_data["indicators"] = ["INVALID_INDICATOR"]  # Deliberate error
                    
                async with self._session.post(endpoint, json=analysis_data) as response:
                    _ = await response.json()
                    
            elif activity == UserActivityType.DOWNLOAD_REPORT:
                endpoint = self.endpoints.get("reports", "http://localhost:8080/api/reports")
                params = {
                    "type": "performance",
                    "period": "1w",
                    "format": "csv"
                }
                
                if error_injected:
                    params["format"] = "invalid_format"  # Deliberate error
                    
                async with self._session.get(endpoint, params=params) as response:
                    _ = await response.read()
                    
            elif activity == UserActivityType.CONFIGURE_ALERTS:
                endpoint = self.endpoints.get("alerts", "http://localhost:8080/api/alerts")
                alert_data = {
                    "symbol": "EUR/USD",
                    "condition": "price_above",
                    "value": round(random.uniform(1.0, 1.5), 4),
                    "notification_type": "email"
                }
                
                if error_injected:
                    alert_data["condition"] = "invalid_condition"  # Deliberate error
                    
                async with self._session.post(endpoint, json=alert_data) as response:
                    _ = await response.json()
                    
            elif activity == UserActivityType.VIEW_CHARTS:
                endpoint = self.endpoints.get("charts", "http://localhost:8080/api/charts")
                params = {
                    "symbol": "EUR/USD",
                    "timeframe": "1h",
                    "from": "2023-01-01",
                    "to": "2023-01-31"
                }
                
                if error_injected:
                    params["timeframe"] = "invalid_timeframe"  # Deliberate error
                    
                async with self._session.get(endpoint, params=params) as response:
                    _ = await response.json()
                    
            elif activity == UserActivityType.EXECUTE_STRATEGY:
                endpoint = self.endpoints.get("strategies", "http://localhost:8080/api/strategies")
                strategy_data = {
                    "name": "MovingAverageCrossover",
                    "parameters": {
                        "fast_period": 10,
                        "slow_period": 30
                    },
                    "symbols": ["EUR/USD"],
                    "timeframe": "1h"
                }
                
                if error_injected:
                    strategy_data["parameters"] = {}  # Deliberate error
                    
                async with self._session.post(endpoint, json=strategy_data) as response:
                    _ = await response.json()
                    
            elif activity == UserActivityType.LOGOUT:
                endpoint = self.endpoints.get("auth", "http://localhost:8080/api/auth/logout")
                
                async with self._session.post(endpoint) as response:
                    _ = await response.text()
            
            # Record successful request
            self._stats["successful_requests"] += 1
            
        except Exception as e:
            # Record failed request
            self._stats["failed_requests"] += 1
            logger.error(f"Error performing activity {activity.value}: {str(e)}")
        finally:
            # Record response time
            response_time = time.time() - start_time
            self._stats["response_times"].append(response_time)
    
    async def stop(self) -> None:
        """Stop the load generation."""
        self._stop_event.set()
        
        # Wait for all active tasks to complete (with timeout)
        if self._active_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._active_tasks, return_exceptions=True),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                for task in self._active_tasks:
                    if not task.done():
                        task.cancel()
        
        # Close the HTTP session
        if self._session:
            await self._session.close()
            self._session = None
        
        self._stats["end_time"] = datetime.now()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the current load test."""
        stats = self._stats.copy()
        
        # Calculate derived statistics
        total_requests = stats["total_requests"]
        if total_requests > 0:
            stats["success_rate"] = stats["successful_requests"] / total_requests
            stats["error_rate"] = stats["failed_requests"] / total_requests
        
        response_times = stats["response_times"]
        if response_times:
            stats["avg_response_time"] = sum(response_times) / len(response_times)
            stats["min_response_time"] = min(response_times)
            stats["max_response_time"] = max(response_times)
            
            # Calculate percentiles
            stats["p50_response_time"] = np.percentile(response_times, 50)
            stats["p90_response_time"] = np.percentile(response_times, 90)
            stats["p95_response_time"] = np.percentile(response_times, 95)
            stats["p99_response_time"] = np.percentile(response_times, 99)
        
        # Calculate test duration
        if stats["start_time"] and stats["end_time"]:
            duration = (stats["end_time"] - stats["start_time"]).total_seconds()
            stats["duration_seconds"] = duration
            if duration > 0 and total_requests > 0:
                stats["requests_per_second"] = total_requests / duration
        
        return stats


class SyncUserLoadGenerator(UserLoadGenerator):
    """Synchronous implementation of user load generator for simpler use cases."""
    
    def __init__(self, 
                 load_profile: LoadProfile,
                 endpoints: Dict[str, str],
                 credentials: Optional[Dict[str, str]] = None,
                 stress_level: StressLevel = StressLevel.MODERATE):
        """Initialize the synchronous user load generator."""
        super().__init__(load_profile, endpoints, credentials, stress_level)
        self._executor = ThreadPoolExecutor(max_workers=max(100, self.load_profile.concurrent_users))
        self._running = False
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": [],
            "active_sessions": 0,
            "completed_sessions": 0,
            "start_time": None,
            "end_time": None
        }
        self._stats_lock = asyncio.Lock()
    
    async def start(self) -> None:
        """Start the load generation in a separate event loop."""
        loop = asyncio.get_event_loop()
        asyncio.run_coroutine_threadsafe(self._async_start(), loop)
    
    async def _async_start(self) -> None:
        """Internal method to start load generation."""
        if self._running:
            return  # Already started
            
        self._running = True
        self._stats["start_time"] = datetime.now()
        self._stop_event.clear()
        
        # Convert to async to reuse the patterns from AsyncUserLoadGenerator
        if self.load_profile.connection_pattern == "constant":
            await self._start_constant_load()
        elif self.load_profile.connection_pattern == "cyclic":
            await self._start_cyclic_load()
        elif self.load_profile.connection_pattern == "random":
            await self._start_random_load()
        elif self.load_profile.connection_pattern == "spike":
            await self._start_spike_load()
        else:
            # Default to constant load
            await self._start_constant_load()
    
    async def stop(self) -> None:
        """Stop the load generation."""
        self._stop_event.set()
        self._running = False
        self._stats["end_time"] = datetime.now()
        self._executor.shutdown(wait=False)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the current load test."""
        async with self._stats_lock:
            stats = self._stats.copy()
        
        # Calculate derived statistics
        total_requests = stats["total_requests"]
        if total_requests > 0:
            stats["success_rate"] = stats["successful_requests"] / total_requests
            stats["error_rate"] = stats["failed_requests"] / total_requests
        
        response_times = stats["response_times"]
        if response_times:
            stats["avg_response_time"] = sum(response_times) / len(response_times)
            stats["min_response_time"] = min(response_times)
            stats["max_response_time"] = max(response_times)
            
            # Calculate percentiles
            stats["p50_response_time"] = np.percentile(response_times, 50)
            stats["p90_response_time"] = np.percentile(response_times, 90)
            stats["p95_response_time"] = np.percentile(response_times, 95)
            stats["p99_response_time"] = np.percentile(response_times, 99)
        
        # Calculate test duration
        if stats["start_time"] and stats["end_time"]:
            duration = (stats["end_time"] - stats["start_time"]).total_seconds()
            stats["duration_seconds"] = duration
            if duration > 0 and total_requests > 0:
                stats["requests_per_second"] = total_requests / duration
        
        return stats
    
    # Implementation of the load patterns
    # Use the same pattern logic as AsyncUserLoadGenerator
    async def _start_constant_load(self) -> None:
        """Start a constant user load."""
        # Ramp up phase - similar to async version but using ThreadPoolExecutor
        # ...
        pass
    
    async def _start_cyclic_load(self) -> None:
        """Start a cyclical user load that varies over time."""
        # Similar to async version but using ThreadPoolExecutor
        # ...
        pass
    
    async def _start_random_load(self) -> None:
        """Start a random user load pattern."""
        # Similar to async version but using ThreadPoolExecutor
        # ...
        pass
    
    async def _start_spike_load(self) -> None:
        """Start a spike load pattern with sudden surges in users."""
        # Similar to async version but using ThreadPoolExecutor
        # ...
        pass


def create_default_load_profile(concurrent_users: int = 100) -> LoadProfile:
    """Create a default load profile for testing."""
    return LoadProfile(
        concurrent_users=concurrent_users,
        ramp_up_time=30.0,  # 30 seconds to ramp up
        steady_state_time=300.0,  # 5 minutes of steady state
        ramp_down_time=30.0,  # 30 seconds to ramp down
        user_distribution={
            UserType.RETAIL: 0.6,  # 60% retail users
            UserType.INSTITUTIONAL: 0.2,  # 20% institutional users
            UserType.ALGORITHMIC: 0.1,  # 10% algorithmic users
            UserType.HFT: 0.05,  # 5% high-frequency traders
            UserType.API_CONSUMER: 0.05  # 5% API consumers
        },
        connection_pattern="constant",  # Constant connection pattern
        geographic_distribution={
            "us-east": 0.3,
            "us-west": 0.2,
            "europe": 0.3,
            "asia": 0.2
        }
    )


def create_default_endpoints() -> Dict[str, str]:
    """Create a default set of endpoints for testing."""
    base_url = "http://localhost:8080/api"
    return {
        "auth": f"{base_url}/auth",
        "market_data": f"{base_url}/market-data",
        "orders": f"{base_url}/orders",
        "portfolio": f"{base_url}/portfolio",
        "analysis": f"{base_url}/analysis",
        "reports": f"{base_url}/reports",
        "alerts": f"{base_url}/alerts",
        "charts": f"{base_url}/charts",
        "strategies": f"{base_url}/strategies"
    }


async def run_sample_load_test() -> None:
    """Run a sample load test with the AsyncUserLoadGenerator."""
    # Create load profile
    load_profile = create_default_load_profile(concurrent_users=50)
    
    # Create endpoints
    endpoints = create_default_endpoints()
    
    # Create generator
    generator = AsyncUserLoadGenerator(
        load_profile=load_profile,
        endpoints=endpoints,
        stress_level=StressLevel.MODERATE
    )
    
    # Start load test
    logger.info("Starting load test...")
    await generator.start()
    
    # Run for a specific duration
    duration = load_profile.ramp_up_time + load_profile.steady_state_time + load_profile.ramp_down_time
    logger.info(f"Load test running for {duration} seconds")
    
    # Wait for the load test to complete
    await asyncio.sleep(duration)
    
    # Stop the load test
    await generator.stop()
    
    # Get and print stats
    stats = await generator.get_stats()
    logger.info("Load test completed:")
    logger.info(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the sample load test
    asyncio.run(run_sample_load_test())
