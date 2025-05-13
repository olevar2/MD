"""
End-to-end demonstration of news/sentiment integration with RL pipeline.

This example shows how to:
1. Create a broker simulator
2. Set up a news and sentiment simulator
3. Integrate both with the enhanced RL environment
4. Train an RL agent that responds to news events
5. Evaluate the agent's performance
"""

import os
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from trading_gateway_service.simulation.forex_broker_simulator import ForexBrokerSimulator
from trading_gateway_service.simulation.news_sentiment_simulator import (
    NewsAndSentimentSimulator, NewsEventType, NewsImpactLevel, SentimentLevel
)
from core.enhanced_rl_env import EnhancedForexTradingEnv
from models.rl_model_factory import RLModelFactory


def create_simulators():
    """Create broker and news simulators."""
    print("Creating simulators...")
    
    # Create broker simulator with configuration
    broker_sim = ForexBrokerSimulator(
        balance=10000.0,
        leverage=100,
        symbols=["EUR/USD", "GBP/USD"],
        fee_percent=0.001
    )
    
    # Create news and sentiment simulator
    news_sim = NewsAndSentimentSimulator(seed=42)
    
    # Generate random economic calendar for the next week
    print("Generating news calendar...")
    start_date = datetime.now()
    end_date = start_date + timedelta(days=7)
    
    events = news_sim.generate_random_economic_calendar(
        start_date=start_date,
        end_date=end_date,
        currency_pairs=["EUR/USD", "GBP/USD"],
        num_events=30
    )
    
    # Set some initial sentiment levels
    news_sim.set_sentiment("EUR", SentimentLevel.BULLISH)
    news_sim.set_sentiment("GBP", SentimentLevel.NEUTRAL)
    
    print(f"Generated {len(events)} news events for the next week")
    
    return broker_sim, news_sim


def create_environments(broker_sim, news_sim):
    """Create RL environments with and without news integration."""
    print("Creating RL environments...")
    
    # Create environment with news integration
    env_with_news = EnhancedForexTradingEnv(
        broker_simulator=broker_sim,
        symbol="EUR/USD",
        timeframes=["1m", "5m", "15m"],
        lookback_periods=30,
        include_news_sentiment=True,
        news_sentiment_simulator=news_sim,
        reward_mode="risk_adjusted",
        max_position_size=1.0,
        episode_timesteps=500
    )
    
    # Create identical environment but without news integration for comparison
    env_without_news = EnhancedForexTradingEnv(
        broker_simulator=ForexBrokerSimulator(
            balance=10000.0,
            leverage=100,
            symbols=["EUR/USD", "GBP/USD"],
            fee_percent=0.001
        ),  # New simulator to avoid interference
        symbol="EUR/USD",
        timeframes=["1m", "5m", "15m"],
        lookback_periods=30,
        include_news_sentiment=False,
        reward_mode="risk_adjusted",
        max_position_size=1.0,
        episode_timesteps=500
    )
    
    return env_with_news, env_without_news


def train_models(env_with_news, env_without_news):
    """Train RL models with and without news integration."""
    print("Training models...")
    
    # Create PPO models using factory
    factory = RLModelFactory()
    
    model_with_news = factory.create_model(
        algorithm="PPO",
        environment=env_with_news,
        params={
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "verbose": 1,
            "tensorboard_log": "./logs/ppo_news_aware/"
        }
    )
    
    model_without_news = factory.create_model(
        algorithm="PPO",
        environment=env_without_news,
        params={
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "verbose": 1,
            "tensorboard_log": "./logs/ppo_standard/"
        }
    )
    
    # Train models
    print("Training news-aware model...")
    model_with_news.learn(total_timesteps=50000)
    
    print("Training standard model...")
    model_without_news.learn(total_timesteps=50000)
    
    print("Training complete!")
    
    # Save models
    os.makedirs("models", exist_ok=True)
    model_with_news.save("models/ppo_news_aware")
    model_without_news.save("models/ppo_standard")
    
    return model_with_news, model_without_news


def evaluate_models(env_with_news, env_without_news, model_with_news, model_without_news):
    """Evaluate and compare models with and without news integration."""
    print("Evaluating models...")
    
    # Reset environments for evaluation
    env_with_news.reset()
    
    # Track performance metrics for visualization
    rewards = []
    equity = []
    news_impacts = []
    
    # Evaluate for 1000 steps
    obs = env_with_news.reset()
    for i in range(1000):
        action, _states = model_with_news.predict(obs, deterministic=True)
        obs, reward, done, info = env_with_news.step(action)
        
        rewards.append(reward)
        equity.append(info["account_summary"]["equity"])
        
        # Calculate news impact for visualization
        if env_with_news.news_sentiment_simulator:
            impact = env_with_news.news_sentiment_simulator.calculate_price_impact(
                env_with_news.symbol, 1.0
            )
            news_impacts.append(abs(impact["price_change_pct"]) * 100)  # Convert to percentage
        else:
            news_impacts.append(0)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot equity curve
    plt.subplot(2, 1, 1)
    plt.plot(equity, label='Equity')
    plt.title('Trading Results with News Integration')
    plt.ylabel('Account Equity')
    plt.grid(True)
    plt.legend()
    
    # Plot news impact
    plt.subplot(2, 1, 2)
    plt.plot(news_impacts, color='red', label='News Impact')
    plt.xlabel('Steps')
    plt.ylabel('News Impact (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/news_sentiment_rl_results.png')
    
    print(f"Final equity with news integration: ${equity[-1]:.2f}")
    print(f"Plot saved to output/news_sentiment_rl_results.png")
    
    # Compare performance metrics
    print("\nPerformance Comparison:")
    
    # Evaluate standard model for comparison
    mean_reward_standard, _ = evaluate_policy(model_without_news, env_without_news, n_eval_episodes=10)
    mean_reward_news_aware, _ = evaluate_policy(model_with_news, env_with_news, n_eval_episodes=10)
    
    print(f"Mean reward (standard model): {mean_reward_standard:.2f}")
    print(f"Mean reward (news-aware model): {mean_reward_news_aware:.2f}")
    print(f"Improvement: {((mean_reward_news_aware / mean_reward_standard) - 1) * 100:.2f}%")


def analyze_news_adaptation():
    """Analyze how the RL agent adapts to specific news events."""
    print("\nAnalyzing news adaptation strategies...")
    
    # Create simulators
    broker_sim, news_sim = create_simulators()
    
    # Create environment
    env = EnhancedForexTradingEnv(
        broker_simulator=broker_sim,
        symbol="EUR/USD",
        news_sentiment_simulator=news_sim,
        include_news_sentiment=True
    )
    
    # Load the trained model
    model = PPO.load("models/ppo_news_aware")
    
    # Create specific news events for testing adaptation
    high_impact_event = {
        "title": "Central Bank Rate Decision",
        "event_type": NewsEventType.CENTRAL_BANK,
        "impact_level": NewsImpactLevel.HIGH,
        "price_impact": 0.01,  # 1% price impact
        "volatility_impact": 3.0  # Triples volatility
    }
    
    medium_impact_event = {
        "title": "GDP Report",
        "event_type": NewsEventType.ECONOMIC_DATA,
        "impact_level": NewsImpactLevel.MEDIUM,
        "price_impact": 0.005,  # 0.5% price impact
        "volatility_impact": 1.5  # 50% increase in volatility
    }
    
    low_impact_event = {
        "title": "Minor Economic Release",
        "event_type": NewsEventType.ECONOMIC_DATA,
        "impact_level": NewsImpactLevel.LOW,
        "price_impact": 0.001,  # 0.1% price impact
        "volatility_impact": 1.1  # 10% increase in volatility
    }
    
    # Test each event type
    for event_config in [high_impact_event, medium_impact_event, low_impact_event]:
        print(f"\nTesting adaptation to {event_config['title']} ({event_config['impact_level'].name}):")
        
        # Reset environment
        obs = env.reset()
        
        # Add the test event at current time + 5 minutes
        event_time = broker_sim.current_time + timedelta(minutes=5)
        
        event = NewsEvent(
            event_id=f"test_{event_config['event_type'].name}",
            event_type=event_config['event_type'],
            impact_level=event_config['impact_level'],
            timestamp=event_time,
            currencies_affected=["EUR/USD"],
            title=event_config['title'],
            price_impact=event_config['price_impact'],
            volatility_impact=event_config['volatility_impact'],
            duration_minutes=60
        )
        
        news_sim.add_news_event(event)
        
        # Track agent behavior
        actions_before_event = []
        actions_during_event = []
        
        # Run simulation steps
        for i in range(20):  # 20 steps
            # Get model prediction
            action, _states = model.predict(obs, deterministic=True)
            
            # Execute step
            obs, reward, done, info = env.step(action)
            
            # Record actions based on time
            if broker_sim.current_time < event_time:
                actions_before_event.append(action[0])
            else:
                actions_during_event.append(action[0])
        
        # Analyze behavior change
        if len(actions_before_event) > 0 and len(actions_during_event) > 0:
            avg_position_before = sum(actions_before_event) / len(actions_before_event)
            avg_position_during = sum(actions_during_event) / len(actions_during_event)
            position_change = abs(avg_position_during - avg_position_before)
            
            print(f"  - Average position before event: {avg_position_before:.3f}")
            print(f"  - Average position during event: {avg_position_during:.3f}")
            print(f"  - Position size change: {position_change:.3f}")
            
            if position_change > 0.2:
                print("  - Agent significantly adapted its strategy during the event")
            elif position_change > 0.05:
                print("  - Agent moderately adapted its strategy during the event")
            else:
                print("  - Agent maintained similar strategy during the event")


def main():
    """Run the end-to-end news sentiment RL pipeline."""
    # Create simulators
    broker_sim, news_sim = create_simulators()
    
    # Create environments
    env_with_news, env_without_news = create_environments(broker_sim, news_sim)
    
    # Train models
    model_with_news, model_without_news = train_models(env_with_news, env_without_news)
    
    # Evaluate models
    evaluate_models(env_with_news, env_without_news, model_with_news, model_without_news)
    
    # Analyze news adaptation
    analyze_news_adaptation()


if __name__ == "__main__":
    main()
