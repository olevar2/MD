@echo off
cd /d "d:\MD\forex_trading_platform"
python -m pip install pytest pytest-asyncio
python -m pytest strategy-execution-engine\tests\strategies\test_causal_enhanced_strategy.py -v
pause
