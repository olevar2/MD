"""
SQLAlchemy database models for the data pipeline service.
"""
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION

# Define the metadata
metadata = sa.MetaData()

# OHLCV data table
ohlcv_table = sa.Table(
    'ohlcv',
    metadata,
    sa.Column('symbol', sa.String, nullable=False),
    sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
    sa.Column('timeframe', sa.String, nullable=False),
    sa.Column('open', DOUBLE_PRECISION, nullable=False),
    sa.Column('high', DOUBLE_PRECISION, nullable=False),
    sa.Column('low', DOUBLE_PRECISION, nullable=False),
    sa.Column('close', DOUBLE_PRECISION, nullable=False),
    sa.Column('volume', DOUBLE_PRECISION, nullable=False),
    sa.UniqueConstraint('symbol', 'timestamp', 'timeframe', name='ohlcv_symbol_timestamp_timeframe_key'),
    # Time index is created by TimescaleDB hypertable
)

# Tick data table
tick_data_table = sa.Table(
    'tick_data',
    metadata,
    sa.Column('symbol', sa.String, nullable=False),
    sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
    sa.Column('bid', DOUBLE_PRECISION, nullable=False),
    sa.Column('ask', DOUBLE_PRECISION, nullable=False),
    sa.Column('bid_volume', DOUBLE_PRECISION, nullable=True),
    sa.Column('ask_volume', DOUBLE_PRECISION, nullable=True),
    sa.UniqueConstraint('symbol', 'timestamp', name='tick_data_symbol_timestamp_key'),
    # Time index is created by TimescaleDB hypertable
)

# Instruments table
instruments_table = sa.Table(
    'instruments',
    metadata,
    sa.Column('symbol', sa.String, primary_key=True, nullable=False),
    sa.Column('name', sa.String, nullable=False),
    sa.Column('type', sa.String, nullable=False),
    sa.Column('pip_size', DOUBLE_PRECISION, nullable=False),
    sa.Column('min_lot_size', DOUBLE_PRECISION, nullable=False),
    sa.Column('max_lot_size', DOUBLE_PRECISION, nullable=False),
    sa.Column('lot_step', DOUBLE_PRECISION, nullable=False),
    sa.Column('commission', DOUBLE_PRECISION, nullable=True),
    sa.Column('swap_long', DOUBLE_PRECISION, nullable=True),
    sa.Column('swap_short', DOUBLE_PRECISION, nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
)

# Trading hours table
trading_hours_table = sa.Table(
    'trading_hours',
    metadata,
    sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
    sa.Column('symbol', sa.String, nullable=False),
    sa.Column('day_of_week', sa.Integer, nullable=False),
    sa.Column('open_time', sa.String, nullable=False),
    sa.Column('close_time', sa.String, nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    sa.ForeignKeyConstraint(['symbol'], ['instruments.symbol'], name='fk_trading_hours_symbol'),
    sa.UniqueConstraint('symbol', 'day_of_week', name='trading_hours_symbol_day_key'),
)