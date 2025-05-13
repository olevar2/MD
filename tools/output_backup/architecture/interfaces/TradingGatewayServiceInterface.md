# TradingGatewayServiceInterface

*Generated on 2025-05-13 05:58:22*

## Description

Interface for trading-gateway-service service.

## File

`trading_gateway_service_interface.py`

## Methods

### get_status() -> Dict

Get the status of the service.

Returns:
    Service status information

#### Returns

- Dict

### execute_trade(trade_request: Dict) -> Dict

Execute a trade.

Args:
    trade_request: Trade request details
Returns:
    Trade execution result

#### Parameters

- **trade_request** (Dict)

#### Returns

- Dict

### get_trade_status(trade_id: str) -> Dict

Get the status of a trade.

Args:
    trade_id: Trade identifier
Returns:
    Trade status information

#### Parameters

- **trade_id** (str)

#### Returns

- Dict

