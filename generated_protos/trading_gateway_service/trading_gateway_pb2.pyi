import common_pb2 as _common_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class OrderType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ORDER_TYPE_UNSPECIFIED: _ClassVar[OrderType]
    MARKET: _ClassVar[OrderType]
    LIMIT: _ClassVar[OrderType]

class OrderStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ORDER_STATUS_UNSPECIFIED: _ClassVar[OrderStatus]
    PENDING: _ClassVar[OrderStatus]
    FILLED: _ClassVar[OrderStatus]
    PARTIALLY_FILLED: _ClassVar[OrderStatus]
    CANCELLED: _ClassVar[OrderStatus]
    REJECTED: _ClassVar[OrderStatus]
ORDER_TYPE_UNSPECIFIED: OrderType
MARKET: OrderType
LIMIT: OrderType
ORDER_STATUS_UNSPECIFIED: OrderStatus
PENDING: OrderStatus
FILLED: OrderStatus
PARTIALLY_FILLED: OrderStatus
CANCELLED: OrderStatus
REJECTED: OrderStatus

class OrderRequest(_message.Message):
    __slots__ = ("order_id", "instrument_symbol", "order_type", "quantity", "price", "timestamp")
    ORDER_ID_FIELD_NUMBER: _ClassVar[int]
    INSTRUMENT_SYMBOL_FIELD_NUMBER: _ClassVar[int]
    ORDER_TYPE_FIELD_NUMBER: _ClassVar[int]
    QUANTITY_FIELD_NUMBER: _ClassVar[int]
    PRICE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    order_id: _common_pb2.UUID
    instrument_symbol: str
    order_type: OrderType
    quantity: float
    price: float
    timestamp: _common_pb2.Timestamp
    def __init__(self, order_id: _Optional[_Union[_common_pb2.UUID, _Mapping]] = ..., instrument_symbol: _Optional[str] = ..., order_type: _Optional[_Union[OrderType, str]] = ..., quantity: _Optional[float] = ..., price: _Optional[float] = ..., timestamp: _Optional[_Union[_common_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class ExecutionReport(_message.Message):
    __slots__ = ("order_id", "execution_id", "status", "filled_quantity", "average_price", "timestamp", "message")
    ORDER_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    FILLED_QUANTITY_FIELD_NUMBER: _ClassVar[int]
    AVERAGE_PRICE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    order_id: _common_pb2.UUID
    execution_id: _common_pb2.UUID
    status: OrderStatus
    filled_quantity: float
    average_price: float
    timestamp: _common_pb2.Timestamp
    message: str
    def __init__(self, order_id: _Optional[_Union[_common_pb2.UUID, _Mapping]] = ..., execution_id: _Optional[_Union[_common_pb2.UUID, _Mapping]] = ..., status: _Optional[_Union[OrderStatus, str]] = ..., filled_quantity: _Optional[float] = ..., average_price: _Optional[float] = ..., timestamp: _Optional[_Union[_common_pb2.Timestamp, _Mapping]] = ..., message: _Optional[str] = ...) -> None: ...
