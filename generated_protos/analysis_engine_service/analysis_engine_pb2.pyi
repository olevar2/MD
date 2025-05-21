import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AnalysisType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ANALYSIS_TYPE_UNSPECIFIED: _ClassVar[AnalysisType]
    TREND_ANALYSIS: _ClassVar[AnalysisType]
    VOLATILITY_ANALYSIS: _ClassVar[AnalysisType]
    SENTIMENT_ANALYSIS: _ClassVar[AnalysisType]

class AnalysisStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ANALYSIS_STATUS_UNSPECIFIED: _ClassVar[AnalysisStatus]
    PENDING: _ClassVar[AnalysisStatus]
    COMPLETED: _ClassVar[AnalysisStatus]
    FAILED: _ClassVar[AnalysisStatus]
ANALYSIS_TYPE_UNSPECIFIED: AnalysisType
TREND_ANALYSIS: AnalysisType
VOLATILITY_ANALYSIS: AnalysisType
SENTIMENT_ANALYSIS: AnalysisType
ANALYSIS_STATUS_UNSPECIFIED: AnalysisStatus
PENDING: AnalysisStatus
COMPLETED: AnalysisStatus
FAILED: AnalysisStatus

class AnalysisRequest(_message.Message):
    __slots__ = ("request_id", "instrument_symbol", "analysis_type", "start_time", "end_time", "parameters")
    class ParametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    INSTRUMENT_SYMBOL_FIELD_NUMBER: _ClassVar[int]
    ANALYSIS_TYPE_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    END_TIME_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    request_id: _common_pb2.UUID
    instrument_symbol: str
    analysis_type: AnalysisType
    start_time: _common_pb2.Timestamp
    end_time: _common_pb2.Timestamp
    parameters: _containers.ScalarMap[str, str]
    def __init__(self, request_id: _Optional[_Union[_common_pb2.UUID, _Mapping]] = ..., instrument_symbol: _Optional[str] = ..., analysis_type: _Optional[_Union[AnalysisType, str]] = ..., start_time: _Optional[_Union[_common_pb2.Timestamp, _Mapping]] = ..., end_time: _Optional[_Union[_common_pb2.Timestamp, _Mapping]] = ..., parameters: _Optional[_Mapping[str, str]] = ...) -> None: ...

class AnalysisResponse(_message.Message):
    __slots__ = ("request_id", "analysis_id", "status", "summary", "result_data_json", "timestamp", "error")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ANALYSIS_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    RESULT_DATA_JSON_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    request_id: _common_pb2.UUID
    analysis_id: _common_pb2.UUID
    status: AnalysisStatus
    summary: str
    result_data_json: str
    timestamp: _common_pb2.Timestamp
    error: _common_pb2.StandardErrorResponse
    def __init__(self, request_id: _Optional[_Union[_common_pb2.UUID, _Mapping]] = ..., analysis_id: _Optional[_Union[_common_pb2.UUID, _Mapping]] = ..., status: _Optional[_Union[AnalysisStatus, str]] = ..., summary: _Optional[str] = ..., result_data_json: _Optional[str] = ..., timestamp: _Optional[_Union[_common_pb2.Timestamp, _Mapping]] = ..., error: _Optional[_Union[_common_pb2.StandardErrorResponse, _Mapping]] = ...) -> None: ...
