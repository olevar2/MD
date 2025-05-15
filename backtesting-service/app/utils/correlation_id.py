# d:\MD\forex_trading_platform\backtesting-service\app\utils\correlation_id.py
import uuid
from contextvars import ContextVar
from typing import Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

CORRELATION_ID_HEADER = "X-Correlation-ID"

_correlation_id_ctx_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)

def get_correlation_id() -> Optional[str]:
    """Returns the current correlation ID."""
    return _correlation_id_ctx_var.get()

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next
    ) -> Response:
        """Injects correlation ID into the request and response."""
        correlation_id = request.headers.get(CORRELATION_ID_HEADER)
        if not correlation_id:
            correlation_id = str(uuid.uuid4())

        # Store correlation_id in context var
        token = _correlation_id_ctx_var.set(correlation_id)

        response = await call_next(request)
        response.headers[CORRELATION_ID_HEADER] = correlation_id

        # Reset context var
        _correlation_id_ctx_var.reset(token)
        return response