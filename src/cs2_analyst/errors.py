"""Errors written for the model, not for a human reading a stack trace.

Every message says what went wrong AND what to do next, so the calling model
can recover on its own (call another tool, fix an argument) instead of giving up.
"""

from __future__ import annotations


class AnalystError(Exception):
    """Base class. The message is shown to the model verbatim."""


class DemoNotFound(AnalystError):
    pass


class PlayerNotFound(AnalystError):
    pass


class RoundNotFound(AnalystError):
    pass


class ApiError(AnalystError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class MissingApiKey(AnalystError):
    pass
