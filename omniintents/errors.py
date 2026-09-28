"""Typed failures shared by runtime, adapters, and evaluation."""


class OmniIntentsError(Exception):
    code = "runtime_error"


class ValidationError(OmniIntentsError, ValueError):
    code = "validation_error"


class ProviderError(OmniIntentsError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class SelectionRequired(OmniIntentsError):
    code = "selection_required"
