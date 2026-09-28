import json

import pytest
import requests

from omniintents.llm import MockLLMClient
from omniintents.llm.base import LLMResult


@pytest.fixture(autouse=True)
def no_accidental_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Tests must not access live services")
    monkeypatch.setattr(requests.sessions.Session, "request", blocked)


class CaptureClient(MockLLMClient):
    def __init__(self, overrides=None):
        self.requests = []
        self.overrides = overrides or {}

    def generate(self, request):
        self.requests.append(request)
        if request.stage in self.overrides:
            value = self.overrides[request.stage]
            if isinstance(value, Exception):
                raise value
            return LLMResult(value if isinstance(value, str) else json.dumps(value))
        return super().generate(request)


@pytest.fixture
def capture_client():
    return CaptureClient()
