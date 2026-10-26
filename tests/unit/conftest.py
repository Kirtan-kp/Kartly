from __future__ import annotations

from collections.abc import Iterator

import pytest

from tests.llm_support import FakeOpenAIServer


@pytest.fixture
def fake_server() -> Iterator[FakeOpenAIServer]:
    server = FakeOpenAIServer().start()
    yield server
    server.stop()
