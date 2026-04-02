import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
TEST_INPUTS = ROOT / "test_inputs"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vcs_test_helpers import (  # noqa: E402
    make_backend_mock,
    make_prepare_engine_mock,
    make_session,
    parse_sse,
)


@pytest.fixture
def backend_mock():
    return make_backend_mock()


@pytest.fixture
def prepare_engine_mock(backend_mock):
    return make_prepare_engine_mock()


@pytest.fixture
def client(backend_mock, prepare_engine_mock):
    with patch("api.main._backend", backend_mock), \
         patch("api.two_step_router._engine", prepare_engine_mock):
        from api.app import app
        with TestClient(app, raise_server_exceptions=True) as c:
            c.headers.update({"Authorization": "Bearer test.token.value"})
            yield c


@pytest.fixture
def unauth_client(backend_mock, prepare_engine_mock):
    with patch("api.main._backend", backend_mock), \
         patch("api.two_step_router._engine", prepare_engine_mock):
        from api.app import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


@pytest.fixture
def fast_no_sleep(monkeypatch):
    async def _fast_sleep(_):
        return None
    monkeypatch.setattr("api.main.asyncio.sleep", _fast_sleep)
    monkeypatch.setattr("api.prepare_engine.asyncio.sleep", _fast_sleep)
    return True
