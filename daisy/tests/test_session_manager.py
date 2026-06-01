"""Tests for SessionManager — registration, voice routing, broadcast."""

import pytest
from unittest.mock import AsyncMock

from daisy.api.session_manager import SessionManager


@pytest.fixture
def sm():
    return SessionManager()


@pytest.mark.asyncio
async def test_register_creates_session(sm):
    ws = AsyncMock()
    session = await sm.register(ws)
    assert len(session.id) == 8
    assert sm.client_count == 1


@pytest.mark.asyncio
async def test_unregister_removes_session(sm):
    ws = AsyncMock()
    session = await sm.register(ws)
    assert sm.client_count == 1
    await sm.unregister(session.id)
    assert sm.client_count == 0


@pytest.mark.asyncio
async def test_register_twice_different_ids(sm):
    s1 = await sm.register(AsyncMock())
    s2 = await sm.register(AsyncMock())
    assert s1.id != s2.id
    assert sm.client_count == 2


@pytest.mark.asyncio
async def test_unregister_nonexistent_does_not_crash(sm):
    await sm.unregister("deadbeef")  # should not raise


@pytest.mark.asyncio
async def test_activate_voice_first_client(sm):
    ws = AsyncMock()
    session = await sm.register(ws)
    accepted = await sm.activate_voice(session.id)
    assert accepted is True


@pytest.mark.asyncio
async def test_activate_voice_kicks_previous(sm):
    ws1 = AsyncMock()
    ws1.send_json = AsyncMock()
    ws2 = AsyncMock()
    ws2.send_json = AsyncMock()

    s1 = await sm.register(ws1)
    s2 = await sm.register(ws2)

    await sm.activate_voice(s1.id)
    accepted = await sm.activate_voice(s2.id)

    assert accepted is True


def test_route_mic_audio_ignores_non_active(sm):
    sm.route_mic_audio("nonexistent", b"fake_audio")


@pytest.mark.asyncio
async def test_broadcast_json_removes_dead_client(sm):
    ws = AsyncMock()
    ws.send_json = AsyncMock(side_effect=Exception("connection lost"))
    await sm.register(ws)

    await sm.broadcast_json({"type": "test"})
    assert sm.client_count == 0  # dead client removed


@pytest.mark.asyncio
async def test_broadcast_json_multiple_clients(sm):
    ws1 = AsyncMock()
    ws1.send_json = AsyncMock()
    ws2 = AsyncMock()
    ws2.send_json = AsyncMock()

    await sm.register(ws1)
    await sm.register(ws2)

    await sm.broadcast_json({"type": "test"})

    ws1.send_json.assert_called_once_with({"type": "test"})
    ws2.send_json.assert_called_once_with({"type": "test"})
    assert sm.client_count == 2


@pytest.mark.asyncio
async def test_send_to_returns_true_on_success(sm):
    ws = AsyncMock()
    ws.send_json = AsyncMock()
    session = await sm.register(ws)

    result = await sm.send_to(session.id, {"type": "test"})
    assert result is True


@pytest.mark.asyncio
async def test_send_to_returns_false_on_missing_session(sm):
    result = await sm.send_to("deadbeef", {"type": "test"})
    assert result is False


def test_client_count_starts_at_zero(sm):
    assert sm.client_count == 0
