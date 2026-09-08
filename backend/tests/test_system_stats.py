"""GPU 統計 subprocess の停止と回収を検証する。"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from sphere_reconstruct.api import system


class FakeProbe:
    def __init__(self):
        self.returncode = None
        self.killed = False
        self.communications = 0

    async def communicate(self):
        self.communications += 1
        if self.killed:
            self.returncode = -9
        return b"", b""

    def kill(self):
        self.killed = True


@pytest.mark.parametrize("error", [TimeoutError(), asyncio.CancelledError(), OSError("pipe failed")])
async def test_failed_gpu_probe_is_terminated_and_reaped(monkeypatch, error):
    probe = FakeProbe()
    monkeypatch.setattr(system.asyncio, "create_subprocess_exec", AsyncMock(return_value=probe))

    async def fail(awaitable, *, timeout):
        await awaitable
        raise error

    monkeypatch.setattr(system.asyncio, "wait_for", fail)
    if isinstance(error, asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            await system._gpu_stats()
    else:
        assert await system._gpu_stats() == []
    assert probe.killed
    assert probe.communications == 2
    assert probe.returncode is not None


async def test_missing_gpu_probe_returns_unavailable(monkeypatch):
    monkeypatch.setattr(system.asyncio, "create_subprocess_exec", AsyncMock(side_effect=FileNotFoundError()))
    assert await system._gpu_stats() == []
