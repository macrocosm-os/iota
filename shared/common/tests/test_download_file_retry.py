"""download_file must retry transient network errors, not just HTTP 5xx/429.

Guards the fix for miners crashing on truncated R2 downloads: a response whose
body is shorter than its Content-Length raises aiohttp.ClientPayloadError
(ContentLengthError), which previously escaped download_file uncaught and
killed run_miner (17 of 19 fatal miner crashes on research, 2026-07-19/20).
"""

import asyncio

import pytest
from aiohttp import web

from common.utils.s3_utils import download_file

PAYLOAD = b"x" * 64


def _truncating_app(fail_first_n: int, attempts: list[int]) -> web.Application:
    """Serve a truncated body (Content-Length > bytes sent) for the first N requests."""

    async def handler(request: web.Request) -> web.StreamResponse:
        attempts.append(1)
        resp = web.StreamResponse()
        if len(attempts) <= fail_first_n:
            resp.content_length = len(PAYLOAD)
            await resp.prepare(request)
            await resp.write(PAYLOAD[: len(PAYLOAD) // 2])
            request.transport.close()
        else:
            resp.content_length = len(PAYLOAD)
            await resp.prepare(request)
            await resp.write(PAYLOAD)
        return resp

    app = web.Application()
    app.router.add_get("/file", handler)
    return app


async def _serve(app: web.Application):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}/file"


async def _recovers_after_truncation() -> None:
    attempts: list[int] = []
    runner, url = await _serve(_truncating_app(fail_first_n=2, attempts=attempts))
    try:
        data = await download_file(url, max_retries=3)
        assert data == PAYLOAD
        assert len(attempts) == 3  # 2 truncated + 1 clean
    finally:
        await runner.cleanup()


async def _raises_after_max_retries() -> None:
    attempts: list[int] = []
    runner, url = await _serve(_truncating_app(fail_first_n=99, attempts=attempts))
    try:
        with pytest.raises(Exception):
            await download_file(url, max_retries=2)
        assert len(attempts) == 3  # initial + 2 retries, then raise
    finally:
        await runner.cleanup()


def test_download_file_retries_truncated_body():
    asyncio.run(_recovers_after_truncation())


def test_download_file_raises_after_exhausting_retries():
    asyncio.run(_raises_after_max_retries())
