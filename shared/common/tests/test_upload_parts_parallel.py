"""upload_parts now PUTs parts concurrently — the returned list must still be
ordered by PartNumber and the byte ranges must reassemble to the input."""

import asyncio

from aiohttp import web
from common.utils.s3_utils import upload_parts

NUM_PARTS = 5


async def _roundtrip() -> None:
    received: dict[str, bytes] = {}

    async def handler(request: web.Request) -> web.Response:
        received[request.query["part"]] = await request.read()
        return web.Response(headers={"ETag": f'"etag-{request.query["part"]}"'})

    app = web.Application()
    app.router.add_put("/upload", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        port = site._server.sockets[0].getsockname()[1]
        urls = [f"http://127.0.0.1:{port}/upload?part={i}" for i in range(1, NUM_PARTS + 1)]
        data = bytes(range(256)) * 200  # 51200 bytes -> 5 parts of 10240

        parts = await upload_parts(urls=urls, data=data, upload_id="test-upload-id")

        assert [p["PartNumber"] for p in parts] == list(range(1, NUM_PARTS + 1))
        assert [p["ETag"] for p in parts] == [f"etag-{i}" for i in range(1, NUM_PARTS + 1)]
        assert b"".join(received[str(i)] for i in range(1, NUM_PARTS + 1)) == data
    finally:
        await runner.cleanup()


def test_upload_parts_parallel_order_and_reassembly():
    asyncio.run(_roundtrip())
