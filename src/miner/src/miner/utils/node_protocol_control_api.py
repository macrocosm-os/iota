"""
Compatible with your KeySigner shapes:
  - enclave.get_key_id -> returns { keyId, alg:"ES256", publicKeyX963Base64 }
  - enclave.sign       -> request uses { payloadBase64 } and returns { signatureDerBase64, alg:"ES256" }

Features:
  - single WS endpoint /ws
  - hello/welcome handshake
  - request/response correlation (server-initiated requests to host)
  - ping/pong + disconnect handling
  - Pydantic models for envelope + enclave payloads
  - optional token gating (?token=...)
  - optional DER ECDSA verification helper (P-256) using cryptography (if installed)


Electron connects:
  ws://127.0.0.1:8010/ws?token=...  (if you set ALLOWED_TOKEN)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Literal, Optional
from uuid import uuid4
from loguru import logger

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from common.models.api_models import (
    TrainingStateRequest,
    TrainingStateResponse,
    NODE_PROTOCOL_CONTROL_SERVER_CAPABILITIES,
    EnclaveGetKeyIdRequest,
    EnclaveGetKeyIdResponse,
    EnclaveSignRequest,
    EnclaveSignResponse,
    RegisterSetBestRunRequest,
    RegisterSetBestRunResponse,
    RegisterSetStatusRequest,
    RegisterSetStatusResponse,
    RegisterStatus,
    ServerCapability,
)

# -----------------------------
# Protocol constants
# -----------------------------

PROTOCOL_VERSION = 1
KIND = Literal["hello", "welcome", "request", "response", "event", "error", "ping", "pong"]


# Electron would connect with: ws://127.0.0.1:8010/ws?token=...
# Currently not used
ALLOWED_TOKEN: Optional[str] = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# -----------------------------
# Envelope + payload models
# -----------------------------


class ErrorObj(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class Envelope(BaseModel):
    v: int = Field(default=PROTOCOL_VERSION)
    kind: KIND
    id: str = Field(default_factory=lambda: str(uuid4()))
    ts: str = Field(default_factory=now_iso)
    name: str
    reply_to: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[ErrorObj] = None


class HelloData(BaseModel):
    app: str
    appVersion: Optional[str] = None
    capabilities: list[str] = Field(default_factory=list)


class WelcomeData(BaseModel):
    sessionId: str
    serverCapabilities: list[str]


# -----------------------------
# Server implementation
# -----------------------------

Handler = Callable[[Envelope], Awaitable[Envelope | Dict[str, Any] | None] | Envelope | Dict[str, Any] | None]


@dataclass
class ClientState:
    websocket: WebSocket
    session_id: str
    hello: Optional[HelloData] = None


class ProtocolServer:
    def __init__(
        self,
        *,
        allowed_token: Optional[str] = None,
        ping_interval_s: float = 10.0,
        ping_timeout_s: float = 20.0,
        request_timeout_s: float = 30.0,
    ):
        self.allowed_token = allowed_token
        self.ping_interval_s = ping_interval_s
        self.ping_timeout_s = ping_timeout_s
        self.request_timeout_s = request_timeout_s

        self._client: Optional[ClientState] = None
        self._client_lock = asyncio.Lock()

        self._pending: dict[str, asyncio.Future[Envelope]] = {}
        self._handlers: dict[str, Handler] = {}

        self._last_pong: float = 0.0

        self.server_capabilities = NODE_PROTOCOL_CONTROL_SERVER_CAPABILITIES

    def register_handler(self, name: str, handler: Handler) -> None:
        self._handlers[name] = handler

    def _ensure_connected(self) -> ClientState:
        if self._client is None:
            raise RuntimeError("No host connected")
        return self._client

    async def send(self, env: Envelope) -> None:
        client = self._ensure_connected()
        await client.websocket.send_json(env.model_dump())

    async def request(
        self, name: ServerCapability, data: Dict[str, Any], timeout_s: Optional[float] = None
    ) -> Envelope:
        """
        Server -> Host request. Returns host response envelope (kind="response") or raises on error/timeout.
        """
        client = self._ensure_connected()
        req = Envelope(kind="request", name=name, data=data)

        fut: asyncio.Future[Envelope] = asyncio.get_running_loop().create_future()
        self._pending[req.id] = fut
        await client.websocket.send_json(req.model_dump())

        try:
            t = timeout_s if timeout_s is not None else self.request_timeout_s
            resp = await asyncio.wait_for(fut, timeout=t)
        finally:
            self._pending.pop(req.id, None)

        if resp.kind == "error":
            code = resp.error.code if resp.error else "host_error"
            msg = resp.error.message if resp.error else "Host returned error"
            raise RuntimeError(f"{code}: {msg}")

        return resp

    # --- registration helpers ---

    async def register_set_status(self, *, status: RegisterStatus) -> RegisterSetStatusResponse:
        req = RegisterSetStatusRequest(status=status)
        resp_env = await self.request("register.status", req.model_dump())

        return RegisterSetStatusResponse(status=resp_env.data["status"])

    async def register_set_best_run(self, *, run_id: str) -> RegisterSetBestRunResponse:
        req = RegisterSetBestRunRequest(run_id=run_id)
        resp_env = await self.request("register.best_run", req.model_dump(by_alias=True))

        return RegisterSetBestRunResponse(status=resp_env.data["status"])

    async def training_set_state(
        self, *, state: str, detail: Optional[str] = None, run_id: Optional[str] = None, layer: Optional[int] = None
    ) -> TrainingStateResponse:
        req = TrainingStateRequest(state=state, detail=detail, run_id=run_id, layer=layer)
        resp_env = await self.request("training.state", req.model_dump(by_alias=True, exclude_none=True))
        return TrainingStateResponse(status=resp_env.data["status"])

    # --- enclave helpers (KeySigner-aligned) ---

    async def enclave_get_key_id(
        self,
        *,
        purpose: str,
        preferred_algorithms: Optional[list[str]] = None,
        dp_keychain: Optional[bool] = None,
    ) -> EnclaveGetKeyIdResponse:
        req = EnclaveGetKeyIdRequest(
            purpose=purpose,
            preferredAlgorithms=preferred_algorithms or [],
            dpKeychain=dp_keychain,
        ).model_dump(exclude_none=True)

        resp_env = await self.request("enclave.get_key_id", req)
        return EnclaveGetKeyIdResponse(
            key_id=resp_env.data["keyId"],
            public_key_base64=resp_env.data["publicKeyX963Base64"],
            alg=resp_env.data["alg"],
        )

    async def enclave_sign(
        self,
        *,
        key_id: str,
        payload: bytes,
        dp_keychain: Optional[bool] = None,
    ) -> EnclaveSignResponse:
        req = EnclaveSignRequest(
            keyId=key_id,
            payloadBase64=payload,
            alg="ES256",
            dpKeychain=dp_keychain,
        ).model_dump(exclude_none=True)

        logger.debug(f"Sending sign request: {req}")
        resp_env = await self.request("enclave.sign", req)
        return EnclaveSignResponse(
            key_id=resp_env.data["keyId"],
            signature_der_base64=resp_env.data["signatureDerBase64"],
            alg=resp_env.data["alg"],
            public_key_base64=resp_env.data["publicKeyX963Base64"],
        )

    async def enclave_get_key_then_sign(
        self,
        *,
        purpose: str,
        payload: bytes,
        preferred_algorithms: Optional[list[str]] = None,
        dp_keychain: Optional[bool] = None,
    ) -> tuple[EnclaveGetKeyIdResponse, EnclaveSignResponse]:
        init = await self.enclave_get_key_id(
            purpose=purpose,
            preferred_algorithms=preferred_algorithms,
            dp_keychain=dp_keychain,
        )
        sig = await self.enclave_sign(key_id=init.key_id, payload=payload, dp_keychain=dp_keychain)
        return init, sig

    # --- internal helpers ---

    async def _send_error(
        self,
        ws: WebSocket,
        *,
        name: str,
        reply_to: Optional[str],
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        env = Envelope(
            kind="error",
            name=name,
            reply_to=reply_to,
            error=ErrorObj(code=code, message=message, details=details),
        )
        await ws.send_json(env.model_dump())

    async def _send_response(self, ws: WebSocket, *, name: str, reply_to: str, data: Dict[str, Any]) -> None:
        env = Envelope(kind="response", name=name, reply_to=reply_to, data=data)
        await ws.send_json(env.model_dump())

    async def _ping_loop(self, ws: WebSocket) -> None:
        self._last_pong = asyncio.get_running_loop().time()
        while True:
            await asyncio.sleep(self.ping_interval_s)
            ping = Envelope(kind="ping", name="ping", data={})
            await ws.send_json(ping.model_dump())

            now = asyncio.get_running_loop().time()
            if (now - self._last_pong) > self.ping_timeout_s:
                await ws.close(code=1011)  # internal error
                return

    async def websocket_session(self, ws: WebSocket, token: Optional[str]) -> None:
        if self.allowed_token is not None and token != self.allowed_token:
            await ws.close(code=1008)  # policy violation
            return

        await ws.accept()

        # Single active client
        async with self._client_lock:
            if self._client is not None:
                try:
                    await self._client.websocket.close(code=1012)  # service restart
                except Exception:
                    pass
                self._client = None

            self._client = ClientState(websocket=ws, session_id=str(uuid4()))

        ping_task = asyncio.create_task(self._ping_loop(ws))

        try:
            # Require hello first
            raw = await ws.receive_json()
            try:
                env = Envelope(**raw)
            except ValidationError as e:
                await self._send_error(
                    ws,
                    name="hello",
                    reply_to=None,
                    code="bad_request",
                    message="Invalid envelope",
                    details={"validation": str(e)},
                )
                await ws.close(code=1003)
                return

            if env.kind != "hello" or env.name != "hello":
                await self._send_error(
                    ws,
                    name="hello",
                    reply_to=env.id,
                    code="bad_request",
                    message="Expected hello as first message",
                )
                await ws.close(code=1003)
                return

            if env.v != PROTOCOL_VERSION:
                await self._send_error(
                    ws,
                    name="hello",
                    reply_to=env.id,
                    code="unsupported",
                    message=f"Unsupported protocol version {env.v}",
                    details={"supported": PROTOCOL_VERSION},
                )
                await ws.close(code=1002)
                return

            try:
                hello = HelloData(**env.data)
            except ValidationError as e:
                await self._send_error(
                    ws,
                    name="hello",
                    reply_to=env.id,
                    code="bad_request",
                    message="Invalid hello payload",
                    details={"validation": str(e)},
                )
                await ws.close(code=1003)
                return

            async with self._client_lock:
                if self._client is not None:
                    self._client.hello = hello

            welcome = Envelope(
                kind="welcome",
                name="welcome",
                data=WelcomeData(
                    sessionId=self._ensure_connected().session_id,
                    serverCapabilities=self.server_capabilities,
                ).model_dump(),
            )
            await ws.send_json(welcome.model_dump())

            # Main loop
            while True:
                raw = await ws.receive_json()
                try:
                    msg = Envelope(**raw)
                except ValidationError as e:
                    await self._send_error(
                        ws,
                        name="protocol",
                        reply_to=None,
                        code="bad_request",
                        message="Invalid envelope",
                        details={"validation": str(e)},
                    )
                    continue

                # Heartbeat
                if msg.kind == "pong":
                    self._last_pong = asyncio.get_running_loop().time()
                    continue
                if msg.kind == "ping":
                    pong = Envelope(kind="pong", name="pong", reply_to=msg.id, data={})
                    await ws.send_json(pong.model_dump())
                    continue

                # Resolve server-initiated requests
                if msg.kind in ("response", "error") and msg.reply_to:
                    fut = self._pending.get(msg.reply_to)
                    if fut and not fut.done():
                        fut.set_result(msg)
                    continue

                # Host-initiated requests (optional)
                if msg.kind == "request":
                    handler = self._handlers.get(msg.name)
                    if handler is None:
                        await self._send_error(
                            ws,
                            name=msg.name,
                            reply_to=msg.id,
                            code="unsupported",
                            message=f"No handler for {msg.name}",
                        )
                        continue

                    try:
                        result = handler(msg)
                        if asyncio.iscoroutine(result):
                            result = await result  # type: ignore
                    except Exception as e:
                        await self._send_error(ws, name=msg.name, reply_to=msg.id, code="internal", message=str(e))
                        continue

                    if result is None:
                        await self._send_response(ws, name=msg.name, reply_to=msg.id, data={"ok": True})
                    elif isinstance(result, Envelope):
                        if result.kind not in ("response", "error"):
                            await self._send_error(
                                ws,
                                name=msg.name,
                                reply_to=msg.id,
                                code="internal",
                                message="Handler returned invalid envelope kind",
                            )
                            continue
                        result.reply_to = msg.id
                        result.name = msg.name
                        await ws.send_json(result.model_dump())
                    else:
                        await self._send_response(ws, name=msg.name, reply_to=msg.id, data=result)  # type: ignore
                    continue

                if msg.kind == "event":
                    continue

                await self._send_error(
                    ws,
                    name="protocol",
                    reply_to=msg.id,
                    code="bad_request",
                    message=f"Unexpected message kind {msg.kind}",
                )

        except WebSocketDisconnect:
            pass
        finally:
            ping_task.cancel()

            async with self._client_lock:
                if self._client and self._client.websocket is ws:
                    self._client = None

            # Fail pending requests
            for req_id, fut in list(self._pending.items()):
                if not fut.done():
                    fut.set_result(
                        Envelope(
                            kind="error",
                            name="protocol",
                            reply_to=req_id,
                            error=ErrorObj(code="disconnected", message="Host disconnected"),
                        )
                    )
            self._pending.clear()


# -----------------------------
# FastAPI app wiring
# -----------------------------

app = FastAPI(title="Node Protocol of Control Server (WS, ES256 KeySigner-aligned)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

protocol = ProtocolServer(allowed_token=ALLOWED_TOKEN)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket, token: Optional[str] = Query(default=None)):
    await protocol.websocket_session(ws, token=token)


@app.get("/health")
async def health() -> Dict[str, Any]:
    connected = False
    hello = None
    session_id = None
    try:
        c = protocol._ensure_connected()
        connected = True
        hello = c.hello.model_dump() if c.hello else None
        session_id = c.session_id
    except Exception:
        pass
    return {"ok": True, "connected": connected, "sessionId": session_id, "client": hello}


@app.post("/enclave/get_key_id")
async def http_enclave_get_key_id(req: EnclaveGetKeyIdRequest) -> Dict[str, Any]:
    try:
        resp = await protocol.enclave_get_key_id(
            purpose=req.purpose,
            preferred_algorithms=req.preferredAlgorithms,
            dp_keychain=req.dpKeychain,
        )
        return resp.model_dump()
    except RuntimeError as e:
        # e.g. "No host connected" or host error/timeout
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/enclave/sign")
async def http_enclave_sign(req: EnclaveSignRequest) -> Dict[str, Any]:
    try:
        resp = await protocol.enclave_sign(
            key_id=req.keyId,
            payload=req.payloadBase64,
            dp_keychain=req.dpKeychain,
        )
        return resp.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/register/status")
async def http_register_set_status(req: RegisterSetStatusRequest) -> Dict[str, Any]:
    try:
        resp = await protocol.register_set_status(status=req.status)
        return resp.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/register/best_run")
async def http_register_set_best_run(req: RegisterSetBestRunRequest) -> Dict[str, Any]:
    try:
        resp = await protocol.register_set_best_run(run_id=req.run_id)
        return resp.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/training/state")
async def http_training_set_state(req: TrainingStateRequest) -> Dict[str, Any]:
    try:
        resp = await protocol.training_set_state(state=req.state, detail=req.detail, run_id=req.run_id, layer=req.layer)
        return resp.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


def start_protocol_server(port: int = 8010, host: str = "127.0.0.1") -> None:
    """
    Blocking runner suitable for a sidecar process launched by Electron / CLI.
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="warning")
