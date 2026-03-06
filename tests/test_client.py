"""Unit tests for aiobedrock Client."""

# pylint: disable=protected-access,no-member,wrong-import-position,no-name-in-module
# pylint: disable=too-few-public-methods,missing-function-docstring,missing-class-docstring

import asyncio
import base64
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import orjson
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aiobedrock import BedrockClientError, BedrockStreamError
from aiobedrock.main import Client


def _make_client() -> Client:
    return Client.__new__(Client)  # type: ignore[call-arg]


class DummyResponse:
    def __init__(self, body: bytes = b"ok", status: int = 200):
        self.status = status
        self._body = body

    async def read(self) -> bytes:
        return self._body

    async def text(self) -> str:
        return self._body.decode("utf-8", errors="replace")


class DummyContext:
    def __init__(self, response: Any):
        self._response = response

    async def __aenter__(self) -> Any:
        return self._response

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


def _stub_client(
    response_body: bytes = b"ok", status: int = 200
) -> Tuple[Any, Dict[str, Any]]:
    client: Any = _make_client()
    client.region_name = "us-east-1"
    client.credentials = object()
    client._max_retries = 0
    client._retry_backoff = 0.0
    client._retry_backoff_cap = 0.0
    client._retry_statuses = ()
    client._max_concurrency = None
    client._request_semaphore = None
    client._credential_lock = None
    client._client_timeout = None
    client.assume_role_arn = None
    client.session = None
    client.connector = None
    client._connector_kwargs = {"limit": 100}

    captured: Dict[str, Any] = {"calls": []}

    async def _no_creds() -> None:
        return None

    def fake_signed(**kwargs: Any) -> Dict[str, str]:
        captured["calls"].append(kwargs)
        return {"Authorization": "test"}

    resp = DummyResponse(body=response_body, status=status)

    def fake_request(**kwargs: Any) -> DummyContext:
        captured["request"] = kwargs
        return DummyContext(resp)

    async def fake_handle_error(_response: DummyResponse) -> None:
        if _response.status != 200:
            raise BedrockClientError(
                _response.status, "TestError", await _response.text()
            )

    client._ensure_valid_credentials = _no_creds
    client._signed_request = fake_signed
    client._request = fake_request
    client._handle_error_response = fake_handle_error

    return client, captured


def _stub_get_client(
    response_body: bytes = b"{}",
) -> Tuple[Any, Dict[str, Any]]:
    client: Any = _make_client()
    client.region_name = "us-east-1"
    client.credentials = object()
    client.assume_role_arn = None
    client._client_timeout = None
    client.session = None
    client.connector = None
    client._connector_kwargs = {"limit": 100}

    captured: Dict[str, Any] = {}

    async def _no_creds() -> None:
        return None

    def fake_signed(**kwargs: Any) -> Dict[str, str]:
        captured["signed"] = kwargs
        return {"Authorization": "test"}

    class _FakeGetContext:
        async def __aenter__(self) -> DummyResponse:
            return DummyResponse(body=response_body)

        async def __aexit__(self, *_args: Any) -> bool:
            return False

    class _FakeSession:
        closed = False

        def get(self, **_kwargs: Any) -> _FakeGetContext:
            captured["get_kwargs"] = _kwargs
            return _FakeGetContext()

    client._ensure_valid_credentials = _no_creds
    client._signed_request = fake_signed
    client.session = _FakeSession()

    return client, captured


def test_process_event_message_decodes_base64_payload() -> None:
    client = _make_client()
    payload_bytes = base64.b64encode(b"hello world").decode()
    payload_json = f'{{"bytes": "{payload_bytes}"}}'.encode()
    message = {
        "headers": {
            ":message-type": "event",
            ":event-type": "chunk",
            ":content-type": "application/json",
        },
        "payload": payload_json,
    }
    assert client._process_event_message(message) == b"hello world"


def test_process_event_message_returns_none_on_exception() -> None:
    client = _make_client()
    message = {
        "headers": {
            ":message-type": "exception",
            ":exception-type": "SomeError",
            ":content-type": "application/json",
        },
        "payload": b'{"message": "failing"}',
    }
    assert client._process_event_message(message) is None


def test_normalize_headers_extracts_values() -> None:
    client = _make_client()

    class _HeaderValue:
        def __init__(self, value: str):
            self.value = value

    normalized = client._normalize_headers(
        {"plain": "value", "wrapped": _HeaderValue("wrapped-value")}
    )
    assert normalized == {"plain": "value", "wrapped": "wrapped-value"}


def test_invoke_sagemaker_endpoint_builds_headers() -> None:
    client: Any = _make_client()
    client.region_name = "us-west-2"
    client.credentials = object()

    captured: Dict[str, Any] = {}

    async def _no_creds() -> None:
        return None

    def fake_signed(**kwargs: Any) -> Dict[str, str]:
        captured["kwargs"] = kwargs
        return {"Authorization": "test"}

    def fake_request(**kwargs: Any) -> DummyContext:
        captured["request"] = kwargs
        return DummyContext(DummyResponse())

    async def fake_handle_error(_r: Any) -> None:
        return None

    client._ensure_valid_credentials = _no_creds
    client._signed_request = fake_signed
    client._request = fake_request
    client._handle_error_response = fake_handle_error

    result = asyncio.run(
        client.invoke_sagemaker_endpoint(
            "demo-endpoint",
            body=b"data",
            content_type="application/json",
            accept="application/json",
            custom_attributes="attr",
            target_variant="variant",
            headers={"X-Custom": "value"},
        )
    )

    assert result == b"ok"
    sk = captured["kwargs"]
    assert sk["service"] == "sagemaker"
    assert sk["accept"] == "application/json"
    assert sk["contentType"] == "application/json"
    assert sk["extra_headers"]["X-Amzn-SageMaker-Custom-Attributes"] == "attr"
    assert sk["extra_headers"]["X-Amzn-SageMaker-Target-Variant"] == "variant"
    assert sk["extra_headers"]["X-Custom"] == "value"
    assert captured["request"]["url"].endswith("/endpoints/demo-endpoint/invocations")


def test_lazy_connector_and_session_creation() -> None:
    client: Any = _make_client()
    client.connector = None
    client.session = None
    client._connector_kwargs = {
        "limit": 100,
        "ttl_dns_cache": 3600,
        "use_dns_cache": True,
        "enable_cleanup_closed": True,
    }
    client._client_timeout = None

    async def _test() -> None:
        session = client._ensure_session()
        assert client.connector is not None
        assert session is not None
        assert client._ensure_session() is session
        await session.close()

    asyncio.run(_test())


def test_signed_request_service_tier_header() -> None:
    import boto3  # pylint: disable=import-outside-toplevel

    client = _make_client()
    client.region_name = "us-east-1"
    client.credentials = boto3.Session(region_name="us-east-1").get_credentials()

    headers = client._signed_request(
        credentials=client.credentials,
        url="https://bedrock-runtime.us-east-1.amazonaws.com/model/test/invoke",
        method="POST",
        body=b'{"prompt": "hello"}',
        region_name="us-east-1",
        serviceTier="default",
    )

    assert headers["X-Amzn-Bedrock-Service-Tier"] == "default"
    assert "X-Amzn-Bedrock-ServiceTier" not in headers


def test_build_converse_body_all_fields() -> None:
    client = _make_client()

    body = client._build_converse_body(
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
        system=[{"text": "You are helpful."}],
        inferenceConfig={"maxTokens": 100},
        toolConfig={"tools": []},
        guardrailConfig={"guardrailIdentifier": "g1", "guardrailVersion": "1"},
        additionalModelRequestFields={"extra": True},
        additionalModelResponseFieldPaths=["/stop"],
        promptVariables={"var1": {"text": "val"}},
        requestMetadata={"key": "value"},
        performanceConfig={"latency": "optimized"},
        serviceTier={"type": "default"},
        outputConfig={
            "textFormat": {"type": "json_schema", "structure": {"jsonSchema": {}}}
        },
    )

    assert body["system"] == [{"text": "You are helpful."}]
    assert body["inferenceConfig"] == {"maxTokens": 100}
    assert body["serviceTier"] == {"type": "default"}
    assert "outputConfig" in body

    minimal = client._build_converse_body(
        messages=[{"role": "user", "content": [{"text": "hi"}]}]
    )
    assert "system" not in minimal
    assert "outputConfig" not in minimal


def test_converse_sends_output_config() -> None:
    client, captured = _stub_client(response_body=b'{"output": {}}')

    result = asyncio.run(
        client.converse(
            modelId="test-model",
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
            outputConfig={
                "textFormat": {"type": "json_schema", "structure": {"jsonSchema": {}}}
            },
        )
    )

    assert result == b'{"output": {}}'
    sent_body = orjson.loads(captured["request"]["data"])
    assert "outputConfig" in sent_body
    assert "/converse" in captured["request"]["url"]


def test_converse_stream_sends_output_config() -> None:
    client: Any = _make_client()
    client.region_name = "us-east-1"
    client.credentials = object()
    client._max_retries = 0
    client._retry_backoff = 0.0
    client._retry_backoff_cap = 0.0
    client._retry_statuses = ()
    client._max_concurrency = None
    client._request_semaphore = None
    client._credential_lock = None
    client._client_timeout = None
    client.assume_role_arn = None
    client.session = None
    client.connector = None
    client._connector_kwargs = {"limit": 100}

    captured: Dict[str, Any] = {}

    class _EmptyContent:
        async def iter_chunked(self, _size: int) -> Any:
            return
            yield  # noqa: E501

    class _StreamResponse:
        status = 200
        content = _EmptyContent()

    async def _noop() -> None:
        return None

    async def _noop_err(_r: Any) -> None:
        return None

    client._ensure_valid_credentials = _noop
    client._signed_request = lambda **_kw: {"Authorization": "test"}

    def fake_request(**kw: Any) -> DummyContext:
        captured["request"] = kw
        return DummyContext(_StreamResponse())

    client._request = fake_request
    client._handle_error_response = _noop_err

    async def _test() -> None:
        async for _ in client.converse_stream(
            modelId="test-model",
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
            outputConfig={
                "textFormat": {"type": "json_schema", "structure": {"jsonSchema": {}}}
            },
        ):
            pass

    asyncio.run(_test())

    sent_body = orjson.loads(captured["request"]["data"])
    assert "outputConfig" in sent_body
    assert "/converse-stream" in captured["request"]["url"]


def test_count_tokens_converse_style() -> None:
    client, captured = _stub_client(response_body=b'{"inputTokens": 42}')

    result = asyncio.run(
        client.count_tokens(
            modelId="test-model",
            messages=[{"role": "user", "content": [{"text": "hello"}]}],
            system=[{"text": "Be brief."}],
        )
    )

    assert result == {"inputTokens": 42}
    sent = orjson.loads(captured["request"]["data"])
    assert sent["input"]["converse"]["messages"][0]["content"] == [{"text": "hello"}]
    assert sent["input"]["converse"]["system"] == [{"text": "Be brief."}]
    assert "/count-tokens" in captured["request"]["url"]


def test_count_tokens_invoke_model_style() -> None:
    client, captured = _stub_client(response_body=b'{"inputTokens": 10}')

    invoke_body = '{"anthropic_version": "bedrock-2023-05-31", "messages": []}'
    result = asyncio.run(
        client.count_tokens(modelId="test-model", invokeModelBody=invoke_body)
    )

    assert result == {"inputTokens": 10}
    sent = orjson.loads(captured["request"]["data"])
    assert sent["input"]["invokeModel"]["body"] == invoke_body


def test_count_tokens_raises_without_input() -> None:
    client, _ = _stub_client()
    with pytest.raises(ValueError, match="Provide either"):
        asyncio.run(client.count_tokens(modelId="test-model"))


def test_apply_guardrail() -> None:
    client, captured = _stub_client(
        response_body=b'{"action": "NONE", "outputs": [{"text": "ok"}]}'
    )

    result = asyncio.run(
        client.apply_guardrail(
            guardrailIdentifier="gr-123",
            guardrailVersion="1",
            source="INPUT",
            content=[{"text": {"text": "Hello"}}],
            outputScope="interventions",
        )
    )

    assert result["action"] == "NONE"
    sent = orjson.loads(captured["request"]["data"])
    assert sent["source"] == "INPUT"
    assert sent["content"] == [{"text": {"text": "Hello"}}]
    assert sent["outputScope"] == "interventions"
    assert "/guardrail/gr-123/version/1/apply" in captured["request"]["url"]


def test_start_async_invoke() -> None:
    client, captured = _stub_client(
        response_body=b'{"invocationArn": "arn:aws:bedrock:us-east-1:123:async-invoke/abc"}'
    )

    result = asyncio.run(
        client.start_async_invoke(
            modelId="test-model",
            modelInput={"messages": [{"role": "user", "content": "hi"}]},
            outputDataConfig={"s3OutputDataConfig": {"s3Uri": "s3://bucket/output"}},
            clientRequestToken="token-123",
            tags=[{"key": "env", "value": "test"}],
        )
    )

    assert "invocationArn" in result
    sent = orjson.loads(captured["request"]["data"])
    assert sent["modelId"] == "test-model"
    assert sent["clientRequestToken"] == "token-123"
    assert sent["tags"] == [{"key": "env", "value": "test"}]
    assert "/async-invoke" in captured["request"]["url"]


def test_get_async_invoke() -> None:
    client, captured = _stub_get_client(response_body=b'{"status": "Completed"}')

    result = asyncio.run(client.get_async_invoke(invocationArn="abc-123"))

    assert result == {"status": "Completed"}
    assert captured["signed"]["method"] == "GET"
    assert "/async-invoke/abc-123" in captured["signed"]["url"]


def test_list_async_invokes() -> None:
    client, captured = _stub_get_client(response_body=b'{"invocations": []}')

    result = asyncio.run(
        client.list_async_invokes(
            statusEquals="Completed",
            maxResults=10,
            sortBy="SubmissionTime",
            sortOrder="Descending",
        )
    )

    assert result == {"invocations": []}
    url = captured["signed"]["url"]
    assert "statusEquals=Completed" in url
    assert "maxResults=10" in url


def test_converse_raises_on_error_status() -> None:
    client, _ = _stub_client(response_body=b"Throttled", status=429)

    with pytest.raises(BedrockClientError) as exc_info:
        asyncio.run(
            client.converse(
                modelId="test-model",
                messages=[{"role": "user", "content": [{"text": "hi"}]}],
            )
        )

    assert exc_info.value.status == 429


def test_process_converse_stream_event() -> None:
    client = _make_client()

    result = client._process_converse_stream_event(
        {
            "headers": {
                ":message-type": "event",
                ":event-type": "contentBlockDelta",
                ":content-type": "application/json",
            },
            "payload": b'{"delta": {"text": "Hello"}, "contentBlockIndex": 0}',
        }
    )
    assert result == {
        "contentBlockDelta": {"delta": {"text": "Hello"}, "contentBlockIndex": 0}
    }

    with pytest.raises(BedrockStreamError):
        client._process_converse_stream_event(
            {
                "headers": {
                    ":message-type": "exception",
                    ":exception-type": "throttlingException",
                    ":content-type": "application/json",
                },
                "payload": b'{"message": "Rate exceeded"}',
            }
        )
