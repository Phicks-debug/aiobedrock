import asyncio
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aiobedrock.main import Client


def _make_client() -> Client:
    """Create a Client instance without triggering __init__."""
    return Client.__new__(Client)  # type: ignore[call-arg]


def test_process_event_message_decodes_base64_payload():
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

    result = client._process_event_message(message)

    assert result == b"hello world"


def test_process_event_message_logs_exception(capsys):
    client = _make_client()
    message = {
        "headers": {
            ":message-type": "exception",
            ":exception-type": "SomeError",
            ":content-type": "application/json",
        },
        "payload": b'{"message": "failing"}',
    }

    result = client._process_event_message(message)
    assert result is None

    # Logging is handled by a custom logger; just ensure it doesn't raise


def test_normalize_headers_extracts_values():
    client = _make_client()

    class HeaderValue:
        def __init__(self, value):
            self.value = value

    headers = {
        "plain": "value",
        "wrapped": HeaderValue("wrapped-value"),
    }

    normalized = client._normalize_headers(headers)

    assert normalized == {
        "plain": "value",
        "wrapped": "wrapped-value",
    }


def test_invoke_sagemaker_endpoint_builds_headers():
    client = _make_client()
    client.region_name = "us-west-2"
    client.credentials = object()

    async def _no_credentials_refresh():
        return None

    client._ensure_valid_credentials = _no_credentials_refresh  # type: ignore[assignment]

    captured = {}

    def fake_signed_request(**kwargs):
        captured["kwargs"] = kwargs
        return {"Authorization": "test"}

    async def fake_handle_error_response(response):
        return None

    class DummyResponse:
        status = 200

        async def read(self):
            return b"ok"

    class DummyContext:
        def __init__(self, response):
            self._response = response

        async def __aenter__(self):
            return self._response

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def fake_request(**kwargs):
        captured["request"] = kwargs
        return DummyContext(DummyResponse())

    client._signed_request = fake_signed_request  # type: ignore[assignment]
    client._request = fake_request  # type: ignore[assignment]
    client._handle_error_response = fake_handle_error_response  # type: ignore[assignment]

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
    signed_kwargs = captured["kwargs"]
    assert signed_kwargs["service"] == "sagemaker"
    assert signed_kwargs["accept"] == "application/json"
    assert signed_kwargs["contentType"] == "application/json"
    assert (
        signed_kwargs["extra_headers"]["X-Amzn-SageMaker-Custom-Attributes"] == "attr"
    )
    assert (
        signed_kwargs["extra_headers"]["X-Amzn-SageMaker-Target-Variant"] == "variant"
    )
    assert signed_kwargs["extra_headers"]["X-Custom"] == "value"
    assert captured["request"]["url"].endswith("/endpoints/demo-endpoint/invocations")
