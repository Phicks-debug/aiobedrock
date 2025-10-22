import base64

import orjson
import pytest

from aiobedrock.main import BedrockStreamError, Client


def _make_client() -> Client:
    """Create a Client instance without triggering __init__."""
    return Client.__new__(Client)  # type: ignore[call-arg]


def test_process_event_message_decodes_base64_payload():
    client = _make_client()
    payload_bytes = base64.b64encode(b"hello world").decode()
    message = {
        "headers": {
            ":message-type": "event",
            ":event-type": "chunk",
            ":content-type": "application/json",
        },
        "payload": orjson.dumps({"bytes": payload_bytes}),
    }

    result = client._process_event_message(message)

    assert result == b"hello world"


def test_process_event_message_raises_on_exception_type():
    client = _make_client()
    message = {
        "headers": {
            ":message-type": "exception",
            ":exception-type": "SomeError",
            ":content-type": "application/json",
        },
        "payload": b'{"message": "failing"}',
    }

    with pytest.raises(BedrockStreamError) as exc_info:
        client._process_event_message(message)

    assert "SomeError" in str(exc_info.value)


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
