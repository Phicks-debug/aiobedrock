import boto3
import base64
import orjson
import aiohttp
import struct
import io
from binascii import crc32  # For CRC validation

from botocore.awsrequest import AWSRequest
from botocore.auth import SigV4Auth

# Constants from botocore.eventstream
_PRELUDE_LENGTH = 12
_MAX_HEADERS_LENGTH = 128 * 1024
_MAX_PAYLOAD_LENGTH = 24 * 1024 * 1024


class Client:
    def __init__(self, region_name):
        self.region_name = region_name
        conn = aiohttp.TCPConnector(
            limit=10000,
            ttl_dns_cache=3600,
            use_dns_cache=True,
            enable_cleanup_closed=True,
            verify_ssl=True,
        )
        self.session = aiohttp.ClientSession(connector=conn)
        boto3_session = boto3.Session(region_name=region_name)
        self.credentials = boto3_session.get_credentials()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        if self.session:
            await self.session.close()

    async def invoke_model(self, body: str, modelId: str, **kwargs):
        url = f"https://bedrock-runtime.{self.region_name}.amazonaws.com/model/{modelId}/invoke"  # noqa: E501
        headers = self.__signed_request(
            body=body,
            url=url,
            method="POST",
            credentials=self.credentials,
            region_name=self.region_name,
            **kwargs,
        )
        try:
            async with self.session.post(
                url=url,
                headers=headers,
                data=body,
            ) as res:
                if res.status == 200:
                    return await res.read()
                elif res.status == 403:
                    e = await res.text()
                    raise Exception(f"403 AccessDeniedException: {e}")
                elif res.status == 500:
                    e = await res.text()
                    raise Exception(f"500 InternalServerException: {e}")
                elif res.status == 424:
                    e = await res.text()
                    raise Exception(f"424 ModelErrorException: {e}")
                elif res.status == 408:
                    e = await res.text()
                    raise Exception(f"408 ModelTimeoutException: {e}")
                elif res.status == 429:
                    e = await res.text()
                    raise Exception(f"429 ThrottlingException: {e}")
                else:
                    e = await res.text()
                    raise Exception(f"{res.status}: {e}")
        except Exception as e:
            raise Exception(f"Error invoke model: {e}")

    async def invoke_model_with_response_stream(
        self, body: str, modelId: str, **kwargs
    ):
        url = f"https://bedrock-runtime.{self.region_name}.amazonaws.com/model/{modelId}/invoke-with-response-stream"  # noqa: E501
        headers = self.__signed_request(
            body=body,
            url=url,
            method="POST",
            credentials=self.credentials,
            region_name=self.region_name,
            **kwargs,
        )

        event_buffer = b""
        try:
            async with self.session.post(
                url=url,
                headers=headers,
                data=body,
            ) as res:
                if res.status == 200:
                    async for http_chunk, _ in res.content.iter_chunks():
                        event_buffer += http_chunk
                        while True:
                            parsed_event_payload, consumed_length = (
                                self._try_parse_message_from_buffer(
                                    event_buffer,
                                )
                            )
                            if parsed_event_payload is not None:
                                yield parsed_event_payload
                                event_buffer = event_buffer[consumed_length:]
                            else:
                                # Not enough data for a full message yet,
                                # or buffer is empty and stream ended
                                break
                    # After the loop, try to
                    # parse any remaining data in the buffer
                    while event_buffer:
                        parsed_event_payload, consumed_length = (
                            self._try_parse_message_from_buffer(event_buffer)
                        )
                        if parsed_event_payload is not None:
                            yield parsed_event_payload
                            event_buffer = event_buffer[consumed_length:]
                        else:
                            if (
                                event_buffer
                            ):  # Should not happen if stream ended cleanly
                                print(
                                    f"Warning: Unparsed data left in buffer: {event_buffer[:100].hex()}"  # noqa: E501
                                )
                            break
                elif res.status == 403:
                    e = await res.text()
                    raise Exception(f"403 AccessDeniedException: {e}")
                elif res.status == 500:
                    e = await res.text()
                    raise Exception(f"500 InternalServerException: {e}")
                elif res.status == 424:
                    e = await res.text()
                    raise Exception(f"424 ModelErrorException: {e}")
                elif res.status == 408:
                    e = await res.text()
                    raise Exception(f"408 ModelTimeoutException: {e}")
                elif res.status == 429:
                    e = await res.text()
                    raise Exception(f"429 ThrottlingException: {e}")
                else:
                    e = await res.text()
                    raise Exception(f"{res.status}: {e}")
        except Exception as e:
            raise Exception(f"Error invoke model with response stream: {e}")

    @staticmethod
    def _validate_crc(
        data_bytes: bytes, expected_crc: int, initial_crc_val: int = 0
    ) -> None:
        calculated_crc = crc32(data_bytes, initial_crc_val) & 0xFFFFFFFF
        if calculated_crc != expected_crc:
            raise ValueError(
                f"CRC mismatch. Expected: {expected_crc:#010x}, Calculated: {calculated_crc:#010x}"  # noqa: E501
            )

    def _try_parse_message_from_buffer(
        self, buffer: bytes
    ) -> tuple[dict | bytes | None, int]:
        """
        Tries to parse a complete
        event stream message from the beginning of the buffer.
        Returns (parsed_payload_dict_or_bytes_or_None, consumed_bytes_count).
        The parsed_payload is the final decoded content
        (e.g., from the "bytes" field as bytes, or a dict for metadata).
        """
        if len(buffer) < _PRELUDE_LENGTH:
            return None, 0  # Not enough data for prelude

        try:
            # Parse Prelude
            total_length = struct.unpack(">I", buffer[0:4])[0]
            headers_length = struct.unpack(">I", buffer[4:8])[0]
            prelude_crc = struct.unpack(">I", buffer[8:12])[0]

            # Validate Prelude CRC
            self._validate_crc(buffer[0:8], prelude_crc)

            if (
                headers_length < 0 or headers_length > _MAX_HEADERS_LENGTH
            ):  # headers_length can be 0
                raise ValueError(
                    f"Headers length {headers_length} is invalid"
                    f"or exceeds max {_MAX_HEADERS_LENGTH}"
                )

            payload_length = (
                total_length - headers_length - _PRELUDE_LENGTH - 4
            )  # 4 for message CRC
            if (
                payload_length < 0 or payload_length > _MAX_PAYLOAD_LENGTH
            ):  # payload_length can be 0
                raise ValueError(f"Invalid payload length: {payload_length}")

            if len(buffer) < total_length:
                return None, 0  # Not enough data for the full message

            # Full message is available in buffer
            message_bytes = buffer[:total_length]

            # Validate Message CRC
            # The message CRC is calculated over the entire message
            # up to (but not including) the CRC itself.
            message_content_for_crc = message_bytes[:-4]
            expected_message_crc = struct.unpack(
                ">I", message_bytes[total_length - 4 : total_length]  # noqa: E203 E501
            )[0]
            self._validate_crc(message_content_for_crc, expected_message_crc)

            # Parse Headers
            headers_stream = io.BytesIO(
                message_bytes[
                    _PRELUDE_LENGTH : _PRELUDE_LENGTH  # noqa: E203
                    + headers_length  # noqa: E501
                ]
            )
            parsed_headers = {}
            current_headers_pos = 0
            while current_headers_pos < headers_length:
                header_name_len = headers_stream.read(1)[0]
                current_headers_pos += 1
                header_name = headers_stream.read(
                    header_name_len,
                ).decode("utf-8")
                current_headers_pos += header_name_len
                header_type = headers_stream.read(1)[0]
                current_headers_pos += 1

                if header_type == 7:  # String
                    value_len = struct.unpack(">H", headers_stream.read(2))[0]
                    current_headers_pos += 2
                    value = headers_stream.read(value_len).decode("utf-8")
                    current_headers_pos += value_len
                    parsed_headers[header_name] = value
                # Add other header type parsing if needed,
                # similar to botocore.eventstream.EventStreamHeaderParser
                # For Bedrock, :event-type (initial-response, chunk, error)
                # and :content-type are common.
                else:  # Minimal handling for other types for now
                    # Based on
                    # botocore.eventstream.EventStreamHeaderParser._HEADER_TYPE_MAP
                    # and DecodeUtils
                    if header_type == 0 or header_type == 1:  # True / False
                        pass  # No value bytes
                    elif header_type == 2:  # byte
                        headers_stream.read(1)
                        current_headers_pos += 1
                    elif header_type == 3:  # short
                        headers_stream.read(2)
                        current_headers_pos += 2
                    elif header_type == 4:  # integer
                        headers_stream.read(4)
                        current_headers_pos += 4
                    elif (
                        header_type == 5 or header_type == 8
                    ):  # long or timestamp (int64)
                        headers_stream.read(8)
                        current_headers_pos += 8
                    elif header_type == 6:  # byte_array
                        value_len = struct.unpack(
                            ">H", headers_stream.read(2)
                        )[  # noqa: E501
                            0
                        ]
                        current_headers_pos += 2
                        headers_stream.read(value_len)
                        current_headers_pos += value_len
                    elif header_type == 9:  # uuid
                        headers_stream.read(16)
                        current_headers_pos += 16
                    else:
                        raise ValueError(
                            f"Unsupported header type: {header_type}"
                        )  # noqa: E501

            # Extract and Parse Payload
            payload_bytes = message_bytes[
                _PRELUDE_LENGTH + headers_length : total_length - 4  # noqa: E203 E501
            ]

            content_type = parsed_headers.get(
                ":content-type", "application/json"
            )  # Default for Bedrock

            if "application/json" in content_type:
                if not payload_bytes:  # Empty payload for some events
                    return {}, total_length

                payload_json_str = payload_bytes.decode("utf-8")
                payload_data = orjson.loads(payload_json_str)

                # For Bedrock, the actual data is often in a 'bytes' field,
                # base64 encoded.
                if "bytes" in payload_data:
                    decoded_final_payload = base64.b64decode(
                        payload_data["bytes"],
                    )
                    return (
                        decoded_final_payload,
                        total_length,
                    )  # Return the decoded bytes
                # It might also be in chunk.bytes
                elif (
                    "chunk" in payload_data
                    and isinstance(payload_data["chunk"], dict)
                    and "bytes" in payload_data["chunk"]
                ):
                    decoded_final_payload = base64.b64decode(
                        payload_data["chunk"]["bytes"]
                    )
                    return decoded_final_payload, total_length
                else:
                    # If no 'bytes' field, return the parsed JSON dict itself
                    # (e.g. for metadata events)
                    return payload_data, total_length
            else:
                # For other content types, return raw payload bytes
                return payload_bytes, total_length

        except struct.error as e:  # Not enough data for a struct unpack
            print(f"Struct error, likely partial message: {e}")
            return None, 0
        except ValueError as e:  # CRC mismatch or other parsing error
            print(
                f"ValueError during parsing: {e} on buffer (first 100 bytes): {buffer[:100].hex()}"  # noqa: E501
            )
            # Attempt to discard the message
            # if total_length is known and seems plausible
            if "total_length" in locals() and 0 < total_length <= len(buffer):
                print(
                    f"Attempting to discard {total_length}"
                    " bytes from buffer after error."
                )
                # Return an error object instead of raw bytes to signal issues
                return {
                    "error": str(e),
                    "action": "discarded_message",
                    "length": total_length,
                }, total_length
            # If total_length is unknown or invalid, it's harder to recover.
            # Returning an error
            # and consuming 0 means the buffer won't advance,
            # potentially stalling.
            # A more robust strategy might involve
            # trying to find the next valid prelude.
            # For now, signal error and don't consume
            # if total_length is unreliable.
            return {
                "error": str(e),
                "action": "parse_failed_no_reliable_length",
            }, 0

    @staticmethod
    def __signed_request(
        credentials,
        url: str,
        method: str,
        body: str,
        region_name: str,
        **kwargs,
    ):
        request = AWSRequest(method=method, url=url, data=body)
        request.headers.add_header(
            "Host",
            url.split("/")[2],
        )
        # For invoke-with-response-stream, X-Amzn-Bedrock-Accept
        # might be needed for the stream part
        # and Accept for the initial HTTP response.
        # The service expects "application/vnd.amazon.eventstream"
        # for the stream itself,
        # but the initial HTTP negotiation might use application/json
        # for errors.
        if "invoke-with-response-stream" in url:
            request.headers.add_header(
                "Accept",
                "application/vnd.amazon.eventstream",
            )
            # The X-Amzn-Bedrock-Accept header is
            # not standard for this operation.
            # The primary Accept header should indicate the event stream.
        elif kwargs.get("accept"):
            request.headers.add_header(
                "Accept",
                kwargs.get("accept"),
            )
        else:
            request.headers.add_header(
                "Accept",
                "application/json",
            )

        if kwargs.get("contentType"):
            request.headers.add_header(
                "Content-Type",
                kwargs.get("contentType"),
            )
        else:
            request.headers.add_header(
                "Content-Type",
                "application/json",
            )
        if kwargs.get("trace"):
            request.headers.add_header(
                "X-Amzn-Bedrock-Trace",
                kwargs.get("trace"),
            )
        else:
            request.headers.add_header(
                "X-Amzn-Bedrock-Trace",
                "DISABLED",
            )
        if kwargs.get("guardrailIdentifier"):
            request.headers.add_header(
                "X-Amzn-Bedrock-GuardrailIdentifier",
                kwargs.get("guardrailIdentifier"),
            )
        if kwargs.get("guardrailVersion"):
            request.headers.add_header(
                "X-Amzn-Bedrock-GuardrailVersion",
                kwargs.get("guardrailVersion"),
            )
        if kwargs.get("performanceConfigLatency"):
            request.headers.add_header(
                "X-Amzn-Bedrock-PerformanceConfig-Latency",
                kwargs.get("performanceConfigLatency"),
            )
        SigV4Auth(credentials, "bedrock", region_name).add_auth(request)
        return dict(request.headers)
