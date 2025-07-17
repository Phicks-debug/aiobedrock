import io
import boto3
import base64
import orjson
import aiohttp
import struct
import logsim
from datetime import datetime, timezone

from botocore.awsrequest import AWSRequest
from botocore.auth import SigV4Auth
from botocore.eventstream import EventStreamMessage, NoInitialResponseError

from typing import AsyncGenerator, Dict, Any, Union

log = logsim.CustomLogger()


class Client:
    def __init__(self, region_name: str, assume_role_arn: str = None):
        self.region_name = region_name
        self.assume_role_arn = assume_role_arn
        self.connector = aiohttp.TCPConnector(
            limit=10000,
            ttl_dns_cache=3600,
            use_dns_cache=True,
            enable_cleanup_closed=True,
        )
        self.session = None
        self.expiration = None
        self.access_key = None
        self.secret_key = None
        self.session_token = None

        # Initialize credentials
        self._refresh_credentials()

    def _refresh_credentials(self):
        """Refresh AWS credentials, handling role assumption if needed"""
        if self.assume_role_arn:
            sts_client = boto3.client("sts")
            response = sts_client.assume_role(
                RoleArn=self.assume_role_arn,
                RoleSessionName="aiobedrock",
            )

            # Extract temporary credentials
            credentials = response["Credentials"]
            self.access_key = credentials["AccessKeyId"]
            self.secret_key = credentials["SecretAccessKey"]
            self.session_token = credentials["SessionToken"]
            self.expiration = credentials["Expiration"]

            log.info(f"Refreshed credentials, expires at: {self.expiration}")

            # Create session with temporary credentials
            boto3_session = boto3.Session(
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                aws_session_token=self.session_token,
                region_name=self.region_name,
            )
        else:
            # Use default credentials
            boto3_session = boto3.Session(region_name=self.region_name)

        self.credentials = boto3_session.get_credentials()

    def _are_credentials_expired(self) -> bool:
        """Check if the current credentials are expired or about to expire"""
        if not self.assume_role_arn or not self.expiration:
            return False

        # Check if credentials expire within the next 5 minutes
        current_time = datetime.now(timezone.utc)
        expiration_time = self.expiration

        # Handle timezone-aware expiration time
        if expiration_time.tzinfo is None:
            expiration_time = expiration_time.replace(tzinfo=timezone.utc)

        time_until_expiration = expiration_time - current_time
        return time_until_expiration.total_seconds() < 300  # 5 minutes

    def _ensure_valid_credentials(self):
        """Ensure credentials are valid, refreshing if necessary"""
        if not self.assume_role_arn:
            return
        if self._are_credentials_expired():
            log.info("Credentials expired or expiring soon, refreshing...")
            self._refresh_credentials()

    async def __aenter__(self):
        if self.session is None:
            self.session = aiohttp.ClientSession(connector=self.connector)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        if self.session is not None:
            await self.session.close()
            self.session = None

    async def invoke_model(self, body: str, modelId: str, **kwargs) -> bytes:
        """Invoke a model and return the response body as bytes"""
        # Ensure credentials are valid before making the request
        self._ensure_valid_credentials()

        url = f"https://bedrock-runtime.{self.region_name}.amazonaws.com/model/{modelId}/invoke"  # noqa: E501
        headers = self._signed_request(
            body=body,
            url=url,
            method="POST",
            credentials=self.credentials,
            region_name=self.region_name,
            **kwargs,
        )

        async with self.session.post(
            url=url,
            headers=headers,
            data=body,
        ) as res:
            await self._handle_error_response(res)
            return await res.read()

    async def invoke_model_with_response_stream(
        self, body: str, modelId: str, **kwargs
    ) -> AsyncGenerator[Union[Dict[str, Any], bytes], None]:
        """
        Invoke a model with streaming response with memory management
        """
        # Ensure credentials are valid before making the request
        self._ensure_valid_credentials()

        url = f"https://bedrock-runtime.{self.region_name}.amazonaws.com/model/{modelId}/invoke-with-response-stream"  # noqa: E501
        headers = self._signed_request(
            body=body,
            url=url,
            method="POST",
            credentials=self.credentials,
            region_name=self.region_name,
            **kwargs,
        )

        async with self.session.post(
            url=url,
            headers=headers,
            data=body,
        ) as res:
            await self._handle_error_response(res)

            # Process the stream using botocore's EventStreamMessage
            message_buffer = b""

            async for chunk in res.content.iter_chunked(8192):
                message_buffer += chunk

                # Parse complete messages from the buffer
                while True:
                    try:
                        # Try to parse a message from the current buffer
                        message, consumed = self._parse_event_stream_message(
                            message_buffer
                        )

                        if message is None:
                            # Not enough data for a complete message
                            break

                        # Remove consumed bytes from buffer
                        message_buffer = message_buffer[consumed:]

                        # Process the parsed message
                        processed_content = self._process_event_message(
                            message,
                        )
                        if processed_content is not None:
                            yield processed_content

                    except Exception as e:
                        log.error(
                            f"Error parsing event stream message: {e}"
                        )  # noqa:E501
                        # Try to recover by discarding some data
                        if len(message_buffer) > 1024:
                            message_buffer = message_buffer[512:]
                        else:
                            break

            # Process any remaining complete messages in the buffer
            while message_buffer:
                try:
                    message, consumed = self._parse_event_stream_message(
                        message_buffer,
                    )
                    if message is None or consumed == 0:
                        break

                    message_buffer = message_buffer[consumed:]
                    processed_content = self._process_event_message(message)
                    if processed_content is not None:
                        yield processed_content

                except Exception as e:
                    log.error(f"Error processing remaining buffer: {e}")
                    break

    def _parse_event_stream_message(
        self, buffer: bytes
    ) -> tuple[EventStreamMessage, int]:
        """
        Parse an EventStreamMessage from the buffer using botocore's parser
        """
        if len(buffer) < 12:  # Minimum message size (prelude)
            return None, 0

        try:
            # Use botocore's EventStreamMessage to parse
            # Create a BytesIO stream for the message parser
            stream = io.BytesIO(buffer)

            try:
                message = EventStreamMessage.from_response_dict(
                    {"body": stream},
                    None,
                )

                # Calculate how many bytes were consumed
                consumed = stream.tell()
                return message, consumed

            except NoInitialResponseError:
                return None, 0
            except Exception:
                # Try the manual approach as fallback
                return self._manual_parse_message(buffer)

        except Exception as e:
            log.error(f"Failed to parse event stream message: {e}")
            return None, 0

    def _manual_parse_message(
        self,
        buffer: bytes,
    ) -> tuple[Dict[str, Any], int]:
        """Fallback manual parsing when botocore parsing fails"""
        if len(buffer) < 12:
            return None, 0

        try:
            # Parse the prelude
            total_length = struct.unpack(">I", buffer[0:4])[0]
            headers_length = struct.unpack(">I", buffer[4:8])[0]

            if len(buffer) < total_length:
                return None, 0

            # Extract the payload (skip prelude, headers, and trailing CRC)
            payload_start = 12 + headers_length
            payload_end = total_length - 4
            payload = buffer[payload_start:payload_end]

            # Create a simple message structure
            message = {"headers": {}, "payload": payload}

            return message, total_length

        except Exception as e:
            log.error(f"Manual parsing failed: {e}")
            return None, 0

    def _process_event_message(
        self,
        message,
    ) -> Union[Dict[str, Any], bytes, None]:
        """Process an individual event message"""
        try:
            # Handle EventStreamMessage objects
            if hasattr(message, "headers") and hasattr(message, "payload"):
                headers = getattr(message, "headers", {})
                payload = getattr(message, "payload", b"")
            # Handle dict-like objects from manual parsing
            elif isinstance(message, dict):
                headers = message.get("headers", {})
                payload = message.get("payload", b"")
            else:
                log.warning(f"Unexpected message type: {type(message)}")
                return None

            # Get content type from headers
            content_type = "application/json"  # Default
            if hasattr(headers, "get"):
                content_type_header = headers.get(":content-type")
                if content_type_header and hasattr(
                    content_type_header,
                    "value",
                ):
                    content_type = content_type_header.value

            if not payload:
                return {}

            # Handle JSON content
            if "application/json" in content_type:
                try:
                    if isinstance(payload, bytes):
                        payload_str = payload.decode("utf-8")
                    else:
                        payload_str = str(payload)

                    payload_data = orjson.loads(payload_str)

                    # Extract base64-encoded bytes if present
                    if "bytes" in payload_data:
                        return base64.b64decode(payload_data["bytes"])
                    elif (
                        "chunk" in payload_data
                        and isinstance(payload_data["chunk"], dict)
                        and "bytes" in payload_data["chunk"]
                    ):
                        return base64.b64decode(payload_data["chunk"]["bytes"])
                    else:
                        return payload_data

                except (UnicodeDecodeError, orjson.JSONDecodeError) as e:
                    log.error(f"Failed to parse JSON payload: {e}")
                    return payload
            else:
                # Return raw bytes for non-JSON content
                return payload

        except Exception as e:
            log.error(f"Error processing event message: {e}")
            return None

    async def _handle_error_response(self, response: aiohttp.ClientResponse):
        """Handle HTTP error responses"""
        if response.status == 200:
            return

        error_text = await response.text()
        error_map = {
            403: "AccessDeniedException",
            408: "ModelTimeoutException",
            424: "ModelErrorException",
            429: "ThrottlingException",
            500: "InternalServerException",
        }

        error_type = error_map.get(response.status, "UnknownException")
        raise Exception(f"{response.status} {error_type}: {error_text}")

    def _signed_request(
        self,
        credentials,
        url: str,
        method: str,
        body: str,
        region_name: str,
        **kwargs,
    ) -> Dict[str, str]:
        """Create a signed AWS request"""
        request = AWSRequest(method=method, url=url, data=body)
        request.headers.add_header("Host", url.split("/")[2])

        # Set appropriate headers based on the endpoint
        if "invoke-with-response-stream" in url:
            request.headers.add_header(
                "Accept",
                "application/vnd.amazon.eventstream",
            )
        else:
            request.headers.add_header(
                "Accept", kwargs.get("accept", "application/json")
            )

        request.headers.add_header(
            "Content-Type", kwargs.get("contentType", "application/json")
        )
        request.headers.add_header(
            "X-Amzn-Bedrock-Trace", kwargs.get("trace", "DISABLED")
        )

        # Optional headers
        optional_headers = [
            (
                "guardrailIdentifier",
                "X-Amzn-Bedrock-GuardrailIdentifier",
            ),
            (
                "guardrailVersion",
                "X-Amzn-Bedrock-GuardrailVersion",
            ),
            (
                "performanceConfigLatency",
                "X-Amzn-Bedrock-PerformanceConfig-Latency",
            ),
        ]

        for kwarg_key, header_name in optional_headers:
            if kwargs.get(kwarg_key):
                request.headers.add_header(header_name, kwargs.get(kwarg_key))

        # Sign the request
        SigV4Auth(credentials, "bedrock", region_name).add_auth(request)
        return dict(request.headers)
