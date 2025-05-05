import boto3
import base64
import orjson
import aiohttp
import struct
import io

from botocore.awsrequest import AWSRequest
from botocore.auth import SigV4Auth


class Client:
    def __init__(self, region_name):
        self.region_name = region_name

        # Initialize the aiohttp session
        conn = aiohttp.TCPConnector(
            limit=10000,
            ttl_dns_cache=3600,
            use_dns_cache=True,
            enable_cleanup_closed=True,
            verify_ssl=True,
        )
        self.session = aiohttp.ClientSession(connector=conn)

        # Initialize the boto3 session
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

        try:
            async with self.session.post(
                url=url,
                headers=headers,
                data=body,
            ) as res:
                if res.status == 200:
                    async for chunk, _ in res.content.iter_chunks():
                        yield self.__parse_chunk_async(chunk)
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
        if kwargs.get("accept"):
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

    @staticmethod
    def __parse_chunk_async(chunk: bytes) -> dict:
        """Parse an AWS Event Stream chunk into a usable message.

        The event stream format consists of:
        - 4 bytes: Total message length
        - 4 bytes: Headers length
        - 4 bytes: CRC checksum for the prelude
        - Headers content
        - Message content
        - 4 bytes: CRC checksum for the entire message

        Returns:
            dict: The decoded message content
        """
        if not chunk:
            return None

        try:
            # Use a BytesIO to parse the binary format
            stream = io.BytesIO(chunk)

            # Read the prelude (first 12 bytes)
            total_length = struct.unpack(">I", stream.read(4))[
                0
            ]  # Big-endian 4-byte unsigned int
            headers_length = struct.unpack(">I", stream.read(4))[0]
            # Skip the prelude CRC
            stream.read(4)

            # Parse headers
            headers = {}
            headers_end_pos = 12 + headers_length
            while stream.tell() < headers_end_pos:
                header_name_len = stream.read(1)[0]  # Single byte length
                header_name = stream.read(header_name_len).decode("utf-8")
                header_type = stream.read(1)[0]  # Header value type

                # Handle different header value types
                if header_type == 7:  # String
                    # 2-byte length for string values
                    value_len = struct.unpack(">H", stream.read(2))[0]
                    value = stream.read(value_len).decode("utf-8")
                    headers[header_name] = value
                else:
                    # Skip other header types for now
                    # Type 0-6: bool, byte, short, int, long, timestamp, uuid
                    type_lengths = {0: 0, 1: 1, 2: 2, 3: 4, 4: 8, 5: 8, 6: 16}
                    if header_type in type_lengths:
                        stream.read(type_lengths[header_type])

            # Read the payload (everything between headers and the final CRC)
            # Total - headers - prelude(12) - end CRC(4)
            payload_length = total_length - headers_length - 16
            payload = stream.read(payload_length)

            # Parse the JSON payload
            content = payload.decode("utf-8")

            # Find the JSON object with the "bytes" field
            payload_json = orjson.loads(content)
            if "bytes" in payload_json:
                # Decode base64-encoded message
                bytes_content = payload_json["bytes"]
                decoded = base64.b64decode(bytes_content)
                return decoded
            return payload_json

        except Exception as e:
            # On error, return the raw chunk
            return {"error": str(e), "raw": chunk}
