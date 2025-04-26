import re
import boto3
import base64
import orjson
import aiohttp

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
    def __parse_chunk_async(chunk: dict) -> bytes:
        # Decode the chunk with 'ignore' to skip invalid bytes
        message = chunk.decode("utf-8", errors="ignore")

        json_pattern = re.compile(r'{"bytes":"(.*?)"}')

        # Find all JSON objects in the message
        matches = json_pattern.finditer(message)

        for m in matches:
            p_out = orjson.loads(m.group(0))
            # Decode the response json contents.
            return base64.b64decode(p_out["bytes"])

        # Return nothing if nothing is parsed
        return b""
