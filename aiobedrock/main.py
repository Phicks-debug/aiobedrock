import asyncio
import base64
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
    overload,
)
from urllib.parse import urlparse

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime.type_defs import (
        GuardrailConfigurationTypeDef,
        GuardrailStreamConfigurationTypeDef,
        InferenceConfigurationTypeDef,
        MessageTypeDef,
        PerformanceConfigurationTypeDef,
        SystemContentBlockTypeDef,
        ToolConfigurationTypeDef,
    )
    from typing import TypedDict

    # ServiceTierConfigTypeDef for converse/converse_stream endpoints
    class ServiceTierConfigTypeDef(TypedDict, total=False):
        type: Literal["priority", "default", "flex", "reserved"]

import aiohttp
import boto3
import logsim
import orjson
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.eventstream import EventStreamBuffer


class BedrockStreamError(Exception):
    """Raised when the Bedrock event stream surfaces an error payload."""

    def __init__(self, message_type: str, exception_type: str, detail: str):
        self.message_type = message_type
        self.exception_type = exception_type or "UnknownException"
        self.detail = detail
        super().__init__(
            f"Bedrock Error ({self.message_type}/{self.exception_type}): {self.detail}"  # noqa: E501
        )


class BedrockClientError(Exception):
    """Raised when the Bedrock Invoke API returns a non-success HTTP."""

    def __init__(self, status: int, error_type: str, detail: str):
        self.status = status
        self.error_type = error_type
        self.detail = detail
        super().__init__(f"{status} {error_type}: {detail}")


log = logsim.CustomLogger()


class Client:
    def __init__(
        self,
        region_name: str,
        assume_role_arn: Optional[str] = None,
        *,
        profile_name: Optional[str] = None,
        aws_account_id: Optional[str] = None,
        max_connections: int = 10000,
        request_timeout: Optional[float] = None,
        max_concurrency: Optional[int] = None,
        max_retries: int = 2,
        retry_backoff: float = 0.5,
        max_backoff: float = 6.0,
        retry_statuses: Optional[Sequence[int]] = None,
    ):
        self.region_name = region_name
        self.assume_role_arn = assume_role_arn
        self.profile_name = profile_name
        self.aws_account_id = aws_account_id
        self.connector = aiohttp.TCPConnector(
            limit=max_connections,
            ttl_dns_cache=3600,
            use_dns_cache=True,
            enable_cleanup_closed=True,
        )
        self.session = None
        self.expiration = None
        self.access_key = None
        self.secret_key = None
        self.session_token = None
        self.credentials = None

        self._client_timeout = (
            aiohttp.ClientTimeout(total=request_timeout) if request_timeout else None  # noqa: E501
        )
        self._max_concurrency = (
            max_concurrency if max_concurrency and max_concurrency > 0 else None  # noqa: E501
        )
        self._max_retries = max(0, max_retries)
        self._retry_backoff = max(0.0, retry_backoff)
        self._retry_backoff_cap = (
            max(max_backoff, self._retry_backoff)
            if max_backoff > 0
            else max(self._retry_backoff, 0.0)
        )
        self._retry_statuses = (
            tuple(retry_statuses)
            if retry_statuses
            else (
                408,
                424,
                429,
                500,
                502,
                503,
                504,
            )
        )
        self._request_semaphore: Optional[asyncio.Semaphore] = None
        self._credential_lock: Optional[asyncio.Lock] = None

        # Initialize credentials
        self._refresh_credentials_sync()

    def _refresh_credentials_sync(self):
        """Refresh AWS credentials, handling role assumption if needed"""
        if self.assume_role_arn:
            # Create STS client using profile if specified
            if self.profile_name:
                base_session = boto3.Session(profile_name=self.profile_name)
                sts_client = base_session.client("sts")
            else:
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
            # Use profile credentials if specified, otherwise use default
            boto3_session = boto3.Session(
                profile_name=self.profile_name,
                region_name=self.region_name,
            )

        self.credentials = boto3_session.get_credentials()

    async def _refresh_credentials(self):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._refresh_credentials_sync)

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

    async def _ensure_valid_credentials(self):
        """Ensure credentials are valid, refreshing if necessary"""
        if self.credentials is None:
            await self._refresh_credentials()
            return

        if not self.assume_role_arn:
            return

        if not self._are_credentials_expired():
            return

        if self._credential_lock is None:
            self._credential_lock = asyncio.Lock()

        async with self._credential_lock:
            if self._are_credentials_expired():
                log.info("Credentials expired or expiring soon, refreshing...")
                await self._refresh_credentials()

    def _ensure_session(self) -> aiohttp.ClientSession:
        """Create the shared aiohttp session on first use."""

        if self.session is None or getattr(self.session, "closed", False):
            session_kwargs: Dict[str, Any] = {"connector": self.connector}
            if self._client_timeout is not None:
                session_kwargs["timeout"] = self._client_timeout
            self.session = aiohttp.ClientSession(**session_kwargs)
        return self.session

    async def __aenter__(self):
        self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        if self.session is not None:
            await self.session.close()
            self.session = None

    async def _acquire_request_slot(self) -> Optional[asyncio.Semaphore]:
        if self._max_concurrency is None:
            return None

        if self._request_semaphore is None:
            self._request_semaphore = asyncio.Semaphore(self._max_concurrency)

        await self._request_semaphore.acquire()
        return self._request_semaphore

    @asynccontextmanager
    async def _request(
        self,
        url: str,
        headers: Dict[str, str],
        data: Union[str, bytes],
    ):
        limiter = await self._acquire_request_slot()
        session = self._ensure_session()
        post_kwargs: Dict[str, Any] = {
            "url": url,
            "headers": headers,
            "data": data,
        }
        if self._client_timeout is not None:
            post_kwargs["timeout"] = self._client_timeout

        try:
            async with session.post(**post_kwargs) as response:
                yield response
        finally:
            if limiter is not None:
                limiter.release()

    def _should_retry(self, exc: Exception, attempt: int) -> bool:
        if attempt >= self._max_retries:
            return False

        if isinstance(exc, BedrockClientError):
            return exc.status in self._retry_statuses

        if isinstance(exc, (aiohttp.ClientError, asyncio.TimeoutError)):
            return True

        return False

    async def _sleep_backoff(self, attempt: int) -> None:
        if self._retry_backoff <= 0:
            if attempt > 0:
                await asyncio.sleep(0)
            return

        delay = min(
            self._retry_backoff * (2**attempt),
            self._retry_backoff_cap,
        )
        if delay > 0:
            await asyncio.sleep(delay)

    async def invoke_model(self, body: str, modelId: str, **kwargs) -> bytes:
        """Invoke a model and return the response body as bytes"""
        url = f"https://bedrock-runtime.{self.region_name}.amazonaws.com/model/{modelId}/invoke"  # noqa: E501

        attempt = 0
        while True:
            await self._ensure_valid_credentials()
            headers = self._signed_request(
                body=body,
                url=url,
                method="POST",
                credentials=self.credentials,
                region_name=self.region_name,
                **kwargs,
            )

            try:
                async with self._request(url=url, headers=headers, data=body) as res:  # noqa: E501
                    await self._handle_error_response(res)
                    return await res.read()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self._should_retry(exc, attempt):
                    raise
                await self._sleep_backoff(attempt)
                attempt += 1

    async def invoke_sagemaker_endpoint(
        self,
        endpoint_name: str,
        *,
        body: Union[str, bytes],
        content_type: Optional[str] = None,
        accept: Optional[str] = None,
        custom_attributes: Optional[str] = None,
        target_variant: Optional[str] = None,
        target_model: Optional[str] = None,
        target_container_hostname: Optional[str] = None,
        target_channel: Optional[str] = None,
        inference_component: Optional[str] = None,
        inference_id: Optional[str] = None,
        endpoint_config_name: Optional[str] = None,
        invocation_timeout_seconds: Optional[int] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> bytes:
        """Invoke a SageMaker endpoint asynchronously."""

        url = (
            f"https://runtime.sagemaker.{self.region_name}.amazonaws.com"
            f"/endpoints/{endpoint_name}/invocations"
        )

        attempt = 0
        while True:
            await self._ensure_valid_credentials()

            extra_headers: Dict[str, str] = {}

            header_mappings = {
                "X-Amzn-SageMaker-Custom-Attributes": custom_attributes,
                "X-Amzn-SageMaker-Target-Variant": target_variant,
                "X-Amzn-SageMaker-Target-Model": target_model,
                "X-Amzn-SageMaker-Target-Container-Hostname": target_container_hostname,  # noqa: E501
                "X-Amzn-SageMaker-Target-Channel": target_channel,
                "X-Amzn-SageMaker-Inference-Components": inference_component,
                "X-Amzn-SageMaker-Inference-Id": inference_id,
                "X-Amzn-SageMaker-Endpoint-Config-Name": endpoint_config_name,
                "X-Amzn-SageMaker-Invocation-Timeout-Seconds": (
                    str(invocation_timeout_seconds)
                    if invocation_timeout_seconds is not None
                    else None
                ),
            }

            for key, value in header_mappings.items():
                if value is not None:
                    extra_headers[key] = value

            if headers:
                for key, value in headers.items():
                    if value is not None:
                        extra_headers[key] = value

            signed_headers = self._signed_request(
                body=body,
                url=url,
                method="POST",
                credentials=self.credentials,
                region_name=self.region_name,
                service="sagemaker",
                accept=accept,
                contentType=content_type,
                extra_headers=extra_headers if extra_headers else None,
            )

            try:
                async with self._request(
                    url=url,
                    headers=signed_headers,
                    data=body,
                ) as res:
                    await self._handle_error_response(res)
                    return await res.read()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self._should_retry(exc, attempt):
                    raise
                await self._sleep_backoff(attempt)
                attempt += 1

    async def invoke_model_with_response_stream(
        self, body: str, modelId: str, **kwargs
    ) -> AsyncGenerator[Union[Dict[str, Any], bytes], None]:
        """
        Invoke a model with streaming response
        """
        url = f"https://bedrock-runtime.{self.region_name}.amazonaws.com/model/{modelId}/invoke-with-response-stream"  # noqa: E501
        attempt = 0
        emitted = False

        while True:
            await self._ensure_valid_credentials()
            headers = self._signed_request(
                body=body,
                url=url,
                method="POST",
                credentials=self.credentials,
                region_name=self.region_name,
                **kwargs,
            )

            try:
                async with self._request(url=url, headers=headers, data=body) as res:  # noqa: E501
                    await self._handle_error_response(res)

                    event_stream_buffer = EventStreamBuffer()

                    async for chunk in res.content.iter_chunked(8192):
                        event_stream_buffer.add_data(chunk)

                        while True:
                            try:
                                message = event_stream_buffer.next()
                                processed_content = self._process_event_message(  # noqa: E501
                                    message=message,
                                )
                                if processed_content is not None:
                                    emitted = True
                                    yield processed_content
                            except StopIteration:
                                break
                            except Exception as e:
                                log.error(f"Error processing event: {e}")
                                break

                    while True:
                        try:
                            message = event_stream_buffer.next()
                            processed_content = self._process_event_message(
                                message,
                            )
                            if processed_content is not None:
                                emitted = True
                                yield processed_content
                        except StopIteration:
                            return
                        except Exception as e:
                            log.error(f"Error processing buffer: {e}")
                            return
            except asyncio.CancelledError:
                raise
            except BedrockStreamError:
                raise
            except Exception as exc:
                if emitted or not self._should_retry(exc, attempt):
                    raise
                await self._sleep_backoff(attempt)
                attempt += 1

    @overload
    async def invoke_many(
        self,
        requests: Iterable[Mapping[str, Any]],
        *,
        concurrency: Optional[int] = None,
        return_exceptions: Literal[False] = False,
    ) -> Sequence[bytes]: ...

    @overload
    async def invoke_many(
        self,
        requests: Iterable[Mapping[str, Any]],
        *,
        concurrency: Optional[int] = None,
        return_exceptions: Literal[True],
    ) -> Sequence[Union[bytes, BaseException]]: ...

    async def invoke_many(
        self,
        requests: Iterable[Mapping[str, Any]],
        *,
        concurrency: Optional[int] = None,
        return_exceptions: bool = False,
    ) -> Sequence[Union[bytes, BaseException]]:
        """Invoke multiple requests concurrently."""

        request_list = list(requests)
        if not request_list:
            return []

        limiter: Optional[asyncio.Semaphore] = None
        if concurrency and concurrency > 0:
            limiter = asyncio.Semaphore(concurrency)

        async def _invoke(entry: Mapping[str, Any]) -> bytes:
            if "body" not in entry or "modelId" not in entry:
                raise ValueError("Each request must include 'body' and 'modelId' keys")  # noqa: E501

            body_value = entry["body"]
            model_id = entry["modelId"]
            extra = {
                key: value
                for key, value in entry.items()
                if key not in {"body", "modelId"}
            }

            if limiter is not None:
                async with limiter:
                    return await self.invoke_model(
                        body=body_value, modelId=model_id, **extra
                    )

            return await self.invoke_model(
                body=body_value,
                modelId=model_id,
                **extra,
            )

        tasks = [asyncio.create_task(_invoke(item)) for item in request_list]
        try:
            if return_exceptions:
                gathered = await asyncio.gather(*tasks, return_exceptions=True)
                results: Sequence[Union[bytes, BaseException]] = tuple(gathered)  # noqa: E501
            else:
                gathered = await asyncio.gather(*tasks, return_exceptions=False)  # noqa: E501
                results = tuple(gathered)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()

        return results

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

            headers_dict = self._normalize_headers(headers)

            message_type = headers_dict.get(":message-type")
            exception_type = headers_dict.get(":exception-type")
            event_type = headers_dict.get(":event-type")

            if isinstance(payload, memoryview):
                payload = payload.tobytes()
            elif isinstance(payload, bytearray):
                payload = bytes(payload)

            # Get content type from headers
            content_type = "application/json"  # Default
            content_type = headers_dict.get(":content-type", content_type)

            if message_type in {"exception", "error"} or (
                message_type == "event"
                and event_type
                in {
                    "error",
                    "exception",
                    "modelInvocationError",
                }
            ):
                detail = self._decode_payload_text(payload)
                raise BedrockStreamError(
                    message_type=message_type or "event",
                    exception_type=exception_type or event_type or "UnknownException",  # noqa: E501
                    detail=detail,
                )

            if not payload:
                return {}

            # Handle JSON content
            if "application/json" in content_type:
                try:
                    payload_data = orjson.loads(
                        self._ensure_text_payload(
                            payload=payload,
                        )
                    )

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

    def _normalize_headers(self, headers) -> Dict[str, Any]:
        """Convert botocore header objects into plain Python values."""

        if headers is None:
            return {}

        if isinstance(headers, dict):
            items = headers.items()
        elif hasattr(headers, "items"):
            items = headers.items()
        else:
            try:
                items = list(headers)
            except TypeError:
                return {}

        normalized: Dict[str, Any] = {}
        for key, value in items:
            if hasattr(value, "value"):
                normalized[key] = value.value
            else:
                normalized[key] = value

        return normalized

    def _decode_payload_text(self, payload: Union[bytes, str]) -> str:
        if isinstance(payload, bytes):
            return payload.decode("utf-8", errors="replace")
        if isinstance(payload, memoryview):
            return payload.tobytes().decode("utf-8", errors="replace")
        if isinstance(payload, bytearray):
            return bytes(payload).decode("utf-8", errors="replace")
        return str(payload)

    def _ensure_text_payload(self, payload: Union[str, bytes]) -> str:
        if isinstance(payload, str):
            return payload
        if isinstance(payload, bytes):
            return payload.decode("utf-8")
        if isinstance(payload, memoryview):
            return payload.tobytes().decode("utf-8")
        if isinstance(payload, bytearray):
            return bytes(payload).decode("utf-8")
        if hasattr(payload, "read"):
            return self._ensure_text_payload(payload.read())
        return str(payload)

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
            503: "ServiceUnavailableException",
        }

        error_type = error_map.get(response.status, "UnknownException")
        raise BedrockClientError(response.status, error_type, error_text)

    def _signed_request(
        self,
        credentials,
        url: str,
        method: str,
        body: Union[str, bytes],
        region_name: str,
        *,
        service: str = "bedrock",
        extra_headers: Optional[Mapping[str, str]] = None,
        **kwargs,
    ) -> Dict[str, str]:
        """Create a signed AWS request"""
        if credentials is None:
            raise RuntimeError("AWS credentials are not initialized")

        if hasattr(credentials, "get_frozen_credentials"):
            credentials = credentials.get_frozen_credentials()

        request = AWSRequest(method=method, url=url, data=body)
        request.headers.add_header("Host", urlparse(url).netloc)

        accept_header = kwargs.get("accept")
        content_type = kwargs.get("contentType")

        if service == "bedrock":
            # Set appropriate headers based on the endpoint
            if "invoke-with-response-stream" in url or "converse-stream" in url:
                request.headers.add_header(
                    "Accept",
                    "application/vnd.amazon.eventstream",
                )
            else:
                request.headers.add_header(
                    "Accept", accept_header or "application/json"
                )

            request.headers.add_header(
                "Content-Type", content_type or "application/json"
            )
            request.headers.add_header(
                "X-Amzn-Bedrock-Trace", kwargs.get("trace", "DISABLED")
            )

            # Optional headers specific to Bedrock
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
                (
                    "serviceTier",
                    "X-Amzn-Bedrock-ServiceTier",
                ),
            ]

            for kwarg_key, header_name in optional_headers:
                value = kwargs.get(kwarg_key)
                if value:
                    request.headers.add_header(
                        header_name,
                        value,
                    )
        else:
            if accept_header is not None:
                request.headers.add_header("Accept", accept_header)
            if content_type is not None:
                request.headers.add_header("Content-Type", content_type)

        if extra_headers:
            for key, value in extra_headers.items():
                if value is not None:
                    request.headers.add_header(key, value)

        # Sign the request
        SigV4Auth(credentials, service, region_name).add_auth(request)
        return dict(request.headers)

    async def converse(
        self,
        modelId: str,
        messages: Union[Sequence["MessageTypeDef"], Sequence[Mapping[str, Any]]],
        *,
        system: Optional[
            Union[Sequence["SystemContentBlockTypeDef"], Sequence[Mapping[str, Any]]]
        ] = None,
        inferenceConfig: Optional[
            Union["InferenceConfigurationTypeDef", Mapping[str, Any]]
        ] = None,
        toolConfig: Optional[
            Union["ToolConfigurationTypeDef", Mapping[str, Any]]
        ] = None,
        guardrailConfig: Optional[
            Union["GuardrailConfigurationTypeDef", Mapping[str, Any]]
        ] = None,
        additionalModelRequestFields: Optional[Mapping[str, Any]] = None,
        additionalModelResponseFieldPaths: Optional[Sequence[str]] = None,
        promptVariables: Optional[Mapping[str, Any]] = None,
        requestMetadata: Optional[Mapping[str, str]] = None,
        performanceConfig: Optional[
            Union["PerformanceConfigurationTypeDef", Mapping[str, Any]]
        ] = None,
        serviceTier: Optional[
            Union["ServiceTierConfigTypeDef", Mapping[str, Any]]
        ] = None,
    ) -> bytes:
        """
        Invoke a model using the Converse API and return the response as bytes.

        The Converse API provides a consistent interface across all Bedrock models
        that support messages.
        """
        url = f"https://bedrock-runtime.{self.region_name}.amazonaws.com/model/{modelId}/converse"  # noqa: E501

        # Build request body - only include non-None values
        body_dict: Dict[str, Any] = {"messages": list(messages)}

        if system is not None:
            body_dict["system"] = list(system)
        if inferenceConfig is not None:
            body_dict["inferenceConfig"] = inferenceConfig
        if toolConfig is not None:
            body_dict["toolConfig"] = toolConfig
        if guardrailConfig is not None:
            body_dict["guardrailConfig"] = guardrailConfig
        if additionalModelRequestFields is not None:
            body_dict["additionalModelRequestFields"] = additionalModelRequestFields
        if additionalModelResponseFieldPaths is not None:
            body_dict["additionalModelResponseFieldPaths"] = list(
                additionalModelResponseFieldPaths
            )
        if promptVariables is not None:
            body_dict["promptVariables"] = promptVariables
        if requestMetadata is not None:
            body_dict["requestMetadata"] = requestMetadata
        if performanceConfig is not None:
            body_dict["performanceConfig"] = performanceConfig
        if serviceTier is not None:
            body_dict["serviceTier"] = serviceTier

        body = orjson.dumps(body_dict)

        attempt = 0
        while True:
            await self._ensure_valid_credentials()
            headers = self._signed_request(
                body=body,
                url=url,
                method="POST",
                credentials=self.credentials,
                region_name=self.region_name,
            )

            try:
                async with self._request(url=url, headers=headers, data=body) as res:
                    await self._handle_error_response(res)
                    return await res.read()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self._should_retry(exc, attempt):
                    raise
                await self._sleep_backoff(attempt)
                attempt += 1

    def _process_converse_stream_event(
        self,
        message,
    ) -> Optional[Dict[str, Any]]:
        """Process a ConverseStream event message and return typed event dict."""
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

            headers_dict = self._normalize_headers(headers)

            message_type = headers_dict.get(":message-type")
            exception_type = headers_dict.get(":exception-type")
            event_type = headers_dict.get(":event-type")

            if isinstance(payload, memoryview):
                payload = payload.tobytes()
            elif isinstance(payload, bytearray):
                payload = bytes(payload)

            # Handle errors
            if message_type in {"exception", "error"} or (
                message_type == "event"
                and event_type
                in {
                    "error",
                    "exception",
                    "modelStreamErrorException",
                    "internalServerException",
                    "validationException",
                    "throttlingException",
                    "serviceUnavailableException",
                }
            ):
                detail = self._decode_payload_text(payload)
                raise BedrockStreamError(
                    message_type=message_type or "event",
                    exception_type=exception_type or event_type or "UnknownException",
                    detail=detail,
                )

            if not payload:
                return None

            # Parse JSON payload directly for ConverseStream events
            try:
                payload_data = orjson.loads(payload)

                # Return the event with its type as key
                # ConverseStream events: messageStart, contentBlockStart,
                # contentBlockDelta, contentBlockStop, messageStop, metadata
                if event_type:
                    return {event_type: payload_data}
                return payload_data

            except orjson.JSONDecodeError as e:
                log.error(f"Failed to parse JSON payload: {e}")
                return None

        except BedrockStreamError:
            raise
        except Exception as e:
            log.error(f"Error processing converse stream event: {e}")
            return None

    async def converse_stream(
        self,
        modelId: str,
        messages: Union[Sequence["MessageTypeDef"], Sequence[Mapping[str, Any]]],
        *,
        system: Optional[
            Union[Sequence["SystemContentBlockTypeDef"], Sequence[Mapping[str, Any]]]
        ] = None,
        inferenceConfig: Optional[
            Union["InferenceConfigurationTypeDef", Mapping[str, Any]]
        ] = None,
        toolConfig: Optional[
            Union["ToolConfigurationTypeDef", Mapping[str, Any]]
        ] = None,
        guardrailConfig: Optional[
            Union["GuardrailStreamConfigurationTypeDef", Mapping[str, Any]]
        ] = None,
        additionalModelRequestFields: Optional[Mapping[str, Any]] = None,
        additionalModelResponseFieldPaths: Optional[Sequence[str]] = None,
        promptVariables: Optional[Mapping[str, Any]] = None,
        requestMetadata: Optional[Mapping[str, str]] = None,
        performanceConfig: Optional[
            Union["PerformanceConfigurationTypeDef", Mapping[str, Any]]
        ] = None,
        serviceTier: Optional[
            Union["ServiceTierConfigTypeDef", Mapping[str, Any]]
        ] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Invoke a model using the ConverseStream API with streaming response.

        Yields event dictionaries like:
        - {"messageStart": {"role": "assistant"}}
        - {"contentBlockDelta": {"delta": {"text": "..."}, "contentBlockIndex": 0}}
        - {"messageStop": {"stopReason": "end_turn"}}
        - {"metadata": {"usage": {...}, "metrics": {...}}}
        """
        url = f"https://bedrock-runtime.{self.region_name}.amazonaws.com/model/{modelId}/converse-stream"  # noqa: E501

        # Build request body - only include non-None values
        body_dict: Dict[str, Any] = {"messages": list(messages)}

        if system is not None:
            body_dict["system"] = list(system)
        if inferenceConfig is not None:
            body_dict["inferenceConfig"] = inferenceConfig
        if toolConfig is not None:
            body_dict["toolConfig"] = toolConfig
        if guardrailConfig is not None:
            body_dict["guardrailConfig"] = guardrailConfig
        if additionalModelRequestFields is not None:
            body_dict["additionalModelRequestFields"] = additionalModelRequestFields
        if additionalModelResponseFieldPaths is not None:
            body_dict["additionalModelResponseFieldPaths"] = list(
                additionalModelResponseFieldPaths
            )
        if promptVariables is not None:
            body_dict["promptVariables"] = promptVariables
        if requestMetadata is not None:
            body_dict["requestMetadata"] = requestMetadata
        if performanceConfig is not None:
            body_dict["performanceConfig"] = performanceConfig
        if serviceTier is not None:
            body_dict["serviceTier"] = serviceTier

        body = orjson.dumps(body_dict)

        attempt = 0
        emitted = False

        while True:
            await self._ensure_valid_credentials()
            headers = self._signed_request(
                body=body,
                url=url,
                method="POST",
                credentials=self.credentials,
                region_name=self.region_name,
            )

            try:
                async with self._request(url=url, headers=headers, data=body) as res:
                    await self._handle_error_response(res)

                    event_stream_buffer = EventStreamBuffer()

                    async for chunk in res.content.iter_chunked(8192):
                        event_stream_buffer.add_data(chunk)

                        while True:
                            try:
                                message = event_stream_buffer.next()
                                processed_content = self._process_converse_stream_event(
                                    message
                                )
                                if processed_content is not None:
                                    emitted = True
                                    yield processed_content
                            except StopIteration:
                                break
                            except BedrockStreamError:
                                raise
                            except Exception as e:
                                log.error(f"Error processing event: {e}")
                                break

                    # Process remaining buffer
                    while True:
                        try:
                            message = event_stream_buffer.next()
                            processed_content = self._process_converse_stream_event(
                                message
                            )
                            if processed_content is not None:
                                emitted = True
                                yield processed_content
                        except StopIteration:
                            return
                        except BedrockStreamError:
                            raise
                        except Exception as e:
                            log.error(f"Error processing buffer: {e}")
                            return
            except asyncio.CancelledError:
                raise
            except BedrockStreamError:
                raise
            except Exception as exc:
                if emitted or not self._should_retry(exc, attempt):
                    raise
                await self._sleep_backoff(attempt)
                attempt += 1

    @overload
    async def converse_many(
        self,
        requests: Iterable[Mapping[str, Any]],
        *,
        concurrency: Optional[int] = None,
        return_exceptions: Literal[False] = False,
    ) -> Sequence[bytes]: ...

    @overload
    async def converse_many(
        self,
        requests: Iterable[Mapping[str, Any]],
        *,
        concurrency: Optional[int] = None,
        return_exceptions: Literal[True],
    ) -> Sequence[Union[bytes, BaseException]]: ...

    async def converse_many(
        self,
        requests: Iterable[Mapping[str, Any]],
        *,
        concurrency: Optional[int] = None,
        return_exceptions: bool = False,
    ) -> Sequence[Union[bytes, BaseException]]:
        """
        Invoke multiple converse requests concurrently.

        Each request dict must contain 'modelId' and 'messages' keys.
        Optional keys: system, inferenceConfig, toolConfig, guardrailConfig,
        additionalModelRequestFields, additionalModelResponseFieldPaths,
        promptVariables, requestMetadata, performanceConfig, serviceTier.
        """
        request_list = list(requests)
        if not request_list:
            return []

        limiter: Optional[asyncio.Semaphore] = None
        if concurrency and concurrency > 0:
            limiter = asyncio.Semaphore(concurrency)

        async def _converse(entry: Mapping[str, Any]) -> bytes:
            if "modelId" not in entry or "messages" not in entry:
                raise ValueError(
                    "Each request must include 'modelId' and 'messages' keys"
                )

            model_id = entry["modelId"]
            messages = entry["messages"]
            kwargs = {
                k: v for k, v in entry.items() if k not in {"modelId", "messages"}
            }

            if limiter is not None:
                async with limiter:
                    return await self.converse(
                        modelId=model_id, messages=messages, **kwargs
                    )

            return await self.converse(modelId=model_id, messages=messages, **kwargs)

        tasks = [asyncio.create_task(_converse(item)) for item in request_list]
        try:
            if return_exceptions:
                gathered = await asyncio.gather(*tasks, return_exceptions=True)
                results: Sequence[Union[bytes, BaseException]] = tuple(gathered)
            else:
                gathered = await asyncio.gather(*tasks, return_exceptions=False)
                results = tuple(gathered)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()

        return results
