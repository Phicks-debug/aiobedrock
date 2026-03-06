# aiobedrock

[![PyPI version](https://img.shields.io/pypi/v/aiobedrock.svg)](https://pypi.org/project/aiobedrock/)
[![Python versions](https://img.shields.io/pypi/pyversions/aiobedrock.svg)](https://pypi.org/project/aiobedrock/)
[![License](https://img.shields.io/github/license/Phicks-debug/aiobedrock.svg)](https://github.com/Phicks-debug/aiobedrock/blob/main/LICENSE)

An asynchronous Python client for AWS Bedrock, providing non-blocking access to Amazon's foundation model service.

## Features

- **Fully Asynchronous**: Non-blocking API calls using `aiohttp`
- **Low Overhead**: Minimal dependencies with efficient implementation
- **Converse API**: Unified API for all Bedrock models with structured messages
- **Streaming Support**: Stream responses for real-time AI model interactions
- **Structured Output**: JSON schema output support via `outputConfig`
- **Guardrail Integration**: Support for AWS Bedrock Guardrails, including standalone `apply_guardrail`
- **Token Counting**: Count tokens before sending requests with `count_tokens`
- **Async Invocations**: Background model invocations with S3 output via `start_async_invoke`
- **Service Tier Support**: Configure processing tiers (priority, default, flex, reserved)
- **AWS SigV4 Auth**: Proper AWS authentication for secure API calls
- **Batch Processing**: Concurrent batch invocations with `invoke_many` and `converse_many`
- **Error Handling**: Comprehensive error handling with `BedrockClientError` and `BedrockStreamError`
- **Type Hints**: Optional type checking support with `mypy-boto3-bedrock-runtime`

## Installation

```bash
pip install aiobedrock
```

For type checking support (optional):

```bash
pip install aiobedrock[types]
```

## Requirements

- Python 3.9 or later (tested through Python 3.14)
- AWS credentials configured in your environment
- boto3 1.42.29 or newer (installed automatically via dependencies)

## Quick Start

### Converse API (Recommended)

The Converse API provides a unified interface for all Bedrock models:

```python
import json
import asyncio
from aiobedrock import Client

async def main():
    async with Client(region_name="us-west-2") as client:
        messages = [
            {
                "role": "user",
                "content": [{"text": "What is the capital of France?"}],
            }
        ]

        response = await client.converse(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            messages=messages,
            inferenceConfig={
                "maxTokens": 1024,
                "temperature": 0.7,
            },
        )

        result = json.loads(response.decode("utf-8"))
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
```

### Converse Streaming

```python
import asyncio
from aiobedrock import Client

async def main():
    async with Client(region_name="us-west-2") as client:
        messages = [
            {
                "role": "user",
                "content": [{"text": "Tell me a short story about a robot."}],
            }
        ]

        print("Assistant: ", end="", flush=True)

        async for event in client.converse_stream(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            messages=messages,
            inferenceConfig={
                "maxTokens": 1024,
                "temperature": 0.7,
            },
        ):
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                if "text" in delta:
                    print(delta["text"], end="", flush=True)
            elif "messageStop" in event:
                print(f"\n[Stop reason: {event['messageStop'].get('stopReason')}]")
            elif "metadata" in event:
                usage = event["metadata"].get("usage", {})
                print(f"[Tokens - Input: {usage.get('inputTokens')}, Output: {usage.get('outputTokens')}]")

if __name__ == "__main__":
    asyncio.run(main())
```

### Structured JSON Output

Use `outputConfig` to enforce JSON schema output:

```python
import json
import asyncio
from aiobedrock import Client

async def main():
    async with Client(region_name="us-west-2") as client:
        response = await client.converse(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            messages=[
                {
                    "role": "user",
                    "content": [{"text": "List 3 capitals in Europe."}],
                }
            ],
            outputConfig={
                "textFormat": {
                    "type": "json_schema",
                    "structure": {
                        "jsonSchema": {
                            "type": "object",
                            "properties": {
                                "capitals": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                }
                            },
                        }
                    },
                }
            },
        )
        print(json.loads(response.decode("utf-8")))

if __name__ == "__main__":
    asyncio.run(main())
```

### Token Counting

Count tokens before sending a request:

```python
import asyncio
from aiobedrock import Client

async def main():
    async with Client(region_name="us-west-2") as client:
        result = await client.count_tokens(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            messages=[
                {
                    "role": "user",
                    "content": [{"text": "What is the capital of France?"}],
                }
            ],
            system=[{"text": "Be concise."}],
        )
        print(f"Input tokens: {result['inputTokens']}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Apply Guardrail

Evaluate content against a guardrail without invoking a model:

```python
import asyncio
import json
from aiobedrock import Client

async def main():
    async with Client(region_name="us-west-2") as client:
        result = await client.apply_guardrail(
            guardrailIdentifier="YOUR_GUARDRAIL_ID",
            guardrailVersion="1",
            source="INPUT",
            content=[{"text": {"text": "Is this content safe?"}}],
        )
        print(f"Action: {result['action']}")
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
```

### Async Invocation

Start a background model invocation with output stored in S3:

```python
import asyncio
import json
import time
from aiobedrock import Client

async def main():
    async with Client(region_name="us-west-2") as client:
        result = await client.start_async_invoke(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            modelInput={
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Write a haiku."}],
                    }
                ],
            },
            outputDataConfig={
                "s3OutputDataConfig": {"s3Uri": "s3://YOUR_BUCKET/output/"}
            },
        )
        arn = result["invocationArn"]
        print(f"Started: {arn}")

        # Poll for completion
        while True:
            status = await client.get_async_invoke(invocationArn=arn)
            if status.get("status") in ("Completed", "Failed"):
                print(json.dumps(status, indent=2, default=str))
                break
            time.sleep(5)

        # List recent invocations
        invocations = await client.list_async_invokes(maxResults=5)
        print(json.dumps(invocations, indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(main())
```

### Basic Model Invocation (More control)

For direct model invocation with model-specific request formats:

```python
import json
import asyncio
from aiobedrock import Client

async def main():
    async with Client(region_name="YOUR_AWS_REGION") as client:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "temperature": 0.7,
            "top_p": 0.9,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What can you do?"},
                    ],
                }
            ],
        }

        response = await client.invoke_model(
            body=json.dumps(body),
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            accept="application/json",
            contentType="application/json",
        )

        print(json.loads(response.decode("utf-8")))

if __name__ == "__main__":
    asyncio.run(main())
```

### Streaming Response (More control)

```python
import json
import asyncio
from aiobedrock import Client

async def main():
    async with Client(region_name="YOUR_AWS_REGION") as client:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "temperature": 0.7,
            "top_p": 0.9,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What can you do?"},
                    ],
                }
            ],
        }

        async for chunk in client.invoke_model_with_response_stream(
            body=json.dumps(body),
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            accept="application/json",
            contentType="application/json",
        ):
            print(json.loads(chunk.decode("utf-8")))

if __name__ == "__main__":
    asyncio.run(main())
```

### Using Guardrails with Invoke

```python
import json
import asyncio
from aiobedrock import Client

async def main():
    async with Client(region_name="YOUR_AWS_REGION") as client:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "temperature": 0.7,
            "top_p": 0.9,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What can you do?"},
                    ],
                }
            ],
        }

        response = await client.invoke_model(
            body=json.dumps(body),
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            accept="application/json",
            contentType="application/json",
            guardrailIdentifier="arn:aws:bedrock:YOUR_REGION:YOUR_ACCOUNT_ID:guardrail/YOUR_GUARDRAIL_ID",
            guardrailVersion="LATEST",
        )

        print(json.loads(response.decode("utf-8")))

if __name__ == "__main__":
    asyncio.run(main())
```

## API Reference

### Client

```python
Client(
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
)
```

Creates a new Bedrock client instance. The underlying `aiohttp` session and
TCP connector are created lazily on first use, so the client can safely be
constructed outside of an async context.

- **region_name**: AWS region where Bedrock is available (e.g., "us-east-1", "us-west-2")
- **assume_role_arn**: Optional ARN of an IAM role to assume for cross-account access
- **profile_name**: Optional AWS profile name from `~/.aws/credentials`
- **aws_account_id**: Optional AWS account ID
- **max_connections**: Maximum number of connections in the pool (default: 10000)
- **request_timeout**: Optional request timeout in seconds
- **max_concurrency**: Optional maximum concurrent requests
- **max_retries**: Maximum number of retry attempts (default: 2)
- **retry_backoff**: Initial backoff delay in seconds (default: 0.5)
- **max_backoff**: Maximum backoff delay in seconds (default: 6.0)
- **retry_statuses**: HTTP status codes to retry (default: 408, 424, 429, 500, 502, 503, 504)

### Methods

#### converse

```python
async converse(
    modelId: str,
    messages: Sequence[MessageTypeDef],
    *,
    system: Optional[Sequence[SystemContentBlockTypeDef]] = None,
    inferenceConfig: Optional[InferenceConfigurationTypeDef] = None,
    toolConfig: Optional[ToolConfigurationTypeDef] = None,
    guardrailConfig: Optional[GuardrailConfigurationTypeDef] = None,
    additionalModelRequestFields: Optional[Mapping[str, Any]] = None,
    additionalModelResponseFieldPaths: Optional[Sequence[str]] = None,
    promptVariables: Optional[Mapping[str, Any]] = None,
    requestMetadata: Optional[Mapping[str, str]] = None,
    performanceConfig: Optional[PerformanceConfigurationTypeDef] = None,
    serviceTier: Optional[ServiceTierConfigTypeDef] = None,
    outputConfig: Optional[Mapping[str, Any]] = None,
) -> bytes
```

Invokes a Bedrock model using the Converse API and returns the complete response as bytes.

- **modelId**: Bedrock model identifier
- **messages**: List of conversation messages with `role` and `content`
- **system**: Optional system prompts
- **inferenceConfig**: Optional inference parameters (`maxTokens`, `temperature`, `topP`, `stopSequences`)
- **toolConfig**: Optional tool configuration for function calling
- **guardrailConfig**: Optional guardrail configuration
- **additionalModelRequestFields**: Optional model-specific parameters
- **performanceConfig**: Optional performance configuration (`latency`: "standard" or "optimized")
- **serviceTier**: Optional service tier (`type`: "priority", "default", "flex", or "reserved")
- **outputConfig**: Optional structured output config (e.g., `{"textFormat": {"type": "json_schema", "structure": {"jsonSchema": {...}}}}`)

#### converse_stream

```python
async converse_stream(
    modelId: str,
    messages: Sequence[MessageTypeDef],
    *,
    # Same optional parameters as converse()
) -> AsyncGenerator[Dict[str, Any], None]
```

Invokes a Bedrock model using the ConverseStream API and yields streaming events.

Event types:
- `{"messageStart": {"role": "assistant"}}` - Message started
- `{"contentBlockStart": {...}}` - Content block started
- `{"contentBlockDelta": {"delta": {"text": "..."}, "contentBlockIndex": 0}}` - Text delta
- `{"contentBlockStop": {"contentBlockIndex": 0}}` - Content block completed
- `{"messageStop": {"stopReason": "end_turn"}}` - Message completed
- `{"metadata": {"usage": {...}, "metrics": {...}}}` - Usage metadata

#### converse_many

```python
async converse_many(
    requests: Iterable[Mapping[str, Any]],
    *,
    concurrency: Optional[int] = None,
    return_exceptions: bool = False,
) -> Sequence[Union[bytes, BaseException]]
```

Runs multiple converse invocations concurrently while preserving the order of results.
Each entry in `requests` must include `modelId` and `messages`; any additional key/value
pairs are forwarded to `converse`.

#### count_tokens

```python
async count_tokens(
    modelId: str,
    *,
    messages: Optional[Sequence[MessageTypeDef]] = None,
    system: Optional[Sequence[SystemContentBlockTypeDef]] = None,
    invokeModelBody: Optional[Union[str, bytes]] = None,
) -> Dict[str, Any]
```

Counts tokens for a given input without invoking the model.

Provide either `messages` (+ optional `system`) for a Converse-style count,
or `invokeModelBody` for an InvokeModel-style count. Returns a dict with `inputTokens`.

#### apply_guardrail

```python
async apply_guardrail(
    guardrailIdentifier: str,
    guardrailVersion: str,
    source: str,
    content: Sequence[Mapping[str, Any]],
    *,
    outputScope: Optional[str] = None,
) -> Dict[str, Any]
```

Evaluates content against a guardrail without invoking a model. Returns the
guardrail assessment including `action` ("NONE" or "GUARDRAIL_INTERVENED"),
`outputs`, and `assessments`.

- **source**: "INPUT" or "OUTPUT"
- **content**: List of content blocks to evaluate

#### start_async_invoke

```python
async start_async_invoke(
    modelId: str,
    modelInput: Mapping[str, Any],
    outputDataConfig: Mapping[str, Any],
    *,
    clientRequestToken: Optional[str] = None,
    tags: Optional[Sequence[Mapping[str, str]]] = None,
) -> Dict[str, Any]
```

Starts an asynchronous (background) model invocation. The result is written
to S3 when complete. Returns a dict containing `invocationArn`.

#### get_async_invoke

```python
async get_async_invoke(invocationArn: str) -> Dict[str, Any]
```

Gets the status and details of an asynchronous invocation.

#### list_async_invokes

```python
async list_async_invokes(
    *,
    submitTimeAfter: Optional[str] = None,
    submitTimeBefore: Optional[str] = None,
    statusEquals: Optional[str] = None,
    maxResults: Optional[int] = None,
    nextToken: Optional[str] = None,
    sortBy: Optional[str] = None,
    sortOrder: Optional[str] = None,
) -> Dict[str, Any]
```

Lists asynchronous invocations with optional filters.

#### invoke_model

```python
async invoke_model(body: str, modelId: str, **kwargs) -> bytes
```

Invokes a Bedrock model and returns the complete response.

- **body**: JSON string with model parameters and prompt
- **modelId**: Bedrock model identifier
- **kwargs**: Optional parameters
  - **accept**: Accept header (default: "application/json")
  - **contentType**: Content-Type header (default: "application/json")
  - **trace**: Tracing level: "ENABLED", "ENABLED_FULL" or "DISABLED" (default: "DISABLED")
  - **guardrailIdentifier**: ARN of the guardrail to use
  - **guardrailVersion**: Version of the guardrail (e.g., "1" or "LATEST")
  - **performanceConfigLatency**: "standard" or "optimized"
  - **serviceTier**: "priority", "default", "flex", or "reserved"

#### invoke_model_with_response_stream

```python
async invoke_model_with_response_stream(body: str, modelId: str, **kwargs) -> AsyncGenerator[Union[Dict[str, Any], bytes], None]
```

Invokes a Bedrock model and returns an asynchronous generator. The generator
yields either parsed JSON objects or raw byte chunks depending on the payload.
Parameters are the same as `invoke_model`.

#### invoke_many

```python
async invoke_many(requests: Iterable[Mapping[str, Any]], *, concurrency: Optional[int] = None, return_exceptions: bool = False) -> Sequence[Union[bytes, Exception]]
```

Runs multiple invocations concurrently while preserving the order of results.
Each entry in `requests` must include `body` (JSON string) and `modelId`; any
additional key/value pairs are forwarded to `invoke_model`.

See `example/invoke_many.py` for a complete usage example.

#### invoke_sagemaker_endpoint

```python
async invoke_sagemaker_endpoint(
    endpoint_name: str,
    *,
    body: Union[str, bytes],
    content_type: Optional[str] = None,
    accept: Optional[str] = None,
    # ... additional SageMaker-specific headers
) -> bytes
```

Invokes a SageMaker endpoint asynchronously.

#### close

```python
async close()
```

Closes the aiohttp session.

## Supported Models

aiobedrock supports all models available on AWS Bedrock and AWS SageMaker.
Ensure you have appropriate permissions to access these models in your AWS account.

## Error Handling

The client raises two exception types, both importable from `aiobedrock`:

```python
from aiobedrock import BedrockClientError, BedrockStreamError
```

**`BedrockClientError`** is raised on non-200 HTTP responses:

| Status | Error Type |
|--------|-----------|
| 403 | AccessDeniedException |
| 408 | ModelTimeoutException |
| 424 | ModelErrorException |
| 429 | ThrottlingException |
| 500 | InternalServerException |
| 503 | ServiceUnavailableException |

**`BedrockStreamError`** is raised when a streaming response contains an error event
from Bedrock (e.g., `ModelStreamError`). The exception includes the error type
and payload returned by the service.

For more details, refer to the [AWS Bedrock API documentation](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html).

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
