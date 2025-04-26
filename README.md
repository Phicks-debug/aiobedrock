# aiobedrock

[![PyPI version](https://img.shields.io/pypi/v/aiobedrock.svg)](https://pypi.org/project/aiobedrock/)
[![Python versions](https://img.shields.io/pypi/pyversions/aiobedrock.svg)](https://pypi.org/project/aiobedrock/)
[![License](https://img.shields.io/github/license/Phicks-debug/aiobedrock.svg)](https://github.com/Phicks-debug/aiobedrock/blob/main/LICENSE)

An asynchronous Python client for AWS Bedrock, providing non-blocking access to Amazon's foundation model service.

## Features

- **Fully Asynchronous**: Non-blocking API calls using `aiohttp`
- **Low Overhead**: Minimal dependencies with efficient implementation
- **Streaming Support**: Stream responses for real-time AI model interactions
- **Guardrail Integration**: Support for AWS Bedrock Guardrails
- **AWS SigV4 Auth**: Proper AWS authentication for secure API calls
- **Error Handling**: Comprehensive error handling with descriptive exceptions

## Installation

```bash
pip install aiobedrock
```

## Requirements

- Python 3.9 or later
- AWS credentials configured in your environment

## Quick Start

### Basic Model Invocation

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

### Streaming Response

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

### Using Guardrails

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
Client(region_name: str)
```

Creates a new Bedrock client instance.

- **region_name**: AWS region where Bedrock is available (e.g., "us-east-1", "us-west-2", "ap-southeast-1")

### Methods

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
  - **performanceConfigLatency**: Performance configuration for latency. Valid values are "standard" or "optimized".

#### invoke_model_with_response_stream

```python
async invoke_model_with_response_stream(body: str, modelId: str, **kwargs) -> AsyncGenerator[bytes, None]
```

Invokes a Bedrock model and returns an asynchronous generator that yields response chunks.

- Parameters are the same as `invoke_model`

#### close

```python
async close()
```

Closes the aiohttp session.

## Supported Models

aiobedrock supports all models available on AWS Bedrock.

Ensure you have appropriate permissions to access these models in your AWS account.

## Error Handling

The client provides detailed error messages for common Bedrock API errors:

- 403: AccessDeniedException
- 500: InternalServerException
- 424: ModelErrorException
- 408: ModelTimeoutException
- 429: ThrottlingException

For more error details, refer to the [AWS Bedrock API documentation](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html).

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
