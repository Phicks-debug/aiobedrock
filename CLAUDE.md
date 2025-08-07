# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

aiobedrock is an asynchronous Python client library for AWS Bedrock, providing non-blocking access to Amazon's foundation model service. The library offers both regular and streaming model invocations with full AWS SigV4 authentication.

## Architecture

### Core Components

- **Client** (`aiobedrock/main.py:18`): Main async client class that manages HTTP connections and AWS authentication
- **Authentication**: Uses boto3 sessions with SigV4Auth for secure AWS API calls
- **Streaming Parser**: Custom event stream message parser with botocore integration and manual fallback
- **Error Handling**: Comprehensive HTTP error mapping for Bedrock-specific exceptions

### Key Features

- Fully asynchronous using aiohttp with connection pooling
- Streaming response support via `invoke_model_with_response_stream`
- AWS Bedrock Guardrails integration
- Performance configuration options (latency optimization)
- Custom logging via logsim dependency

## Common Development Commands

### Package Installation
```bash
pip install -e .                    # Install in development mode
pip install -r requirements.txt     # Install dependencies
```

### Running Examples
```bash
cd example/
python invoke_model.py              # Basic model invocation
python invoke_model_with_response_stream.py  # Streaming example
python guardrail.py                 # Guardrail usage example
python nova_model_streaming.py      # Nova model streaming
```

### Package Building and Distribution
```bash
python setup.py sdist bdist_wheel   # Build distribution packages
pip install build && python -m build  # Modern build approach
```

## Key Dependencies

- **aiohttp>=3.11.16**: Async HTTP client for AWS API calls
- **boto3>=1.38.21**: AWS SDK for authentication and request signing
- **orjson>=3.10.16**: Fast JSON parsing for response handling
- **logsim>=0.2.32**: Custom logging library

## AWS Configuration

The client requires AWS credentials configured in your environment:
- AWS credentials file (~/.aws/credentials)
- Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
- IAM roles (when running on AWS infrastructure)

Regional endpoints are automatically constructed based on the `region_name` parameter.

## Response Handling

### Regular Invocation
Returns raw bytes that need to be JSON decoded:
```python
response = await client.invoke_model(body=json.dumps(body), modelId="...")
result = json.loads(response.decode("utf-8"))
```

### Streaming Invocation
Returns async generator yielding either parsed JSON objects or raw bytes:
```python
async for chunk in client.invoke_model_with_response_stream(...):
    # chunk can be Dict[str, Any] or bytes
```

## Error Types

The client maps HTTP status codes to Bedrock-specific exceptions:
- 403: AccessDeniedException
- 408: ModelTimeoutException  
- 424: ModelErrorException
- 429: ThrottlingException
- 500: InternalServerException

## Testing

No formal test suite is present. Examples in `example/` directory serve as integration tests and usage demonstrations.