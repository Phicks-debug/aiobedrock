import asyncio
import json

from aiobedrock import Client


async def main():
    async with Client(
        region_name="us-east-1",
        # assume_role_arn="arn:aws:iam::130506138320:role/bedrock-cross-account-role",  # noqa: E501
    ) as client:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Bạn có thể làm được những gì?",
                        },
                    ],
                }
            ],
        }

        async for chunk in client.invoke_model_with_response_stream(
            body=json.dumps(body),
            modelId="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            accept="application/json",
            contentType="application/json",
            trace="ENABLED_FULL",
            serviceTier="flex",
        ):
            if isinstance(chunk, bytes):
                print(chunk.decode("utf-8"))


if __name__ == "__main__":
    asyncio.run(main())
