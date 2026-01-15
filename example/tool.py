import asyncio
import json

import aiobedrock


async def main():
    async with aiobedrock.Client(
        region_name="us-east-1",
        profile_name="130506138320_dev.policy.custom",
    ) as client:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "anthropic_beta": [
                "context-management-2025-06-27",
            ],
            "max_tokens": 16000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is the newest model release from google?",  # noqa: E501
                        },
                    ],
                }
            ],
            "tools": [
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
            ],
        }

        async for chunk in client.invoke_model_with_response_stream(
            body=json.dumps(body),
            modelId="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
            accept="application/json",
            contentType="application/json",
        ):
            print(chunk)


if __name__ == "__main__":
    asyncio.run(main())
