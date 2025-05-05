import json
import asyncio

from aiobedrock import Client


async def main():
    async with Client(region_name="ap-southeast-1") as client:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "top_k": 40,
            "temperature": 0.7,
            "top_p": 0.9,
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
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            accept="application/json",
            contentType="application/json",
            trace="ENABLED_FULL",
        ):
            print(chunk)


if __name__ == "__main__":
    asyncio.run(main())
