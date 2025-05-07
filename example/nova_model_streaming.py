import json
import asyncio

from aiobedrock import Client


async def main():
    async with Client(region_name="ap-southeast-1") as client:
        body = {
            "schemaVersion": "messages-v1",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": "Kể cho tôi câu truyện về Bob, dài"},
                    ],
                },
            ],
            "system": [{"text": "Your name is BOB"}],
            "inferenceConfig": {
                "maxTokens": 2048,
                "topP": 1,
                "topK": 1,
                "temperature": 1,
            },
        }

        async for chunk in client.invoke_model_with_response_stream(
            body=json.dumps(body),
            modelId="apac.amazon.nova-micro-v1:0",
            accept="application/json",
            contentType="application/json",
            trace="ENABLED_FULL",
        ):
            c = json.loads(chunk)
            if c.get("contentBlockDelta"):
                print(
                    c["contentBlockDelta"]["delta"]["text"],
                    flush=True,
                    end="",
                )


if __name__ == "__main__":
    asyncio.run(main())
