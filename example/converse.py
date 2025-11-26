import asyncio
import json

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

        # Extract the assistant's response text
        if "output" in result and "message" in result["output"]:
            content = result["output"]["message"]["content"]
            for block in content:
                if "text" in block:
                    print(f"\nAssistant: {block['text']}")


if __name__ == "__main__":
    asyncio.run(main())
