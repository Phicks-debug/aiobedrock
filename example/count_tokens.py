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
