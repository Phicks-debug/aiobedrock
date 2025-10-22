"""Example demonstrating the invoke_many helper for parallel Bedrock calls."""

import asyncio
import json
from typing import Any, Dict

from aiobedrock import Client


async def main() -> None:
    # Prepare three lightweight chat prompts to run in parallel.
    requests: list[Dict[str, Any]] = []
    for topic in ("capabilities", "latency", "pricing", "models", "updated"):
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Give short overview Claude's {topic}.",
                        }
                    ],
                }
            ],
        }

        requests.append(
            {
                "body": json.dumps(body),
                "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
                "accept": "application/json",
                "contentType": "application/json",
            }
        )

    async with Client(
        region_name="ap-southeast-1",
        max_concurrency=5,
    ) as client:
        responses = await client.invoke_many(requests)

    # Results come back in the same order as the requests list.
    for idx, response in enumerate(responses, start=1):
        print(f"=== Response {idx} ===")
        print(json.loads(response.decode("utf-8")))


if __name__ == "__main__":
    asyncio.run(main())
