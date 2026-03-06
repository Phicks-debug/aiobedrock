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

        print(json.dumps(result, indent=2))
        print(f"Action: {result['action']}")


if __name__ == "__main__":
    asyncio.run(main())
