import asyncio
import json
import time

from aiobedrock import Client


async def main():
    async with Client(region_name="us-west-2") as client:
        # Start an async invocation
        result = await client.start_async_invoke(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            modelInput={
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Write a haiku about clouds."}],
                    }
                ],
            },
            outputDataConfig={
                "s3OutputDataConfig": {"s3Uri": "s3://YOUR_BUCKET/output/"}
            },
        )

        arn = result["invocationArn"]
        print(f"Started async invoke: {arn}")

        # Poll for completion
        while True:
            status = await client.get_async_invoke(invocationArn=arn)
            state = status.get("status", "Unknown")
            print(f"Status: {state}")
            if state in ("Completed", "Failed"):
                print(json.dumps(status, indent=2, default=str))
                break
            time.sleep(5)

        # List recent invocations
        invocations = await client.list_async_invokes(maxResults=5)
        for inv in invocations.get("asyncInvokeSummaries", []):
            print(f"  {inv['invocationArn']} -> {inv['status']}")


if __name__ == "__main__":
    asyncio.run(main())
