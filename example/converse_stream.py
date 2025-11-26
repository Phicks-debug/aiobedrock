import asyncio

from aiobedrock import Client


async def main():
    async with Client(region_name="us-west-2") as client:
        messages = [
            {
                "role": "user",
                "content": [{"text": "Tell me a short story about a robot."}],
            }
        ]

        print("Assistant: ", end="", flush=True)

        async for event in client.converse_stream(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            messages=messages,
            inferenceConfig={
                "maxTokens": 1024,
                "temperature": 0.7,
            },
        ):
            # Handle different event types
            if "messageStart" in event:
                # Message started, role is available
                pass
            elif "contentBlockStart" in event:
                # Content block started (e.g., for tool use)
                pass
            elif "contentBlockDelta" in event:
                # Content delta - print streaming text
                delta = event["contentBlockDelta"].get("delta", {})
                if "text" in delta:
                    print(delta["text"], end="", flush=True)
            elif "contentBlockStop" in event:
                # Content block completed
                pass
            elif "messageStop" in event:
                # Message completed
                stop_reason = event["messageStop"].get("stopReason")
                print(f"\n\n[Stop reason: {stop_reason}]")
            elif "metadata" in event:
                # Metadata with usage info
                metadata = event["metadata"]
                usage = metadata.get("usage", {})
                print(
                    f"[Tokens - Input: {usage.get('inputTokens')}, Output: {usage.get('outputTokens')}]"
                )


if __name__ == "__main__":
    asyncio.run(main())
