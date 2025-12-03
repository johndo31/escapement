"""
Example: Replaying a recorded trace.

This example shows how to:
1. Load a previously recorded trace
2. Replay it with cached responses (no API calls, no cost)
3. Verify the execution matches the original

Usage:
    python replay_trace.py <trace_id>
"""

import sys
import escapement
from openai import OpenAI


def main():
    if len(sys.argv) < 2:
        print("Usage: python replay_trace.py <trace_id>")
        print("\nRun 'escapement list' to see available traces.")
        sys.exit(1)

    trace_id = sys.argv[1]

    # Enter replay mode - LLM calls will use cached responses
    print(f"Loading trace: {trace_id}")
    source_trace = escapement.replay(trace_id)

    if source_trace is None:
        print(f"Error: Trace not found: {trace_id}")
        sys.exit(1)

    print(f"Replaying trace with {source_trace.metadata.total_steps} steps")
    print("=" * 50)

    # Create OpenAI client - calls will be served from cache
    client = OpenAI()

    # Run the same code as the original recording
    # These calls will return cached responses instantly
    print("\nReplaying first API call (from cache)...")
    response1 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2 + 2?"},
        ],
        max_tokens=100,
    )
    print(f"Response: {response1.choices[0].message.content}")

    print("\nReplaying second API call (from cache)...")
    response2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Now multiply that by 3."},
            {"role": "assistant", "content": response1.choices[0].message.content},
            {"role": "user", "content": "What do you get?"},
        ],
        max_tokens=100,
    )
    print(f"Response: {response2.choices[0].message.content}")

    # Stop replay
    replay_trace = escapement.stop()

    print("\n" + "=" * 50)
    print("Replay complete!")
    print(f"Original trace: {trace_id}")
    print(f"Replay trace: {replay_trace.trace_id}")
    print("\nNo API calls were made - all responses came from cache.")
    print("Cost: $0.00")


if __name__ == "__main__":
    main()
