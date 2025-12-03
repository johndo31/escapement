"""
Example: Time-travel debugging with fork.

This example shows how to:
1. Replay a trace up to a certain point
2. Fork to live mode to try a different approach
3. Debug without re-running expensive setup steps
"""

import sys
import escapement
from openai import OpenAI


def main():
    if len(sys.argv) < 2:
        print("Usage: python time_travel.py <trace_id>")
        print("\nFirst run basic_recording.py to create a trace.")
        sys.exit(1)

    trace_id = sys.argv[1]

    # Start in replay mode
    print(f"Loading trace for time-travel debugging: {trace_id}")
    source_trace = escapement.replay(trace_id, strict=False)

    if source_trace is None:
        print(f"Error: Trace not found: {trace_id}")
        sys.exit(1)

    print(f"Source trace has {source_trace.metadata.total_steps} steps")
    print("=" * 50)

    client = OpenAI()

    # Replay the first call (from cache)
    print("\n[REPLAY] First API call (from cache, $0 cost)...")
    response1 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2 + 2?"},
        ],
        max_tokens=100,
    )
    print(f"  Response: {response1.choices[0].message.content}")

    # Now FORK to live mode - we want to try a different second prompt
    print("\n" + "=" * 50)
    print("FORKING TO LIVE MODE")
    print("From here, calls go to the real API")
    print("=" * 50)
    escapement.fork()

    # This call will go to the real API because we forked
    # and the prompt is different from the recorded one
    print("\n[LIVE] Trying a different follow-up question...")
    response2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": response1.choices[0].message.content},
            {"role": "user", "content": "What if we square that result instead?"},
        ],
        max_tokens=100,
    )
    print(f"  Response: {response2.choices[0].message.content}")

    # Stop and see results
    final_trace = escapement.stop()

    print("\n" + "=" * 50)
    print("Time-travel debugging complete!")
    print(f"New trace created: {final_trace.trace_id}")
    print(f"Fork point: step {final_trace.fork_point}")
    print("\nThis technique lets you:")
    print("  1. Skip expensive setup steps (replay from cache)")
    print("  2. Try different approaches at the failure point (live)")
    print("  3. Iterate quickly without burning API credits")


if __name__ == "__main__":
    main()
