"""
Example: Loop detection preventing runaway agents.

This example shows how Escapement's loop detection feature
automatically kills agents that get stuck in infinite loops,
preventing expensive API bills.
"""

import escapement
from escapement.config import EscapementConfig
from escapement.interceptor import LoopDetectedError
from openai import OpenAI


def main():
    # Configure with aggressive loop detection (for demo purposes)
    config = EscapementConfig(
        loop_detection_enabled=True,
        max_identical_calls=3,  # Kill after 3 identical calls
    )

    trace = escapement.init(
        name="loop-detection-demo",
        config=config,
    )
    print(f"Started trace: {trace.trace_id}")
    print(f"Loop detection: enabled (max {config.max_identical_calls} identical calls)")
    print("=" * 50)

    client = OpenAI()

    # Simulate a buggy agent that keeps making the same request
    # In real life, this might be a ReAct loop that can't find an answer
    try:
        for i in range(10):
            print(f"\nAttempt {i + 1}: Making API call...")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": "Search for: how to fix this bug"}
                ],
                max_tokens=50,
            )
            print(f"  Response: {response.choices[0].message.content[:50]}...")

    except LoopDetectedError as e:
        print("\n" + "=" * 50)
        print("LOOP DETECTED!")
        print(f"  Trace ID: {e.trace_id}")
        print(f"  Request hash: {e.repeated_hash}")
        print(f"\nThe agent was killed before burning through your API budget.")
        print(f"Trace saved for debugging: {e.trace_id}")
        print("\nRun: escapement show", e.trace_id)


if __name__ == "__main__":
    main()
