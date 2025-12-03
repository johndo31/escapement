"""
Basic example: Recording an OpenAI agent execution.

This example shows how to:
1. Initialize Escapement to start recording
2. Run OpenAI API calls that get automatically captured
3. Stop recording and get the trace ID
"""

import escapement
from openai import OpenAI


def main():
    # Initialize Escapement - this starts recording all LLM calls
    trace = escapement.init(
        name="basic-example",
        tags=["example", "openai"],
    )
    print(f"Started recording trace: {trace.trace_id}")

    # Create OpenAI client - Escapement automatically intercepts calls
    client = OpenAI()

    # Make some API calls - these are automatically recorded
    print("\nMaking first API call...")
    response1 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2 + 2?"},
        ],
        max_tokens=100,
    )
    print(f"Response: {response1.choices[0].message.content}")

    print("\nMaking second API call...")
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

    # Stop recording
    completed_trace = escapement.stop()
    print(f"\nRecording complete!")
    print(f"Trace ID: {completed_trace.trace_id}")
    print(f"Total steps: {completed_trace.metadata.total_steps}")
    print(f"Total tokens: {completed_trace.metadata.total_tokens}")

    print(f"\nTo replay this trace, run:")
    print(f"  python replay_trace.py {completed_trace.trace_id}")


if __name__ == "__main__":
    main()
