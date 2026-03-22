#!/usr/bin/env python3
"""Memory demonstration with thread isolation."""

from langchain_core.messages import HumanMessage

from src.agent import create_agent
from src.checkpointer import get_checkpointer


def print_separator(char="="):
    print("\n" + char * 70 + "\n")


def run_conversation(agent, thread_id: str, messages: list[str]):
    config = {"configurable": {"thread_id": thread_id}}
    current_step = 0

    print(f"🧵 Thread: {thread_id}")
    print_separator("-")

    for user_msg in messages:
        print(f"👤 User: {user_msg}")
        print("🤖 Assistant: ", end="")

        result = agent.invoke(
            {
                "messages": [HumanMessage(content=user_msg)],
                "current_step": current_step,
            },
            config=config,
        )

        last_message = result["messages"][-1]
        print(last_message.content)
        current_step = result["current_step"]
        print()

    print(f"💾 Total messages in this thread: {len(result['messages'])}")
    print(f"📊 Total steps: {current_step}")


def main():
    print("🧠 LangGraph Agent - Memory Demonstration")
    print("Showcasing thread isolation and conversation persistence")
    print_separator()

    with get_checkpointer(setup=True) as checkpointer:
        agent = create_agent(checkpointer)

        print("👩 Starting conversation with Alice (thread: alice-chat)")
        print_separator()

        run_conversation(
            agent,
            thread_id="alice-chat",
            messages=[
                "Hi! My name is Alice. What's 10 + 5?",
                "Great! Now multiply that by 2",
            ],
        )

        print_separator()

        print("👨 Starting conversation with Bob (thread: bob-chat)")
        print_separator()

        run_conversation(
            agent,
            thread_id="bob-chat",
            messages=[
                "Hello! I'm Bob. Can you tell me what time it is?",
                "Thanks! What's 100 divided by 4?",
            ],
        )

        print_separator()

        print("👩 Returning to Alice's conversation...")
        print_separator()

        run_conversation(
            agent,
            thread_id="alice-chat",
            messages=[
                "Do you remember my name?",
                "What was the result of our last calculation?",
            ],
        )

        print_separator()

        print("👨 Returning to Bob's conversation...")
        print_separator()

        run_conversation(
            agent,
            thread_id="bob-chat",
            messages=[
                "Do you remember what I asked you to calculate?",
            ],
        )

        print_separator()

        print("✨ Memory Demonstration Complete!")
        print()
        print("Key observations:")
        print("  1. Each thread maintains separate memory")
        print("  2. Agent remembers context within each thread")
        print("  3. Conversations persist in PostgreSQL")
        print("  4. Thread isolation prevents information leakage")
        print()
        print("💡 Run this script again - conversations remain in database!")
        print_separator()


if __name__ == "__main__":
    main()
