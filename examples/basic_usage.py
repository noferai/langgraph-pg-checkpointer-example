#!/usr/bin/env python3
"""Basic usage example demonstrating state persistence."""

from langchain_core.messages import HumanMessage

from src.agent import create_agent
from src.checkpointer import get_checkpointer


def print_separator():
    print("\n" + "=" * 70 + "\n")


def main():
    print("🤖 LangGraph Agent - Basic Usage Example")
    print_separator()

    print("📦 Setting up agent with PostgreSQL checkpointer...")
    with get_checkpointer(setup=True) as checkpointer:
        agent = create_agent(checkpointer)
        print("✓ Agent ready!")
        print_separator()

        config = {"configurable": {"thread_id": "example-conversation-1"}}

        print("👤 User: What is 15 multiplied by 7?")
        print("\n🤖 Assistant:")

        result = agent.invoke(
            {
                "messages": [HumanMessage(content="What is 15 multiplied by 7?")],
                "current_step": 0,
            },
            config=config,
        )

        last_message = result["messages"][-1]
        print(last_message.content)
        print(f"\n📊 Steps taken: {result['current_step']}")
        print_separator()

        print("👤 User: Now divide that result by 3")
        print("\n🤖 Assistant:")

        result = agent.invoke(
            {
                "messages": [HumanMessage(content="Now divide that result by 3")],
                "current_step": result["current_step"],
            },
            config=config,
        )

        last_message = result["messages"][-1]
        print(last_message.content)
        print(f"\n📊 Steps taken: {result['current_step']}")
        print_separator()

        print("👤 User: What was the first calculation we did?")
        print("\n🤖 Assistant:")

        result = agent.invoke(
            {
                "messages": [HumanMessage(content="What was the first calculation we did?")],
                "current_step": result["current_step"],
            },
            config=config,
        )

        last_message = result["messages"][-1]
        print(last_message.content)
        print(f"\n📊 Steps taken: {result['current_step']}")
        print(f"💾 Total messages in conversation: {len(result['messages'])}")
        print_separator()

        print("✨ Example complete!")
        print("\nNote: Conversation saved in PostgreSQL.")
        print("Restart this script to continue the same conversation.")


if __name__ == "__main__":
    main()
