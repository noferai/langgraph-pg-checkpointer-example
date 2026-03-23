#!/usr/bin/env python3
"""Async concurrent conversations demonstration.

Shows the key advantage of async: multiple conversations run in parallel,
not sequentially. Compare with memory_demo.py where threads run one after another.

Key differences from sync examples:
- AsyncPostgresSaver instead of PostgresSaver
- async with / await instead of with / sync call
- asyncio.gather() to run conversations concurrently
"""

import asyncio
import time

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.agent import create_agent
from src.checkpointer import get_database_url


def print_separator(char="="):
    print("\n" + char * 70 + "\n")


async def run_conversation(agent, thread_id: str, messages: list[str]) -> tuple[str, int]:
    """Run a single conversation asynchronously, return thread_id and step count."""
    config = {"configurable": {"thread_id": thread_id}}
    current_step = 0
    result = None

    print(f"🧵 [{thread_id}] Starting")

    for user_msg in messages:
        print(f"👤 [{thread_id}] {user_msg}")

        result = await agent.ainvoke(
            {
                "messages": [HumanMessage(content=user_msg)],
                "current_step": current_step,
            },
            config=config,
        )

        last_message = result["messages"][-1]
        print(f"🤖 [{thread_id}] {last_message.content}")
        current_step = result["current_step"]

    print(f"✅ [{thread_id}] Done — {len(result['messages'])} messages saved to PostgreSQL")
    return thread_id, current_step


async def main():
    print("⚡ LangGraph Agent - Async Concurrent Conversations")
    print("Three users interact with the agent simultaneously via asyncio.gather()")
    print_separator()

    db_url = get_database_url()

    # AsyncPostgresSaver works with the same connection string as PostgresSaver.
    # psycopg[binary] already supports async — no extra dependencies needed.
    async with AsyncPostgresSaver.from_conn_string(db_url) as checkpointer:
        await checkpointer.setup()
        agent = create_agent(checkpointer)

        conversations = [
            (
                "async-carol",
                [
                    "Hi! I'm Carol. What's 256 divided by 4?",
                    "Multiply that result by 3",
                ],
            ),
            (
                "async-dave",
                [
                    "Hello, I'm Dave. What time is it right now?",
                    "Thanks! What's 17 multiplied by 6?",
                ],
            ),
            (
                "async-eve",
                [
                    "Hey, I'm Eve. What's 100 + 200 + 300?",
                    "Divide that by 6",
                ],
            ),
        ]

        print("🚀 Launching all three conversations concurrently...\n")
        start = time.perf_counter()

        # All conversations run IN PARALLEL — LLM calls overlap in time.
        # In memory_demo.py the same three threads run sequentially.
        results = await asyncio.gather(*[run_conversation(agent, thread_id, msgs) for thread_id, msgs in conversations])

        elapsed = time.perf_counter() - start

        print_separator()
        print(f"⏱️  All conversations finished in {elapsed:.1f}s (concurrent)")
        print()
        print("Key observations:")
        print("  1. All LLM requests were in-flight at the same time")
        print("  2. Each thread_id has isolated memory in PostgreSQL")
        print("  3. AsyncPostgresSaver handles concurrent checkpoint writes safely")
        print("  4. No extra dependencies — psycopg[binary] supports async natively")
        print()
        print("💡 Run again — conversations persist in the database across restarts")
        print_separator()


if __name__ == "__main__":
    asyncio.run(main())
