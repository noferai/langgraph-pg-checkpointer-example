#!/usr/bin/env python3
"""Human-in-the-loop demonstration with interrupts."""

from langchain_core.messages import HumanMessage

from src.agent import create_agent_graph
from src.checkpointer import get_checkpointer


def print_separator(char="="):
    print("\n" + char * 70 + "\n")


def main():
    print("👤 LangGraph Agent - Human-in-the-Loop Demo")
    print("Demonstrating interrupts for human oversight")
    print_separator()

    with get_checkpointer(setup=True) as checkpointer:
        workflow = create_agent_graph()
        agent = workflow.compile(
            checkpointer=checkpointer,
            interrupt_before=["tools"],
        )

        config = {"configurable": {"thread_id": "hitl-demo"}}

        print("👤 User requests: 'Calculate 999 divided by 3'")
        print_separator()

        result = agent.invoke(
            {
                "messages": [HumanMessage(content="Calculate 999 divided by 3")],
                "current_step": 0,
            },
            config=config,
        )

        print("⏸️  Agent PAUSED before executing tools!")
        print("\n📋 Current state:")
        last_message = result["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print(f"   Tool to be called: {last_message.tool_calls[0]['name']}")
            print(f"   Arguments: {last_message.tool_calls[0]['args']}")

        print("\n💭 Human review: This looks safe to execute.")
        print("   Approving tool execution...")
        print_separator()

        result = agent.invoke(None, config=config)

        print("▶️  Execution resumed!")
        last_message = result["messages"][-1]
        print(f"🤖 Final answer: {last_message.content}")
        print_separator()

        print("👤 User requests: 'Multiply 50 by 20'")
        print_separator()

        result = agent.invoke(
            {
                "messages": [HumanMessage(content="Multiply 50 by 20")],
                "current_step": result["current_step"],
            },
            config=config,
        )

        print("⏸️  Agent PAUSED before executing tools!")
        print("\n📋 Current state:")
        last_message = result["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print(f"   Tool to be called: {last_message.tool_calls[0]['name']}")
            print(f"   Arguments: {last_message.tool_calls[0]['args']}")

        print("\n💭 Human review: Approved, continuing...")
        print_separator()

        result = agent.invoke(None, config=config)
        last_message = result["messages"][-1]
        print(f"🤖 Final answer: {last_message.content}")
        print_separator()

        print("✨ Human-in-the-Loop Demo Complete!")
        print()
        print("Key capabilities:")
        print("  1. Interrupts - Agent pauses before executing tools")
        print("  2. Review - Human can inspect planned actions")
        print("  3. Approval - Continue execution by invoking with None")
        print("  4. State persistence - All state saved in PostgreSQL")
        print()
        print("Advanced use cases:")
        print("  - Use interrupt_after=['tools'] to review tool results")
        print("  - Modify state before resuming to change behavior")
        print("  - Implement approval workflows for sensitive operations")
        print_separator()


if __name__ == "__main__":
    main()
