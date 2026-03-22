#!/usr/bin/env python3
"""Recipe Assistant — step-by-step cooking with progress saved in PostgreSQL.

Demonstrates:
  - Custom State       : RecipeState with recipe fields and progress tracking
  - Progress Tracking  : current step persisted between invocations
  - Session Resume     : continue cooking after application restart
  - State Inspection   : agent.get_state() to read saved progress
  - update_state()     : explicit checkpoint update to advance progress
  - Time Travel        : roll back to previous checkpoints via get_state_history()
"""

import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from src.checkpointer import get_checkpointer

load_dotenv()

# ---------------------------------------------------------------------------
# Recipe data
# ---------------------------------------------------------------------------

CARBONARA = {
    "name": "Pasta Carbonara",
    "ingredients": [
        "spaghetti 400g",
        "bacon/pancetta 200g",
        "eggs 4",
        "parmesan 100g",
        "garlic 2 cloves",
        "black pepper",
        "salt",
    ],
    "steps": [
        "Bring water to a boil — large pot 4–5 litres, add plenty of salt",
        "Cook spaghetti 8–10 minutes until al dente",
        "Dice bacon and fry in a dry pan until crispy (5–7 min)",
        "Whisk eggs with grated parmesan and freshly ground black pepper in a bowl",
        "Drain spaghetti, reserving ½ cup of pasta water; toss with bacon off the heat, "
        "add the egg-cheese mixture and stir until creamy",
    ],
}

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class RecipeState(TypedDict):
    """State for the recipe assistant agent."""

    recipe_name: str
    current_step: int  # 1..N — active step; 0 — recipe not started
    total_steps: int
    ingredients: list[str]
    steps: list[str]
    completed_steps: list[int]
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


def get_llm() -> ChatOpenAI:
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError("LLM_API_KEY not found in environment")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    base_url = os.getenv("LLM_BASE_URL")
    kwargs: dict = dict(model=model, api_key=api_key, temperature=0)
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------


def recipe_agent_node(state: RecipeState, config: RunnableConfig) -> dict:
    llm = get_llm()

    recipe_name = state.get("recipe_name", "")
    current_step = state.get("current_step", 0)
    total_steps = state.get("total_steps", 0)
    steps = state.get("steps", [])
    completed = state.get("completed_steps", [])
    ingredients = state.get("ingredients", [])

    steps_text = "\n".join(
        f"  {'✅' if (i + 1) in completed else '🔲'} Step {i + 1}/{total_steps}: {s}" for i, s in enumerate(steps)
    )
    progress = f"Current active step: {current_step}/{total_steps}" if current_step > 0 else "Recipe not started yet"
    completed_text = f"Completed steps: {completed}" if completed else "No steps completed yet"

    system_content = f"""You are a friendly cooking assistant helping the user prepare "{recipe_name}".

CURRENT PROGRESS:
{progress}
{completed_text}

INGREDIENTS: {", ".join(ingredients)}

RECIPE STEPS:
{steps_text}

INSTRUCTIONS:
- Guide the user step by step: present the active step and wait for confirmation
- When the user says "done" / "ready" / "yes" — congratulate them and confirm the step is complete
- Answer questions about ingredients or previous steps in detail
- If all steps are done — congratulate the user on finishing the dish!
- Be concise and friendly"""

    messages = state["messages"]
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]
    response = llm.invoke([SystemMessage(content=system_content)] + non_system, config)

    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def create_recipe_graph() -> StateGraph:
    workflow = StateGraph(RecipeState)
    workflow.add_node("agent", recipe_agent_node)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)
    return workflow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def print_separator(char: str = "=") -> None:
    print("\n" + char * 70 + "\n")


def say(role: str, text: str) -> None:
    print(f"{role} {text}")
    print()


# ---------------------------------------------------------------------------
# Session 1: start cooking
# ---------------------------------------------------------------------------

THREAD_ID = "recipe-carbonara-demo"


def session_1() -> None:
    """Session 1: launch the app, start the recipe, complete step 1."""
    print("📱 SESSION 1 (10:00)  — launching app, starting to cook")
    print_separator()

    with get_checkpointer(setup=True) as checkpointer:
        agent = create_recipe_graph().compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": THREAD_ID}}

        # First request: initialise the full recipe state
        result = agent.invoke(
            {
                "recipe_name": CARBONARA["name"],
                "current_step": 1,
                "total_steps": len(CARBONARA["steps"]),
                "ingredients": CARBONARA["ingredients"],
                "steps": CARBONARA["steps"],
                "completed_steps": [],
                "messages": [HumanMessage(content="I want to make Pasta Carbonara! Where do I start?")],
            },
            config=config,
        )
        say("👤", "I want to make Pasta Carbonara! Where do I start?")
        say("🤖", result["messages"][-1].content)
        print_separator("-")

        # User confirms step 1
        result = agent.invoke(
            {"messages": [HumanMessage(content="Water is boiling and salted, done!")]},
            config=config,
        )
        say("👤", "Water is boiling and salted, done!")
        say("🤖", result["messages"][-1].content)

        # Advance progress: step 1 complete, move to step 2
        agent.update_state(config, {"current_step": 2, "completed_steps": [1]})
        print("✅ update_state() → current_step=2, completed_steps=[1]")
        print("   Progress saved to PostgreSQL")

    print()
    print("📴 User closes the application...")
    print("   [APP STOPPED — PostgreSQL retains all progress]")


# ---------------------------------------------------------------------------
# Session 2: restart, resume
# ---------------------------------------------------------------------------


def session_2() -> None:
    """Session 2: app restarted, resume from the saved checkpoint."""
    print_separator()
    print("📱 SESSION 2 (10:15) — restarting the application")
    print("   [NEW PROCESS — state restored from PostgreSQL]")
    print_separator()

    with get_checkpointer() as checkpointer:
        agent = create_recipe_graph().compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": THREAD_ID}}

        # Inspect the saved state
        saved = agent.get_state(config)
        step = saved.values.get("current_step", 0)
        total = saved.values.get("total_steps", 0)
        completed = saved.values.get("completed_steps", [])
        recipe = saved.values.get("recipe_name", "")
        msg_count = len(saved.values.get("messages", []))

        print("📊 get_state() — restored from PostgreSQL:")
        print(f"   Recipe:          {recipe}")
        print(f"   Current step:    {step}/{total}")
        print(f"   Completed steps: {completed}")
        print(f"   Messages so far: {msg_count}")
        print_separator("-")

        # Resume from where we left off
        result = agent.invoke(
            {"messages": [HumanMessage(content="Let's continue!")]},
            config=config,
        )
        say("👤", "Let's continue!")
        say("🤖", result["messages"][-1].content)
        print_separator("-")

        result = agent.invoke(
            {"messages": [HumanMessage(content="Spaghetti is ready, what's next?")]},
            config=config,
        )
        say("👤", "Spaghetti is ready, what's next?")
        say("🤖", result["messages"][-1].content)

        agent.update_state(config, {"current_step": 3, "completed_steps": [1, 2]})
        print("✅ update_state() → current_step=3, completed_steps=[1, 2]")
        print_separator("-")

        # Off-step question — agent answers using full recipe context
        result = agent.invoke(
            {"messages": [HumanMessage(content="Wait, remind me — how should I handle the eggs?")]},
            config=config,
        )
        say("👤", "Wait, remind me — how should I handle the eggs?")
        say("🤖", result["messages"][-1].content)


# ---------------------------------------------------------------------------
# Time Travel: checkpoint history and rollback
# ---------------------------------------------------------------------------


def time_travel_demo() -> None:
    """Time Travel: inspect checkpoint history and resume from step 1."""
    print_separator()
    print("⏰ TIME TRAVEL — checkpoint history and rollback")
    print_separator()

    with get_checkpointer() as checkpointer:
        agent = create_recipe_graph().compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": THREAD_ID}}

        history = list(agent.get_state_history(config))
        print(f"📜 get_state_history() → {len(history)} checkpoints:\n")

        for i, snap in enumerate(history[:6]):
            step = snap.values.get("current_step", "?")
            done = snap.values.get("completed_steps", [])
            msgs = len(snap.values.get("messages", []))
            cid = snap.config["configurable"].get("checkpoint_id", "")[:12]
            print(f"  [{i + 1}] checkpoint={cid}…  step={step}  completed={done}  messages={msgs}")

        # Find the earliest checkpoint at step 1 — the very beginning of the recipe
        target = next(
            (s for s in reversed(history) if s.values.get("current_step") == 1),
            None,
        )

        if target:
            t_cid = target.config["configurable"].get("checkpoint_id", "")[:12]
            print(f"\n🔍 Found recipe start checkpoint: checkpoint={t_cid}…")
            print("   Rolling back and asking a question from that point in time...")
            print_separator("-")

            # Resume execution from the historical checkpoint
            past_config = target.config
            result = agent.invoke(
                {"messages": [HumanMessage(content="What ingredients do I need?")]},
                config=past_config,
            )
            say("👤", "What ingredients do I need?")
            say("🤖", result["messages"][-1].content)

            print("💡 The agent answers in the context of the recipe START (step 1),")
            print("   unaware of any steps completed after this checkpoint.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("🍝  Recipe Assistant — LangGraph PostgreSQL Checkpointer Demo")
    print("=" * 70)

    session_1()
    session_2()
    time_travel_demo()

    print_separator()
    print("✨ Recipe Assistant Demo Complete!")
    print()
    print("LangGraph + PostgreSQL capabilities demonstrated:")
    print("  1. Custom State      — RecipeState with recipe fields and progress")
    print("  2. Progress Tracking — current_step / completed_steps across invocations")
    print("  3. Session Resume    — continue after restart via a new with-block")
    print("  4. State Inspection  — agent.get_state() to read saved state from DB")
    print("  5. update_state()    — explicit checkpoint update to advance progress")
    print("  6. Time Travel       — get_state_history() + resume from a past checkpoint")
    print()
    print("Key API:")
    print("  agent.invoke(state, config)           — invoke the agent")
    print("  agent.get_state(config)               — current state from DB")
    print("  agent.update_state(config, values)    — update the checkpoint")
    print("  agent.get_state_history(config)       — full checkpoint history")
    print_separator()


if __name__ == "__main__":
    main()
