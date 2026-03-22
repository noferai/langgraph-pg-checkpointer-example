import os
from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from .state import AgentState
from .tools import ALL_TOOLS

load_dotenv()


def get_llm():
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError("LLM_API_KEY not found in environment")

    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    base_url = os.getenv("LLM_BASE_URL")

    if base_url:
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0,
        )
    else:
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=0,
        )


def agent_node(state: AgentState, config: RunnableConfig) -> AgentState:
    llm = get_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    messages = state["messages"]
    if not messages or not any(isinstance(m, SystemMessage) for m in messages):
        system_msg = SystemMessage(
            content=(
                "You are a helpful assistant with access to various tools. "
                "Use the tools when needed to help answer questions. "
                "Be concise and friendly."
            )
        )
        messages = [system_msg] + messages

    response = llm_with_tools.invoke(messages, config)
    current_step = state.get("current_step", 0)

    return {
        "messages": [response],
        "current_step": current_step + 1,
    }


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "__end__"


def create_agent_graph():
    """Create the agent graph with START -> agent -> tools/END loop."""
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(ALL_TOOLS))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "__end__": END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow


def create_agent(checkpointer):
    """Create and compile agent with checkpointer."""
    workflow = create_agent_graph()
    return workflow.compile(checkpointer=checkpointer)
