# LangGraph PostgreSQL Checkpointer Example

Minimal working example of a LangGraph AI agent with PostgreSQL checkpointer, demonstrating all persistence and short-term memory capabilities.

## 🎯 What is this?

This project demonstrates:

- ✅ **Persistent state** via PostgreSQL checkpointer
- ✅ **Short-term memory** within thread scope
- ✅ **Thread isolation** - different `thread_id` for different conversations
- ✅ **Human-in-the-loop** - interrupts for action verification
- ✅ **Tool calling** - agent uses tools (calculator, time)
- ✅ **State graph** with nodes and conditional edges
- ✅ **Custom state** - domain-specific TypedDict with progress tracking
- ✅ **Time travel** - inspect history and resume from any checkpoint

## 📁 Project Structure

```
langgraph-pg-checkpointer-example/
├── src/
│   ├── __init__.py
│   ├── agent.py          # Main agent graph
│   ├── tools.py          # Tools (calculator, time, etc.)
│   ├── state.py          # State definition
│   └── checkpointer.py   # PostgreSQL checkpointer
├── examples/
│   ├── basic_usage.py       # Basic usage
│   ├── memory_demo.py       # Memory and thread isolation demo
│   ├── human_in_loop.py     # Human-in-the-loop with interrupts
│   └── recipe_assistant.py  # Custom state, progress tracking, time travel
├── docker-compose.yml     # PostgreSQL container
├── .env.example          # Configuration example
└── pyproject.toml        # Project dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install dependencies with uv
uv sync
```

### 2. Environment Setup

```bash
# Copy configuration example
cp .env.example .env

# Edit .env and configure your LLM provider:
# - LLM_API_KEY: Your API key
# - LLM_MODEL: Model name (e.g., gpt-4o-mini)
# - LLM_BASE_URL: API endpoint (optional, defaults to OpenAI)
```

### 3. Start PostgreSQL

```bash
# Start PostgreSQL in Docker
docker-compose up -d

# Check status
docker-compose ps
```

### 4. Run Examples

```bash
# Basic usage
python examples/basic_usage.py

# Memory demonstration
python examples/memory_demo.py

# Human-in-the-loop
python examples/human_in_loop.py

# Recipe Assistant (custom state, session resume, time travel)
python examples/recipe_assistant.py
```

## 💡 Usage Examples

### Basic Usage

```python
from src.agent import create_agent
from src.checkpointer import get_checkpointer
from langchain_core.messages import HumanMessage

with get_checkpointer(setup=True) as checkpointer:
    agent = create_agent(checkpointer)

    config = {"configurable": {"thread_id": "my-conversation"}}

    result = agent.invoke(
        {
            "messages": [HumanMessage(content="What is 15 * 7?")],
            "current_step": 0,
        },
        config=config,
    )

    print(result["messages"][-1].content)
```

### Working with Memory

```python
# First conversation
config1 = {"configurable": {"thread_id": "user-alice"}}
agent.invoke({"messages": [HumanMessage("My name is Alice")]}, config1)

# Second conversation (independent)
config2 = {"configurable": {"thread_id": "user-bob"}}
agent.invoke({"messages": [HumanMessage("My name is Bob")]}, config2)

# Return to first conversation - agent remembers context!
agent.invoke({"messages": [HumanMessage("What's my name?")]}, config1)
# Response: "Your name is Alice"
```

### Human-in-the-loop

```python
from src.agent import create_agent_graph
from src.checkpointer import get_checkpointer

with get_checkpointer(setup=True) as checkpointer:
    workflow = create_agent_graph()
    agent = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["tools"],  # Pause before executing tools
    )

    config = {"configurable": {"thread_id": "review-thread"}}

    # First invocation - agent stops before tool execution
    result = agent.invoke(
        {"messages": [HumanMessage("Calculate 100 / 5")]},
        config=config,
    )
    # State: INTERRUPTED

    # Check which tool will be called
    last_msg = result["messages"][-1]
    print(last_msg.tool_calls)  # [{"name": "divide", "args": {"a": 100, "b": 5}}]

    # Approve and continue
    result = agent.invoke(None, config=config)
    # Execution continues
```

### Recipe Assistant

```python
# Session 1: start cooking
with get_checkpointer(setup=True) as checkpointer:
    agent = create_recipe_graph().compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "recipe-demo"}}

    agent.invoke(
        {
            "recipe_name": "Pasta Carbonara",
            "current_step": 1,
            "total_steps": 5,
            "ingredients": [...],
            "steps": [...],
            "completed_steps": [],
            "messages": [HumanMessage("I want to make Pasta Carbonara! Where do I start?")],
        },
        config=config,
    )

    # Step done — advance progress
    agent.update_state(config, {"current_step": 2, "completed_steps": [1]})

# Session 2: restart — resume from the same place
with get_checkpointer() as checkpointer:
    agent = create_recipe_graph().compile(checkpointer=checkpointer)

    saved = agent.get_state(config)
    print(saved.values["current_step"])  # 2 — restored from PostgreSQL

    agent.invoke({"messages": [HumanMessage("Let's continue!")]}, config=config)

# Time Travel: roll back to a previous checkpoint
history = list(agent.get_state_history(config))
past_config = history[-1].config  # earliest checkpoint
agent.invoke({"messages": [HumanMessage("What ingredients do I need?")]}, config=past_config)
```

## 🔧 Architecture

### Agent Graph

```
┌─────────┐
│  START  │
└────┬────┘
     │
     ▼
┌─────────┐
│  agent  │ ◄─────┐
└────┬────┘       │
     │            │
     ▼            │
┌───────────┐     │
│ should_   │     │
│ continue? │     │
└─────┬─────┘     │
      │           │
   ┌──┴──┐        │
   │     │        │
   ▼     ▼        │
 tools  END       │
   │              │
   └──────────────┘
```

### State

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_step: int
```

- `messages`: Message history with automatic appending (reducer: `add_messages`)
- `current_step`: Step counter for tracking progress

### Nodes

1. **agent**: LLM decides next action
2. **tools**: Tool execution (calculator, time, etc.)

### Tools

- `add(a, b)` - addition
- `multiply(a, b)` - multiplication
- `divide(a, b)` - division
- `get_current_time()` - current time
- `save_note(note)` - save note (simulated)

## 🗄️ PostgreSQL Checkpointer

### Capabilities

1. **Persistence**: All state saved in PostgreSQL
2. **Thread isolation**: Each `thread_id` has its own history
3. **Checkpoint history**: Can rollback to previous states
4. **Fault tolerance**: Recovery after failures
5. **Human-in-the-loop**: Interrupt support

### Database Structure

Tables created on initialization:
- `checkpoints` - main table with state (JSONB)
- `checkpoint_blobs` - for large binary data (BYTEA)

### Configuration

```python
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/langgraph
```

## 🎓 Key Concepts

### Short-term Memory

- Memory within a single `thread_id`
- Agent remembers entire conversation context
- Automatic save after each step
- Context loading when continuing conversation

### Thread Isolation

- Different `thread_id` = different conversations
- Complete data isolation
- Perfect for multi-user applications

### Interrupts

- Pause execution at critical points
- `interrupt_before=["tools"]` - before tools
- `interrupt_after=["tools"]` - after tools
- State modification capability

## 🌐 Supported LLM Providers

This project works with any OpenAI-compatible API endpoint:

- **OpenAI** - Direct OpenAI API
- **OpenRouter** - Access to 280+ models from various providers
- **Groq** - Fast inference with open models
- **DeepSeek** - High-performance Chinese LLM
- **vLLM** - Self-hosted inference server
- Any other OpenAI-compatible endpoint

See `.env.example` for configuration examples for each provider.

## 🛠️ Development

### Requirements

- Python >= 3.13
- [uv](https://github.com/astral-sh/uv) package manager
- PostgreSQL (via Docker)
- API key for your chosen LLM provider

## 📝 License

MIT

**Created to demonstrate LangGraph capabilities with PostgreSQL checkpointer**
