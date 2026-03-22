from datetime import datetime
from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def save_note(note: str) -> str:
    """Save a note (simulated)."""
    return f"Note saved: '{note}' at {datetime.now().strftime('%H:%M:%S')}"


ALL_TOOLS = [add, multiply, divide, get_current_time, save_note]
