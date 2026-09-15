from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic mathematical expression."""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Calculation error: {e}"


@tool
def get_system_status(service: str) -> str:
    """Return the simulated health status of a service."""
    statuses = {
        "api": "healthy, 2 replicas running",
        "worker": "degraded, 1 of 3 replicas unhealthy",
        "database": "healthy, latency 12ms",
    }

    return statuses.get(service.lower(), "service not found")
