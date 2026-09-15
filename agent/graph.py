from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from .state import AgentState
from .retrieval import retrieve
from .tools import calculator, get_system_status
from .llm import GeminiLLM



tools = {
    "calculator": calculator,
    "get_system_status": get_system_status,
}


llm = GeminiLLM()

def retrieve_node(state: AgentState):
    context = retrieve(state["query"])

    state["recorder"].record(
        "retrieval",
        input=state["query"],
        output=context,
        agent="retrieval",
    )

    return {
        "retrieved_context": context
    }

def researcher_node(state: AgentState):
    prompt = f"""
You are the researcher sub-agent.

Answer the user's query.

You have access to these tools:

1. calculator(expression)
2. get_system_status(service)

If a tool is required, respond ONLY with JSON:

{{
    "tool": "calculator",
    "input": {{
        "expression": "847 * 293"
    }}
}}

OR:

{{
    "tool": "get_system_status",
    "input": {{
        "service": "api"
    }}
}}

If no tool is required, respond normally.

User query:
{state["query"]}

Retrieved context:
{state["retrieved_context"]}

Previous tool result:
{state["tool_result"]}
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    content = response.content

    if isinstance(content, list):
        content = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )

    state["recorder"].record(
        "llm_call",
        agent="researcher",
        input=prompt,
        output=content,
    )

    return {
        "research": content
    }


def tool_node(state: AgentState):
    research = state["research"]

    # Simple parser for our demo tool protocol.
    import json

    try:
        data = json.loads(research)
    except json.JSONDecodeError:
        return {
            "tool_result": ""
        }

    tool_name = data.get("tool")
    tool_input = data.get("input", {})

    if tool_name not in tools:
        return {
            "tool_result": f"Unknown tool: {tool_name}"
        }

    try:
        state["recorder"].record(
            "tool_call",
            agent="researcher",
            tool=tool_name,
            input=tool_input,
        )

        result = tools[tool_name].invoke(tool_input)

        state["recorder"].record(
            "tool_result",
            agent="researcher",
            tool=tool_name,
            output=str(result),
        )

        return {
            "tool_result": str(result)
        }

    except Exception as e:
        return {
            "tool_result": f"Tool error: {e}"
        }


def route_after_research(state: AgentState):
    import json

    try:
        data = json.loads(state["research"])
    except json.JSONDecodeError:
        return "reviewer"

    if data.get("tool") in tools:
        return "tools"

    return "reviewer"

def reviewer_node(state: AgentState):
    prompt = f"""
You are the reviewer sub-agent.

Review the research and tool result for correctness.

User query:
{state["query"]}

Research:
{state["research"]}

Tool result:
{state["tool_result"]}

Retrieved context:
{state["retrieved_context"]}
"""

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):
        content = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )

    state["recorder"].record(
        "llm_call",
        agent="reviewer",
        input=prompt,
        output=content,
    )

    return {
        "review": content
    }

def answer_node(state: AgentState):
    prompt = f"""
Produce the final answer to the user's query.

Query:
{state["query"]}

Research:
{state["research"]}

Tool result:
{state["tool_result"]}

Review:
{state["review"]}
"""

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):
        content = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )

    state["recorder"].record(
        "llm_call",
        agent="answer",
        input=prompt,
        output=content,
    )

    return {
        "answer": content
    }

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieval", retrieve_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("tools", tool_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("answer", answer_node)

    graph.add_edge(START, "retrieval")
    graph.add_edge("retrieval", "researcher")

    graph.add_conditional_edges(
        "researcher",
        route_after_research,
        {
            "tools": "tools",
            "reviewer": "reviewer",
        },
    )

    graph.add_edge("tools", "researcher")
    graph.add_edge("reviewer", "answer")
    graph.add_edge("answer", END)

    return graph.compile()
